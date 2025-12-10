import torch
import torch.nn as nn
import torchsde

class FictitiousControlledSDE(nn.Module):
    """
    Wrapper for Fictitious Play. 
    Allows setting an 'active_agent'; all other agents are treated as 
    part of the fixed environment (detached).
    """

    def __init__(
        self,
        score_model,
        optimality_criterion,
        control_agents,
        aggregator,
        sde,
        agent_keys,
        optimality_target,
        image_shape,
    ):
        super().__init__()
        self.score_model = score_model
        self.optimality_criterion = optimality_criterion
        self.aggregator = aggregator
        self.sde = sde
        self.optimality_target = optimality_target
        self.agent_keys = list(agent_keys)
        
        self.C, self.H, self.W = image_shape
        self.state_dim_per_agent = self.C * self.H * self.W 
        self.total_dim = len(self.agent_keys) * self.state_dim_per_agent + 2

        self.control_agents = nn.ModuleDict({str(k): m for k, m in control_agents.items()})
        
        # Configuration for Stability
        self.noise_type = "diagonal"
        self.sde_type = "stratonovich" 
        
        # Mutable state for Fictitious Play
        self.active_agent_key = None 

    def set_active_agent(self, key):
        """Sets which agent is currently learning. Others are frozen."""
        if key is not None and key not in self.agent_keys:
            raise ValueError(f"Agent {key} not found.")
        self.active_agent_key = key

    def _unpack_state(self, y):
        B = y.shape[0]
        states = {}
        offset = 0
        for key in self.agent_keys:
            flat = y[:, offset : offset + self.state_dim_per_agent]
            states[key] = flat.view(B, self.C, self.H, self.W)
            offset += self.state_dim_per_agent
        c_ctrl = y[:, offset : offset + 1]
        c_opt  = y[:, offset + 1 : offset + 2]
        return states, c_ctrl, c_opt

    def _pack_state(self, states, c_ctrl, c_opt):
        B = c_ctrl.shape[0]
        flats = []
        for key in self.agent_keys:
            flats.append(states[key].view(B, -1))
        flats.append(c_ctrl.view(B, 1))
        flats.append(c_opt.view(B, 1))
        return torch.cat(flats, dim=1)

    def f(self, t, y):
        B = y.shape[0]

        t_phys = 1.0 - t
        t_phys = torch.clamp(t_phys, min=1e-5)
        batch_time = torch.full((B,), t_phys, device=y.device)

        states, _, _ = self._unpack_state(y)
        
        g = self.sde.diffusion_coeff(batch_time)
        g_sq = (g ** 2).view(B, 1, 1, 1)
        
        Y_t = self.aggregator([states[k] for k in self.agent_keys])

        controls, scores, x0_hats = {}, {}, {}
        
        # --- 1. Compute Controls (The Fictitious Logic) ---
        for key in self.agent_keys:
            ctrl_input = torch.cat([states[key], Y_t], dim=1)
            
            if key == self.active_agent_key:
                u = self.control_agents[str(key)](ctrl_input, batch_time)
            else:
                with torch.no_grad():
                    u = self.control_agents[str(key)](ctrl_input, batch_time).detach()

            controls[key] = u.clamp(-50.0, 50.0)
            s_k = self.score_model(states[key], batch_time).clamp(-100.0, 100.0)
            scores[key] = s_k

            # Tweedie
            current_std = self.sde.marginal_prob_std(batch_time).view(B, 1, 1, 1)
            x0_hats[key] = states[key] + (current_std ** 2) * s_k

        # --- 2. Running Costs ---
        Y_0_hat = self.aggregator([x0_hats[k] for k in self.agent_keys])
        dc_opt = self.optimality_criterion.get_running_optimality_loss(
            Y_0_hat, self.optimality_target
        )
        if dc_opt.numel() == 1: dc_opt = dc_opt.expand(B)
        dc_opt = dc_opt.view(B, 1)
     
        active_u = controls[self.active_agent_key].view(B, -1)
        dc_ctrl = active_u.pow(2).mean(dim=1, keepdim=True)

        # --- 3. Drift ---
        drift_states = {}
        for key in self.agent_keys:
            drift_states[key] = g_sq * scores[key] + controls[key]

        return self._pack_state(drift_states, dc_ctrl, dc_opt)

    def g(self, t, y):
        B = y.shape[0]
        t_phys = 1.0 - t
        batch_time = torch.full((B,), t_phys, device=y.device)
        g_val = self.sde.diffusion_coeff(batch_time)
        
        g_list = []
        for _ in self.agent_keys:
            g_list.append(g_val.view(B, 1).expand(B, self.state_dim_per_agent))
        g_list.append(torch.zeros(B, 2, device=y.device))
        return torch.cat(g_list, dim=1)


def fictitious_train_control_adjoint(
    score_model,
    optimality_criterion,
    control_agents,
    aggregator,
    sde,
    optimality_target,
    num_steps,
    batch_size,
    device,
    eps,
    lambda_reg,
    inner_iters,
    running_optimality_reg,
    learning_rate,
    image_dim,
    debug=False,
):
    score_model.eval()
    optimality_criterion.eval()
    agent_keys = sorted(control_agents.keys())
    
    ts = torch.linspace(0.0, 1.0 - eps, num_steps, device=device)
    dt = (ts[1] - ts[0]).item()
    
    sde_ctrl = FictitiousControlledSDE(
        score_model, optimality_criterion, control_agents, aggregator,
        sde, agent_keys, optimality_target, image_dim
    ).to(device)

    loss_dict = {}
    info_dict = {}

    # --- Round Robin Loop ---
    for player_key in agent_keys:
        
        # 1. Manage Modes & Optimizer Persistence
        # We attach the optimizer to the agent object to prevent resetting 
        # momentum if this function is called repeatedly in a loop.
        if not hasattr(control_agents[player_key], '_optimizer'):
             control_agents[player_key]._optimizer = torch.optim.Adam(
                 control_agents[player_key].parameters(), lr=learning_rate
             )
        optimizer = control_agents[player_key]._optimizer

        # Set active/eval modes
        for key in agent_keys:
            if key == player_key:
                control_agents[key].train()
            else:
                control_agents[key].eval()
        
        sde_ctrl.set_active_agent(player_key)
        
        loss_sum = 0.0
        
        # --- Inner Optimization Loop ---
        for _ in range(inner_iters):
            optimizer.zero_grad()
            
            # Initial State
            initial_std = sde.marginal_prob_std(torch.tensor([1.0], device=device))
            x0_dict = {
                k: torch.randn(batch_size, *image_dim, device=device) * initial_std
                for k in agent_keys
            }
            
            # Pack y0
            flats = [x0_dict[k].view(batch_size, -1) for k in agent_keys]
            flats.append(torch.zeros(batch_size, 1, device=device)) 
            flats.append(torch.zeros(batch_size, 1, device=device))
            y0 = torch.cat(flats, dim=1)

            # Integration (Midpoint is safer than Euler, faster than SRK)
            ys = torchsde.sdeint_adjoint(
                sde_ctrl, y0, ts, method="midpoint", dt=dt, options={'step_size': dt}
            )

            # Losses
            y_T = ys[-1]
            states_final, c_ctrl_final, c_opt_final = sde_ctrl._unpack_state(y_T)
            Y_final = aggregator([states_final[k] for k in agent_keys])
            term_loss = optimality_criterion.get_terminal_optimality_loss(
                Y_final, optimality_target
            )

            total_loss = (
                lambda_reg * c_ctrl_final.mean()
                + running_optimality_reg * c_opt_final.mean()
                + term_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(control_agents[player_key].parameters(), 1.0)
            optimizer.step()
            
            loss_sum += total_loss.item()
            
            if debug:
                info_dict[player_key] = {
                    'loss': total_loss.item(),
                    'term': term_loss.item(),
                    'ctrl': c_ctrl_final.mean().item()
                }

        loss_dict[player_key] = loss_sum / inner_iters

    return loss_dict, info_dict