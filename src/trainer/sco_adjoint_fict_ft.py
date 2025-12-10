import torch
import torch.nn as nn
import torchsde

class MultiAgentControlledSDE(nn.Module):
    def __init__(self, score_model, optimality_criterion, control_agents,
                 aggregator, sde, agent_keys, optimality_target, image_shape=(1,28,28)):
        super().__init__()
        self.score_model = score_model
        self.optimality_criterion = optimality_criterion
        self.aggregator = aggregator
        self.sde = sde
        self.optimality_target = optimality_target
        self.agent_keys = list(agent_keys)
        self.num_agents = len(self.agent_keys)

        self.C, self.H, self.W = image_shape
        self.state_dim_per_agent = self.C * self.H * self.W
        self.total_dim = self.num_agents * self.state_dim_per_agent + 2

        self.control_agents = nn.ModuleDict({str(k): m for k, m in control_agents.items()})
        self.noise_type = "diagonal"        
        self.sde_type = "stratonovich" 
        self.active_agent_key = None

    def set_active_agent(self, key):
        self.active_agent_key = key

    def _unpack_state(self, y):
        B, D = y.shape
        states = {}
        offset = 0
        for k in self.agent_keys:
            flat = y[:, offset:offset + self.state_dim_per_agent]
            states[k] = flat.view(B, self.C, self.H, self.W)
            offset += self.state_dim_per_agent
        return states, y[:, offset:offset+1], y[:, offset+1:offset+2]

    def _pack_state(self, states, c_ctrl, c_opt):
        B = next(iter(states.values())).shape[0]
        out = []
        for k in self.agent_keys:
            x = states[k].view(B, self.state_dim_per_agent)
            out.append(x)
        out.append(c_ctrl)
        out.append(c_opt)
        return torch.cat(out, dim=1)

    def f(self, t, y):
        B = y.shape[0]
        t_phys = 1.0 - t
        t_phys = torch.clamp(t_phys, min=1e-5)
        batch_time = torch.full((B,), t_phys, device=y.device, dtype=y.dtype)

        states, _, _ = self._unpack_state(y)

        g = self.sde.diffusion_coeff(batch_time)
        g_sq = (g ** 2).view(B, 1, 1, 1)

        Y_t = self.aggregator([states[k] for k in self.agent_keys])
        current_std = self.sde.marginal_prob_std(batch_time).view(B, 1, 1, 1)

        controls, scores, x0_hats = {}, {}, {}
        for k in self.agent_keys:
            x = states[k]
            ctrl_in = torch.cat([x, Y_t], dim=1)            
            if k == self.active_agent_key:
                u = self.control_agents[str(k)](ctrl_in, batch_time)
            else:
                with torch.no_grad():
                    u = self.control_agents[str(k)](ctrl_in, batch_time).detach()
            
            u = u.clamp(-50.0, 50.0)
            s = self.score_model(x, batch_time).clamp(-100.0, 100.0)

            controls[k] = u
            scores[k] = s
            x0_hats[k] = x + (current_std ** 2) * s

        Y0_hat = self.aggregator([x0_hats[k] for k in self.agent_keys])
        run_vals = self.optimality_criterion.get_running_optimality_loss(
            Y0_hat, self.optimality_target
        )

        if run_vals.numel() == 1: run_vals = run_vals.expand(B)
        dc_opt = run_vals.view(B, 1)

        active_u = controls[self.active_agent_key].view(B, -1)
        dc_ctrl = active_u.pow(2).mean(dim=1, keepdim=True)

        drift_states = {k: g_sq * scores[k] + controls[k] for k in self.agent_keys}

        return self._pack_state(drift_states, dc_ctrl, dc_opt)

    def g(self, t, y):
        B = y.shape[0]
        t_phys = 1.0 - t
        batch_time = torch.full((B,), t_phys, device=y.device, dtype=y.dtype)
        g_base = self.sde.diffusion_coeff(batch_time)

        chunks = []
        for _ in self.agent_keys:
            chunks.append(g_base[:, None].expand(B, self.state_dim_per_agent))
        chunks.append(torch.zeros(B, 2, device=y.device, dtype=y.dtype))
        return torch.cat(chunks, dim=1)


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

    sde_ctrl = MultiAgentControlledSDE(
        score_model, optimality_criterion, control_agents,
        aggregator, sde, agent_keys, optimality_target,
        image_shape=image_dim
    ).to(device)

    loss_dict, info_dict = {}, {}

    for active_key in agent_keys:

        # 1. Optimizer Setup
        if not hasattr(control_agents[active_key], '_optimizer'):
            control_agents[active_key]._optimizer = torch.optim.Adam(
                control_agents[active_key].parameters(), lr=learning_rate)
            # EMA Setup
            control_agents[active_key]._ema_params = [
                p.detach().clone() for p in control_agents[active_key].parameters()
            ]

        optimizer = control_agents[active_key]._optimizer
        ema_params = control_agents[active_key]._ema_params

        # 2. Modes: Train Active, Eval Others
        for k in agent_keys:
            if k == active_key:
                control_agents[k].train()
            else:
                control_agents[k].eval()
        
        sde_ctrl.set_active_agent(active_key)

        loss_acc = 0.
        for _ in range(inner_iters):

            backup = {}
            for k in agent_keys:
                if k != active_key and hasattr(control_agents[k], '_ema_params'):
                    backup[k] = [p.clone() for p in control_agents[k].parameters()]
                    with torch.no_grad():
                        for p, ep in zip(control_agents[k].parameters(),
                                         control_agents[k]._ema_params):
                            p.copy_(ep)

            # 4. Simulation
            # Re-init state (random noise)
            std = sde.marginal_prob_std(torch.tensor([1.0], device=device))
            x0 = {k: torch.randn(batch_size, *image_dim, device=device) * std
                  for k in agent_keys}

            y0 = torch.cat(
                [x0[k].view(batch_size, -1) for k in agent_keys] +
                [torch.zeros(batch_size, 1, device=device)] +
                [torch.zeros(batch_size, 1, device=device)],
                dim=1
            )

            # Integration
            ys = torchsde.sdeint_adjoint(
                sde_ctrl, y0, ts,
                method="midpoint", 
                dt=dt,
                options={'step_size': dt}
            )

            y_T = ys[-1]
            states_f, c_ctrl_f, c_opt_f = sde_ctrl._unpack_state(y_T)
            Y_final = aggregator([states_f[k] for k in agent_keys])

            term_loss = optimality_criterion.get_terminal_optimality_loss(
                Y_final, optimality_target)

            ctrl_loss = c_ctrl_f.mean() 
            run_loss = c_opt_f.mean()

            total_loss = (
                lambda_reg * ctrl_loss +
                running_optimality_reg * run_loss +
                term_loss
            )

            # 5. Update
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(control_agents[active_key].parameters(), 1.0)
            optimizer.step()

            # 6. Update Active Agent's EMA
            beta = 0.995
            with torch.no_grad():
                for p, ep in zip(control_agents[active_key].parameters(), ema_params):
                    ep.mul_(beta).add_(p, alpha=1. - beta)

            # 7. Restore Opponent Weights (so they can be trained in their turn)
            for k, vals in backup.items():
                with torch.no_grad():
                    for p, v in zip(control_agents[k].parameters(), vals):
                        p.copy_(v)

            loss_acc += total_loss.item()
            
            if debug:
                info_dict[active_key] = {'loss': total_loss.item(), 'ctrl': ctrl_loss.item()}

        loss_dict[active_key] = loss_acc / inner_iters

    return loss_dict, info_dict