import torch
import torch.nn as nn
import torchsde

import torch
import torch.nn as nn
import torchsde


import torch
import torch.nn as nn
import torchsde

import torch
import torch.nn as nn
import torchsde


class MultiAgentControlledSDE(nn.Module):
    """
    Controlled reverse-time SDE for multiple agents, with an adjoint-friendly
    flat state y ∈ ℝ^{B×D}.

    Layout of the last dimension:

        y = [ x_0_flat, x_1_flat, ..., x_{N-1}_flat, c_ctrl, c_opt ]

    where:
        - x_i_flat: flattened image state of agent i (size = C * H * W)
        - c_ctrl: cumulative control energy ∫ Σ_i ||u_i||² dt  (per sample)
        - c_opt:  cumulative running optimality ∫ c(Ŷ_0(t), t) dt (per sample)
    """

    def __init__(
        self,
        score_model,
        optimality_criterion,
        control_agents,       # dict, can have int keys
        aggregator,
        sde,
        agent_keys,           # list / sorted keys of agents
        optimality_target,
        image_shape=(1, 28, 28),
    ):
        super().__init__()

        self.score_model = score_model
        self.optimality_criterion = optimality_criterion
        self.aggregator = aggregator
        self.sde = sde
        self.optimality_target = optimality_target

        # Agent indexing and dimensions
        self.agent_keys = list(agent_keys)
        self.num_agents = len(self.agent_keys)

        self.C, self.H, self.W = image_shape
        self.state_dim_per_agent = self.C * self.H * self.W 
        self.total_dim = self.num_agents * self.state_dim_per_agent + 2

        self.control_agents = nn.ModuleDict(
            {str(k): m for k, m in control_agents.items()}
        )
        self.noise_type = "diagonal"
        self.sde_type = "ito"


    def _unpack_state(self, y):
        """
        y: [B, D]
        returns:
            states: dict key -> [B, 1, H, W]
            c_ctrl: [B, 1]
            c_opt : [B, 1]
        """
        B, D = y.shape
        assert D == self.total_dim, f"Expected D={self.total_dim}, got {D}"

        states = {}
        offset = 0
        for key in self.agent_keys:
            flat = y[:, offset:offset + self.state_dim_per_agent]  # [B, M]
            x = flat.view(B, self.C, self.H, self.W)               # [B,1,H,W]
            states[key] = x
            offset += self.state_dim_per_agent

        c_ctrl = y[:, offset:offset + 1]       # [B, 1]
        c_opt  = y[:, offset + 1:offset + 2]   # [B, 1]

        return states, c_ctrl, c_opt

    def _pack_state(self, states, c_ctrl, c_opt):
        """
        states: dict key -> [B, 1, H, W]
        c_ctrl, c_opt: [B, 1]
        returns:
            y: [B, D]
        """
        B = next(iter(states.values())).shape[0]
        flats = []

        for key in self.agent_keys:
            x = states[key]                    # [B,1,H,W]
            flats.append(x.view(B, self.state_dim_per_agent))  # [B,M]

        flats.append(c_ctrl.view(B, 1))        # [B,1]
        flats.append(c_opt.view(B, 1))         # [B,1]

        return torch.cat(flats, dim=1)         # [B, D]

    # ---------------------------------------------------------
    # Drift f(t, y) and diffusion g(t, y)
    # ---------------------------------------------------------

    def f(self, t, y):
        """
        Reverse-time drift + cost ODEs.

        Input:
            t: scalar solver time s ∈ [0, 1 - eps]
            y: [B, D]
        Output:
            drift: [B, D]
        """
        B, D = y.shape
        device = y.device
        dtype = y.dtype

        # Map solver-time s to physical diffusion time t_phys ∈ [eps, 1]
        t_phys = 1.0 - t
        batch_time = torch.full((B,), t_phys, device=device, dtype=dtype)

        states, c_ctrl, c_opt = self._unpack_state(y)

        # Diffusion coefficient and reverse-time drift factor
        g = self.sde.diffusion_coeff(batch_time)          # [B]
        g_sq = (g ** 2)[:, None, None, None]              # [B,1,1,1]

        # Aggregate current agent states
        Y_t = self.aggregator([states[k] for k in self.agent_keys])

        controls = {}
        scores = {}
        x0_hats = {}

        # Controls + scores for each agent
        for key in self.agent_keys:
            x_k = states[key]                             # [B,1,H,W]
            ctrl_input = torch.cat([x_k, Y_t], dim=1)     # concat over channels
            ctrl_net = self.control_agents[str(key)]
            controls[key] = ctrl_net(ctrl_input, batch_time)   # [B,1,H,W]
            scores[key] = self.score_model(x_k, batch_time)    # [B,1,H,W]

        # Tweedie estimator: x0_hat = x_t + σ_t^2 * score
        current_std = self.sde.marginal_prob_std(batch_time)[:, None, None, None]
        for key in self.agent_keys:
            x0_hats[key] = states[key] + (current_std ** 2) * scores[key]

        # Running optimality integrand.
        Y_0_hat = self.aggregator([x0_hats[k] for k in self.agent_keys])
        running_vals = self.optimality_criterion.get_running_optimality_loss(
            Y_0_hat, self.optimality_target
        )

        drift_states = {}
        for key in self.agent_keys:
            drift_states[key] = g_sq * scores[key] + controls[key]

        # ---- Drift for cumulative control cost c_ctrl ----
        # per-sample control energy: Σ_i E[||u_i||²] over image dims
        control_energy = 0.0
        for key in self.agent_keys:
            u = controls[key]  # [B,1,H,W]
            control_energy = control_energy + u.pow(2).mean(dim=(1, 2, 3), keepdim=True)  # [B,1]

        dc_ctrl_dt = control_energy                         # [B,1]

        # ---- Drift for running optimality cost c_opt ----
        # Force shape [B,1] no matter what
        if running_vals.dim() == 0:
            # scalar -> broadcast to all samples
            running_vals = running_vals.expand(B, 1)             # [B,1]
        elif running_vals.dim() == 1:
            # [B] -> [B,1]
            assert running_vals.shape[0] == B, f"running_vals has wrong batch dim: {running_vals.shape}"
            running_vals = running_vals.view(B, 1)               # [B,1]
        elif running_vals.dim() == 2:
            # [B,1] already
            assert running_vals.shape == (B, 1), f"running_vals must be [B,1], got {running_vals.shape}"
        else:
            raise RuntimeError(f"Unexpected running_vals shape: {running_vals.shape}")

        dc_opt_dt = running_vals   # [B,1]

        # Pack back to flat drift
        drift = self._pack_state(drift_states, dc_ctrl_dt, dc_opt_dt)  # [B,D]
        return drift

    def g(self, t, y):
        """
        Diagonal diffusion on agent coordinates, zero on costs.

        Input:
            t: scalar solver time s
            y: [B, D]
        Output:
            diffusion: [B, D]
        """
        B, D = y.shape
        device = y.device
        dtype = y.dtype

        # Map solver-time s to physical time t_phys
        t_phys = 1.0 - t
        batch_time = torch.full((B,), t_phys, device=device, dtype=dtype)

        g_base = self.sde.diffusion_coeff(batch_time)     # [B]

        g_vec_list = []
        for _ in self.agent_keys:
            # same magnitude for each pixel coord of this agent
            g_agent = g_base[:, None].expand(B, self.state_dim_per_agent)  # [B,M]
            g_vec_list.append(g_agent)

        # Zero noise for the two cost coordinates
        g_cost = torch.zeros(B, 2, device=device, dtype=dtype)
        g_vec_list.append(g_cost)

        g_full = torch.cat(g_vec_list, dim=1)             # [B,D]
        return g_full


def train_control_adjoint(
    score_model,
    optimality_criterion,
    control_agents,
    aggregator,
    sde,
    optimizer,
    optimality_target,
    num_steps,
    batch_size,
    device,
    eps,
    lambda_reg,
    running_optimality_reg,
    image_shape=(1, 28, 28),
    debug=False,
):
    """
    Single training step for multi-agent control policies using torchsde.sdeint_adjoint.
    """

    # --- modes ---
    score_model.eval()
    optimality_criterion.eval()
    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided for training.")
    for k in agent_keys:
        control_agents[k].train()

    optimizer.zero_grad()
    eps_val = float(eps)
    assert 0.0 <= eps_val < 1.0, f"eps must be in [0,1), got {eps_val}"

    # --- time grid in solver-time s ---
    # s ∈ [0, 1 - eps], strictly increasing
    ts = torch.linspace(0.0, 1.0 - eps_val, num_steps, device=device)
    if not (ts[1:] > ts[:-1]).all():
        raise RuntimeError(f"ts is not strictly increasing: {ts}")

    dt = (ts[1] - ts[0]).item()
    C, H, W = image_shape
    t0_phys = 1.0
    initial_time = torch.full((batch_size,), t0_phys, device=device)
    initial_std = sde.marginal_prob_std(initial_time)[:, None, None, None]  # [B,1,1,1]
    # Agent initial states x_i(1) ~ N(0, σ^2 I)
    x0_dict = {
        key: torch.randn(batch_size, C, H, W, device=device) * initial_std
        for key in agent_keys
    }

    state_dim_per_agent = C * H * W
    flats = []
    for key in agent_keys:
        x0 = x0_dict[key].view(batch_size, state_dim_per_agent)  # [B, M]
        flats.append(x0)

    # Costs start at zero
    c_ctrl0 = torch.zeros(batch_size, 1, device=device)
    c_opt0 = torch.zeros(batch_size, 1, device=device)
    flats.append(c_ctrl0)
    flats.append(c_opt0)

    # y0: [B, D]
    y0 = torch.cat(flats, dim=1)

    sde_ctrl = MultiAgentControlledSDE(
        score_model=score_model,
        optimality_criterion=optimality_criterion,
        control_agents=control_agents,
        aggregator=aggregator,
        sde=sde,
        agent_keys=agent_keys,
        optimality_target=optimality_target,
        image_shape=image_shape,
    ).to(device)

    # --- forward integration via adjoint ---
    # ys: [num_steps, B, D]
    ys = torchsde.sdeint_adjoint(
        sde_ctrl,
        y0,
        ts,
        method="srk",
        dt=dt,  # smaller internal step for stability
    )

    # final state y_T: [B, D]
    y_T = ys[-1]
    # unpack final state
    states_final, c_ctrl_final, c_opt_final = sde_ctrl._unpack_state(y_T)
    # aggregate final images across agents
    Y_final = aggregator([states_final[k] for k in agent_keys])
    
    # terminal optimality loss
    optimality_loss = optimality_criterion.get_terminal_optimality_loss(
        Y_final, optimality_target
    )
    # --- compute losses ---
    # integrated costs (empirical expectations)
    control_cost = c_ctrl_final.mean()        # ≈ ∫ E[ Σ ||u||² ] dt
    running_opt_cost = c_opt_final.mean()     # ≈ ∫ E[ c(Ŷ_0, t) ] dt
    # final SOC objective
    total_loss = (
        lambda_reg * control_cost
        + running_optimality_reg * running_opt_cost
        + optimality_loss
    )
    # --- backward & update ---
    total_loss.backward()

    if debug:
        print(
            f"ctrl_cost={control_cost.item():.3f}, "
            f"run_cost={running_opt_cost.item():.3f}, "
            f"terminal={optimality_loss.item():.3f}"
        )

    info = {}
    if debug:
        info["cumulative_control_loss"] = (lambda_reg * control_cost).item()
        info["cumulative_optimality_loss"] = (running_optimality_reg * running_opt_cost).item()
        info["optimality_loss"] = optimality_loss.item()
        info["grad_norms"] = {}
        for k in agent_keys:
            gn = torch.nn.utils.clip_grad_norm_(control_agents[k].parameters(), 1.0)
            info["grad_norms"][k] = float(gn)
    else:
        for k in agent_keys:
            torch.nn.utils.clip_grad_norm_(control_agents[k].parameters(), 1.0)

    optimizer.step()
    return float(total_loss.item()), info