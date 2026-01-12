import torch

from collections import defaultdict
from src.trainer.utils import compute_grad_norms


def train_control_bptt(
    aggregator,
    batch_size,
    control_agents,
    device,
    enable_optimality_loss_on_processes,
    eps,
    image_dim,
    lambda_reg,
    num_steps,
    optimality_criterion,
    optimality_target,
    optimizer,
    running_optimality_reg,
    score_model,
    sde,
    debug=False,
):

    """
    Single training step for multi-agent control policies via BPTT
    under a stochastic optimal control (SOC) objective.
    """
    
    # --- Modes ---
    # Set models to appropriate modes
    score_model.eval()
    optimality_criterion.eval() # NOTE: optimality criterion is always shared accross agents.
    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided for training.")
    for key in agent_keys:
        control_agents[key].train()
    # Reset gradients.
    optimizer.zero_grad()

    # --- Time discretisation ---
    # Example: steps=[1.0, 0.9, ..., eps]. Length = num_steps.
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    if len(time_steps) < 2:
        raise ValueError("num_steps must be at least 2 to compute a diffusion step.")
    # # Reverse-time grid: t_0 = 1, t_{K-1} ≈ eps, uniform step size.
    # step_size = time_steps[0] - time_steps[1]

    # --- Initialisation ---
    # Initialize dynamics parameters.
    initial_time = torch.full((batch_size,), time_steps[0], device=device)
    initial_std = sde.marginal_prob_std(initial_time)[:, None, None, None]
    # Initialize system states for all agents.
    # Here we assume that each states has a control agent associated to it.
    system_states = {
        key: torch.randn(batch_size, *image_dim, device=device) * initial_std
        for key in agent_keys
    }
    # Initialize info dict for debugging if needed
    info = {}
    # Initialize cumulative losses
    cumulative_control_loss = torch.tensor(0.0, device=device)
    cumulative_optimality_loss = torch.tensor(0.0, device=device)

    # --- Forward simulation (BPTT through time) ---
    # Iterate over time steps of the diffusion process - forward simulation
    for t_idx in range(len(time_steps) - 1):

        t_current = time_steps[t_idx]
        t_next = time_steps[t_idx + 1]
        
        # Step size
        step_size = t_current - t_next
        batch_time_step = torch.full((batch_size,), t_current, device=device)

        # Diffusion coefficients
        g = sde.diffusion_coeff(batch_time_step)
        g_sq = (g**2)[:, None, None, None]
        g_noise = g[:, None, None, None]

        # Aggregate current agent states
        Y_t = aggregator([system_states[key] for key in agent_keys])

        # Compute controls and scores for all agents
        controls, scores, x0_hats = {}, {}, {}
        for key in agent_keys:
            # Compute control input by concatenating the system state of the current agent with the aggregated state.
            control_input = torch.cat([system_states[key], Y_t], dim=1)
            controls[key] = control_agents[key](control_input, batch_time_step)
            # NOTE: assume that the score model is shared across agents. 
            # This will need to be modifid as we expect to have different score models per agent in the future.
            scores[key] = score_model(system_states[key], batch_time_step)

        # --- Running Optimality Cost ---
        if running_optimality_reg > 0:
            # Compute denoised estimates for all agents (TWEEDY ESTIMATOR).
            current_std = sde.marginal_prob_std(batch_time_step)[:, None, None, None]
            for key in agent_keys:
                x0_hats[key] = system_states[key] + (current_std**2) * scores[key]
            # Tweedie estimator for each agent:
            # Aggregate the denoised estimates across agents for running optimality loss.
            Y_0_hat = aggregator([x0_hats[key] for key in agent_keys])
            # Compute running optimality loss.
            cumulative_optimality_loss += optimality_criterion.get_running_state_loss(
                Y_0_hat, optimality_target, processes=[x0_hats[key] for key in agent_keys] if enable_optimality_loss_on_processes else None) * step_size
    
        # Progress system dynamics for all agents (Euler–Maruyama)
        noise_scale = torch.sqrt(step_size) * g_noise
        for key in agent_keys:
            drift = g_sq * scores[key]  # reverse SDE drift term
            mean_state = system_states[key] + (drift + controls[key]) * step_size
            # Update system state with Euler-Maruyama step.
            system_states[key] = mean_state + noise_scale * torch.randn_like(system_states[key])

            # Control energy (averaged over batch)
            cumulative_control_loss += torch.mean(controls[key] ** 2) * step_size

    # --- Terminal cost on Y_1 ---
    Y_final = aggregator([system_states[key] for key in agent_keys])
    # Compute terminal optimality loss.
    optimality_loss = optimality_criterion.get_terminal_state_loss(
        Y_final,
        optimality_target,
        processes=[system_states[key] for key in agent_keys] if enable_optimality_loss_on_processes else None)
    # Compute overall SOC loss to backpropagate.
    total_loss = (
        lambda_reg * cumulative_control_loss
        + optimality_loss
        + running_optimality_reg * cumulative_optimality_loss
    )
    # --- Backprop & update ---
    total_loss.backward()

    if debug:
        info ['cumulative_control_loss'] = cumulative_control_loss.item()
        info ['cumulative_optimality_loss'] = cumulative_optimality_loss.item()
        info ['optimality_loss'] = optimality_loss.item()
        info['agents_control_mean'] = {
            k: controls[k].detach().mean().item() for k in agent_keys
        }        
        info ['grad_norms'] = compute_grad_norms(control_agents)

    for key in agent_keys:
        torch.nn.utils.clip_grad_norm_(control_agents[key].parameters(), 1.0)
    
    # Update control agent parameters.
    optimizer.step()
    return total_loss.item(), info


def fictitious_train_control_bptt(
    score_model,
    optimality_criterion,
    control_agents,
    aggregator,
    sde,
    optimality_target,
    num_steps,
    inner_iters,
    batch_size,
    device,
    eps,
    lambda_reg,
    running_optimality_reg,
    learning_rate,
    image_dim,
    debug=False,
):
    """Fictitious-play style training of an arbitrary number of control policies."""
    
    # --- Modes ---
    # Set models to appropriate modes
    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided for fictitious training.")

    # Setup optimizers for each control agent. Note: each agent has its own optimizer.
    optimizers = {
        key: torch.optim.Adam(control_agents[key].parameters(), lr=learning_rate)
        for key in agent_keys
    }

    # Initialize info dict for debugging if needed
    local_info = {}

    def _train_single_control_policy(player_key):
        """Train a single control policy while keeping others fixed."""
            
        # Active player -> Train
        # Fixed players -> Eval (affects Dropout/BatchNorm if present)
        for key in agent_keys:
            if key == player_key:
                control_agents[key].train()
            else:
                control_agents[key].eval()

        optimizer = optimizers[player_key]
        # Reset gradients.
        optimizer.zero_grad()

        # --- Initialisation ---
        # Initialize dynamics parameters.
        time_steps = torch.linspace(1.0, eps, num_steps, device=device)
        if len(time_steps) < 2:
            raise ValueError("num_steps must be at least 2 to compute a diffusion step.")

        initial_time = torch.full((batch_size,), time_steps[0], device=device)
        initial_std = sde.marginal_prob_std(initial_time)[:, None, None, None]
        
        # Initialize all agents
        agent_states = {
            key: torch.randn(batch_size, *image_dim, device=device) * initial_std
            for key in agent_keys
        }

        cumulative_control_loss = torch.tensor(0.0, device=device)
        cumulative_optimality_loss = torch.tensor(0.0, device=device)
        for t_idx in range(len(time_steps) - 1):
    
            t_curr = time_steps[t_idx]
            t_next = time_steps[t_idx + 1]
            step_size = t_curr - t_next
            
            batch_time_step = torch.full((batch_size,), t_curr, device=device)

            # Diffusion coefficients
            g = sde.diffusion_coeff(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            g_noise = g[:, None, None, None]

            # Aggregate current agent states
            Y_t = aggregator([agent_states[key] for key in agent_keys])
        
            # Compute controls and scores for all agents
            controls = {}
            for key in agent_keys:
                control_input = torch.cat([agent_states[key], Y_t], dim=1)
                if key == player_key:
                    controls[key] = control_agents[key](control_input, batch_time_step)
                else:
                    # Other agents are fixed during this training step.
                    # No gradient computation for their control policies.
                    with torch.no_grad():
                        controls[key] = control_agents[key](control_input, batch_time_step).detach()

            scores = {}
            for key in agent_keys:
                scores[key] = score_model(agent_states[key], batch_time_step)

            if running_optimality_reg > 0:
                current_std = sde.marginal_prob_std(batch_time_step)[:, None, None, None]
                # Compute denoised estimates for all agents (TWEEDY ESTIMATOR).
                x0_hats = {
                    key: agent_states[key] + (current_std**2) * scores[key]
                    for key in agent_keys
                }
                # Tweedie estimator for each agent:
                #     \hat x_0 = x_t + σ_t^2 * score(x_t, t)
                # Aggregate the denoised estimates across agents for running optimality loss.
                Y_0_hat = aggregator([x0_hats[key] for key in agent_keys])

                # Running optimality cost ∫c(Ŷ_0(t),t) dt
                # Compute running optimality loss.
                cumulative_optimality_loss += optimality_criterion.get_running_optimality_loss(
                    Y_0_hat, optimality_target) * step_size
            
            # Progress system dynamics for all agents (Euler–Maruyama)
            noise_scale = torch.sqrt(step_size) * g_noise
            for key in agent_keys:
                drift = g_sq * scores[key]
                mean_state = agent_states[key] + (drift + controls[key]) * step_size
                if key == player_key:
                    agent_states[key] = mean_state + noise_scale * torch.randn_like(agent_states[key])
                else:
                    with torch.no_grad():
                        agent_states[key] = mean_state + noise_scale * torch.randn_like(agent_states[key])

            cumulative_control_loss += torch.mean(controls[player_key] ** 2) * step_size
        
        # --- Terminal cost on Y_1 ---
        Y_final = aggregator([agent_states[key] for key in agent_keys])
        optimality_loss = optimality_criterion.get_terminal_optimality_loss(Y_final, optimality_target)

        # Compute overall SOC loss to backpropagate.
        total_loss = (
            lambda_reg * cumulative_control_loss
            + optimality_loss
            + running_optimality_reg * cumulative_optimality_loss
        )
        # --- Backprop & update ---
        total_loss.backward()

        if debug:
            local_info ['cumulative_control_loss'] = cumulative_control_loss.item()
            local_info ['cumulative_optimality_loss'] = cumulative_optimality_loss.item()
            local_info ['optimality_loss'] = optimality_loss.item()
            local_info ['agents'] = [controls[key] for key in agent_keys]
            local_info ['grad_norms'] = compute_grad_norms(control_agents)
        
        torch.nn.utils.clip_grad_norm_(control_agents[player_key].parameters(), 1.0)
        # Update control agent parameters.
        optimizer.step()
        return total_loss.item(), local_info

    loss_dict = {}
    info_dict = {}  # per-agent averaged info
    for key in agent_keys:
        loss_sum = 0.0
        scalar_info_sums = defaultdict(float)
        last_non_scalar_info = {}

        for _ in range(inner_iters):
            loss_value, info_step = _train_single_control_policy(player_key=key)
            loss_sum += loss_value

            if debug and info_step:
                for k, v in info_step.items():
                    # average only numeric scalars
                    if isinstance(v, (int, float)):
                        scalar_info_sums[k] += float(v)
                    else:
                        # keep the latest non-scalar entries (e.g. agents, grad_norms)
                        last_non_scalar_info[k] = v

        avg_loss = loss_sum / inner_iters
        loss_dict[key] = avg_loss

        if debug:
            avg_info = {k: v / inner_iters for k, v in scalar_info_sums.items()}
            avg_info.update(last_non_scalar_info)  # attach non-averaged info
            info_dict[key] = avg_info

    return loss_dict, info_dict if debug else {}