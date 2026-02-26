import torch
import copy

from collections import defaultdict
from src.trainer.utils import compute_grad_norms
from src.samplers.diff_dyms import get_tweedy_estimate
from src.guidance import compute_vectorized_guidance_grads


def control_wise_adjoint_matching(
    aggregator,
    batch_size,
    control_agents,
    device,
    eps,
    image_dim,
    inner_iters,
    learning_rate,
    num_steps,
    optimality_criterion,
    optimality_target,
    running_state_cost_scaling,
    score_model,
    sde,
    terminal_state_cost_scaling,
    reuse_forward_adjoint_steps=1,
    ema_decay=0.0,
    debug=False,
):
    """Fictitious-play style training with an adjoint-matching surrogate for VP/VE reverse SDE controls."""
    score_model.eval()
    optimality_criterion.eval()

    if terminal_state_cost_scaling <= 0.0 and running_state_cost_scaling <= 0.0:
        raise ValueError("At least one of terminal_state_cost_scaling or running_state_cost_scaling must be > 0.")
    if terminal_state_cost_scaling != running_state_cost_scaling:
        raise  Warning("terminal_state_cost_scaling and running_state_cost_scaling are not equal")
    
    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided for fictitious training.")
    reuse_forward_adjoint_steps = int(reuse_forward_adjoint_steps)
    if reuse_forward_adjoint_steps < 1:
        raise ValueError("reuse_forward_adjoint_steps must be >= 1.")
    ema_decay = float(ema_decay)
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("ema_decay must be in [0, 1).")

    optimizers = {
        key: torch.optim.Adam(control_agents[key].parameters(), lr=learning_rate)
        for key in agent_keys
    }
    ema_base_agents = {}
    if ema_decay > 0.0:
        for key in agent_keys:
            ema_agent = copy.deepcopy(control_agents[key]).to(device)
            ema_agent.eval()
            ema_agent.requires_grad_(False)
            ema_base_agents[key] = ema_agent

    def _ema_update_base(player_key):
        if ema_decay <= 0.0:
            return
        with torch.no_grad():
            for ema_param, src_param in zip(
                ema_base_agents[player_key].parameters(),
                control_agents[player_key].parameters(),
            ):
                ema_param.mul_(ema_decay).add_(src_param, alpha=1.0 - ema_decay)

    def _train_single_control_policy_adjoint_matching(player_key):
        """Train one control policy while keeping other agents fixed."""
        for key in agent_keys:
            if key == player_key:
                control_agents[key].train()
            else:
                control_agents[key].eval()

        optimizer = optimizers[player_key]

        time_steps = torch.linspace(1.0, eps, num_steps, device=device)
        if ema_decay > 0.0:
            base_agent = ema_base_agents[player_key]
        else:
            base_agent = copy.deepcopy(control_agents[player_key]).to(device)
            base_agent.eval()
            base_agent.requires_grad_(False)
        if len(time_steps) < 2:
            raise ValueError("num_steps must be at least 2 to compute a diffusion step.")

        initial_time = torch.full((batch_size,), time_steps[0], device=device)
        initial_std = sde.marginal_prob_std(initial_time)[:, None, None, None]
        agent_states = {
            key: torch.randn(batch_size, *image_dim, device=device) * initial_std
            for key in agent_keys
        }

        state_traj = {key: [] for key in agent_keys}
        time_traj = []
        step_traj = []
        g_traj = []
        player_guidance_traj = []

        cumulative_optimality_loss = torch.tensor(0.0, device=device)

        # --- Forward trajectories --- DETACHED
        for t_idx in range(len(time_steps) - 1):
            t_curr = time_steps[t_idx]
            t_next = time_steps[t_idx + 1]
            step_size = t_curr - t_next
            batch_time_step = torch.full((batch_size,), t_curr, device=device)

            for key in agent_keys:
                state_traj[key].append(agent_states[key].detach())
            time_traj.append(batch_time_step.detach())
            step_traj.append(step_size.detach())

            g = sde.diffusion_coeff(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            g_noise = g[:, None, None, None]
            g_traj.append(g_noise.detach())

            with torch.no_grad():
                Y_t = aggregator([agent_states[key] for key in agent_keys])
                scores = {
                    key: score_model(agent_states[key], batch_time_step)
                    for key in agent_keys
                }

            grad_inputs = {}
            if running_state_cost_scaling > 0:
                x0_hats = {
                    key: get_tweedy_estimate(sde, agent_states[key], batch_time_step, scores[key])
                    for key in agent_keys
                }
                with torch.no_grad():
                    Y_0_hat = aggregator([x0_hats[key] for key in agent_keys])
                    cumulative_optimality_loss += running_state_cost_scaling * (
                        optimality_criterion.get_running_state_loss(
                            Y_0_hat,
                            optimality_target,
                        )
                        * step_size
                    )
                grad_inputs = compute_vectorized_guidance_grads(
                    x0_hats=x0_hats,
                    agent_keys=agent_keys,
                    aggregator=aggregator,
                    optimality_criterion=optimality_criterion,
                    optimality_target=optimality_target,
                )

            if running_state_cost_scaling > 0:
                player_guidance_traj.append(grad_inputs[player_key].detach())
            else:
                player_guidance_traj.append(torch.zeros_like(agent_states[player_key]))

            controls = {}
            for key in agent_keys:
                grad_input = (
                    grad_inputs[key]
                    if running_state_cost_scaling > 0
                    else torch.zeros_like(agent_states[key])
                )
                control_input = torch.cat([agent_states[key], Y_t, grad_input], dim=1)
                with torch.no_grad():
                    controls[key] = control_agents[key](control_input, batch_time_step).detach()

            noise_scale = torch.sqrt(step_size) * g_noise
            for key in agent_keys:
                drift_rev = -sde.f(agent_states[key], batch_time_step) + g_sq * scores[key]
                mean_state = agent_states[key] + (drift_rev + g_noise * controls[key]) * step_size
                agent_states[key] = mean_state + noise_scale * torch.randn_like(agent_states[key])

        final_states = {key: agent_states[key].detach() for key in agent_keys}

        # --- Terminal adjoint initial condition: a_T = d(terminal loss)/dx_T ---
        # This is to be used as the initial condition for the reverse-time adjoint recursion and is derived from the optimality criterion terminal loss.
        x_terminal = final_states[player_key].detach().requires_grad_(True)
        final_state_list = [
            x_terminal if key == player_key else final_states[key]
            for key in agent_keys
        ]
        Y_final = aggregator(final_state_list)
        terminal_loss_for_adjoint = terminal_state_cost_scaling * optimality_criterion.get_terminal_state_loss(
            Y_final,
            optimality_target,
        )
        # Loss minimization convention: a_T = + dL_T / dx_T.
        adjoint_next = torch.autograd.grad(
            terminal_loss_for_adjoint,  
            x_terminal,
            create_graph=False,
        )[0].detach()

        # --- Lean adjoint recursion --- DETACHED
        num_transitions = len(time_traj)
        adjoint_traj = [None] * num_transitions
        for k in reversed(range(num_transitions)):
            t_k = time_traj[k]
            step_size = step_traj[k]
            g_k = sde.diffusion_coeff(t_k)[:, None, None, None]
            g_k_sq = (sde.diffusion_coeff(t_k)**2)[:, None, None, None]

            x_k = state_traj[player_key][k].detach().requires_grad_(True)
            state_list = [
                x_k if key == player_key else state_traj[key][k]
                for key in agent_keys
            ]
            Y_t = aggregator(state_list)

            if running_state_cost_scaling > 0:
                grad_input = player_guidance_traj[k]
            else:
                grad_input = torch.zeros_like(x_k)
            control_input = torch.cat([x_k, Y_t, grad_input], dim=1)

            score_k = score_model(x_k, t_k)
            # Base field is the frozen pretrained reverse diffusion drift (no learned control).
            base_drift_k = -sde.f(x_k, t_k) + g_k_sq * score_k + g_k * base_agent(control_input, t_k)

            vjp = torch.autograd.grad(
                base_drift_k,
                x_k,
                grad_outputs=adjoint_next,
                retain_graph=running_state_cost_scaling > 0,
            )[0]
            running_grad_x_t = torch.zeros_like(vjp)
            if running_state_cost_scaling > 0:
                x0_hats_for_running = {}
                for key in agent_keys:
                    if key == player_key:
                        x0_hats_for_running[key] = get_tweedy_estimate(sde, x_k, t_k, score_k)
                    else:
                        with torch.no_grad():
                            score_other = score_model(state_traj[key][k], t_k)
                            x0_hats_for_running[key] = get_tweedy_estimate(
                                sde,
                                state_traj[key][k],
                                t_k,
                                score_other,
                            ).detach()
                running_loss_k = running_state_cost_scaling * optimality_criterion.get_running_state_loss(
                    aggregator([x0_hats_for_running[key] for key in agent_keys]),
                    optimality_target,
                )
                running_grad_x_t = running_state_cost_scaling * torch.autograd.grad(
                    running_loss_k,
                    x_k,
                )[0]

            adjoint_curr = (adjoint_next + step_size * (vjp + running_grad_x_t)).detach()
            adjoint_traj[k] = adjoint_curr
            adjoint_next = adjoint_curr

        # Terminal metric for monitoring (constant across reused optimization steps).
        with torch.no_grad():
            terminal_optimality_loss = terminal_state_cost_scaling * optimality_criterion.get_terminal_state_loss(
                aggregator([final_states[key] for key in agent_keys]),
                optimality_target,
            )

        total_loss_sum = torch.tensor(0.0, device=device)
        adjoint_matching_loss_sum = torch.tensor(0.0, device=device)
        cumulative_control_loss_sum = torch.tensor(0.0, device=device)
        mean_residual_norm_sum = torch.tensor(0.0, device=device)
        for _ in range(reuse_forward_adjoint_steps):
            optimizer.zero_grad()
            # --- Adjoint Matching objective (trajectories/adjoints) ---
            adjoint_matching_loss = torch.tensor(0.0, device=device)
            cumulative_control_loss = torch.tensor(0.0, device=device)
            mean_residual_norm = torch.tensor(0.0, device=device)
            for k in range(num_transitions):
                t_k = time_traj[k]
                step_size = step_traj[k]
                g_k = g_traj[k]
                g_k_sq = g_k**2

                state_list = [state_traj[key][k] for key in agent_keys]
                Y_t = aggregator(state_list)
                grad_input = (
                    player_guidance_traj[k]
                    if running_state_cost_scaling > 0
                    else torch.zeros_like(state_traj[player_key][k])
                )
                control_input = torch.cat([state_traj[player_key][k], Y_t, grad_input], dim=1)

                finetune_control = control_agents[player_key](control_input, t_k)
                residual = finetune_control / g_k + g_k * adjoint_traj[k]
                adjoint_matching_loss += torch.mean(residual**2) * step_size
                
                # This losses are for infos and diagnostics, not used for optimization in this version.
                cumulative_control_loss += torch.mean(finetune_control**2) * step_size / len(agent_keys)
                mean_residual_norm += residual.flatten(1).norm(dim=1).mean()

            mean_residual_norm = mean_residual_norm / max(num_transitions, 1)
            adjoint_matching_loss.backward()

            torch.nn.utils.clip_grad_norm_(control_agents[player_key].parameters(), 1.0)
            optimizer.step()
            _ema_update_base(player_key)

            adjoint_matching_loss_sum += adjoint_matching_loss.detach()
            cumulative_control_loss_sum += cumulative_control_loss.detach()
            mean_residual_norm_sum += mean_residual_norm.detach()

        total_loss_avg = total_loss_sum / reuse_forward_adjoint_steps
        adjoint_matching_loss_avg = adjoint_matching_loss_sum / reuse_forward_adjoint_steps
        cumulative_control_loss_avg = cumulative_control_loss_sum / reuse_forward_adjoint_steps
        mean_residual_norm_avg = mean_residual_norm_sum / reuse_forward_adjoint_steps

        info = {}
        if debug:
            info["total_loss"] = total_loss_avg.item()
            info["adjoint_matching_loss"] = adjoint_matching_loss_avg.item()
            info["mean_residual_norm"] = mean_residual_norm_avg.item()
            info["cumulative_control_loss"] = cumulative_control_loss_avg.item()
            info["cumulative_optimality_loss"] = cumulative_optimality_loss.item()
            info["terminal_optimality_loss"] = terminal_optimality_loss.item()
            info["reuse_forward_adjoint_steps"] = reuse_forward_adjoint_steps
            info["ema_decay"] = ema_decay
            info["grad_norms"] = compute_grad_norms(control_agents)

        return total_loss_avg.item(), info

    loss_dict = {}
    info_dict = {}
    for key in agent_keys:
        loss_sum = 0.0
        scalar_info_sums = defaultdict(float)
        last_non_scalar_info = {}

        for _ in range(inner_iters):
            loss_value, info_step = _train_single_control_policy_adjoint_matching(player_key=key)
            loss_sum += loss_value

            if debug and info_step:
                for k, v in info_step.items():
                    if isinstance(v, (int, float)):
                        scalar_info_sums[k] += float(v)
                    else:
                        last_non_scalar_info[k] = v

        loss_dict[key] = loss_sum / inner_iters
        if debug:
            avg_info = {k: v / inner_iters for k, v in scalar_info_sums.items()}
            avg_info.update(last_non_scalar_info)
            info_dict[key] = avg_info

    return loss_dict, info_dict if debug else {}

