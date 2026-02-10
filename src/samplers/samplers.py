from pathlib import Path
from datetime import datetime
from copy import deepcopy

from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch

from src.samplers.diff_dyms import get_tweedy_estimate

def _save_debug_states(info_agg, info_per_agent, agent_keys, debug_dir=None, save_last_only=True):
    """Persist aggregated and per-agent states for offline inspection."""

    if save_last_only:
        print("Saving only the last time step of debug states.")
    
    if debug_dir:
        save_root = Path(debug_dir)
    else:
        try:
            # keep all debug artifacts under the Hydra job output directory (works in multirun)
            save_root = Path(HydraConfig.get().runtime.output_dir) / "debug_states"
        except Exception:
            save_root = Path("debug_states")

    save_dir = save_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_last_only:
        info_agg_to_save = [deepcopy(info_agg[-1])]
        info_per_agent_to_save = {key: [deepcopy(info_per_agent[key][-1])] for key in agent_keys}
    else:
        info_agg_to_save = info_agg
        info_per_agent_to_save = info_per_agent

    np.savez_compressed(save_dir / "aggregated.npz", data=np.stack(info_agg_to_save, axis=0))
    for key in agent_keys:
        np.savez_compressed(save_dir / f"{key}_states.npz", data=np.stack(info_per_agent_to_save[key], axis=0))

    return save_dir


def euler_maruyama_sampler(
    score_model,
    sde,
    batch_size=64,
    num_steps=500,
    device="cuda",
    eps=1e-3,
    shape=(1, 28, 28),
):
    """
    Euler-Maruyama sampler for reverse-time SDE using your SDE wrapper.
    Assumes score_model outputs score ∇_x log p_t(x).
    Time runs from 1 -> eps with positive step_size.
    """
    score_model.eval()

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    if len(time_steps) < 2:
        raise ValueError("num_steps must be >= 2")
    step_size = time_steps[0] - time_steps[1]  # positive

    # init at t=1
    t0 = torch.ones(batch_size, device=device)
    init_std = sde.marginal_prob_std(t0).view(-1, *([1] * len(shape)))
    x = torch.randn(batch_size, *shape, device=device) * init_std

    with torch.no_grad():
        mean_x = x
        for t in time_steps:
            batch_t = torch.full((batch_size,), t, device=device)

            g = sde.diffusion_coeff(batch_t)                      # [B]
            g_view = g.view(-1, *([1] * len(shape)))
            g_sq_view = (g ** 2).view(-1, *([1] * len(shape)))

            score = score_model(x, batch_t)                       # same shape as x

            # reverse-time drift under positive step_size convention:
            drift = -sde.f(x, batch_t) + g_sq_view * score

            mean_x = x + drift * step_size
            x = mean_x + torch.sqrt(step_size) * g_view * torch.randn_like(x)

    return mean_x


def euler_maruyama_controlled_sampler(
    score_model,
    control_agents,
    aggregator,
    sde,
    image_dim: tuple = (1, 28, 28),
    batch_size=8,
    num_steps=500,
    device="cuda",
    eps=1e-3,
    debug: bool = False,
    debug_dir=None,
    save_debug_info=False,
    use_grad_guidance: bool = True,          # include grad channel like in training
    optimality_criterion=None,               # needed if use_grad_guidance=True
    optimality_target=None,                  # needed if use_grad_guidance=True
    ):

    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided to controlled sampler.")

    if use_grad_guidance and (optimality_criterion is None or optimality_target is None):
        raise ValueError("If use_grad_guidance=True, provide optimality_criterion and optimality_target.")

    t = torch.ones(batch_size, device=device)
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    if len(time_steps) < 2:
        raise ValueError("num_steps must be at least 2 for Euler-Maruyama integration.")
    step_size = time_steps[0] - time_steps[1]

    states = {
        key: torch.randn(batch_size, *image_dim, device=device)
        * sde.marginal_prob_std(t)[:, None, None, None]
        for key in agent_keys
    }

    if debug:
        info_per_agent = {key: [] for key in agent_keys}
        info_controls = {key: [] for key in agent_keys}
        info_agg = []

    with torch.no_grad():
        for idx in range(len(time_steps)):
            batch_time_step = torch.full((batch_size,), time_steps[idx], device=device)

            g = sde.diffusion_coeff(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            noise_scale = torch.sqrt(step_size) * g[:, None, None, None]

            Y_t = aggregator([states[key] for key in agent_keys])

            # --- compute x0_hats once (no extra score pass) ---
            scores = {k: score_model(states[k], batch_time_step) for k in agent_keys}

            x0_hats = None
            if use_grad_guidance:
                # Tweedie: x0_hat = x_t + sigma^2 * score(x_t,t) (or your get_tweedy_estimate)
                sigma2 = (sde.marginal_prob_std(batch_time_step) ** 2)[:, None, None, None]
                x0_hats = {k: states[k] + sigma2 * scores[k] for k in agent_keys}

            controls = {}
            for key in agent_keys:
                if use_grad_guidance:
                    # guidance gradient w.r.t. x0_hat (no extra score forward)
                    with torch.enable_grad():
                        x0_guidance = x0_hats[key].detach().requires_grad_(True)

                        Y0_hat_guidance = aggregator([
                            x0_guidance if kk == key else x0_hats[kk].detach()
                            for kk in agent_keys
                        ])

                        loss = optimality_criterion.get_running_state_loss(
                            Y0_hat_guidance,
                            optimality_target,
                        )

                        grad_input = torch.autograd.grad(loss, x0_guidance, create_graph=False)[0].detach()
                else:
                    grad_input = torch.zeros_like(states[key])

                control_input = torch.cat([states[key], Y_t, grad_input], dim=1)
                controls[key] = control_agents[key](control_input, batch_time_step)

            for key in agent_keys:
                drift = -sde.f(states[key], batch_time_step) + g_sq * scores[key]
                states[key] = (
                    states[key]
                    + (drift + g[:, None, None, None] * controls[key]) * step_size
                    + noise_scale * torch.randn_like(states[key])
                )

            if debug:
                info_agg.append(Y_t.clone().detach().cpu().numpy())
                for key in agent_keys:
                    info_per_agent[key].append(states[key].clone().detach().cpu().numpy())
                    info_controls[key].append(controls[key].clone().detach().cpu().numpy())

    if debug:
        for key in agent_keys:
            assert len(info_per_agent[key]) == len(time_steps)
        assert len(info_agg) == len(time_steps)

        final_agg = aggregator([states[k] for k in agent_keys])

        if save_debug_info:
            save_dir = _save_debug_states(info_agg, info_per_agent, agent_keys, debug_dir)
        else:
            save_dir = None

        return final_agg, {
            "per_agent": info_per_agent,
            "aggregated": info_agg,
            "controls": info_controls,
            "save_dir": str(save_dir) if save_debug_info else None,
        }
    else:
        return aggregator([states[k] for k in agent_keys])
    


def euler_maruyama_dps_sampler(
    score_models,
    aggregator,
    sde,
    optimality_loss,             
    target,
    image_dim: tuple = (1, 28, 28),
    batch_size=64,
    num_steps=500,
    device="cuda",
    eps=1e-3,
    guidance_scale: float = 1.0,
    debug: bool = False,
):
    agent_keys = list(score_models.keys())
    if not agent_keys:
        raise ValueError("score_models is empty.")

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    if len(time_steps) < 2:
        raise ValueError("num_steps must be at least 2 for Euler-Maruyama integration.")
    step_size = time_steps[0] - time_steps[1]

    t0 = torch.ones(batch_size, device=device)
    init_std = sde.marginal_prob_std(t0).view(-1, 1, 1, 1)

    states = {
        key: torch.randn(batch_size, *image_dim, device=device) * init_std
        for key in agent_keys
    }

    for k in agent_keys:
        score_models[k].eval()

    if debug:
        info_per_agent = {k: [] for k in agent_keys}
        info_agg = []

    for idx in range(len(time_steps) - 1):
        t_cur = time_steps[idx]
        batch_t = torch.full((batch_size,), t_cur, device=device)

        g = sde.diffusion_coeff(batch_t)                  # [B]
        g_view = g.view(-1, 1, 1, 1)
        g_sq_view = (g ** 2).view(-1, 1, 1, 1)
        noise_scale = torch.sqrt(step_size) * g_view

        with torch.enable_grad():
            for k in agent_keys:
                states[k] = states[k].detach().requires_grad_(True)

            scores = {k: score_models[k](states[k], batch_t) for k in agent_keys}
            x0_hats = {
                k: get_tweedy_estimate(sde, states[k], batch_t, scores[k])
                for k in agent_keys
            }

            Y0_hat = aggregator([x0_hats[k] for k in agent_keys])
            loglik_scalar = optimality_loss.get_running_state_loss(Y0_hat, target, processes = None)
            grads = torch.autograd.grad(
                outputs=loglik_scalar,
                inputs=[states[k] for k in agent_keys],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            guidance = {k: grads[i] for i, k in enumerate(agent_keys)}

        for k in agent_keys:
            drift = (
                -sde.f(states[k].detach(), batch_t)
                + g_sq_view * scores[k].detach()
                - g_sq_view * guidance_scale * guidance[k].detach()
            )

            mean_state = states[k].detach() + drift * step_size
            states[k] = mean_state + noise_scale * torch.randn_like(mean_state)

        if debug:
            with torch.no_grad():
                for k in agent_keys:
                    info_per_agent[k].append(states[k].detach().cpu())
                info_agg.append(aggregator([states[j] for j in agent_keys]).detach().cpu())

    final_agg = aggregator([states[k] for k in agent_keys])

    if debug:
        return final_agg, {
            "per_agent": info_per_agent,
            "aggregated": info_agg,
        }
    else:
        return final_agg