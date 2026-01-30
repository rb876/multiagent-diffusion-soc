from pathlib import Path
from datetime import datetime
from copy import deepcopy

from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch


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
  marginal_prob_std,
  diffusion_coeff,
  batch_size=64,
  num_steps=500,
  device='cuda',
  eps=1e-3
):
	"""
	Generate samples from score-based models with the Euler-Maruyama solver.
	"""
	t = torch.ones(batch_size, device=device)
	init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
	* marginal_prob_std(t)[:, None, None, None]
	time_steps = torch.linspace(1., eps, num_steps, device=device)
	step_size = time_steps[0] - time_steps[1]
	x = init_x

	with torch.no_grad():
		for time_step in range(len(time_steps)):
			batch_time_step = torch.ones(batch_size, device=device) * time_steps[time_step]
			g = diffusion_coeff(batch_time_step)
			mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
			x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

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
):
    
    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided to controlled sampler.")

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
            batch_time_step = torch.ones(batch_size, device=device) * time_steps[idx]

            g = sde.diffusion_coeff(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            noise_scale = torch.sqrt(step_size) * g[:, None, None, None]

            Y_t = aggregator([states[key] for key in agent_keys])

            controls = {}
            for key in agent_keys:
                control_input = torch.cat([states[key], Y_t], dim=1)
                controls[key] = control_agents[key](control_input, batch_time_step)

            for key in agent_keys:
                drift = g_sq * score_model(states[key], batch_time_step)
                states[key] = (
                    states[key]
                    + (drift + g[:, None, None, None] * controls[key]) * step_size
                    + noise_scale * torch.randn_like(states[key])
                )

            if debug:
                # aggregated snapshot at this step
                info_agg.append(Y_t.clone().detach().cpu().numpy())
                # per-agent snapshots
                for key in agent_keys:
                    info_per_agent[key].append(states[key].clone().detach().cpu().numpy())
                # control signals
                for key in agent_keys:
                    info_controls[key].append(controls[key].clone().detach().cpu().numpy())

    if debug:
        for key in agent_keys:
            assert len(info_per_agent[key]) == len(time_steps)
        assert len(info_agg) == len(time_steps)
        final_agg = aggregator([states[k] for k in agent_keys])

        # save debug states as numpy arrays for further analysis as well as the aggregated state
        save_dir = _save_debug_states(info_agg, info_per_agent, agent_keys, debug_dir)

        return final_agg, {
            "per_agent": info_per_agent,
            "aggregated": info_agg,
            "controls": info_controls,
            "save_dir": str(save_dir),
        }
    else:
        return aggregator([states[k] for k in agent_keys])