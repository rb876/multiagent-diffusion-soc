import torch

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
        print("Initial States:")
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
                    + (drift + controls[key]) * step_size
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
        return aggregator([states[k] for k in agent_keys]), {
            "per_agent": info_per_agent,
            "aggregated": info_agg,
            "controls": info_controls,
        }
    else:
        return aggregator([states[k] for k in agent_keys])