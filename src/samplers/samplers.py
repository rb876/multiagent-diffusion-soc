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
    control_net_1,
    control_net_2,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=8,
    num_steps=500,
    device='cuda',
    eps=1e-3
):
    t = torch.ones(batch_size, device=device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    # Initialize separate latent codes
    x1 = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    x2 = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    mask_top = torch.zeros((1, 1, 28, 28), device=device); mask_top[:, :, :14, :] = 1.0
    mask_bot = torch.zeros((1, 1, 28, 28), device=device); mask_bot[:, :, 14:, :] = 1.0

    with torch.no_grad():
        for time_step in range(len(time_steps)):
            batch_time_step = torch.ones(batch_size, device=device) * time_steps[time_step]
    
            g = diffusion_coeff(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            
            Y_t = (x1 * mask_top) + (x2 * mask_bot)
            
            u1 = control_net_1(torch.cat([x1, Y_t], dim=1), batch_time_step)
            u2 = control_net_2(torch.cat([x2, Y_t], dim=1), batch_time_step)
                        
            drift1 = g_sq * score_model(x1, batch_time_step)
            x1 = x1 + (drift1 + u1) * step_size + torch.sqrt(step_size) * g[:,None,None,None] * torch.randn_like(x1)
            
            drift2 = g_sq * score_model(x2, batch_time_step)
            x2 = x2 + (drift2 + u2) * step_size + torch.sqrt(step_size) * g[:,None,None,None] * torch.randn_like(x2)

    return (x1 * mask_top) + (x2 * mask_bot)