import torch
import numpy as np


def marginal_prob_std(t, sigma, device: str = "cuda"):
    """
    Compute the mean and standard deviation of $p(x(t) | x(0))$.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device, dtype=torch.float32)
    else:
        t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 0:
        t = t.unsqueeze(0)

    sigma_scalar = float(sigma)
    sigma_tensor = torch.as_tensor(sigma_scalar, device=device, dtype=t.dtype)
    log_sigma = torch.as_tensor(np.log(sigma_scalar), device=device, dtype=t.dtype)

    numerator = torch.pow(sigma_tensor, 2 * t) - 1.0
    denominator = 2.0 * log_sigma
    return torch.sqrt(numerator / denominator)


def diffusion_coeff(t, sigma, device: str = "cuda"):
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device, dtype=torch.float32)
    else:
        t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 0:
        t = t.unsqueeze(0)

    sigma_tensor = torch.as_tensor(float(sigma), device=device, dtype=t.dtype)
    return torch.pow(sigma_tensor, t)
