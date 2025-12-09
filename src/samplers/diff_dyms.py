import math
import torch


def _to_tensor(t, device: str = "cuda", dtype=torch.float32):
    """Ensure t is a 1D tensor on the given device."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device, dtype=dtype)
    else:
        t = t.to(device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    return t


class SDE:
    """
    Wrapper for VE / VP SDEs providing:
        - marginal_prob_std(t)
        - diffusion_coeff(t)

    mode = "VE": sigma-based VE SDE
    mode = "VP": VP SDE with beta_min/beta_max
    """

    def __init__(
        self,
        mode: str = "VP",
        *,
        sigma: float = 25,          # for VE
        beta_min: float = 0.1,      # for VP
        beta_max: float = 20.0,     # for VP
        device: str = "cuda",
    ) -> None:
        self.mode = mode
        self.device = device

        if mode == "VE":
            if sigma is None:
                raise ValueError("sigma must be provided for VE SDE.")
            self.sigma = float(sigma)
            self.marginal_prob_std, self.diffusion_coeff = make_ve_sde(
                sigma=self.sigma, device=device
            )
        elif mode == "VP":
            self.beta_min = float(beta_min)
            self.beta_max = float(beta_max)
            self.marginal_prob_std, self.diffusion_coeff = make_vp_sde(
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                device=device,
            )
        else:
            raise ValueError(f"Unknown SDE mode: {mode}")


# -------- VE SDE --------
def make_ve_sde(sigma: float, device: str = "cuda"):
    """
    VE SDE with:
        g(t) = sigma^t
        std(t) = sqrt( (sigma^(2t) - 1) / (2 log sigma) )
    """

    sigma_scalar = float(sigma)
    log_sigma = math.log(sigma_scalar)
    sigma_tensor = torch.tensor(sigma_scalar, device=device)

    def marginal_prob_std(t):
        t_ = _to_tensor(t, device=device)
        numerator = sigma_tensor.pow(2.0 * t_) - 1.0
        denominator = 2.0 * log_sigma
        return torch.sqrt(numerator / denominator)

    def diffusion_coeff(t):
        t_ = _to_tensor(t, device=device)
        return sigma_tensor.pow(t_)

    return marginal_prob_std, diffusion_coeff


# -------- VP SDE --------
def make_vp_sde(beta_min: float = 0.1, beta_max: float = 20.0, device: str = "cuda"):
    """
    VP SDE with:
        dx = -1/2 β(t) x dt + sqrt(β(t)) dW_t
        β(t) = β_min + t (β_max - β_min)
        std(t) = sqrt(1 - α(t)^2), α from integrated β(t)
    """

    beta_min = float(beta_min)
    beta_max = float(beta_max)

    def _log_alpha(t):
        # log α(t)
        return -0.25 * (beta_max - beta_min) * t**2 - 0.5 * beta_min * t

    def marginal_prob_std(t):
        t_ = _to_tensor(t, device=device)
        log_alpha = _log_alpha(t_)
        alpha = torch.exp(log_alpha)
        return torch.sqrt(1.0 - alpha**2)

    def diffusion_coeff(t):
        t_ = _to_tensor(t, device=device)
        beta_t = beta_min + t_ * (beta_max - beta_min)
        return torch.sqrt(beta_t)

    return marginal_prob_std, diffusion_coeff