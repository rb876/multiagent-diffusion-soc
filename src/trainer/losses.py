import torch

def loss_fn(model, x0, sde, eps=1e-5):
    """
    The loss function for training score-based generative models.
    """
    t = torch.rand(x0.shape[0], device=x0.device) * (1. - eps) + eps
    z = torch.randn_like(x0)

    std = sde.marginal_prob_std(t)
    std_view = std.view(-1, 1, 1, 1)

    if sde.mode == "VE":
        x_t = x0 + z * std_view
    else:  # VP
        alpha_view = sde.alpha(t).view(-1, 1, 1, 1)
        x_t = alpha_view * x0 + z * std_view

    score = model(x_t, t)          # should output score ∇ log p_t(x_t)
    target = -z / std_view         # ∇ log p(x_t | x0)

    loss = (std_view**2 * (score - target).pow(2)).reshape(x0.shape[0], -1).sum(dim=1).mean()
    return loss