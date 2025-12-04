def compute_grad_norms(modules_dict):
    """Compute L2 grad norm per module in a dict {name: nn.Module}."""
    norms = {}
    for key, module in modules_dict.items():
        total_sq = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            pn = p.grad.detach().data.norm(2)
            total_sq += pn.item() ** 2
        norms[key] = total_sq ** 0.5
    return norms