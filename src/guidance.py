import torch


def compute_vectorized_guidance_grads(
    x0_hats,
    agent_keys,
    aggregator,
    optimality_criterion,
    optimality_target,
):
    """Compute per-agent guidance grads d running_loss / d x0_hat in one autograd pass."""
    # .detach() to prevent gradients from flowing into the score model through the tweedy estimates, we only want gradients w.r.t. the control agents parameters.
    # We need a tensor with no autograd history, then torch.stack([...]) creates a new tensor (new storage) from those values, and 
    # requires_grad_(True) to enable gradient tracking (a new leaf tensor for guidance gradient computation) for the resulting tensor.
    with torch.enable_grad():
        x0_stack = torch.stack([x0_hats[key].detach() for key in agent_keys], dim=0).requires_grad_(True)
        Y0_hat_guidance = aggregator(list(x0_stack.unbind(0)))
        running_loss = optimality_criterion.get_running_state_loss(
            Y0_hat_guidance,
            optimality_target,
        )
        grad_stack = torch.autograd.grad(
            running_loss,
            x0_stack,
            create_graph=False,
        )[0].detach()
    return {key: grad_stack[idx] for idx, key in enumerate(agent_keys)}
