from src.samplers.samplers import euler_maruyama_controlled_sampler

def generate_and_plot_samples(
    score_model,
    control_agents,
    classifier,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    sample_batch_size: int = 64,
    num_steps: int = 500,
    device: str = 'cuda',
    eps: float = 1e-3,
):
    """Generate samples using controlled Euler-Maruyama. """

    # Set all networks in eval mode
    score_model.eval()
    classifier.eval()
    # Set all control agents in eval mode
    for control_net in control_agents.values():
        control_net.eval()

    # Generate samples
    samples = euler_maruyama_controlled_sampler(
        score_model=score_model,
        control_agents=control_agents,
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        batch_size=sample_batch_size,
        num_steps=num_steps,
        device=device,
        eps=eps,
    )

    samples = samples.clamp(0.0, 1.0)
    return samples