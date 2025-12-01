from src.samplers.samplers import euler_maruyama_controlled_sampler

def generate_and_plot_samples(
    score_model,
    control_net_1,
    control_net_2,
    classifier,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    sample_batch_size=64,
    num_steps=500,
    device='cuda',
    eps=1e-3,
):
    """Generate samples using Controlled Euler-Maruyama and plot with predicted class labels."""

    # eval mode
    score_model.eval()
    control_net_1.eval()
    control_net_2.eval()
    classifier.eval()

    # sampling
    samples = euler_maruyama_controlled_sampler(
        score_model=score_model,
        control_net_1=control_net_1,
        control_net_2=control_net_2,
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        batch_size=sample_batch_size,
        num_steps=num_steps,
        device=device,
        eps=eps,
    )

    samples = samples.clamp(0.0, 1.0)

    return samples