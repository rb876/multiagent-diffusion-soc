import os

from src.samplers.samplers import euler_maruyama_controlled_sampler
from hydra.core.hydra_config import HydraConfig


def plot_sample_sequences(sample_sequences, agent_keys, save_path_prefix: str, stride: int = 25):
    import matplotlib.pyplot as plt

    num_steps = len(sample_sequences["aggregated"])
    def plot_step(img, title, out_path):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        # Image
        im = axes[0].imshow(img, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Image")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Histogram
        axes[1].hist(img.flatten(), bins=50)
        axes[1].set_title("Pixel histogram")
        axes[1].set_xlabel("Pixel value")

        fig.suptitle(title)
        plt.tight_layout()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
    
    plot_steps = list(range(0, num_steps, stride))
    plot_steps.append(num_steps - 1)            # ensure final
    plot_steps = sorted(set(plot_steps))        # dedupe + sort
    for key in agent_keys:
        for step in plot_steps:
            img = sample_sequences["per_agent"][key][step][0, 0]
            plot_step(
                img,
                title=f"State {key} — Step {step+1}",
                out_path=os.path.join(save_path_prefix, f"state_{key}_step_{step+1}.png"),
            )
    for key in agent_keys:
        for step in plot_steps:
            img = sample_sequences["controls"][key][step][0, 0]
            plot_step(
                img,
                title=f"Controls — Step {step+1}",
                out_path=os.path.join(save_path_prefix, f"controls_{key}_step_{step+1}.png"),
            )
    for step in plot_steps:
        img = sample_sequences["aggregated"][step][0, 0]
        plot_step(
            img,
            title=f"Aggregated — Step {step+1}",
            out_path=os.path.join(save_path_prefix, f"aggregated_step_{step+1}.png"),
        )
    


def generate_and_plot_samples(
    score_model,
    control_agents,
    aggregator,
    sde,
    image_dim = (1, 28, 28),
    sample_batch_size: int = 64,
    num_steps: int = 500,
    device: str = 'cuda',
    eps: float = 1e-3,
    debug: bool = False,
    step: int = 0,
):
    """Generate samples using controlled Euler-Maruyama. """

    # Set all networks in eval mode
    score_model.eval()
    
    # Set all control agents in eval mode
    for control_net in control_agents.values():
        control_net.eval()

    # Generate samples
    samples = euler_maruyama_controlled_sampler(
        score_model=score_model,
        control_agents=control_agents,
        aggregator=aggregator,
        sde=sde,
        image_dim=image_dim,
        batch_size=sample_batch_size,
        num_steps=num_steps,
        device=device,
        eps=eps,
        debug=debug,
    )
    if debug:
        samples, info = samples  # type: ignore
        out_dir = HydraConfig.get().runtime.output_dir
        plot_sample_sequences(
            sample_sequences=info,
            agent_keys=sorted(control_agents.keys()),
            save_path_prefix=os.path.join(out_dir, "samples_debug_plots_{step}".format(step=step)),
            stride=25,
        )

    samples = samples.clamp(0.0, 1.0)
    return samples