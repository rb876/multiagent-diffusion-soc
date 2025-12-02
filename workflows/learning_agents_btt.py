import functools
from typing import Dict, Any

import hydra
import torch
import wandb
from torchvision.utils import make_grid
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.models.registry import get_model_by_name
from src.samplers.diff_dyms import marginal_prob_std, diffusion_coeff
from src.trainer.soc_btt_ft import train_control_btt
from src.utils import generate_and_plot_samples


def _load_state(module: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(to_absolute_path(checkpoint_path), map_location=device)
    if isinstance(checkpoint, Dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict, strict=False)


@hydra.main(config_path="../configs", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    soc_config = cfg.exps.soc
    sigma = cfg.exps.get("diffusion", {}).get("sigma", 25.0)

    wandb_run = None
    wandb_module = None
    wandb_cfg = cfg.exps.get("wandb")
    if wandb_cfg is not None:
        wandb_dict = OmegaConf.to_container(wandb_cfg, resolve=True)
        enabled = wandb_dict.pop("enabled", True)
        if enabled:
            import wandb as wandb_lib

            wandb_module = wandb_lib
            wandb_kwargs = {k: v for k, v in wandb_dict.items() if v not in (None, "", [], {})}
            wandb_run = wandb_module.init(**wandb_kwargs)
            wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    # Load score model, classifier, and control nets
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)
    score_model_cfg = OmegaConf.to_container(cfg.exps.score_model, resolve=True)
    score_model_name = score_model_cfg.pop("name")
    score_model = get_model_by_name(
        score_model_name,
        marginal_prob_std=marginal_prob_std_fn,
        **score_model_cfg
    ).to(device)
    score_model.eval()
    score_model.requires_grad_(False)
    _load_state(score_model, soc_config.path_to_score_model_checkpoint, device)

    # Load classifier
    classifier_cfg = OmegaConf.to_container(cfg.exps.classifier_model, resolve=True)
    classifier_name = classifier_cfg.pop("name")
    classifier = get_model_by_name(
        classifier_name, 
        **classifier_cfg
    ).to(device)
    classifier.eval()
    _load_state(classifier, soc_config.path_to_classifier_checkpoint, device)

    # Initialize control nets
    control_cfg = OmegaConf.to_container(cfg.exps.control_net_model, resolve=True)
    control_name = control_cfg.pop("name")

    control_agents = {}
    for i in range(soc_config.num_control_agents):
        control_agents[i] = get_model_by_name(
            control_name, marginal_prob_std=marginal_prob_std_fn, **control_cfg
        ).to(device)
        control_agents[i].train()

    control_parameters = [param for net in control_agents.values() for param in net.parameters()]
    optimizer = torch.optim.Adam(
        control_parameters,
        lr=soc_config.learning_rate
    )

    from tqdm.auto import tqdm
    pbar = tqdm(range(soc_config.iters), desc="Training control policy")
    for step in pbar:
        loss = train_control_btt(
            batch_size=soc_config.batch_size,
            classifier=classifier,
            control_agents=control_agents,
            device=device,
            diffusion_coeff_fn=diffusion_coeff_fn,
            eps=soc_config.eps,
            lambda_reg=soc_config.lambda_reg,
            marginal_prob_std_fn=marginal_prob_std_fn,
            num_steps=soc_config.train_num_steps,
            optimizer=optimizer,
            running_class_reg=soc_config.running_class_reg,
            score_model=score_model,
            target_digit=soc_config.target_digit,
        )
        loss_value = float(loss)
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        if wandb_run is not None:
            wandb_module.log(
                {
                    "train/loss": loss_value,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/target_digit": soc_config.target_digit,
                    "iteration": step + 1,
                },
                step=step + 1,
            )

        should_eval = soc_config.eval_every and (
            step % soc_config.eval_every == 0 or step == soc_config.iters - 1
        )
        if should_eval and generate_and_plot_samples is not None:
            samples = generate_and_plot_samples(
                classifier=classifier,
                control_agents=control_agents,
                device=str(device),
                diffusion_coeff_fn=diffusion_coeff_fn,
                marginal_prob_std_fn=marginal_prob_std_fn,
                num_steps=soc_config.sample_num_steps,
                sample_batch_size=soc_config.sample_batch_size,
                score_model=score_model,
            )
            if wandb_run is not None:
                log_payload: Dict[str, Any] = {"eval/iteration": step + 1}
                wandb_module.log(log_payload, step=step + 1)
                grid = make_grid(samples.detach().cpu(), nrow=8, normalize=True, value_range=(0.0, 1.0))
                wandb.log({"eval/samples": wandb.Image(grid)}, step=step + 1)


    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":  
    main()