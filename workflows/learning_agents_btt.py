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
    control_net_1 = get_model_by_name(
        control_name, marginal_prob_std=marginal_prob_std_fn, **control_cfg
    ).to(device)
    control_net_2 = get_model_by_name(
        control_name, marginal_prob_std=marginal_prob_std_fn, **control_cfg
    ).to(device)
    control_net_1.train()
    control_net_2.train()

    optimizer = torch.optim.Adam(
        list(control_net_1.parameters()) + list(control_net_2.parameters()),
        lr=soc_config.learning_rate,
    )

    iters = soc_config.iters
    train_steps = soc_config.get("train_num_steps", 25)
    sample_steps = soc_config.get("sample_num_steps", soc_config.num_diffusion_steps)
    target_digit = soc_config.get("target_digit", 0)
    eval_every = soc_config.get("eval_every", 100)
    sample_batch_size = soc_config.get("sample_batch_size", 64)
    eps = soc_config.get("eps", 1e-3)

    from tqdm.auto import tqdm
    pbar = tqdm(range(iters), desc="Training control policy")
    for step in pbar:
        loss = train_control_btt(
            score_model,
            classifier,
            control_net_1,
            control_net_2,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            optimizer,
            target_digit=target_digit,
            num_steps=train_steps,
            batch_size=soc_config.batch_size,
            device=device,
            eps=eps,
            lambda_reg=soc_config.lambda_reg,
        )
        loss_value = float(loss)
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        if wandb_run is not None:
            wandb_module.log(
                {
                    "train/loss": loss_value,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/target_digit": target_digit,
                    "iteration": step + 1,
                },
                step=step + 1,
            )

        should_eval = eval_every and (step % eval_every == 0 or step == iters - 1)
        if should_eval and generate_and_plot_samples is not None:
            samples = generate_and_plot_samples(
                score_model,
                control_net_1,
                control_net_2,
                classifier,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                sample_batch_size=sample_batch_size,
                num_steps=sample_steps,
                device=str(device),
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