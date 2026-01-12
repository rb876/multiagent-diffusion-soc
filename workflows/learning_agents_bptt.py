import functools
from typing import Dict, Any

import hydra
import torch
import wandb
from torchvision.utils import make_grid
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.envs.aggregator import ImageMaskAggregator
from src.envs.registry import get_optimality_criterion
from src.models.registry import get_model_by_name
from src.samplers.diff_dyms import SDE
from src.trainer.soc_adjoint_ft import train_control_adjoint
from src.trainer.soc_bptt_ft import train_control_bptt
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

    # Load score model, classifier, and control nets.
    sde = SDE(mode= cfg.exps.sde.name, sigma=cfg.exps.sde.sigma, device=device)
    score_model_cfg = OmegaConf.to_container(cfg.exps.score_model, resolve=True)
    score_model_name = score_model_cfg.pop("name")
    score_model = get_model_by_name(
        score_model_name,
        marginal_prob_std=sde.marginal_prob_std,
        **score_model_cfg
    ).to(device)
    score_model.eval()
    # Freezes the model’s parameters so they don’t get gradients or updates. 
    # It does not stop autograd from building a graph or computing gradients with respect to other tensors.
    score_model.requires_grad_(False)
    _load_state(score_model, soc_config.path_to_score_model_checkpoint, device)

    # Load classifier.
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
            control_name, marginal_prob_std=sde.marginal_prob_std, **control_cfg
        ).to(device)
        control_agents[i].train()

    control_parameters = [param for net in control_agents.values() for param in net.parameters()]
    optimizer = torch.optim.Adam(
        control_parameters,
        lr=soc_config.learning_rate
    )

    # Initialize the aggregator.
    aggregator_cfg = soc_config.aggregator
    aggregator = ImageMaskAggregator(
        img_dims=tuple(cfg.exps.data.loader.img_size),
        num_processes=soc_config.num_control_agents,
        device=device,
        **aggregator_cfg
    )
    # Initialize optimality criterion based on the classifier.
    optimality_criterion = get_optimality_criterion(
        name=soc_config.optimality_criterion.name, 
        classifier=classifier,
        mask=aggregator.mask,
    ).to(device)

    # Select training method
    if soc_config.method == "bptt":
        train_soc_fn = train_control_bptt
    elif soc_config.method == "adjoint":
        train_soc_fn = train_control_adjoint
    else:
        raise ValueError(f"Unknown training method: {soc_config.method}")

    from tqdm.auto import tqdm
    pbar = tqdm(range(soc_config.iters), desc="Training control policy")
    for step in pbar:
        loss, info = train_soc_fn(
            aggregator=aggregator,
            batch_size=soc_config.batch_size,
            control_agents=control_agents,
            debug=soc_config.debug,
            device=device,
            enable_optimality_loss_on_processes=soc_config.enable_optimality_loss_on_processes,
            eps=soc_config.eps,
            image_dim=tuple(cfg.exps.data.loader.img_size),
            lambda_reg=soc_config.lambda_reg,
            num_steps=soc_config.train_num_steps,
            optimality_criterion=optimality_criterion,
            optimality_target=soc_config.optimality_target,
            optimizer=optimizer,
            running_optimality_reg=soc_config.running_optimality_reg,
            score_model=score_model,
            sde=sde,
        )
        loss_value = float(loss)
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        if wandb_run is not None:
            wandb_module.log(
                {
                    "train/loss": loss_value,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/optimality_target": soc_config.optimality_target,
                    "iteration": step + 1,
                },
                step=step + 1,
            )
            if soc_config.get("debug", False) and info:
                log_payload: Dict[str, Any] = {"train/iteration": step + 1}
                for k, v in info.items():
                    if isinstance(v, (float, int)):
                        log_payload[f"train/{k}"] = v
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            log_payload[f"train/{k}/{kk}"] = vv
                wandb_module.log(log_payload, step=step + 1)
        
        should_eval = soc_config.eval_every and (
            step % soc_config.eval_every == 0 or step == soc_config.iters - 1
        )
        if should_eval and generate_and_plot_samples is not None:
            samples = generate_and_plot_samples(
                control_agents=control_agents,
                aggregator=aggregator,
                device=str(device),
                sde=sde,
                num_steps=soc_config.sample_num_steps,
                sample_batch_size=soc_config.sample_batch_size,
                score_model=score_model,
                debug=soc_config.debug,
                step=step,
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