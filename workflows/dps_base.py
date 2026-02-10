from typing import Dict, Any
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

import hydra
import torch
import wandb
import numpy as np

from torchvision.utils import make_grid
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.envs.aggregator import ImageMaskAggregator
from src.envs.registry import get_optimality_criterion
from src.models.registry import get_model_by_name
from src.samplers.diff_dyms import SDE
from src.samplers.samplers import euler_maruyama_dps_sampler

def _load_state(module: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(to_absolute_path(checkpoint_path), map_location=device)
    if isinstance(checkpoint, Dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict, strict=False)


def _to_numpy(obj: Any):
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy(v) for v in obj)
    return np.array(obj)


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
    sde = SDE(mode=cfg.exps.sde.name, device=device)
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
        aggregator=aggregator,
    ).to(device)

    samples, info = euler_maruyama_dps_sampler(
        score_models={key: score_model for key in range(soc_config.num_control_agents)},
        aggregator=aggregator,
        sde=sde,
        optimality_loss=optimality_criterion,             
        target=soc_config.optimality_target,
        guidance_scale=soc_config.guidance_scale,
        batch_size=soc_config.eval_batch_size,
        debug=True,
        )
    # save inside the hydra run directory
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    samples_np = samples.detach().cpu().numpy()
    info_np = _to_numpy(info)
    np.save(hydra_output_dir / "samples.npy", samples_np)
    if isinstance(info_np, dict) and all(isinstance(v, np.ndarray) for v in info_np.values()):
        # save a flat dict of arrays as compressed npz
        np.savez_compressed(hydra_output_dir / "info.npz", **info_np)
    else:
        # fallback for nested structures
        np.save(hydra_output_dir / "info.npy", info_np, allow_pickle=True)
    if wandb_run is not None:
        grid = make_grid(samples.detach().cpu(), nrow=8, normalize=True, value_range=(0.0, 1.0))
        wandb.log({"eval/samples": wandb.Image(grid)}, step=0)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":  
    main()