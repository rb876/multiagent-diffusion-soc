import functools
from pathlib import Path


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.models.registry import get_model_by_name
from src.samplers.diff_dyms import SDE
from src.trainer.datasets import get_dataset_loader
from src.trainer.score_trainer import score_model_trainer
from src.trainer.losses import loss_fn
from src.samplers.samplers import euler_maruyama_sampler


@hydra.main(config_path="../configs", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to train a score-based model using the specified configuration."""
    
    # Extract training configuration
    training_cfg = cfg.exps.training
    device = training_cfg.device

    # Set up output directory such that checkpoints are saved in the hydra run directory
    hydra_run_dir = Path(HydraConfig.get().run.dir)
    checkpoint_subdir = training_cfg.get("checkpoint_dir", "checkpoints")
    output_dir = hydra_run_dir / checkpoint_subdir
    
    # Create the marginal probability std function
    # this defines the VE SDE dynamics
    sde = SDE(mode="VE", sigma=training_cfg.sigma, device=device)

    sigma = training_cfg.sigma
    marginal_prob_std_fn = functools.partial(
        sde.marginal_prob_std,
        sigma=sigma,
        device=device,
    )
    diffusion_coeff_fn = functools.partial(
        sde.diffusion_coeff,
        sigma=sigma,
        device=device,
    )

    # Create the score model
    score_model = get_model_by_name(
        cfg.exps.model.name,
        marginal_prob_std=marginal_prob_std_fn,
        **OmegaConf.to_container(cfg.exps.model, resolve=True)
    )

    # Create the data loader
    if cfg.exps.data.loader is not None:
        loader_args = OmegaConf.to_container(cfg.exps.data.loader, resolve=True)
        data_loader = get_dataset_loader(**loader_args)

    # Prepare trainer kwargs
    wandb_cfg = training_cfg.get("wandb", None)
    trainer_kwargs = {
        "load_checkpoint": training_cfg.get("load_checkpoint", False),
        "path_to_checkpoint": training_cfg.get("path_to_checkpoint"),
        "checkpoint_every": training_cfg.get("checkpoint_every", 10),
        "output_dir": str(output_dir),
        "grad_clip_norm": training_cfg.get("grad_clip_norm"),
        "log_to_wandb": training_cfg.get("log_to_wandb", False),
    }
    # Add scheduler config if specified
    scheduler_cfg = training_cfg.get("scheduler")
    if scheduler_cfg:
        trainer_kwargs["scheduler_config"] = OmegaConf.to_container(scheduler_cfg, resolve=True)
    # Add wandb config if logging to wandb
    if trainer_kwargs["log_to_wandb"]:
        trainer_kwargs["wandb_run_kwargs"] = OmegaConf.to_container(
            wandb_cfg or {},
            resolve=True,
        )
    
    # Building a sampler for evaluating training progress
    euler_maruyama_sampler_partial = functools.partial(
        euler_maruyama_sampler,
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        batch_size=training_cfg.get("eval_batch_size", 64),
        num_steps=training_cfg.get("eval_num_steps", 500),
        device=device,
    )

    # Start training the score model
    score_model_trainer(
        data_loader=data_loader,
        device=device,
        ema_decay=training_cfg.ema_decay,
        lr=training_cfg.learning_rate,
        marginal_prob_std_fn=marginal_prob_std_fn,
        n_epochs=training_cfg.num_epochs,
        score_model=score_model,
        loss_fn=loss_fn,
        eval_fn=euler_maruyama_sampler_partial,
        evaluate_every=training_cfg.get("evaluate_every", 10),
        **trainer_kwargs,
    )


if __name__ == "__main__":
    main()