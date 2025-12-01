from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Adam
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm
from torchvision.utils import make_grid


def score_model_trainer(
    data_loader,
    device,
    ema_decay,
    loss_fn,
    lr,
    marginal_prob_std_fn,
    n_epochs,
    score_model,
    *,
    load_checkpoint: bool = False,
    path_to_checkpoint: Optional[str] = None,
    checkpoint_every: int = 10,
    output_dir: str = "checkpoints",
    grad_clip_norm: Optional[float] = None,
    evaluate_every: int = 10,
    eval_fn: Optional[Any] = None,
    log_to_wandb: bool = True,
    wandb_run_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if data_loader is None:
        raise ValueError("data_loader must not be None.")
    
    
    score_model = score_model.to(device)
    score_model.train()

    optimizer = Adam(score_model.parameters(), lr=lr)
    ema_model = AveragedModel(
        score_model,
        avg_fn=lambda avg_p, p, n: ema_decay * avg_p + (1.0 - ema_decay) * p,
    )

    start_epoch = 1
    global_step = 0

    if load_checkpoint:
        if not path_to_checkpoint:
            raise ValueError("path_to_checkpoint must be provided when load_checkpoint=True.")
        ckpt = torch.load(path_to_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            score_model.load_state_dict(ckpt["model_state"])
            if "ema_state" in ckpt:
                ema_model.load_state_dict(ckpt["ema_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = max(int(ckpt.get("epoch", start_epoch)), 1)
            global_step = int(ckpt.get("global_step", global_step))
        else:
            score_model.load_state_dict(ckpt)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if log_to_wandb:
        import wandb
        wandb_run = wandb.init(**(wandb_run_kwargs or {}))
        wandb.watch(score_model, log="gradients", log_freq=100)

    metrics: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, n_epochs + 1):
        score_model.train()
        avg_loss = 0.0
        num_items = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(score_model.parameters(), grad_clip_norm)

            optimizer.step()

            ema_model.update_parameters(score_model)

            batch_size = x.size(0)
            global_step += batch_size
            avg_loss += loss.item() * batch_size
            num_items += batch_size

            running_avg = avg_loss / max(num_items, 1)
            pbar.set_postfix(loss=f"{running_avg:.6f}")

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/running_loss": running_avg,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        epoch_loss = avg_loss / max(num_items, 1)
        metrics.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "num_items": num_items,
                "global_step": global_step,
            }
        )
        print(f"Epoch [{epoch}/{n_epochs}]  Average Loss: {epoch_loss:.6f}")

        if wandb_run is not None:
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "epoch": epoch,
                    "global_step": global_step,
                },
                step=global_step,
            )

        if evaluate_every and epoch % evaluate_every == 0 and eval_fn is not None:
            samples = eval_fn(score_model)
            if wandb_run is not None:
                grid = make_grid(samples.detach().cpu(), nrow=8, normalize=True, value_range=(0.0, 1.0))
                wandb.log({"eval/samples": wandb.Image(grid)}, step=global_step)

        if checkpoint_every and epoch % checkpoint_every == 0:
            _save_checkpoint(
                output_dir_path / f"epoch-{epoch:04d}.ckpt",
                score_model,
                ema_model,
                optimizer,
                ema_decay,
                epoch,
                global_step,
            )
            _save_ema_checkpoint(
                output_dir_path / f"epoch-{epoch:04d}-ema.ckpt",
                ema_model,
                ema_decay,
                epoch,
                global_step,
            )

    last_epoch = metrics[-1]["epoch"] if metrics else start_epoch - 1
    _save_checkpoint(
        output_dir_path / "latest.ckpt",
        score_model,
        ema_model,
        optimizer,
        ema_decay,
        last_epoch,
        global_step,
    )
    _save_ema_checkpoint(
        output_dir_path / "latest-ema.ckpt",
        ema_model,
        ema_decay,
        last_epoch,
        global_step,
    )

    if wandb_run is not None:
        wandb_run.finish()

    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema_model: AveragedModel,
    optimizer: Adam,
    ema_decay: float,
    epoch: int,
    global_step: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "ema_decay": ema_decay,
    }
    torch.save(checkpoint, path)


def _save_ema_checkpoint(
    path: Path,
    ema_model: AveragedModel,
    ema_decay: float,
    epoch: int,
    global_step: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "ema_state": ema_model.state_dict(),
        "ema_decay": ema_decay,
    }
    torch.save(checkpoint, path)
