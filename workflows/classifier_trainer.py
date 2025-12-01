import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn

from hydra.core.hydra_config import HydraConfig
from src.trainer.datasets import get_dataset_loaders
from src.models.registry import get_model_by_name

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    training_cfg = cfg.exps.training

    model_cfg = OmegaConf.to_container(cfg.exps.model, resolve=True)
    model = get_model_by_name(cfg.exps.model.name, **model_cfg).to(training_cfg.device)

    if cfg.exps.data.loader is not None:
        loader_args = OmegaConf.to_container(cfg.exps.data.loader, resolve=True)
        data_loader = get_dataset_loaders(**loader_args)

    opt = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(training_cfg.num_epochs):
        model.train()
        for x, y in data_loader.train:
            x, y = x.to(training_cfg.device), y.to(training_cfg.device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader.test:
                x, y = x.to(training_cfg.device), y.to(training_cfg.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}: test acc = {correct/total:.4f}", end="\r")
    
    # Set up output directory such that checkpoints are saved in the hydra run directory
    hydra_run_dir = Path(HydraConfig.get().run.dir)
    checkpoint_subdir = training_cfg.get("checkpoint_dir", "checkpoints")
    output_dir = hydra_run_dir / checkpoint_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "cnet.pt")


if __name__ == "__main__":
    main()