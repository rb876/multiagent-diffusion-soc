import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from models import nn
from src.trainer.datasets import get_dataset_loader
from src.models.registry import get_model_by_name

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    training_cfg = cfg.training
    device = training_cfg.device

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = get_model_by_name(cfg.model.name, **model_cfg).to(device)

    data_loader = None
    if cfg.data.loader is not None:
        loader_args = OmegaConf.to_container(cfg.data.loader, resolve=True)
        data_loader = get_dataset_loader(**loader_args)
 
    lr = 1e-3
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}: test acc = {correct/total:.4f}", end="\r")

    torch.save(model.state_dict(), "cnet.pt")


if __name__ == "__main__":
    main()