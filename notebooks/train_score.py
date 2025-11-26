import functools
import torch
import torchvision.transforms as transforms
import numpy as np

from torch.optim import Adam
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from unet import UNetModel

    
def marginal_prob_std(t, sigma, device='cuda'):
  """
  Compute the mean and standard deviation of $p(x(t) | x(0))$.
  """
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device='cuda'):
  """Compute the diffusion coefficient of our SDE.
  """
  return torch.tensor(sigma**t, device=device)


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """
  The loss function for training score-based generative models.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))

  return loss


def main():
    device = 'cuda'
    sigma =  25.0

    cfg = {
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "model_channels": 32,
            "channel_mult": [1, 2, 4],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "max_period": 0.005,
        },
    }

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    UNet_model = UNetModel(marginal_prob_std=marginal_prob_std_fn, **cfg["model"])

    score_model = UNet_model.to(device)

    load_model = False
    if load_model:
        ckpt = torch.load('ckpt.pt', map_location=device)
        score_model.load_state_dict(ckpt)
    
    batch_size =  16
    dataset = MNIST('data/', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    n_epochs = 10
    lr = 1e-4
    ema_decay = 0.999
    score_model.to(device)
    optimizer = Adam(score_model.parameters(), lr=lr)
    ema_model = AveragedModel(
        score_model,
        avg_fn=lambda avg_p, p, n: ema_decay * avg_p + (1.0 - ema_decay) * p,
    )

    for epoch in range(1, n_epochs + 1):
        score_model.train()
        avg_loss = 0.0
        num_items = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for x, _ in pbar:        # ignore labels if unused
            x = x.to(device)

            loss = loss_fn(score_model, x, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA parameters
            ema_model.update_parameters(score_model)

            # Accumulate loss
            batch_size = x.size(0)
            avg_loss += loss.item() * batch_size
            num_items += batch_size

            # show running average loss on the bar
            running_avg = avg_loss / max(num_items, 1)
            pbar.set_postfix(loss=f"{running_avg:.6f}")

        avg_loss /= num_items
        print(f"Epoch [{epoch}/{n_epochs}]  Average Loss: {avg_loss:.6f}")
        if epoch % 10 == 0:	
            torch.save(score_model.state_dict(), f"ckpt_{epoch}.pt")

    # Save checkpoints (current and EMA)
    torch.save(score_model.state_dict(), "ckpt.pt")
    torch.save(ema_model.state_dict(), "ckpt_ema.pt")


if __name__ == "__main__":
    main()