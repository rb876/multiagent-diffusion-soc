from typing import Optional, Sequence

import torch
import torch.nn as nn

def tv(y):
    return (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean() + (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean()

class AveragingConsistencyLoss(nn.Module):
    """
    For the "average" aggregator (each process contributes everywhere):
      L = sum_p mean( rho(x_p - y) )

    This prevents the averaged image y from becoming a ghosty blend of
    inconsistent hypotheses by encouraging all processes to agree with y.
    """

    def __init__(
        self,
        weight: float = 1.0,
        use_charbonnier: bool = True,
        eps: float = 1e-3,
        detach_target: bool = True,  # usually helps avoid "meet-in-the-middle" blur
    ):
        super().__init__()
        self.weight = weight
        self.use_charbonnier = use_charbonnier
        self.eps = eps
        self.detach_target = detach_target

    def _rho(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_charbonnier:
            return torch.sqrt(x * x + self.eps * self.eps)
        return x.abs()

    def forward(self, y: torch.Tensor, processes: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        if processes is None or len(processes) == 0:
            return torch.tensor(0.0, device=y.device)

        y_tgt = y.detach() if self.detach_target else y

        loss = 0.0
        for xp in processes:
            loss = loss + self._rho(xp - y_tgt).mean()

        return self.weight * loss / len(processes)