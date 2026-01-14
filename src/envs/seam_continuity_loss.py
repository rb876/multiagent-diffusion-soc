from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def laplacian2d(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0,  1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0,  1.0, 0.0]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    C = x.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=C)

class SeamContinuityLoss(nn.Module):
    """
    Seam coupling loss between adjacent processes that works with:
      - overlapped split masks (uses overlap rows)
      - non-overlapped split masks (uses a boundary band)

    Assumes vertical split masks (contiguous bands along H), but does NOT assume
    non-overlap or any specific overlap size.
    """

    def __init__(
        self,
        mask: torch.Tensor,          # (P,C,H,W)
        seam_weight: float = 1.0,    # intensity continuity
        grad_weight: float = 0.5,    # optional
        lap_weight: float = 0.1,     # optional
        mask_threshold: float = 0.5,
        boundary_band: int = 1,      # used when no overlap; rows on each side
        use_charbonnier: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        self.seam_weight = seam_weight
        self.grad_weight = grad_weight
        self.lap_weight = lap_weight
        self.mask_threshold = mask_threshold
        self.boundary_band = boundary_band
        self.use_charbonnier = use_charbonnier
        self.eps = eps

    def _robust_l1(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_charbonnier:
            return torch.sqrt(x * x + self.eps * self.eps)
        return x.abs()

    def _rows_active(self, p: int) -> torch.Tensor:
        # boolean mask over H for process p (use one column; your masks are constant over W)
        return (self.mask[p, 0, :, 0] > self.mask_threshold)

    def _overlap_rows(self, p: int) -> torch.Tensor:
        return self._rows_active(p) & self._rows_active(p + 1)

    def _boundary_bands(self, p: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For non-overlap: find boundary between p and p+1 and return matching row indices:
        rows_p: last k rows of p
        rows_q: first k rows of p+1
        """
        a = self._rows_active(p).nonzero(as_tuple=False).flatten()
        b = self._rows_active(p + 1).nonzero(as_tuple=False).flatten()
        if a.numel() == 0 or b.numel() == 0:
            return a, b

        end_p = int(a.max().item())       # last active row of p
        start_q = int(b.min().item())     # first active row of p+1

        k = max(1, int(self.boundary_band))
        rows_p = torch.arange(max(0, end_p - k + 1), end_p + 1, device=self.mask.device)
        rows_q = torch.arange(start_q, min(H, start_q + k), device=self.mask.device)

        # make lengths equal
        L = min(rows_p.numel(), rows_q.numel())
        return rows_p[:L], rows_q[:L]

    def forward(
        self,
        y: torch.Tensor,                               # (B,C,H,W) aggregated (unused for coupling)
        processes: Optional[Sequence[torch.Tensor]] = None,  # length P, each (B,C,H,W)
    ) -> torch.Tensor:
        if processes is None:
            raise ValueError("SeamLoss is intended for coupling mode: pass `processes`.")

        P, C, H, W = self.mask.shape
        if len(processes) != P:
            raise ValueError(f"Expected {P} processes, got {len(processes)}")

        loss = torch.zeros((), device=y.device, dtype=y.dtype)

        # Precompute optional operators once (cheaper)
        LA = LB = None

        for p in range(P - 1):
            A = processes[p]
            B = processes[p + 1]

            # 1) Prefer overlap rows when they exist
            ov = self._overlap_rows(p).nonzero(as_tuple=False).flatten()

            if ov.numel() > 0:
                rows_a = ov
                rows_b = ov
            else:
                # 2) No overlap: use boundary bands
                rows_a, rows_b = self._boundary_bands(p, H)
                if rows_a.numel() == 0 or rows_b.numel() == 0:
                    continue

            # intensity continuity
            jump = A[:, :, rows_a, :] - B[:, :, rows_b, :]
            loss_jump = self._robust_l1(jump).mean()
            loss = loss + self.seam_weight * loss_jump

            if self.grad_weight > 0:
                # vertical gradient consistency (best-effort)
                ra = rows_a
                rb = rows_b
                ra0 = torch.clamp(ra - 1, 0, H - 1)
                rb1 = torch.clamp(rb + 1, 0, H - 1)
                grad_a = A[:, :, ra, :] - A[:, :, ra0, :]
                grad_b = B[:, :, rb1, :] - B[:, :, rb, :]
                loss = loss + self.grad_weight * self._robust_l1(grad_a - grad_b).mean()

            if self.lap_weight > 0:
                LA = laplacian2d(A)
                LB = laplacian2d(B)
                lap_jump = LA[:, :, rows_a, :] - LB[:, :, rows_b, :]
                loss = loss + self.lap_weight * self._robust_l1(lap_jump).mean()

        return loss