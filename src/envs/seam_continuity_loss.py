from typing import Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W)
    returns: (B,C,H,W) discrete Laplacian
    """
    kernel = torch.tensor(
        [[0.0,  1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0,  1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    C = x.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)  # depthwise
    return F.conv2d(x, kernel, padding=1, groups=C)


class SeamContinuityLoss(nn.Module):
    """
    Seam loss that supports:
      - alignment on aggregated Y (processes=None)
      - explicit coupling between processes (processes provided)

    Uses masks (P,C,H,W) to infer vertical segments and seams.
    Assumes each mask is (roughly) a contiguous band along height (true for your "split").
    """

    def __init__(
        self,
        mask: torch.Tensor,          # (P,C,H,W)
        seam_weight: float = 1.0,
        grad_weight: float = 0.05,
        lap_weight: float = 0.0,
        use_charbonnier: bool = True,
        eps: float = 1e-3,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.seam_weight = seam_weight
        self.grad_weight = grad_weight
        self.lap_weight = lap_weight

        self.use_charbonnier = use_charbonnier
        self.eps = eps
        self.mask_threshold = mask_threshold

        self._segments = self._infer_segments_from_masks(mask)
        self._seams = self._build_seams(self._segments)  # list[(row_p, row_q)]

    def _robust_l1(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_charbonnier:
            return torch.sqrt(x * x + self.eps * self.eps)
        return x.abs()

    @staticmethod
    def _infer_segments_from_masks(mask: torch.Tensor) -> list[Tuple[int, int]]:
        P, C, H, W = mask.shape
        segs: list[Tuple[int, int]] = []
        for p in range(P):
            row_active = (mask[p, 0, :, 0] > 0.5).nonzero(as_tuple=False).flatten()
            if row_active.numel() == 0:
                raise ValueError(f"Mask for process {p} has no active pixels.")
            segs.append((int(row_active.min().item()), int(row_active.max().item())))
        return sorted(segs, key=lambda x: x[0])

    @staticmethod
    def _build_seams(segments: list[Tuple[int, int]]) -> list[Tuple[int, int]]:
        seams = []
        for p in range(len(segments) - 1):
            _, e_p = segments[p]
            s_q, _ = segments[p + 1]
            seams.append((e_p, s_q))  # (last row of band p, first row of band p+1)
        return seams

    def _seam_loss_between_rows(self, A: torch.Tensor, row_a: int, B: torch.Tensor, row_b: int) -> torch.Tensor:
        """
        A, B: (B,C,H,W). Compare A at row_a with B at row_b (+ gradient consistency).
        """
        # (1) intensity continuity
        jump = A[:, :, row_a, :] - B[:, :, row_b, :]
        loss_jump = self._robust_l1(jump).mean()

        # (2) vertical gradient continuity (best effort)
        loss_grad = torch.zeros((), device=A.device, dtype=A.dtype)
        H = A.shape[2]
        if row_a - 1 >= 0 and row_b + 1 < H:
            grad_a = A[:, :, row_a, :] - A[:, :, row_a - 1, :]
            grad_b = B[:, :, row_b + 1, :] - B[:, :, row_b, :]
            loss_grad = self._robust_l1(grad_a - grad_b).mean()

        # (3) curvature / Laplacian continuity at seam rows (more semantic stroke smoothness)
        LA = laplacian2d(A)
        LB = laplacian2d(B)
        lap_jump = LA[:, :, row_a, :] - LB[:, :, row_b, :]
        loss_lap = self._robust_l1(lap_jump).mean()

        return self.seam_weight * loss_jump + self.grad_weight * loss_grad + self.lap_weight * loss_lap

    def forward(
        self,
        y: torch.Tensor, # (B,C,H,W) aggregated
    ) -> torch.Tensor:
        # alignment mode: enforce continuity across seams within Y itself.
        loss = torch.zeros((), device=y.device, dtype=y.dtype)
        for (row_top, row_bot) in self._seams:
            loss = loss + self._seam_loss_between_rows(y, row_top, y, row_bot)
        return loss