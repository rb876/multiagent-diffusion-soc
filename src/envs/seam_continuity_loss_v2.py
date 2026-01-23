from typing import Sequence, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian2d(x: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W) -> (B,C,H,W) depthwise discrete Laplacian."""
    kernel = torch.tensor(
        [[0.0,  1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0,  1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    C = x.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=C)


class SeamContinuityLossV2(nn.Module):
    """
    Seam loss that supports overlap properly by penalizing inconsistencies in the *overlap band*.
    Also works without overlap by using a small band around the seam.

    Masks: (P,C,H,W), assumed split along H (contiguous-ish bands).
    """

    def __init__(
        self,
        mask: torch.Tensor,          # (P,C,H,W)
        seam_weight: float = 1.0,
        grad_weight: float = 0.5,
        lap_weight: float = 0.1,
        use_charbonnier: bool = True,
        eps: float = 1e-3,
        mask_threshold: float = 0.5,
        band_width_no_overlap: int = 1,   # rows taken on each side when no overlap exists
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

        self.seam_weight = seam_weight
        self.grad_weight = grad_weight
        self.lap_weight = lap_weight

        self.use_charbonnier = use_charbonnier
        self.eps = eps
        self.mask_threshold = mask_threshold
        self.band_width_no_overlap = band_width_no_overlap

        self._segments = self._infer_segments_from_masks(mask, mask_threshold)
        self._seams = self._build_seams(self._segments)  # list[(p, q, end_p, start_q)]

    def _robust_l1(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_charbonnier:
            return torch.sqrt(x * x + self.eps * self.eps)
        return x.abs()

    @staticmethod
    def _infer_segments_from_masks(mask: torch.Tensor, thr: float) -> list[Tuple[int, int]]:
        P, C, H, W = mask.shape
        segs: list[Tuple[int, int]] = []
        # Use any column where mask might be active; reduce over W for robustness.
        for p in range(P):
            # rows active if any pixel in that row is active
            row_active = (mask[p, 0].max(dim=-1).values > thr).nonzero(as_tuple=False).flatten()
            if row_active.numel() == 0:
                raise ValueError(f"Mask for process {p} has no active pixels.")
            segs.append((int(row_active.min().item()), int(row_active.max().item())))
        return sorted(segs, key=lambda x: x[0])

    @staticmethod
    def _build_seams(segments: list[Tuple[int, int]]) -> list[Tuple[int, int, int, int]]:
        seams = []
        for p in range(len(segments) - 1):
            s_p, e_p = segments[p]
            s_q, e_q = segments[p + 1]
            seams.append((p, p + 1, e_p, s_q))
        return seams

    def _masked_mean(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        m: (1,1,H,W) or (B,1,H,W) broadcastable
        """
        m = m.to(dtype=x.dtype)
        denom = m.sum().clamp_min(self.eps)
        return (x * m).sum() / denom

    def _band_mask_overlap_or_seam(
        self, p: int, q: int, end_p: int, start_q: int, device
    ) -> torch.Tensor:
        """
        Returns m: (1,1,H,W) selecting overlap band if it exists, else a thin seam band.
        """
        P, C, H, W = self.mask.shape
        mp = (self.mask[p:p+1, 0:1] > self.mask_threshold)  # (1,1,H,W) bool
        mq = (self.mask[q:q+1, 0:1] > self.mask_threshold)

        overlap = (mp & mq)  # (1,1,H,W)
        if overlap.any():
            return overlap.to(device=device)

        # No overlap: build a thin band around seam rows.
        bw = int(self.band_width_no_overlap)
        m = torch.zeros((1, 1, H, W), device=device, dtype=torch.bool)

        # include rows [end_p-bw+1 .. end_p] and [start_q .. start_q+bw-1]
        r0 = max(0, end_p - bw + 1)
        r1 = min(H, end_p + 1)
        r2 = max(0, start_q)
        r3 = min(H, start_q + bw)

        m[:, :, r0:r1, :] = True
        m[:, :, r2:r3, :] = True
        return m

    def _loss_in_band(self, A: torch.Tensor, B: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        """
        A,B: (B,C,H,W), band: (1,1,H,W) bool.
        """
        # (1) intensity consistency
        jump = self._robust_l1(A - B)
        loss_jump = self._masked_mean(jump, band)

        # (2) vertical gradient consistency
        # one-step vertical diff; pad to keep shape
        gA = A[:, :, 1:, :] - A[:, :, :-1, :]
        gB = B[:, :, 1:, :] - B[:, :, :-1, :]
        gA = F.pad(gA, (0, 0, 0, 1))
        gB = F.pad(gB, (0, 0, 0, 1))
        loss_grad = self._masked_mean(self._robust_l1(gA - gB), band)

        # (3) Laplacian consistency
        LA = laplacian2d(A)
        LB = laplacian2d(B)
        loss_lap = self._masked_mean(self._robust_l1(LA - LB), band)

        return self.seam_weight * loss_jump + self.grad_weight * loss_grad + self.lap_weight * loss_lap

    def forward(
        self,
        y: torch.Tensor,                               # (B,C,H,W) aggregated
        processes: Optional[Sequence[torch.Tensor]] = None,  # length P, each (B,C,H,W)
    ) -> torch.Tensor:
        device = y.device
        loss = torch.zeros((), device=device, dtype=y.dtype)

        if processes is None:
            # Alignment mode: enforce continuity between adjacent regions within y
            # Use p/q band masks but compare y with y (still meaningful because the mask selects the seam/overlap zone)
            for (p, q, end_p, start_q) in self._seams:
                band = self._band_mask_overlap_or_seam(p, q, end_p, start_q, device=device)
                loss = loss + self._loss_in_band(y, y, band)
            return loss

        if len(processes) != self.mask.shape[0]:
            raise ValueError(f"Expected {self.mask.shape[0]} processes, got {len(processes)}")

        for (p, q, end_p, start_q) in self._seams:
            band = self._band_mask_overlap_or_seam(p, q, end_p, start_q, device=device)
            loss = loss + self._loss_in_band(processes[p], processes[q], band)

        return loss
