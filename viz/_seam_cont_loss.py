import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch

from src.envs.aggregator import ImageMaskAggregator
from src.envs.seam_continuity_loss import SeamContinuityLoss


def image_with_center_zero_digit(
    img_dims: Tuple[int, int, int],  # (C, W, H)
    outer_radius: int = 5,
    thickness: float = 2.0,
    device=None,
) -> torch.Tensor:
    """Shape: (1, C, W, H). Background 0, ring (digit '0') = 1."""
    C, W, H = img_dims
    img = torch.zeros(1, C, W, H, device=device)

    cx, cy = W // 2, H // 2
    yy, xx = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="ij",
    )
    dist = (xx - cy) ** 2 + (yy - cx) ** 2
    outer = dist <= outer_radius**2
    inner = dist < (outer_radius - thickness) ** 2
    ring = outer & ~inner
    img[:, :, ring] = 1.0
    return img


def shift_tensor(x: torch.Tensor, dx: int = 0, dy: int = 0) -> torch.Tensor:
    """
    Zero-padded shift. x: (B,C,W,H)
    dx shifts along H (cols): right +, left -
    dy shifts along W (rows): down +, up -
    """
    B, C, W, H = x.shape
    out = torch.zeros_like(x)

    xs = max(dy, 0)
    xe = W + min(dy, 0)
    ys = max(dx, 0)
    ye = H + min(dx, 0)

    out[:, :, xs:xe, ys:ye] = x[:, :, xs - dy:xe - dy, ys - dx:ye - dx]
    return out


def seam_band_mask(W: int, H: int, row_p: int, row_q: int, band: int, device) -> torch.Tensor:
    """
    Build a mask (1,1,W,H) highlighting exactly the rows the loss compares (±band).
    Note: Here W is the "row" dimension in your tensors (because shape is (B,C,W,H)).
    """
    m = torch.zeros(1, 1, W, H, device=device, dtype=torch.float32)

    def _mark(r: int):
        r0 = max(0, r - band)
        r1 = min(W, r + band + 1)
        m[:, :, r0:r1, :] = 1.0

    _mark(int(row_p))
    _mark(int(row_q))
    return m


def main(
    img_dims: tuple[int, int, int] = (1, 27, 27),
    show: bool = False,
    shift: Optional[Tuple[int, int]] = (0, 0),  # (dx, dy)
    band: int = 1,  # how many rows above/below seam rows to highlight
):
    # --- setup aggregator (defines device + mask) ---
    a = ImageMaskAggregator(
        img_dims=img_dims,
        mask_name="split",
        num_processes=2,
        overlap_size=0,
        use_overlap=False,
        eps=1e-6,
    )
    device = a.device

    # --- seam loss ---
    seam_loss = SeamContinuityLoss(
        mask=a.mask,
        seam_weight=1.0,
        grad_weight=0.5,
        lap_weight=0.1,
        use_charbonnier=True,
        eps=1e-3,
        mask_threshold=0.5,
    )

    # Seam rows used by the loss (for 2 processes, there is exactly 1 seam)
    # Original class stores list[(row_p, row_q)]
    if len(seam_loss._seams) < 1:
        raise RuntimeError("No seams inferred from mask; cannot visualize seam loss.")
    row_p, row_q = seam_loss._seams[0]

    # --- generate two images ON THE SAME DEVICE ---
    state_1 = image_with_center_zero_digit(img_dims, outer_radius=6, thickness=1.5, device=device)
    state_2 = image_with_center_zero_digit(img_dims, outer_radius=6, thickness=1.5, device=device)

    if shift is not None:
        dx, dy = shift
        state_2 = shift_tensor(state_2, dx=dx, dy=dy)

    # --- aggregate ---
    y = a(processes=[state_1, state_2])  # (1,C,W,H)

    # --- compute losses (coupling mode) ---
    loss_same_1 = seam_loss(y=y, processes=[state_1, state_1]).item()
    loss_same_2 = seam_loss(y=y, processes=[state_2, state_2]).item()
    loss_cross = seam_loss(y=y, processes=[state_1, state_2]).item()
    print("Seam continuity loss [same1, same2, cross]:", [loss_same_1, loss_same_2, loss_cross])

    # --- numpy for plotting ---
    x0_np = state_1.squeeze().detach().cpu().numpy()
    x1_np = state_2.squeeze().detach().cpu().numpy()
    y_np = y.squeeze().detach().cpu().numpy()

    # --- plot 1x4: X0, X1, Y (with seam rows), diff band ---
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(1, 3, figsize=(3, 2))

    titles = [
        rf"$X_t^{{u,0}}$  $L={loss_same_1:.3g}$",
        rf"$X_t^{{u,1}}$  $L={loss_same_2:.3g}$",
        rf"$Y_t$  $L={loss_cross:.3g}$",
    ]

    imgs = [x0_np, x1_np, y_np]

    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap="gray_r", interpolation="nearest")
        ax.axis("off")

        # title INSIDE axes → zero gap
        ax.text(
            0.5, 1.0, t,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=6,
        )

    out_path = pathlib.Path(__file__).resolve().parent / f"seam_loss_viz_shift_{shift[0]}_{shift[1]}_band_{band}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main(shift=(0, 0), band=1)
    main(shift=(4, 0), band=1)
    print("This is the seam continuity loss visualization module.")