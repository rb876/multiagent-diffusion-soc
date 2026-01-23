import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from src.envs.aggregator import ImageMaskAggregator


def boundaries_from_mask(mask_pchw: np.ndarray) -> np.ndarray:
    """
    mask_pchw: (P, C, H, W) with 0/1 weights.
    Returns boundary map (H, W) in {0,1} where label changes.
    """
    # collapse channels (they're identical in your split mask anyway)
    m = mask_pchw[:, 0]  # (P, H, W)

    # label by strongest mask at each pixel (ties in overlap go to lowest index)
    labels = np.argmax(m, axis=0).astype(np.int32)  # (H, W)

    # boundary where label changes with 4-neighborhood
    b = np.zeros_like(labels, dtype=bool)
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return b.astype(np.float32), labels


def main(
    img_dims: tuple[int, int, int] = (1, 27, 27),
    num_processes: int = 2,
    overlap_size: int = 0,
    show: bool = False,
):
    use_overlap = overlap_size > 0
    C, H, W = img_dims

    a = ImageMaskAggregator(
        img_dims=img_dims,
        mask_name="split",
        num_processes=num_processes,
        overlap_size=overlap_size,
        use_overlap=use_overlap,
        eps=1e-6,
    )

    # ---- compute seam locations (same logic as create_split_process_masks) ----
    base = H // num_processes
    rem = H % num_processes

    segments = []
    cur = 0
    for i in range(num_processes):
        sz = base + (1 if i < rem else 0)
        start = cur
        end = cur + sz
        segments.append((start, end))
        cur = end

    # seam y positions are between segments: y = end - 0.5 in imshow coordinates
    seam_ys = [end - 0.5 for (_, end) in segments[:-1]]

    # optional: overlap band edges (will be two dashed lines per seam)
    half_ov = overlap_size // 2
    overlap_edge_ys = []
    if use_overlap and overlap_size > 0 and num_processes > 1:
        for (_, end) in segments[:-1]:
            overlap_edge_ys.extend([end - half_ov - 0.5, end + half_ov - 0.5])

    # ---- example process states ----
    processes = [i * torch.ones(1, *img_dims, device=a.device) for i in range(num_processes)]
    agg = a(processes=processes).squeeze().detach().cpu().numpy()
    proc_states = [p.squeeze().detach().cpu().numpy() for p in processes]

    # ---- colormaps ----
    base_cmap = plt.cm.get_cmap("Pastel1")
    cmap_disc = colors.ListedColormap(base_cmap(np.linspace(0, 1, num_processes)))
    bounds = np.arange(-0.5, num_processes + 0.5, 1.0)
    norm_disc = colors.BoundaryNorm(bounds, cmap_disc.N)

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    # ---- plotting ----
    n_plots = num_processes + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3), gridspec_kw={"wspace": 0.02})
    if n_plots == 1:
        axes = [axes]

    titles = [rf"$X^{{u,{i}}}_t$" for i in range(num_processes)] + [r"$Y_t$"]

    # process panels (discrete)
    for i in range(num_processes):
        ax = axes[i]
        ax.imshow(proc_states[i], cmap=cmap_disc, norm=norm_disc, interpolation="nearest")
        ax.set_title(titles[i], fontsize=14)
        ax.axis("off")

    # aggregated panel (continuous) + SINGLE seam line(s)
    ax = axes[-1]
    cont_norm = colors.Normalize(vmin=0, vmax=num_processes - 1)
    ax.imshow(agg, cmap="Pastel1", norm=cont_norm, interpolation="nearest")

    for y in seam_ys:
        ax.axhline(y=y, linewidth=1.0, color="black")

    ax.set_title(titles[-1], fontsize=14)
    ax.axis("off")

    out_path = pathlib.Path(__file__).resolve().parent / f"_agg_state_{num_processes}_{overlap_size}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main(num_processes=2, overlap_size=0)
    main(num_processes=3, overlap_size=0)
    main(num_processes=4, overlap_size=0)
    print("This is the display aggregator module.")