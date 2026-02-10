from typing import Tuple, Sequence
import torch


def create_split_process_masks(
    img_dims: Tuple[int, int, int],  # (C, H, W)
    num_processes: int,
    device=None,
    overlap_size: int = 4,
    use_overlap: bool = True,
) -> torch.Tensor:
    C, H, W = img_dims
    masks = torch.zeros((num_processes, C, H, W), device=device)

    if num_processes < 1:
        raise ValueError("num_processes must be >= 1")

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

    if use_overlap and overlap_size > 0 and num_processes > 1:
        half_ov = overlap_size // 2
        for p, (s, e) in enumerate(segments):
            s_ov = s if p == 0 else max(0, s - half_ov)
            e_ov = e if p == num_processes - 1 else min(H, e + half_ov)
            masks[p, :, s_ov:e_ov, :] = 1.0
    else:
        for p, (s, e) in enumerate(segments):
            masks[p, :, s:e, :] = 1.0

    return masks


class ImageMaskAggregator:
    def __init__(
        self,
        img_dims: tuple[int, int, int],
        mask_name: str,
        num_processes: int,
        device: torch.device = torch.device("cuda"),
        overlap_size: int = 4,
        use_overlap: bool = True,
        eps: float = 1e-6,
    ):
        self.device = device
        self.img_dims = img_dims
        self.mask_name = mask_name
        self.num_processes = num_processes
        self.overlap_size = overlap_size
        self.use_overlap = use_overlap
        self.eps = eps

        def _build_masks() -> torch.Tensor:
            if self.mask_name == "split":
                return create_split_process_masks(
                    device=self.device,
                    img_dims=self.img_dims,
                    num_processes=self.num_processes,
                    overlap_size=self.overlap_size,
                    use_overlap=self.use_overlap,
                )  # (P, C, H, W)
            elif self.mask_name == "average":
                raise NotImplementedError("Average masks not implemented.")
            elif self.mask_name == "random":
                raise NotImplementedError("Random masks not implemented.")
            else:
                raise ValueError(f"Unknown mask name: {self.mask_name}")

        self.mask = _build_masks().to(device)

    def __call__(self, processes: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        processes: sequence length P, each (B, C, H, W)
        Returns: mixed (B, C, H, W)

        If masks overlap, pixels are normalized by sum of mask weights to avoid brightening.
        """
        if len(processes) == 0:
            raise ValueError("`processes` must be non-empty.")

        stacked = torch.stack(processes, dim=0)          # (P, B, C, H, W)
        masks = self.mask.unsqueeze(1)                   # (P, 1, C, H, W)

        weighted = (stacked * masks).sum(dim=0)          # (B, C, H, W)
        denom = masks.sum(dim=0)                         # (1, C, H, W)
        denom = denom.clamp_min(self.eps)                # avoid divide by 0

        return weighted / denom