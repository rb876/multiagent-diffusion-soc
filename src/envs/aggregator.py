from typing import Tuple, Sequence
import torch


def create_split_process_masks(
    img_dims: Tuple[int, int, int],  # (C, H, W)
    num_processes: int,
    device=None,
    use_overlap: bool = True,
    overlap_size: int = 4,
) -> torch.Tensor:
    """
    Split along height into `num_processes` contiguous bands, with optional overlap.
    Returns masks of shape (num_processes, C, H, W).

    - Without overlap: each pixel belongs to exactly one process.
    - With overlap: neighboring bands overlap by `overlap_size` (split across both sides).
    """
    C, H, W = img_dims
    masks = torch.zeros((num_processes, C, H, W), device=device)

    if num_processes < 1:
        raise ValueError("num_processes must be >= 1")

    # --- non-overlapping base segments ---
    base = H // num_processes
    rem = H % num_processes

    segments = []
    cur = 0
    for i in range(num_processes):
        sz = base + (1 if i < rem else 0)
        start = cur
        end = cur + sz  # exclusive
        segments.append((start, end))
        cur = end

    # --- apply optional overlap ---
    if use_overlap and overlap_size > 0 and num_processes > 1:
        half_ov = overlap_size // 2
        for p, (s, e) in enumerate(segments):
            # expand upward (towards smaller h) except for first
            s_ov = s if p == 0 else max(0, s - half_ov)
            # expand downward (towards larger h) except for last
            e_ov = e if p == num_processes - 1 else min(H, e + half_ov)
            masks[p, :, s_ov:e_ov, :] = 1.0
    else:
        # no overlap: just fill base segments
        for p, (s, e) in enumerate(segments):
            masks[p, :, s:e, :] = 1.0

    return masks


class ImageMaskAggregator:
    def __init__(
        self,
        img_dims: tuple[int, int, int],  # (channels, height, width)
        mask_name: str,                  # "split" or "random"
        num_processes: int = 2,
        device: torch.device = torch.device("cuda"),
        use_overlap: bool = True,
        overlap_size: int = 4,
    ):
    
        def _build_masks() -> torch.Tensor:
            if self.mask_name == "split":
                return create_split_process_masks(
                    img_dims=self.img_dims,
                    num_processes=self.num_processes,
                    device=self.device,
                    use_overlap=self.use_overlap,
                    overlap_size=self.overlap_size,
                )  # (P, C, H, W)
            elif self.mask_name == "random":
                pass
            else:
                raise ValueError(f"Unknown mask name: {self.mask_name}")
    
        self.img_dims = img_dims
        self.mask_name = mask_name
        self.device = device
        self.num_processes = num_processes
        self.use_overlap = use_overlap
        self.overlap_size = overlap_size
        self.mask = _build_masks().to(device)


    def __call__(self, processes: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        processes: sequence of tensors, each (B, C, H, W),
                   all same shape.
        Returns:
            mixed: (B, C, H, W)
            where each pixel is taken from exactly one process
            according to the masks.
        """

        if len(processes) == 0:
            raise ValueError("`processes` must be non-empty.")
    
        stacked = torch.stack(processes, dim=0)  # (P, B, C, H, W)
        masks = self.mask.unsqueeze(1)  # (P, 1, C, H, W)

        return (stacked * masks).sum(dim=0)  # (B, C, H, W)