
from src.envs.seam_continuity_loss import SeamContinuityLoss

def get_optimality_criterion(
    name, 
    classifier, 
    aggregator, 
    **kwargs
    ):
    if name == "classifier_ce_with_cooperation": # seamless alignment or cooperation
        from src.envs.optimality import ClassifierCEWithAlignmentOrCooperation
        mask = aggregator.mask
        assert aggregator.mask_name == "split", "Only 'split' mask is supported for seam continuity loss."
        assert aggregator.num_processes >= 2, "At least 2 processes are required for seam continuity loss."
        assert not aggregator.use_overlap, "Overlap masks are not supported for seam continuity loss."
        assert aggregator.overlap_size == 0, "Overlap size must be 0 for seam continuity loss."
        return ClassifierCEWithAlignmentOrCooperation(
            classifier=classifier,
            seam_loss=SeamContinuityLoss(mask=mask),
            **kwargs
        )
    else:
        raise ValueError(f"Optimality criterion '{name}' not recognized.")