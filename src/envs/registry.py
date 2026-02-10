
from src.envs.optimality import ClassifierCEWithConsistency, ClassifierCEWithSmoothness
from src.envs.average_loss_optim import AveragingConsistencyLoss
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
    elif name == "classifier_ce_and_smoothness":  # no seam continuity # this loss is used when the mask is average
        assert aggregator.mask_name == "average", "Only 'average' mask is supported for smoothness loss."
        assert aggregator.num_processes >= 2, "At least 2 processes are required for smoothness loss."
        assert not aggregator.use_overlap, "Overlap masks are not supported for smoothness loss."
        assert aggregator.overlap_size == 0, "Overlap size must be 0 for smoothness loss."
        from src.envs.average_loss_optim import tv
        return ClassifierCEWithSmoothness(
            classifier=classifier,
            smoothness=tv,
            **kwargs
        )
    elif name == "classifier_ce_and_consistency":  # no seam continuity # this loss is used when the mask is average
        assert aggregator.mask_name == "average", "Only 'average' mask is supported for consistency loss."
        assert aggregator.num_processes >= 2, "At least 2 processes are required for consistency loss."
        assert not aggregator.use_overlap, "Overlap masks are not supported for consistency loss."
        assert aggregator.overlap_size == 0, "Overlap size must be 0 for consistency loss."
        return ClassifierCEWithConsistency(
            classifier=classifier,
            consistency=AveragingConsistencyLoss(),
            **kwargs
        )
    else:
        raise ValueError(f"Optimality criterion '{name}' not recognized.")