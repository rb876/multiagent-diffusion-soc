
from src.envs.seam_continuity_loss import SeamContinuityLoss

def get_optimality_criterion(name, classifier, mask, **kwargs):
    if name == "classifier_ce_with_alignment_or_cooperation": # seamless alignment or cooperation
        from src.envs.optimality import ClassifierCEWithAlignmentOrCooperation
        return ClassifierCEWithAlignmentOrCooperation(
            classifier=classifier,
            seam_loss=SeamContinuityLoss(mask=mask),
            **kwargs
        )
    else:
        raise ValueError(f"Optimality criterion '{name}' not recognized.")