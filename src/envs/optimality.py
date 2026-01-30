from abc import ABC, abstractmethod
from typing import Optional, Sequence, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.seam_continuity_loss import SeamContinuityLoss
from src.envs.average_loss_optim import AveragingConsistencyLoss

class OptimalityCriterion(nn.Module, ABC):
    """Abstract interface for computing terminal and running optimality losses."""

    @abstractmethod
    def get_terminal_state_loss(
        self, state: torch.Tensor, target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Loss evaluated at the terminal state."""
        raise NotImplementedError

    @abstractmethod
    def get_running_state_loss(
        self, state: torch.Tensor, target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Loss accumulated along the trajectory."""
        raise NotImplementedError


class ClassifierCEWithAlignmentOrCooperation(nn.Module):
    """
    CE loss on aggregated Y + seam term.
    - If processes is None: seam term aligns Y (stitching smoothness)
    - If processes provided: seam term couples agents explicitly
    """
    def __init__(self, classifier: nn.Module, seam_loss: SeamContinuityLoss, reduction: str = "mean") -> None:
        super().__init__()
        self.classifier = classifier
        self.seam_loss = seam_loss
        self.reduction = reduction

    def _ce(self, y: torch.Tensor, target_label: int) -> torch.Tensor:
        logits = self.classifier(y)
        B = logits.shape[0]
        target = torch.full((B,), int(target_label), device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, target, reduction=self.reduction)

    def loss(self, y: torch.Tensor, target_labels: list[int], processes: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        return self._ce(y, target_labels) + self.seam_loss(y, processes=processes)
    
    def get_terminal_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        if target_labels is None:
            raise ValueError("target_labels must be provided for terminal state loss.")
        return self.loss(state, target_labels) 
    
    def get_running_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor, processes: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        # Running loss only includes seam continuity term.
        return self.loss(state, target_labels, processes=processes)


class ClassifierCEWithSmoothness(nn.Module):
    """
    CE loss on aggregated Y + smoothness term.
    """
    def __init__(self, classifier: nn.Module, smoothness: Callable, alpha: float = 0.01, reduction: str = "mean") -> None:
        super().__init__()
        self.classifier = classifier
        self.smoothness = smoothness
        self.reduction = reduction
        self.alpha = alpha

    def _ce(self, y: torch.Tensor, target_label: int) -> torch.Tensor:
        logits = self.classifier(y)
        B = logits.shape[0]
        target = torch.full((B,), int(target_label), device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, target, reduction=self.reduction)

    def loss(self, y: torch.Tensor, target_labels: list[int]) -> torch.Tensor:
        # Ignore processes for smoothness
        return self._ce(y, target_labels) + 1e-1 * self.smoothness(y)
    
    def get_terminal_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor,
    ) -> torch.Tensor:
        # 
        if target_labels is None:
            raise ValueError("target_labels must be provided for terminal state loss.")
        return self.loss(state, target_labels) 
    
    def get_running_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor, processes: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        # Running loss only includes smoothness term.
        return self.loss(state, target_labels)


class ClassifierCEWithConsistency(nn.Module):
    """
    CE loss on aggregated Y + smoothness term.
    """
    def __init__(self, classifier: nn.Module, consistency: AveragingConsistencyLoss, reduction: str = "mean") -> None:
        super().__init__()
        self.classifier = classifier
        self.consistency = consistency
        self.reduction = reduction

    def _ce(self, y: torch.Tensor, target_label: int) -> torch.Tensor:
        logits = self.classifier(y)
        B = logits.shape[0]
        target = torch.full((B,), int(target_label), device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, target, reduction=self.reduction)

    def loss(self, y: torch.Tensor, target_labels: list[int], processes: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        # Ignore processes for smoothness
        # As we set processes to None in the terminal state loss, consistency is zero there.
        return self._ce(y, target_labels) + self.consistency(y, processes=processes)
    
    def get_terminal_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor,
    ) -> torch.Tensor:
        # 
        if target_labels is None:
            raise ValueError("target_labels must be provided for terminal state loss.")
        return self.loss(state, target_labels) 
    
    def get_running_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor, processes: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        # Running loss only includes smoothness term.
        return self.loss(state, target_labels, processes=processes)
