from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.seam_continuity_loss import SeamContinuityLoss

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
    """
    def __init__(self, classifier: nn.Module, seam_loss: SeamContinuityLoss, reduction: str = "mean") -> None:
        super().__init__()
        self.classifier = classifier
        self.seam_loss = seam_loss
        self.reduction = reduction

    def _ce(self, y: torch.Tensor, target_label: int) -> torch.Tensor:
        logits = self.classifier(y)
        return - F.log_softmax(logits, dim=-1)[:, target_label].mean()

    def loss(self, y: torch.Tensor, target_labels: list[int]) -> torch.Tensor:
        return self._ce(y, target_labels) + self.seam_loss(y)
    
    def get_terminal_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        if target_labels is None:
            raise ValueError("target_labels must be provided for terminal state loss.")
        return self.loss(state, target_labels) 
    
    def get_running_state_loss(
        self, state: torch.Tensor, target_labels: torch.Tensor,
    ) -> torch.Tensor:
        # Running loss only includes seam continuity term.
        return self.loss(state, target_labels)

