from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimalityCriterion(nn.Module, ABC):
    """Abstract interface for computing terminal and running optimality losses."""

    @abstractmethod
    def get_terminal_optimality_loss(
        self, state: torch.Tensor, target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Loss evaluated at the terminal state."""
        raise NotImplementedError

    @abstractmethod
    def get_running_optimality_loss(
        self, state: torch.Tensor, target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Loss accumulated along the trajectory."""
        raise NotImplementedError


class ClassifierCEOptimalityCriterion(OptimalityCriterion):
    """Optimality defined by classifier cross-entropy."""

    def __init__(self, classifier: nn.Module, reduction: str = "mean") -> None:
        super().__init__()
        self.classifier = classifier
        self.reduction = reduction

    def _ce_loss(self, logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        target_labels = torch.full(
            (batch_size,), 
            target_labels, 
            device=logits.device, 
            dtype=torch.long
        )
        return nn.functional.cross_entropy(logits, target_labels, reduction=self.reduction)

    def get_terminal_optimality_loss(
        self, state: torch.Tensor, target_labels: int | list[int]
    ) -> torch.Tensor:
        if len(target_labels) > 1:
            # allowed subset of labels for terminal optimality, i.e., varying targets (0, 1, 4, ...).
            raise NotImplementedError("Terminal optimality with multiple target labels is not implemented.")
        elif len(target_labels) == 1:
            # fixed target label for terminal optimality. E.g., always target class '1'.    
            return self._ce_loss(self.classifier(state), target_labels[0])
        else:
            raise ValueError("Target labels list is empty.")

    def get_running_optimality_loss(
        self, state: torch.Tensor, target_labels: int | list[int]
    ) -> torch.Tensor:
        if len(target_labels) > 1:
            # allowed subset of labels for running optimality, i.e., varying targets (0, 1, 4, ...).
            raise NotImplementedError("Running optimality with multiple target labels is not implemented.")
        elif len(target_labels) == 1:
            # fixed target label for running optimality. E.g., always target class '1'.
            return self._ce_loss(self.classifier(state), target_labels[0])
        else:
            raise ValueError("Target labels list is empty.")
