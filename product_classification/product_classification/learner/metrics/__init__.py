"""Built-in metrics"""
from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score, hamming_loss


class IMetric(ABC):
    """
    A metric that is computed at the end of an epoch during fit or evaluation
    """

    def __init__(self) -> None:
        self._state = 0
        self._counter = 0

    def result(self) -> float:
        """Return the metric as average of cumulated metrics"""
        return self._state / self._counter if self._counter else 0

    def reset_states(self) -> None:
        """Reset states"""
        self._state = 0
        self._counter = 0

    @abstractmethod
    def update_states(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update states

        Args:
            predictions (torch.Tensor): what the network predicted
            targets (torch.Tensor): the ground truth results
        """

    @abstractmethod
    def name(self) -> str:
        """Get name of the metric"""


class Accuracy(IMetric):
    """Calculates how often predictions matches labels"""

    def update_states(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update states at each iteration

        Args:
            predictions (torch.Tensor): tensor of shape B x NbCls, where B is the batch size and
                NbCls is the number of labels, that contains predictions
            targets (torch.Tensor): tensor of shape B x 1 that contains targets, a target is the id
                of the ground truth class

        Raises:
            IndexError: if the batch size in predictions differ from the number of targets
        """
        if predictions.shape[0] != targets.shape[0]:
            raise IndexError(f"Nb of predictions ({predictions.shape[0]}) != targets ({targets.shape[0]})")
        nb_batches = targets.shape[0]
        estimates = predictions.argmax(dim=1).view(nb_batches, -1)
        reshaped_targs = targets.view(nb_batches, -1)
        self._state += (estimates == reshaped_targs).float().mean().item()
        self._counter += 1

    def name(self) -> str:
        """Get name of the metric"""
        return "accuracy"


class FScore(IMetric):
    """Calculates how often predictions matches labels"""

    def update_states(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update states at each iteration

        Args:
            predictions (torch.Tensor): tensor of shape B x NbCls, where B is the batch size and
                NbCls is the number of labels, that contains predictions
            targets (torch.Tensor): tensor of shape B x 1 that contains targets, a target is the id
                of the ground truth class

        Raises
            IndexError: if the batch size in predictions differ from the number of targets

        """
        if predictions.shape[0] != targets.shape[0]:
            raise IndexError(f"Nb of predictions ({predictions.shape[0]}) != targets ({targets.shape[0]})")
        # Transform probability tensor [0.32, 0.65, 0.2, 0.3, ...] to binary: [0, 1, 0, 0, ...] based on threshold = 0.5
        predictions = torch.where(predictions >= 0.5, 1, 0)
        self._state += f1_score(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy(), average="samples")
        self._counter += 1

    def name(self) -> str:
        """Get name of the metric

        Returns
            str: Name of the metric

        """
        return "f1_score"


class HammingLoss(IMetric):
    """Calculates how often predictions matches labels"""

    def update_states(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update states at each iteration

        Args:
            predictions (torch.Tensor): tensor of shape B x NbCls, where B is the batch size and
                NbCls is the number of labels, that contains predictions
            targets (torch.Tensor): tensor of shape B x 1 that contains targets, a target is the id
                of the ground truth class

        Raises
            IndexError: if the batch size in predictions differ from the number of targets

        """
        if predictions.shape[0] != targets.shape[0]:
            raise IndexError(f"Nb of predictions ({predictions.shape[0]}) != targets ({targets.shape[0]})")
        # Transform probability tensor [0.32, 0.65, 0.2, 0.3, ...] to binary: [0, 1, 0, 0, ...] based on threshold = 0.5
        predictions = torch.where(predictions >= 0.5, 1, 0)
        self._state += hamming_loss(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        self._counter += 1

    def name(self) -> str:
        """Get name of the metric

        Returns
            str: Name of the metric

        """
        return "hamming_loss"
