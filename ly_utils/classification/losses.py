##########################################################################################
# Description: loss functions for training classification models
##########################################################################################

import torch
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss  # the regular KL, cross entropy loss

__all__ = ["MultiClassFocalLoss"]


class MultiClassFocalLoss(_Loss):
    """
    This function implements and extends the original binary focal
    loss proposed by Lin et al to multi-class focal loss
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__(reduction=reduction)
        if isinstance(weight, list):
            weight = torch.tensor(weight).float()

        # Initiate a cross entropy module with the weight value
        self.CE = CrossEntropyLoss(weight=weight, reduction="none")
        self.gamma = gamma

    def forward(self, input, target, gamma=None):
        # Calculate the raw loss
        raw_loss = self.CE(input, target)

        b_size = input.size(0)
        prob = input.softmax(dim=1)[torch.arange(b_size), target]

        # Calculate the focal loss by multiplying the factors with the
        # above raw loss value
        if gamma is None:
            gamma = self.gamma

        focal_loss = torch.pow(1.0 - prob, gamma) * raw_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        if self.reduction == "none":
            return focal_loss
