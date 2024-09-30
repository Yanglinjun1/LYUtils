##########################################################################################
# Description: loss functions for training classification models
##########################################################################################

import torch
import monai as mn
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss  # the regular KL, cross entropy loss
from typing import List, Dict


class LYClsLosses:
    """
    Class for calculating losses for multi-branch classification tasks.

    Args:
        losses (List[Dict]): List of dictionaries specifying the losses to be used for each branch.
            Each dictionary should contain the following keys:
            - "name" (str): Name of the loss function.
            - "weight" (float, optional): Weight for the loss function. Default is 1.0.

    Attributes:
        supported_losses (List[str]): List of supported loss function names for checking purpose.
        loss_dict: Dictionary containing the loss functions, each of which will be used for all branches.
        loss_weights: Dictionary containing the weights for each loss function.

    Methods:
        __call__(self, pred: Dict, label: Dict, device) -> Dict: Calculates the loss for each branch
            of the task and returns a dictionary containing loss values for each branch and the sum
            of all losses.

    Example:
        losses = [
            {"name": "MONAIFocalLoss", "weight": 1.0},
            {"name": "AnotherLoss", "weight": 0.5}
        ]
        cls_losses = LYClsLosses(losses)
        result = cls_losses(pred, label, device)
    """

    supported_losses = ["MONAIFocalLoss"]

    def __init__(
        self,
        losses: List[Dict] = [{"name": "MONAIFocalLoss", "weight": 1.0}],
    ):

        loss_dict = dict()
        loss_weights = dict()
        for loss in losses:
            loss_name = loss["name"]
            loss_weight = loss.get("weight", 1.0)

            if loss_name not in self.supported_losses:
                raise ValueError(
                    f"Unsupported metric: {loss_name}. Supported metrics are: {self.supported_losses}"
                )

            if loss_name == "MONAIFocalLoss":
                loss_dict[loss_name] = mn.losses.FocalLoss(
                    to_onehot_y=True, use_softmax=True
                )
                loss_weights[loss_name] = loss_weight

        self.loss_dict = loss_dict
        self.loss_weights = loss_weights

    def __call__(self, pred: Dict, label: Dict, device):
        """to be called in the training/validation step of the lightning module to
        calculate the loss for each branch of the task. Also, the "sum_loss" is calculated
        for backpropagation.

        Args:
            pred (Dict): dictionary containing predictions for each branch of task
            label (Dict): dictionary labels containing labels for each branch of task
            device: to device to calculate the loss; provide self.device from lightning module

        Returns:
            result_dict: dictionary containing loss values for each branch (for logging only)
            and the sum of all losses (for logging and backpropagation)
        """
        result_dict = dict()

        sum_loss = torch.zeros([], device=device)
        # each loss type
        for loss_name, loss_func in self.loss_dict.items():
            loss_weight = self.loss_weights[loss_name]

            current_loss = torch.zeros([], device=device)
            # each branch
            for branch_name, tensor in pred.items():
                # accumulate loss
                loss = loss_func(tensor, label[branch_name])
                current_loss += loss

                # "loss_name" + "_branch_name"
                result_dict[f"{loss_name}_{branch_name}"] = loss.item()

            current_loss /= len(pred)  # average loss across all branches
            sum_loss += loss_weight * current_loss

        result_dict["sum_loss"] = sum_loss

        return result_dict


class MultiClassFocalLoss(_Loss):
    """
    This function implements and extends the original binary focal
    loss proposed by Lin et al to multi-class focal loss.

    This class is not being used anymore, please use MONAIFocalLoss instead.
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
