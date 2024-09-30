##########################################################################################
# Description: loss functions to train segmentation models.
##########################################################################################

import torch
from monai.losses import DiceFocalLoss
from typing import List, Dict


class LYSegLosses:
    """
    Class representing a collection of segmentation loss functions (based on MONAI loss functions).

    Args:
        losses (List[Dict]): A list of dictionaries specifying the loss functions to be used.
            Each dictionary should contain the keys 'name' (str) and 'weight' (float, optional).
            The 'name' key specifies the name of the loss function, and the 'weight' key specifies
            the weight to be applied to the loss function. Default is [{'name': 'DiceFocalLoss', 'weight': 1.0}].
        multi_label (bool): Flag indicating whether the segmentation task is multi-label or multi-class.
            If True, the loss functions will include the background class and use sigmoid activation.
            If False, the loss functions will not include the background class and use softmax activation.
            Default is False.

    Attributes:
        loss_dict (dict): A dictionary mapping loss function names to their corresponding loss function objects.
        loss_weights (dict): A dictionary mapping loss function names to their corresponding weights.

    """

    supported_losses = ["DiceFocalLoss"]

    def __init__(
        self,
        losses: List[Dict] = [{"name": "DiceFocalLoss", "weight": 1.0}],
        multi_label: bool = False,
    ):

        self.multi_label = multi_label
        if multi_label:
            include_background = True
            sigmoid = True
            softmax = False
        else:  # multi_class
            include_background = False
            sigmoid = False
            softmax = True

        loss_dict = dict()
        loss_weights = dict()
        for loss in losses:
            loss_name = loss["name"]
            loss_weight = loss.get("weight", 1.0)

            if loss_name not in self.supported_losses:
                raise ValueError(
                    f"Unsupported metric: {loss_name}. Supported metrics are: {self.supported_losses}"
                )

            if loss_name == "DiceFocalLoss":
                loss_dict[loss_name] = DiceFocalLoss(
                    include_background=include_background,
                    sigmoid=sigmoid,
                    softmax=softmax,
                )
                loss_weights[loss_name] = loss_weight

        self.loss_dict = loss_dict
        self.loss_weights = loss_weights

    def __call__(self, pred, seg, device):
        """
        Calculates the loss for the given predicted segmentation and ground truth segmentation.

        Args:
            pred (torch.Tensor): The predicted segmentation tensor.
            seg (torch.Tensor): The ground truth segmentation tensor.
            device (torch.device): The device on which the tensors are located.

        Returns:
            dict: A dictionary containing the individual losses (for logging) and the sum of all losses
            (for logging and backpropagation).
        """

        result_dict = dict()

        sum_loss = torch.zeros([], device=device)
        for loss_name, loss_func in self.loss_dict.items():
            loss = loss_func(pred, seg)
            loss_weight = self.loss_weights[loss_name]
            result_dict[loss_name] = loss.item()
            sum_loss += loss_weight * loss

        result_dict["sum_loss"] = sum_loss

        return result_dict
