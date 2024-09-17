##########################################################################################
# Description: loss functions to train segmentation models.
##########################################################################################

import torch
from monai.losses import DiceFocalLoss
from typing import List, Dict


class LYSegLosses:

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
            # TODO check loss types
            if loss_name == "DiceFocalLoss":
                loss_dict[loss_name] = DiceFocalLoss(
                    include_background=include_background,
                    sigmoid=sigmoid,
                    softmax=softmax,
                )
                loss_weights[loss_name] = loss_weight

        self.loss_dict = loss_dict
        self.loss_weights = loss_weights

    def __call__(self, pred, seg):
        result_dict = dict()

        sum_loss = torch.zeros([])
        for loss_name, loss_func in self.loss_dict.items():
            loss = loss_func(pred, seg)
            loss_weight = self.loss_weights[loss_name]
            result_dict[loss_name] = loss.item()
            sum_loss += loss_weight * loss

        result_dict["sum_loss"] = sum_loss

        return result_dict
