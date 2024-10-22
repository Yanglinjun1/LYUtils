##########################################################################################
# Description: base lightning class for training any deep learning model.
##########################################################################################

import torch
from collections import OrderedDict
import lightning as L
from ..dl_utils.optimizers import create_optimizer
from ..dl_utils.lr_schedulers import create_LRScheduler


class LYLightningModuleBase(L.LightningModule):
    """
    Base class based on LightningModule for training any deep learning model.
    """

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), **self.opt_params)
        scheduler = create_LRScheduler(optimizer, self.lr_scheduler_params)

        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def log_loss(self):
        raise NotImplementedError

    def log_metric(self):
        raise NotImplementedError

    def process_configurations(self):
        raise NotImplementedError


def create_vanilla_state_dict(ckpt_path, key_name="state_dict"):
    """
    Create a vanilla state dictionary from a checkpoint file.

    Parameters:
    - ckpt_path (str): The path to the checkpoint file.
    - key_name (str): The key name of the state dictionary in the checkpoint file. Default is 'state_dict'.

    Returns:
    - state_dict (OrderedDict): The modified state dictionary.

    """
    checkpoint = torch.load(ckpt_path)
    model_weight = checkpoint[key_name]

    # modify the key names
    state_dict = OrderedDict()
    for key, value in model_weight.items():
        splitted_key = key.split(".")

        new_key = ".".join(splitted_key[1:])  # get rid of the string part, e.g., "model"
        state_dict[new_key] = value

    return state_dict
