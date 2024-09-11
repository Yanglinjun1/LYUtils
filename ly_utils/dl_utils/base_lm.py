##########################################################################################
# Description: base lightning class for training any deep learning model.
##########################################################################################

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
