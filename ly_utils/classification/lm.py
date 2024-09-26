##########################################################################################
# Description: base lightning class for classification model training, validation and
# testing.
##########################################################################################

from .models import create_cls_model
from .losses import LYClsLosses
from .metrics import LYClsMetrics
from ..dl_utils.base_lm import LYLightningModuleBase


class LYClsModelBase(LYLightningModuleBase):
    def __init__(
        self,
        model_cfg,
        train_hyp,
    ):
        super().__init__()

        # configure all the settings
        self.process_configurations(model_cfg, train_hyp)

        # model
        self.model = create_cls_model(**self.model_args)

        # loss
        self.loss_func = LYClsLosses(self.losses)

        # metric
        self.train_metric_func = LYClsMetrics(self.metrics, self.branch_dict)
        self.val_metric_func = LYClsMetrics(self.metrics, self.branch_dict)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]

        label = self.move_label_to_device(label, device=self.device)

        pred = self.model(img)
        loss_dict = self.loss_func(pred, label, device=self.device)

        # logging: loss
        self.log_loss(loss_dict, "train")

        # logging: metric
        self.train_metric_func(pred, label, device=self.device)

        loss = loss_dict["sum_loss"]

        return loss

    def on_training_epoch_end(self):

        result_dict = self.train_metric_func.aggregate()

        # logging: metric
        self.log_metric(result_dict, "train")

        # reset the metric
        self.train_metric_func.reset()

    def validation_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]

        label = self.move_label_to_device(label, device=self.device)

        pred = self.model(img)
        loss = self.loss_func(pred, label, device=self.device)

        # logging: loss
        self.log_loss(loss, "val")

        # logging: metric
        self.val_metric_func(pred, label, device=self.device)

        # logging image: TODO

    def on_validation_epoch_end(self):

        result_dict = self.val_metric_func.aggregate()

        # logging: metric
        self.log_metric(result_dict, "val")

        # reset the metric
        self.val_metric_func.reset()

    def log_loss(self, loss_dict, mode="train"):  # TODO
        for loss_name, loss in loss_dict.items():
            if loss_name == "sum_loss":
                loss = loss.item()
                self.log(
                    f"{mode}_{loss_name}",
                    loss,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{mode}_{loss_name}",
                    loss,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

    def log_metric(self, result_dict, mode="train"):  # TODO
        metric_dict_for_log = dict()
        for name, value in result_dict.items():
            # e.g., "train_side_f1" or "val_view_auc"
            metric_dict_for_log[f"{mode}_{name}"] = value

        self.log_dict(
            metric_dict_for_log,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def process_configurations(self, model_cfg, train_hyp):
        # e.g., 0:'background'
        label_index = model_cfg.get("label_index", None)
        if label_index is None:
            raise ValueError("Please specify the label index in the model configuration!")
        branch_dict = model_cfg["branch_dict"]

        # neural network args
        model_name = model_cfg.get("model_name", "efficientnet-b0")  # TODO
        model_args = model_cfg.get("model_args", {})
        model_args["model_name"] = model_name
        model_args["weights_path"] = train_hyp.get("weights_path", None)
        model_args["label_index"] = label_index
        model_args["branch_dict"] = branch_dict
        model_args["in_channels"] = model_args.get("in_channels", 1)

        # wandb Image for logging classification results TODO

        # loss parameters
        losses = train_hyp.get("losses", [{"name": "MONAIFocalLoss", "weight": 1.0}])

        # metric parameters
        metrics = model_cfg.get("metrics", ["f1", "auc"])

        # optimizer parameters
        opt_params = train_hyp.get("optimizer", {"opt": "Adam", "lr": 1e-3})

        # scheduler parameters
        lr_scheduler_params = train_hyp.get("lr_scheduler", {"scheduler_name": None})

        # assign all as the attributes
        self.label_index = label_index
        self.branch_dict = branch_dict
        self.model_args = model_args
        self.losses = losses
        self.metrics = metrics
        self.opt_params = opt_params
        self.lr_scheduler_params = lr_scheduler_params

    def build_post_processing(self):

        raise NotImplementedError("Please implement the post processing function!")

    def move_label_to_device(self, label, device):
        label_device = dict()
        for branch_name, branch_tensor in label.items():
            label_device[branch_name] = branch_tensor.to(device)

        return label_device
