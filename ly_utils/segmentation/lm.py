##########################################################################################
# Description: base lightning class for segmentation model training, validation and
# testing.
##########################################################################################

import torch
import monai as mn
import lightning as L
from .models import create_seg_model
from .losses import LYSegLosses
from .metrics import LYSegMetrics
from ..dl_utils.optimizers import create_optimizer
from ..dl_utils.lr_schedulers import create_LRScheduler


class LYSegModelBase(L.LightningModule):
    def __init__(
        self,
        model_cfg,
        train_hyp,
    ):
        super().__init__()

        # configure all the settings
        self.process_configurations(model_cfg, train_hyp)

        # model
        self.model = create_seg_model(**self.model_args)

        # loss
        self.loss_func = LYSegLosses(self.losses, self.multi_label)

        # metric
        self.train_metric_func = LYSegMetrics(
            self.metrics, self.label_index, self.multi_label
        )
        self.val_metric_func = LYSegMetrics(
            self.metrics, self.label_index, self.multi_label
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, seg = batch["img"], batch["seg"]

        pred = self.model(img)
        loss_dict = self.loss_func(pred, seg)
        pred_list = [self.post_process(data) for data in self.decollate_batch(pred)]

        # logging: loss
        self.log_loss(loss_dict, "train")

        # logging: metric
        _ = self.train_metric_func(pred_list, seg)

        loss = loss_dict["sum_loss"]

        return loss

    def on_training_epoch_end(self):

        result_dict = self.train_metric_func.aggregate()

        # logging: metric
        self.log_metric(result_dict, "train")

        # reset the metric
        self.train_metric_func.reset()

    def validation_step(self, batch, batch_idx):
        img, seg = batch["img"], batch["seg"]

        pred = self.model(img)
        loss = self.loss_func(pred, seg)
        pred_list = [self.post_process(data) for data in self.decollate_batch(pred)]

        # logging: loss
        self.log_loss(loss, "val")

        # logging: metric
        _ = self.val_metric_func(pred_list, seg)

        # logging: visualization: TODO

    def on_validation_epoch_end(self):

        result_dict = self.val_metric_func.aggregate()

        # logging: metric
        self.log_metric(result_dict, "val")

        # reset the metric
        self.val_metric_func.reset()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), **self.opt_params)
        scheduler = create_LRScheduler(optimizer, self.lr_scheduler_params)

        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def log_loss(self, loss_dict, mode="train"):
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

    def log_metric(self, result_dict, mode="train"):

        for metric, metric_result in result_dict.items():
            metric_dict_for_log = (
                dict()
            )  # to log the same metric, e.g., dice, in one plot

            # for each label
            for ind, label_name in self.label_index.items():
                if ind == 0 and not self.multi_label:
                    continue
                metric_dict_for_log[f"{mode}_{metric}_{label_name}"] = metric_result[
                    ind
                ].item()

            # mean metric across labels
            if self.multi_label:
                mean_metric = metric_result.mean().item()
            else:  # if multi_class, skip background
                mean_metric = metric_result[1:].mean().item()
            metric_dict_for_log[f"{mode}_{metric}_mean"] = mean_metric

            self.log_dict(
                metric_dict_for_log,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

    def process_configurations(self, model_cfg, train_hyp):  # TODO
        # e.g., 0:'background'
        label_index = model_cfg.get("label_index", None)
        if label_index is None:
            raise ValueError("Please specify the label index in the model configuration!")
        out_channels = len(label_index)

        # network
        model_name = model_cfg.get("model_name", "SMPUNet")
        model_args = model_cfg.get("model_args", {})
        model_args["model_name"] = model_name
        model_args["in_channels"] = model_args.get("in_channels", 1)
        if model_args.get("out_channels") != out_channels:
            model_args["out_channels"] = out_channels

        # multi_label or multi_class
        multi_label = model_cfg.get("multi_label", None)
        if multi_label is None:
            raise ValueError("Please specify if the seg model is multi-label!")

        # wandb image visulization; order to overlay labels TODO
        label_overlay_order = model_cfg.get(
            "label_overlay_order", {i: i for i in range(out_channels)}
        )

        # loss parameters
        losses = train_hyp.get("losses", [{"name": "DiceFocalLoss", "weight": 1.0}])

        # metric parameters
        metrics = model_cfg.get("metrics", ["dice", "hausdorff"])

        # optimizer parameters
        opt_params = train_hyp.get("optimizer", {"opt": "Adam", "lr": 1e-3})

        # scheduler parameters
        lr_scheduler_params = train_hyp.get("lr_scheduler", {"scheduler_name": None})

        # post processing
        self.post_process = self.build_post_processing(out_channels, multi_label)
        self.decollate_batch = mn.data.decollate_batch

        # assign all as the attributes
        self.label_index = label_index
        self.label_overlay_order = label_overlay_order
        self.out_channels = out_channels
        self.model_args = model_args
        self.multi_label = multi_label
        self.losses = losses
        self.metrics = metrics
        self.opt_params = opt_params
        self.lr_scheduler_params = lr_scheduler_params

    def build_post_processing(self, num_class, multi_label):
        if multi_label:
            # multi_label: 1) sigmoid, 2) threshold of 0.5
            post_process = mn.transforms.Compose(
                [
                    mn.transforms.Activations(sigmoid=True),
                    mn.transforms.AsDiscrete(threshold=0.5),
                ]
            )
        else:
            # multi_class: 1) softmax, 2) argmax, 3) onehot
            post_process = mn.transforms.Compose(
                [
                    mn.transforms.Activations(softmax=True),
                    mn.transforms.AsDiscrete(argmax=True, to_onehot=num_class),
                ]
            )

        return post_process