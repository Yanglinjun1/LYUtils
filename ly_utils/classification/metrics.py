##########################################################################################
# Description: Utility functions for classification evaluation based on TorchMetrics
##########################################################################################

import torch.nn as nn
from typing import Dict, List, Union
import monai as mn

__all__ = ["LYClsMetrics", "create_metrics", "check_confusion_matrix_metric_name"]


class LYClsMetrics:
    """
    Class for computing and aggregating metrics for multi-branch classification tasks.
    """

    def __init__(self, metric_names, branch_dict: Dict, average: str = "macro"):
        """
        Initializes the LYClsMetrics object. Besides the metric names and the branch dictionary,
        the metric functions will also be created and stored in the object. It is a torch.nn.ModuleDict
        that contains the metric functions for each branch. The metric function for each branch is also
        a torch.nn.ModuleDict that contains the metric functions for each metric. Defaults to use "macro"
        averaging for all metrics and give metric values for each branch.

        Args:
            metric_names (list): List of metric names to compute.
            branch_dict (dict): Dictionary containing the branches and their corresponding metric functions.
        """
        self.metric_names = metric_names
        self.branch_dict = branch_dict
        self.metric_func_dict = create_metrics(self.branch_dict, self.metric_names, average)
        self.average = average

    def __call__(self, pred: Dict, label: Dict, device):
        """
        Computes the metrics for the given predictions and labels.

        Args:
            pred (dict): Dictionary containing the predicted values for each branch.
            label (dict): Dictionary containing the true labels for each branch.
            device: The device to move the metric functions to.
        """
        # move to the same device
        self.metric_func_dict.to(device)

        # over all branches
        for branch_name, branch_metrics in self.metric_func_dict.items():
            pred_branch, label_branch = pred[branch_name], label[branch_name]

            # convert to regular tensor if need to
            if isinstance(pred_branch, mn.data.meta_tensor.MetaTensor):
                pred_branch = pred_branch.as_tensor()

            # over all metrics
            for _, metric_func in branch_metrics.items():
                metric_func.update(pred_branch, label_branch)

    def aggregate(self):
        """
        Aggregates the computed metrics across all branches.

        Returns:
            dict: Dictionary containing the aggregated metric values.
        """
        result_dict = dict()
        # over all branches
        for branch_name, branch_metrics in self.metric_func_dict.items():
            # over all metrics
            for metric_name, metric_func in branch_metrics.items():
                value = metric_func.compute()
                
                if self.average in ["macro", "micro", "weighted"]:
                    result_dict[f"{branch_name}_{metric_name}"] = value.item() # return scalar
                elif self.average in ["none"] or metric_name == "cm":
                    result_dict[f"{branch_name}_{metric_name}"] = value # return tensor (per class)

        return result_dict

    def reset(self):
        """
        Resets the metric functions to their initial state.
        """
        # over all branches
        for _, branch_metrics in self.metric_func_dict.items():
            # over all metrics
            for _, metric_func in branch_metrics.items():
                metric_func.reset()


def create_metrics(
    branch_dict: dict,
    metric_names: Union[str, List[str]] = ["f1"],
    average: str = "macro",
):
    """
    Create a dictionary of metric functions for multi-class classification.

    Args:
        branch_dict (dict): A dictionary mapping branch names to branch indices.
        metric_names (Union[str, List[str]], optional): Names of the metrics to be created. Defaults to ["f1"].
        average (str, optional): Type of averaging to be applied for multi-class metrics. Defaults to "macro".

    Returns:
        nn.ModuleDict: A dictionary of metric functions, where the keys are branch names and the values are
        nn.ModuleDict objects containing the metric functions for each metric name.
    """
    if isinstance(metric_names, str):
        metric_names = [metric_names]

    # check metric names
    metric_names2 = list()
    for metric_name in metric_names:
        metric_name = check_confusion_matrix_metric_name(metric_name)
        metric_names2.append(metric_name)

    metric_func_dict = nn.ModuleDict()
    # each branch
    for branch_name, branch_indices in branch_dict.items():
        num_classes = len(branch_indices)

        branch_metrics = nn.ModuleDict()
        # each metric
        for metric_name in metric_names2:
            if metric_name == "acc":
                from torchmetrics import Accuracy

                metric_func = Accuracy(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "f1":
                from torchmetrics import F1Score

                metric_func = F1Score(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "tpr":
                from torchmetrics import Recall

                metric_func = Recall(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "ppv":
                from torchmetrics import Precision

                metric_func = Precision(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "tnr":
                from torchmetrics import Specificity

                metric_func = Specificity(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "ap":
                from torchmetrics import AveragePrecision

                metric_func = AveragePrecision(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "auc":
                from torchmetrics import AUROC

                metric_func = AUROC(
                    task="multiclass", num_classes=num_classes, average=average
                )
            elif metric_name == "cm":
                from torchmetrics import ConfusionMatrix

                metric_func = ConfusionMatrix(
                    task="multiclass", num_classes=num_classes
                )
            else:
                raise NotImplementedError(
                    f"The metric {metric_name} has not been implemented"
                )
            branch_metrics[metric_name] = metric_func

        metric_func_dict[branch_name] = branch_metrics

    return metric_func_dict


def check_confusion_matrix_metric_name(metric_name: str):
    """
    A function from MONAI to simplify the confusion matrix metric names.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["area_under_curve", "auc"]:
        return "auc"
    if metric_name in ["average_precision", "ap"]:
        return "ap"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in [
        "informedness",
        "bookmaker_informedness",
        "bm",
        "youden_index",
        "youden",
    ]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    if metric_name in ["confusion_matrix", "cm"]:
        return "cm"
    raise NotImplementedError("the metric is not implemented.")
