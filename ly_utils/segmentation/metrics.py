##########################################################################################
# Description: Utility functions for segmentation evaluation.
##########################################################################################


from monai.metrics import DiceMetric, HausdorffDistanceMetric


class LYSegMetrics:
    """
    Class for computing segmentation metrics.

    Args:
        metrics (list, optional): List of metrics to compute. Defaults to ["dice", "hausdorff"].
        label_index (dict, optional): Dictionary mapping label names to indices. Defaults to {"foreground": 1}.
        multi_label (bool, optional): Flag indicating whether the segmentation is multi-label. Defaults to False.
    """

    supported_metrics = ["dice", "hausdorff"]  # for checking purposes

    def __init__(
        self,
        metrics: list = ["dice", "hausdorff"],
        label_index: dict = {"foreground": 1},
        multi_label: bool = False,
    ):
        """
        Initializes the LYSegMetrics object. Supported metrics are "dice" and "hausdorff". include_background
        is set to True. The reduction is set to "mean_batch".


        Args:
            metrics (list, optional): List of metrics to compute. Defaults to ["dice", "hausdorff"].
            label_index (dict, optional): Dictionary mapping label names to indices. Defaults to {"foreground": 1}.
            multi_label (bool, optional): Flag indicating whether the segmentation is multi-label. Defaults to False.
        """

        self.multi_label = multi_label
        self.label_index = label_index.copy()

        metric_dict = dict()
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(
                    f"Unsupported metric: {metric}. Supported metrics are: {self.supported_metrics}"
                )
            if metric == "dice":
                metric_dict[metric] = DiceMetric(
                    include_background=True if multi_label else False, reduction="mean_batch",
                    num_classes=None if multi_label else len(label_index) + 1,
                )
            elif metric == "hausdorff":
                metric_dict[metric] = HausdorffDistanceMetric(
                    include_background=True if multi_label else False, reduction="mean_batch",
                    num_classes=None if multi_label else len(label_index) + 1,
                )
        self.metrics = metrics
        self.metric_dict = metric_dict

    def __call__(self, pred, seg):
        """
        Computes the segmentation metrics.

        Args:
            pred: Predicted segmentation.
            seg: Ground truth segmentation.

        Returns:
            dict: Dictionary containing the computed metrics.
        """

        result_dict = dict()
        for metric, metric_obj in self.metric_dict.items():
            result = metric_obj(pred, seg)
            result_dict[metric] = result

        return result_dict

    def aggregate(self):
        """
        Aggregates the computed metrics. The aggregation is performed according to if the segmentation
        is multi-label or multi-class.

        Returns:
            dict: Dictionary containing the aggregated metrics.
        """

        result_dict = dict()
        for metric, metric_obj in self.metric_dict.items():
            result = dict()  # for current metric, e.g. dice

            # for each label
            result_tensor = metric_obj.aggregate()
            for label, ind in self.label_index.items():
                ind -= 1 # multi-label starts from 0th channel; multi-class starts from 1st label which is 0-index as include_background=False
                result[label] = result_tensor[ind].item()

            # mean metric across labels
            if self.multi_label:
                result["mean"] = result_tensor.mean().item()
            else:
                result["mean"] = result_tensor[1:].mean().item()

            result_dict[metric] = result

        return result_dict

    def reset(self):
        """
        Resets the metrics.

        """
        for metric_obj in self.metric_dict.values():
            metric_obj.reset()
