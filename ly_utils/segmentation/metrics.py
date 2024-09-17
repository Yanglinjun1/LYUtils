##########################################################################################
# Description: Utility functions for segmentation evaluation.
##########################################################################################


from monai.metrics import DiceMetric, HausdorffDistanceMetric


class LYSegMetrics:

    supported_metrics = ["dice", "hausdorff"]

    def __init__(
        self,
        metrics: list = ["dice", "hausdorff"],
        label_index: dict = {"foreground":1},
        multi_label: bool = False,
    ):

        self.multi_label = multi_label
        self.label_index = label_index

        metric_dict = dict()
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(
                    f"Unsupported metric: {metric}. Supported metrics are: {self.supported_metrics}"
                )
            if metric == "dice":
                metric_dict[metric] = DiceMetric(
                    include_background=True, reduction="mean_batch"
                )
            elif metric == "hausdorff":
                metric_dict[metric] = HausdorffDistanceMetric(
                    include_background=True, reduction="mean_batch"
                )
        self.metrics = metrics
        self.metric_dict = metric_dict

    def __call__(self, pred, seg):
        result_dict = dict()
        for metric, metric_obj in self.metric_dict.items():
            result = metric_obj(pred, seg)
            result_dict[metric] = result

        return result_dict

    def aggregate(self):
        result_dict = dict()
        for metric, metric_obj in self.metric_dict.items():
            result = dict()  # for current metric, e.g. dice

            # for each label
            result_tensor = metric_obj.aggregate()
            for label, ind in self.label_index.items():
                if self.multi_label:  # if multi_label, start from 1-1=0; else (multi-class) start with 1
                    ind -= 1
                result[label] = result_tensor[ind].item()

            # mean metric across labels
            if self.multi_label:
                result["mean"] = result_tensor.mean().item()
            else:
                result["mean"] = result_tensor[1:].mean().item()

            result_dict[metric] = result

        return result_dict

    def reset(self):
        for metric_obj in self.metric_dict.values():
            metric_obj.reset()
