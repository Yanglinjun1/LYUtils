##########################################################################################
# Description: functions for transform and collate functions
##########################################################################################

import os
import copy
import numpy as np
import monai as mn
from monai.config import KeysCollection, SequenceStr, DtypeLike
import torch


class DictLabelsD(mn.transforms.Transform):
    """
    A custom MONAI transform to convert a label array into a dictionary format, according to the provided
    arguments "label_index" and "branch_dict". The "branch_dict" is a dictionary mapping branch names to their
    indices in the label array. Each label array is converted into a dictionary where the keys are the branch names
    and the values are the corresponding indexed label converted into a integer tensor (not one-hot for loss calculation
    purpose) using the associated indices. It is only intended to be used for MONAI dictionary data transform and
    only for 1D classification label array, not for imaging or bounding box data.

    Example:
        label_index = {0: "left", 1: "right", 2: "AP": 3: "PA", 4: "Y"}
        branch_dict = {"side": [0, 1], "view2": [2, 3]}
        keys = ["label"]
        transform = DictLabelsD(keys, label_index, branch_dict)
        data = {"label": np.array([0, 1, 0, 0, 1])} # "right", "Y"
        out = transform(data) # {"side": tensor(1), "view2": tensor(2)}

    Args:
        keys (KeysCollection): The keys in the data dictionary that contain label data.
        label_index (dict): A dictionary mapping label names to their corresponding indices.
        branch_dict (dict): A dictionary mapping branch names to their corresponding indices.

    """

    def __init__(self, keys: KeysCollection, label_index: dict, branch_dict: dict):
        super().__init__()
        self.keys = keys
        self.label_index = label_index
        self.label_index_reverse = {value: key for key, value in label_index.items()}
        self.branch_dict = branch_dict

    def __call__(self, data):
        data_copy = copy.deepcopy(data)
        for key in self.keys:
            label = data_copy[key]

            image_label_dict = dict()
            for branch_name, branch_index in self.branch_dict.items():
                label_of_branch = np.argmax(label[branch_index])
                image_label_dict[branch_name] = torch.tensor(
                    label_of_branch, dtype=torch.long
                )
            data_copy[key] = image_label_dict
        return data_copy


class DictLabelCollateD:
    """
    A custom collate function to collate the MONAI dictionary data, e.g., {"img": tensor, "label": dict, ...} in
    the monai or torch data loader. It is only intended to be used for MONAI dictionary data collate and with
    predefined set of keys, including "img" and "label". The "label" key should contain a dictionary with keys as
    the branch names and values as the corresponding label tensors. The function will stack the image tensors and
    labels tensors for each branch name in the dictionary. You should also provide set of keys that should be ignored,
    such as those keys for the bounding box data (which are used for cropping images only and not used for training).
    """

    def __init__(self, branch_names, data_keys, ignore_keys=["xyxy", "xywhn"]):
        """
        args:
            branch_names - a list of task names (string), given by the model yaml file
        """
        self.branch_names = branch_names
        self.data_keys = data_keys
        self.ignore_keys = ignore_keys

    def collate(self, batch):
        """
        args:
            batch - list of ({'img': tensor, 'label': label_dict, ...})

        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # Make image batch
        output = dict()
        for data_key in self.data_keys:
            if data_key in ["img", "image"]:
                ims = map(lambda x: x[data_key], batch)
                stacked_data = torch.stack(tuple(ims), dim=0)  # stack images
            elif data_key in self.ignore_keys:
                continue  # no need to pass the bounding boxes to the model
            elif data_key in ["label"]:
                # Make dictionary of label batch
                stacked_data = dict()
                for branch_name in self.branch_names:
                    lbs = map(lambda x: x[data_key][branch_name], batch)
                    stacked_data[branch_name] = torch.stack(tuple(lbs), dim=0)

            output[data_key] = stacked_data

        return output

    def __call__(self, batch):
        return self.collate(batch)
