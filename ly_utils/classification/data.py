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
    """ """

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
    Created on 05/19/2022, used to concatenate labels stored in the dictionary
    Modifed on 07/30/2024 to process dictionary data
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
