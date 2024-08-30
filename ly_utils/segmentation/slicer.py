##########################################################################################
# Description: Custom function and classes for converting 3D Slicer Segmentations.
##########################################################################################

import copy
import numpy as np
import monai as mn
import slicerio
import nrrd

__all__ = ["load_segmentations", "LoadSegmentationD"]


def load_segmentations(
    path: str, labels: dict, label_order: dict = None
):  # TODO labels -> label_index
    """
    This function reads a given segmentation file in .seg.nrrd format and return
    the numpy array of the segmentation data in one of the two formats:
    1) multi-label: each channel represents one class label; one-hot format
    2) multi-class: one single channel and class label as different integer values
    Originally by Kellen Mulford
    """

    multi_label = False if label_order is None else True

    ## Load Data
    segmentation_info = slicerio.read_segmentation(path, skip_voxels=True)
    voxels, _ = nrrd.read(path)

    ## Add channels dim if not present
    if len(voxels.shape) == 3:
        voxels = np.expand_dims(voxels, axis=0)

    ## Prep Empty Volume
    x = voxels.shape[1]
    y = voxels.shape[2]
    channels = len(segmentation_info["segments"])

    output = np.zeros((x, y, channels))

    ## Loop through layers
    for i, segment in enumerate(segmentation_info["segments"]):

        ## Extract Metadata
        layer = segment["layer"]
        layer_name = segment["name"]
        labelValue = segment["labelValue"]

        ## Set up new layer based on voxel value from segmentation info
        layer_voxels = np.moveaxis(voxels, 0, -1)
        layer_voxels = np.squeeze(layer_voxels, axis=-2)
        indx = (layer_voxels[..., layer] == labelValue).nonzero()
        new_layer = np.zeros(layer_voxels[..., layer].shape)
        new_layer[indx] = labelValue

        ## Assign the new layer to the output based on defined channel order
        # label starts with 1 not zero, minus 1 is needed
        output[
            ...,
            labels[str.lower(layer_name) if multi_label else str.lower(layer_name)] - 1,
        ] = new_layer

    output = np.where(np.moveaxis(output, -1, 0) > 0.0, 1, 0)
    if multi_label:  # return for multi-label segmentation model
        return output
    else:
        output2 = np.zeros((x, y, 1))
        label_order_list = sorted(label_order.items(), key=lambda x: x[1])
        for label_name, _ in label_order_list:
            output2[output[labels[label_name] - 1] == 1, :] = labels[label_name]

        return output2


class LoadSegmentationD(mn.transforms.Transform):
    def __init__(self, keys: list[str], labels: dict, labels_order: dict) -> None:
        super().__init__()
        self.keys = keys
        self.labels = labels
        self.labels_order = labels_order

    def __call__(self, data):
        data_copy = copy.deepcopy(data)
        for key in self.keys:
            if key in data:
                output = load_segmentations(data[key], self.labels, self.labels_order)
                data_copy[key] = output

        return data_copy
