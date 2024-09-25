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
    path: str,
    labels: dict,
    label_order: dict = None,
    target_shape: tuple = None,
    pad: bool = True,
    transpose: bool = True,
):
    """
    This function reads a given segmentation file in .seg.nrrd format and return the numpy array
    of the segmentation data in one of the two formats: multi-label (multi-channel, binary value) or multi-class (
    single-channel, multiple value). The labels argument is needed to specify the channel index at the output array and
    /or pixel value at the single channel output for each class label. The label_order argument is specified
    to define the order that labels overwrite each other, and it's only used when creating multi-class segmentation map.
    The target_shape argument is used to resize the segmentation map to the desired shape. The transpose argument is used
    to swap the image dimension of the voxel array after loading, which is set to True as the nrrd usually swap the axes.

    Original by Kellen Mulford; modified by Linjun Yang

    Args:
        path (str): The path to the segmentation data.
        labels (dict): A dictionary mapping layer names to their corresponding label values.
        label_order (dict, optional): A dictionary specifying the order of the labels. Defaults to None.
        target_shape (tuple, optional): The desired shape of the output segmentation. Defaults to None.
        pad (bool, optional): Whether to pad the output array before resizing. Defaults to True.
        transpose (bool, optional): Whether to swap the image dimension of the voxel array after loading. Defaults to True to process .seg.nrrd files.

    Returns:
        np.ndarray: The processed segmentation data of shape (num_channels, height, width) for multi-label or
        (1, height, width) for multi-class.

    Raises:
        None

    """

    # Check if the segmentation is multi-label or multi-class
    multi_label = True if label_order is None else False

    ## Load Data
    segmentation_info = slicerio.read_segmentation(path, skip_voxels=True)
    voxels, _ = nrrd.read(path)

    ## get mapping of layer names to index in segmentation_info["segments"]
    segments = segmentation_info["segments"]
    segment_names = {segment["name"]: i for i, segment in enumerate(segments)}

    ## Add the first channels dim if not present
    if len(voxels.shape) == 3:
        voxels = np.expand_dims(voxels, axis=0)

    # swap axes if needed; usually needed for .seg.nrrd files
    if transpose:
        voxels = np.transpose(voxels, (0, 2, 1, 3))

    ## Prep Empty Volume
    x = voxels.shape[1]
    y = voxels.shape[2]

    # channels: the number of given labels
    channels = len(labels)
    output = np.zeros((channels, x, y))

    ## Loop through layers
    layer_voxels = np.squeeze(voxels, axis=-1)
    print(f"Processing file {path}")
    for label in labels.keys():

        # (x, y) is the shape of the layer
        new_layer = np.zeros((x, y))

        ## get the segment index for the given label
        segment_ind = segment_names.get(label, None)

        ## assign all zeros if the current label is not in the segmentation file
        if segment_ind is None:
            print(
                f"Label {label} not found in segmentation file. It will not be in the final segmentation array."
            )
            output[labels[label] - 1, ...] = new_layer
            continue

        ## segment info is available for the current label
        segment = segments[segment_ind]
        layer = segment["layer"]
        labelValue = segment["labelValue"]

        ## Set up new layer based on voxel value from segmentation info
        indx = (layer_voxels[layer, ...] == labelValue).nonzero()
        new_layer[indx] = 1

        ## Assign the new layer to the output based on defined channel order
        # label starts with 1 not zero, minus 1 is needed
        output[labels[label] - 1, ...] = new_layer

    # Resize to target shape
    if target_shape is not None:
        if pad:
            side = max(x, y)
            padder = mn.transforms.SpatialPad(
                (side, side), method="symmetric", mode="constant"
            )
            output = padder(output)
        x, y = target_shape  # update the x and y!!!
        resizer = mn.transforms.Resize(target_shape, mode="nearest")
        output = resizer(output)

    # return for multi-label segmentation model
    if multi_label:
        return output
    else:
        output2 = np.zeros((1, x, y))
        label_order_list = sorted(label_order.items(), key=lambda x: x[1])
        for label_name, _ in label_order_list:
            output2[:, output[labels[label_name] - 1] == 1] = labels[label_name]

        return output2


class LoadSegmentationD(mn.transforms.Transform):
    def __init__(
        self,
        keys: list[str],
        labels: dict,
        label_order: dict,
        target_shape: tuple = None,
        transpose: bool = True,
        pad: bool = True,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.labels = labels
        self.label_order = label_order
        self.target_shape = target_shape
        self.transpose = transpose
        self.pad = pad

    def __call__(self, data):
        data_copy = copy.deepcopy(data)
        for key in self.keys:
            if key in data:
                output = load_segmentations(
                    data[key],
                    self.labels,
                    self.label_order,
                    self.target_shape,
                    self.transpose,
                    self.pad,
                )
                data_copy[key] = output

        return data_copy
