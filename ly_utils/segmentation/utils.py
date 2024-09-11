##########################################################################################
# Description: Custom utility functions for training segmentation models.
##########################################################################################

import numpy as np
import torch
from torchvision.utils import make_grid
import wandb


def image2np(image):
    """ "
    For use in create_overlay_log to log wandb Image.
    Adapted from a package.
    """
    res = image.cpu().permute(1, 2, 0).numpy()
    return res[..., 0] if res.shape[2] == 1 else res


def overlay_labels(data, label_overlay_order):
    """
    Overlay labels on the given data according to the specified order. Data is expected to be in the format (channels, height, width).
    The label_overlay_order is a dictionary with the label value as the key and the order as the value.

    Args:
        data (torch.Tensor): The input data of shape (channels, height, width).
        label_overlay_order (dict): The order to overlay labels for wandb Image.

    Returns:
        torch.Tensor: The overlaid data with the first dimension added.
    """

    label_overlay_order_list = [
        (key, value) for key, value in label_overlay_order.items()
    ]
    label_overlay_order_list = sorted(label_overlay_order_list, key=lambda x: x[1])

    overlaid = torch.zeros(data.shape[1:], device=data.device)
    for label_value, _ in label_overlay_order_list:
        channel_index = (
            label_value - 1
        )  # label value is 1-indexed; minus 1 to get channel index
        current_channel = data[channel_index]
        overlaid[current_channel > 0.0] = label_value

    return overlaid[None]


def create_overlay_log(
    img, seg, pred, int2str_dict, label_overlay_order=None, nrow=6, num_image=6
):
    # convert lists of image, segmentations to grid image
    num_image_to_use = min(len(img), num_image)

    img_grid = make_grid(img[:num_image_to_use], nrow=nrow, pad_value=0)

    # multi-class: apply argmax to segmentations;
    if label_overlay_order is None:
        seg_list = [
            torch.argmax(seg[i].data, dim=0, keepdim=True)
            for i in range(num_image_to_use)
        ]
        seg_grid = make_grid(seg_list, nrow=nrow, pad_value=0)[:1]
        pred_list = [
            torch.argmax(pred[i].data, dim=0, keepdim=True)
            for i in range(num_image_to_use)
        ]
        pred_grid = make_grid(pred_list, nrow=nrow, pad_value=0)[:1]
    # multi-label: overlay labels
    else:
        seg_list = [
            overlay_labels(seg[i].data, label_overlay_order)
            for i in range(num_image_to_use)
        ]
        seg_grid = make_grid(seg_list, nrow=nrow, pad_value=0)[:1]
        pred_list = [
            overlay_labels(pred[i].data, label_overlay_order)
            for i in range(num_image_to_use)
        ]
        pred_grid = make_grid(pred_list, nrow=nrow, pad_value=0)[:1]

    img_data = image2np(img_grid.data * 255).astype(np.uint8)
    seg_data = image2np(seg_grid.data).astype(np.uint8)
    pred_data = image2np(pred_grid.data).astype(np.uint8)

    # e.g., 1:'foreground'
    wb_image = wandb.Image(
        img_data,
        masks={
            "prediction": {"mask_data": pred_data, "class_label": int2str_dict},
            "ground truth": {"mask_data": seg_data, "class_label": int2str_dict},
        },
    )

    return wb_image
