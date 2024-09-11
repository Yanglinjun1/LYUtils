# LYUtils

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-red.svg?logo=pytorch)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-brightgreen)
![MONAI](https://img.shields.io/badge/MONAI-blue)

This repository contains the codes for my personal AI/ML/DL projects. **Please use with cautions**.

Developer: Linjun Yang, Ph.D. [wadeyang2@gmail.com]

## Logs

The following is a log of the changes made to the repository.

### 09/11/2024

- Implemented (adapted) the **load_semgentations** function that
    - reads .seg.nrrd segmentation file (by 3D Slicer)
    - converts the segmentation voxel array into a multi-channel array, according to the user-specified label dictionary
    - supports padding, resizing, and transpose (as height and width are usually swapped in .seg.nrrd file)
    - returns the multi-channel array for multi-label segmentation task, or single-channel array for multi-class segmentation task
    - wass wrapped into a MONAI transform: **LoadSegmentationD**
- Modified the **.segmentation.lm.py** module to
    - assign **label_overlay_order** and **in2str_dict** for wandb image logging
    - adjust number of **out_channels** for multi-class segmentation task
- Modified **.segmentation.utils.py** module to
    - create a function called **overlay_labels** to use **label_overlay_order** above to overlay multi-label segmentation map into a single-channel segmentation map for wandb image logging purpose
    - create a unified function called **create_overlay_log** to generate wandb Image for logging for both multi-class and multi-label segmentation models
- To standardize segmentation DL project, in model configuration yaml file, the following fields should be specified:
    - **label_index**
        - **1-index** or starting with 1; it should NOT include background (or 0)
        - It should be **label_name:label_index**, e.g., "foreground":1
        - The label_index value will be the pixel value for multi-class segmentation map; label_value-1 will be the channel index for the multi-label segmentation map
        - **NOTE: you should design this field beforehand, and label_name should match with those in the std_segmentation file used in 3D Slicer segmentation!!**
    - **label_order**
        - It should be **label_name:label_order**, e.g., "hand":1 and "ring":2, meaning the "ring" class will be overlaid on the top of "hand" class
        - It has two usages 1) **LoadSegmentationD** class, and 2) **.segmentation.lm.py**

### 08/30/2024

-  Initial commit.
