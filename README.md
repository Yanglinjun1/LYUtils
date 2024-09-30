# LYUtils

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-red.svg?logo=pytorch)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-brightgreen)
![MONAI](https://img.shields.io/badge/MONAI-blue)

This repository contains the codes for my personal AI/ML/DL projects. **Please use with cautions**.

Developer: Linjun Yang, Ph.D. [wadeyang2@gmail.com]

## Logs

The following is a log of the changes made to the repository.

### 09/30/2024

- Provided docstrings for most functions and classes.

### 09/26/2024

- Cleaned up __init__.py file of **ly_utils.classification**.
- Added **move_label_to_device** function to make sure dictionary-based label is in the GPU device.

### 09/25/2024

- Added **calculate_distances_between_two_point_sets** to measure.py

### 09/23/2024

- Added **chi2_test** in statistic module to do chi-square test
- Added **measure.py** in segmentation module and included common basic measuring functions
- Updated the setup.py
- All added functions to be tested.

### 09/18/2024

- The codes worked for training segmentation models
- **make_deterministic** works
- wandb's **log_dict** cannot log metrics in the same chart.

### 09/17/2024

- Checked and to try **LYLightningTrainer**

### 09/16/2024

- Implemented **LYDataModuleBase** and subclassed it to creat **LYSegDataModuleBase**

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
    - label_index
        - **1-index** or starting with 1; it should NOT include background (or 0)
        - It should be **label_name:label_index**, e.g., "foreground":1
        - The label_index value will be the pixel value for multi-class segmentation map; label_value-1 will be the channel index for the multi-label segmentation map
        - **NOTE: you should design this field beforehand, and label_name should match with those in the std_segmentation file used in 3D Slicer segmentation!!**
    - label_order
        - It should be **label_name:label_order**, e.g., "hand":1 and "ring":2, meaning the "ring" class will be overlaid on the top of "hand" class
        - It has **two usages** 1) LoadSegmentationD class for segmentation, and 2) wandb image creation
- Added **dl_utils.base_lm** and created **LYLightningModuleBase** for basic lightning module.
    - Implemented **configure_optimizers** function
    - Set foward, log_loss, log_metric, and process_configurations with NotImplementedError

### 08/30/2024

-  Initial commit.
