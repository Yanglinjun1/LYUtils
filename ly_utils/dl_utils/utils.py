##########################################################################################
# Description: Script containing utility functions
##########################################################################################

import numpy as np
import os
import random
import torch
import monai as mn


def is_notebook_running():
    """
    Checks if the code is running on a notebook or a Python file.
    Orignally from https://github.com/BardiaKh/PytorchUtils/blob/main/bkh_pytorch_utils/py/utils.py
    """
    try:
        shell = get_ipython().__class__
        if "google.colab._shell.Shell" in str(shell):
            return True
        if shell.__name__ == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell.__name__ == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def make_determinate(random_seed):
    """Use a random_seed to enable deterministic programming for Pytorch, Moani,
      and Numpy and.

    Args:
        random_seed (int): a random_seed to be used by different libraries.
    """
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mn.utils.misc.set_determinism(seed=random_seed)
