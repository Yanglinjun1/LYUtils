##########################################################################################
# Description: Script containing functions/classes to build the lightning trainer for
# model training
##########################################################################################

import os
import datetime
import wandb
import lightning as L
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from .utils import is_notebook_running


def resume_training(resume_dir):  # TODO
    logger_instance = os.path.basename(resume_dir)
    experiment_name = os.path.basename(os.path.dirname(resume_dir))
    if os.path.basename(os.path.dirname(os.path.dirname(resume_dir))) != "pl":
        raise ValueError("The resume directory must be in the pl directory.")
    wandb_id_file = f"{resume_dir}/wandb_run_id.txt"
    try:
        with open(wandb_id_file, "r") as f:
            wandb_id = f.read()
    except FileNotFoundError:
        wandb_id = None
        print("Wandb run ID file not found. Will create a new Wandb run.")

    result = dict()
    result["logger_instance"] = logger_instance
    result["experiment_name"] = experiment_name
    result["wandb_id"] = wandb_id

    return result


class LYLightningTrainer(L.Trainer):
    """class original implemented by Bardia Khosravi. Adapted by Linjun Yang"""

    def __init__(
        self,
        project_name,
        output_directory,
        experiment_name,
        max_epochs=None,
        logger_instance=None,
        check_val_every_n_epoch=1,
        precision="16-mixed",
        devices=-1,
        nodes=1,
        deterministic=True,
        wandb_id=None,
        callback_list=[],
        **kwargs,
    ):
        # output directory
        self.output_directory = output_directory

        # wandb logger
        now = datetime.datetime.now()
        year, month, day, hour, minute = (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
        )
        self.logger_instance = (
            logger_instance
            if logger_instance
            else f"run_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
        )
        self.wandb_project = project_name
        self.group = experiment_name
        self.wandb_id = wandb_id

        # project and instance is required
        if self.logger_instance is None or self.wandb_project is None:
            raise ValueError("Project name or logger instance cannot be None.")

        # Wandb API key is required
        if "WANDB_API_KEY" not in os.environ:
            raise EnvironmentError(
                "Wandb API key not found. Please set the WANDB_API_KEY environment variable."
            )

        wandb_logger = WandbLogger(
            save_dir=f"{self.output_directory}",
            name=self.logger_instance,
            group=self.group,
            project=self.wandb_project,
            offline=False,
            log_model=False,
            id=self.wandb_id,
        )

        # output dirs
        os.makedirs(f"{self.output_directory}/pl", exist_ok=True)
        os.makedirs(f"{self.output_directory}/wandb", exist_ok=True)

        # make dirs for the group (experiment) and the current run
        os.makedirs(f"{self.output_directory}/pl/{self.group}", exist_ok=True)
        os.makedirs(
            f"{self.output_directory}/pl/{self.group}/{self.logger_instance}",
            exist_ok=True,
        )

        super().__init__(
            deterministic=deterministic,
            callbacks=callback_list,
            profiler="simple",
            logger=wandb_logger,
            precision=precision,
            accelerator="gpu",
            devices=devices[0] if is_notebook_running() else devices,
            num_nodes=nodes,
            strategy=(
                "auto"
                if is_notebook_running()
                else DDPStrategy(find_unused_parameters=True)
            ),
            default_root_dir=self.output_directory,
            num_sanity_val_steps=0,
            fast_dev_run=False,
            max_epochs=max_epochs,
            use_distributed_sampler=is_notebook_running() is False and len(devices) != 1,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )
