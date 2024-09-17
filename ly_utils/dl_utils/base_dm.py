##########################################################################################
# Description: base data module class for training any deep learning model.
##########################################################################################

import os
import lightning as L
import monai as mn
import logging


class LYDataModuleBase(L.LightningDataModule):
    """
    Lightning data module for training any deep learning model.
    Ideas adapted from: https://github.com/BardiaKh/PytorchUtils/blob/main/bkh_pytorch_utils/pl/utils.py
    """

    def __init__(
        self,
        collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
        ddp_sampler=False,
        train_ds=None,
        val_ds=None,
        test_ds=None,
        dl_workers=-1,
        batch_size=None,
        val_batch_size=None,
        test_batch_size=None,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False,
    ):
        super().__init__()

        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_collate_fn = val_collate_fn if val_collate_fn is not None else collate_fn
        self.test_batch_size = (
            test_batch_size if test_batch_size is not None else val_batch_size
        )
        self.test_collate_fn = (
            test_collate_fn if test_collate_fn is not None else val_collate_fn
        )

        self.dl_workers = min(os.cpu_count() * 2, 8) if dl_workers == -1 else dl_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler if test_sampler is not None else val_sampler
        self.ddp_sampler = ddp_sampler
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        if train_ds is not None:
            self.set_train_dataset(train_ds)

        if val_ds is not None:
            self.set_val_dataset(val_ds)

        if test_ds is not None:
            self.set_test_dataset(test_ds)

    def prepare_data(self):

        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            if self.train_ds is None:
                raise Exception(
                    "Use the 'set_train_dataset' method to set the training dataset."
                )
            if self.val_ds is None:
                raise Exception(
                    "Use the 'set_val_dataset' method to set the validation dataset."
                )
            logging.info(
                f"""
            Length of training data: {len(self.train_ds)}
            Length of validation data: {len(self.val_ds)}
            """
            )
        elif stage == "test":
            if self.test_ds is None:
                raise Exception(
                    "Use the 'set_test_dataset' method to set the test dataset."
                )
            logging.info(
                f"""
            Length of test data: {len(self.test_ds)}
            """
            )

    def set_train_dataset(self, ds):
        self.train_ds = ds

    def set_val_dataset(self, ds):
        self.val_ds = ds

    def set_test_dataset(self, ds):
        self.test_ds = ds

    def train_dataloader(self):
        if self.train_ds is None:
            raise Exception(
                "Use the 'set_train_dataset' method to set the training dataset."
            )

        return mn.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise Exception(
                "Use the 'set_val_dataset' method to set the validation dataset."
            )
        return mn.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise Exception("Use the 'set_test_dataset' method to set the test dataset.")
        return mn.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=4,
        )
