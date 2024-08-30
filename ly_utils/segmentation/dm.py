##########################################################################################
# Description: lightning data module for segmentation model development.
##########################################################################################

import lightning as L
import monai as mn
import logging


class LYSegDataModuleBase(L.LightningDataModule):
    def __init__(
        self,
        collate_fn=None,
        val_collate_fn=None,
        train_sampler=None,
        val_sampler=None,
        ddp_sampler=False,
        train_ds=None,
        val_ds=None,
        dl_workers=-1,
        batch_size=None,
        val_batch_size=None,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False,
    ):
        super().__init__()

        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_collate_fn = val_collate_fn if val_collate_fn is not None else collate_fn

        self.total_steps = None
        self.last_stepped_step = -1

        self.dl_workers = min(os.cpu_count() * 2, 8) if dl_workers == -1 else dl_workers

        self.train_ds = None
        self.val_ds = None

        self.train_dl = None
        self.val_dl = None

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.ddp_sampler = ddp_sampler
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        if train_ds is not None:
            self.set_train_dataset(train_ds)

        if val_ds is not None:
            self.set_val_dataset(val_ds)

    def prepare_data(self):

        # other parameters
        self.batch_size = self.train_hyp.get("batch_size", 2)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            logging.info(
                f"""
            Length of training data: {len(self.train_ds)}
            Length of validation data: {len(self.val_ds)}
            """
            )
            self.train_ds = mn.data.PersistentDataset(
                data=self.train_files, transform=self.train_transform, cache_dir=CACHE_DIR
            )

            self.val_ds = mn.data.PersistentDataset(
                data=self.val_files, transform=self.val_transform, cache_dir=CACHE_DIR
            )

    def train_dataloader(self):
        return mn.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return mn.data.DataLoader(
            dataset=self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
