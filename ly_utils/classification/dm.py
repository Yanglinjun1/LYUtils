##########################################################################################
# Description: lightning data module for classification model development.
##########################################################################################

from ..dl_utils.base_dm import LYDataModuleBase


class LYClsDataModuleBase(LYDataModuleBase):
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
        super().__init__(
            collate_fn=collate_fn,
            val_collate_fn=val_collate_fn,
            test_collate_fn=test_collate_fn,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            test_sampler=test_sampler,
            ddp_sampler=ddp_sampler,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            dl_workers=dl_workers,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
