##########################################################################################
# Description: functions to create learning rate scheduler for pytorch-based optimizers.
##########################################################################################

import torch.optim as optim

__all__ = ["create_LRScheduler"]


def create_LRScheduler(
    optimizer: optim.Optimizer,
    scheduler_params: dict,
) -> optim.lr_scheduler:
    """A custom function to create a scheduler of a given type;
    Right now, the OneCycleLR, CosineAnnealingLR, and CosineAnnealingWarmRestarts
    are supported.

    Args:
        scheduler_name (str): type name of the pytorch scheduler
        optimizer (optim.Optimizer): the optimizer instanace

    Returns:
        optim.lr_scheduler: the scheduler or None
    """
    scheduler_name = scheduler_params.get("scheduler_name", None)
    max_lr = scheduler_params.get("max_lr", 1e-3)
    if scheduler_name == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr)
    elif scheduler_name == "CosineAnnealingLR":
        T_max = scheduler_params.get("T_max", 5e3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        T_0 = scheduler_params.get("T_0", 200)
        eta_min = scheduler_params.get("eta_min", max_lr / 10)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=eta_min
        )
    elif scheduler_name == "TimmCosineRestartsDecay":
        from timm.scheduler.cosine_lr import CosineLRScheduler

        t_initial = scheduler_params.get("t_initial", 20)
        cycle_decay = scheduler_params.get("cycle_decay", 0.85)
        lr_min = scheduler_params.get("lr_min", 1e-5)
        warmup_t = scheduler_params.get("warmup_t", 5)
        warmup_lr_init = scheduler_params.get("warmup_lr_init", 1e-5)
        cycle_limit = scheduler_params.get("cycle_limit", 401)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cycle_decay,
            cycle_decay=cycle_decay,
            lr_min=lr_min,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            cycle_limit=cycle_limit,
        )
    else:
        import warnings

        if scheduler_name is None:
            warnings.warn(f"The scheduler name is not specified. Return no Scheduler")
        else:
            warnings.warn(
                f"The {scheduler_name} is not included or does not existed. Return no Scheduler"
            )
        scheduler = None

    return scheduler
