import torch.optim.lr_scheduler as lr_scheduler


def get_lr_scheduler(scheduler_type: str, optimizer, **kwargs):
    """
    Returns the corresponding PyTorch learning rate scheduler based on the scheduler_type string.
    Returns None if scheduler_type is "constant".

    Args:
        scheduler_type (str): The type of scheduler ("constant", "cosineannealingwarmrestarts", etc.)
        optimizer: The optimizer to attach the scheduler to.

    Returns:
        The learning rate scheduler instance or None.
    """
    milestones = kwargs.pop('milestones')
    epochs = kwargs.pop('epochs')
    t0 = kwargs.pop('t0')

    if scheduler_type == "constant":
        return None
    elif scheduler_type == "cosineannealingwarmrestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, eta_min=1e-6)
    elif scheduler_type == "multisteplr":
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    elif scheduler_type == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6)
    elif scheduler_type == "cosineannealing":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
