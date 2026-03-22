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
    lr_patience = kwargs.pop('lr_patience')
    lr_gamma = kwargs.pop('lr_gamma')
    lr_factor = kwargs.pop('lr_factor')

    if scheduler_type == "constant":
        return None
    elif scheduler_type == "cosineannealingwarmrestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, eta_min=1e-6)
    elif scheduler_type == "multisteplr":
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)
    elif scheduler_type == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6)
    elif scheduler_type == "cosineannealing":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmUpLR(lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]
