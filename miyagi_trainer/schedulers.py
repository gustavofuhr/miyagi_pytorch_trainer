import torch

def get_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=10,
                                                                    T_mult=args.t_mult,
                                                                    eta_min=0.01,
                                                                    last_epoch=-1)

