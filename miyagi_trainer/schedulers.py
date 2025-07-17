import torch

def get_scheduler(optimizer, args, train_loader=None):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0 if hasattr(args, "T_0") else 10,
            T_mult=args.T_mult if hasattr(args, "T_mult") else 2,
            eta_min=args.eta_min if hasattr(args, "eta_min") else 0.01,
            last_epoch=-1
        )
    elif args.scheduler == "onecycle":
        if train_loader is None:
            raise ValueError("train_loader must be provided for OneCycleLR")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr if hasattr(args, "max_lr") else 0.1,
            steps_per_epoch=len(train_loader),
            epochs=int(args.n_epochs)
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
