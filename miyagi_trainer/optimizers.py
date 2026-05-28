import torch
import timm.optim


def _build_optimizer(params, optimizer_name, weight_decay=1e-4):
    if optimizer_name == "adamp":
        # timm's create_optimizer_v2 expects a model or a param list
        optimizer = timm.optim.AdamP(params, lr=0.01, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=0.05, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer_name}'")
    return optimizer


def get_optimizer(model, optimizer_arg, weight_decay=1e-4):
    return _build_optimizer(list(model.parameters()), optimizer_arg, weight_decay)


def get_optimizer_with_extra_params(model, loss_fn, optimizer_arg, weight_decay=1e-4):
    """
    Build an optimizer whose parameter list includes both the backbone parameters
    and the learnable parameters of the loss function (e.g. the classification weight
    matrix W in ArcFace / CosFace / AdMSoftmax).
    """
    params = list(model.parameters()) + list(loss_fn.parameters())
    return _build_optimizer(params, optimizer_arg, weight_decay)
