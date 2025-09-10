import timm
from torchvision import models
import torch.nn as nn


def get_model(model_name, n_classes, pretrained = True, freeze_all_but_last = False, drop_path_rate: float = 0.0):
    def _freeze_all(model):
        # make every parameter freeze, fc will be redone and unfreeze
        for param in model.parameters():
            param.requires_grad = False

    # always priorize timm if possible
    if model_name in timm.list_models("*"):
        # print("Getting model from timm")
        model = timm.create_model(model_name, pretrained, num_classes=n_classes, drop_path_rate=drop_path_rate)
        if freeze_all_but_last: _freeze_all(model)
    else:
        # print("Getting model from torchvision")
        model = getattr(models, model_name)(pretrained=pretrained)
        if freeze_all_but_last: _freeze_all(model)

        # TODO, why it works when the last layer is not resized!?
        no_features_fc = model.fc.in_features
        model.fc = nn.Linear(no_features_fc, n_classes)


    return model