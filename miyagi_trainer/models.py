import timm
from torchvision import models
import torch.nn as nn


def get_model(model_name, n_classes, pretrained=True, freeze_all_but_last=False,
              drop_path_rate: float = 0.0, embedding_size: int = None):
    """
    Build a backbone model.

    Args:
        model_name:         timm or torchvision model name.
        n_classes:          Number of output classes (used when embedding_size is None).
        pretrained:         Load ImageNet pretrained weights.
        freeze_all_but_last: Freeze all layers except the final head.
        drop_path_rate:     Stochastic depth drop-path rate (timm only).
        embedding_size:     When set, the final FC outputs an embedding of this size
                            instead of class logits.  Use with metric losses (ArcFace,
                            CosFace, AdMSoftmax) whose own classification head operates
                            on these embeddings.
    """
    def _freeze_all(model):
        # make every parameter freeze, fc will be redone and unfreeze
        for param in model.parameters():
            param.requires_grad = False

    output_size = embedding_size if embedding_size is not None else n_classes

    # always priorize timm if possible
    if model_name in timm.list_models("*"):
        # print("Getting model from timm")
        model = timm.create_model(model_name, pretrained, num_classes=output_size, drop_path_rate=drop_path_rate)
        if freeze_all_but_last: _freeze_all(model)
    else:
        # print("Getting model from torchvision")
        model = getattr(models, model_name)(pretrained=pretrained)
        if freeze_all_but_last: _freeze_all(model)

        # TODO, why it works when the last layer is not resized!?
        no_features_fc = model.fc.in_features
        model.fc = nn.Linear(no_features_fc, output_size)

    return model
