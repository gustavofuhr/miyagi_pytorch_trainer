import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import losses



class CustomAnomalyLoss(nn.Module):

    def forward(self, inputs, targets, target2look = 1):
        idx = np.argwhere(np.asarray(targets.cpu() == target2look)).flatten()
        input_only4target = torch.index_select(inputs.cpu(), 0, torch.tensor(idx))
        norms = torch.norm(input_only4target, 2, dim=1)

        return torch.sum(norms)/len(idx)

def get_loss(loss_name):
    if loss_name.startswith("cross_entropy"):
        # support to cross_entropy_label_smoothing_0.2 type of loss
        # this is done mainly because its impossible (right now) to combine parameters in sweeps
        ss = "_label_smoothing_"
        if loss_name.find(ss) >= 0:
            ce_label_smoothing = float(loss_name[loss_name.find(ss)+len(ss):])
        else:
            ce_label_smoothing = 0.0
        loss = nn.CrossEntropyLoss(label_smoothing = ce_label_smoothing)
    elif loss_name == "angular":
        loss = losses.AngularLoss(alpha=40)
    elif loss_name == "custom_anomaly":
        loss = CustomAnomalyLoss()

    return loss