import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


# Set of loss names that operate on embeddings and carry their own classification head.
# These require --embedding_size to be set and their parameters must be added to the optimizer.
METRIC_LOSSES = {"arcface", "cosface", "admsoft"}


def is_metric_loss(loss_name: str) -> bool:
    return loss_name in METRIC_LOSSES


class CustomAnomalyLoss(nn.Module):

    def forward(self, inputs, targets, target2look = 1):
        idx = np.argwhere(np.asarray(targets.cpu() == target2look)).flatten()
        input_only4target = torch.index_select(inputs.cpu(), 0, torch.tensor(idx))
        norms = torch.norm(input_only4target, 2, dim=1)

        return torch.sum(norms)/len(idx)


class DualMarginAdMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax with per-class margins, ported from:
      patchnet_prototype/fas-patchnet/metrics/losses.py

    Uses separate margins for each class (e.g. m_s for spoof=0, m_l for live=1),
    which is the patchnet variant.  Interface mirrors pytorch-metric-learning losses:
    forward(embeddings, labels) and get_logits(embeddings).
    """

    def __init__(self, in_features: int, out_features: int,
                 s: float = 30.0, m_l: float = 0.4, m_s: float = 0.1):
        super().__init__()
        self.s = s
        # margins indexed by class label: [spoof (0), live (1)]
        self.margins = [m_s, m_l]
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.fc.out_features

        W_norm = F.normalize(self.fc.weight, dim=1)   # (C, D)
        x_norm = F.normalize(x, dim=1)                # (N, D)
        wf = x_norm @ W_norm.T                        # (N, C)

        m = torch.tensor(
            [self.margins[int(l)] for l in labels], dtype=x.dtype, device=x.device
        )
        correct_scores = wf[range(len(labels)), labels]
        numerator = self.s * (correct_scores - m)

        # denominator: exp(numerator) + sum of exp(s * other class scores)
        mask = torch.ones_like(wf, dtype=torch.bool)
        mask[range(len(labels)), labels] = False
        other_scores = wf[mask].view(len(labels), -1)
        denominator = torch.exp(numerator) + torch.exp(self.s * other_scores).sum(dim=1)

        L = numerator - torch.log(denominator)
        return -L.mean()

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return scaled cosine similarity scores — usable as class logits for inference."""
        W_norm = F.normalize(self.fc.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        return self.s * (x_norm @ W_norm.T)


def get_loss(loss_name, num_classes=None, embedding_size=None, class_weights=None,
             arcface_margin=28.6, arcface_scale=64,
             cosface_margin=0.35, cosface_scale=64,
             admsoft_m_l=0.4, admsoft_m_s=0.1, admsoft_scale=30.0):

    if loss_name.startswith("cross_entropy"):
        # support to cross_entropy_label_smoothing_0.2 type of loss
        # this is done mainly because its impossible (right now) to combine parameters in sweeps
        ss = "_label_smoothing_"
        if loss_name.find(ss) >= 0:
            ce_label_smoothing = float(loss_name[loss_name.find(ss)+len(ss):])
            print("Loss with label smoothing:", ce_label_smoothing)
        else:
            ce_label_smoothing = 0.0
        if class_weights is not None:
            print("Using class weights for cross entropy loss:", class_weights)
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ce_label_smoothing)
        else:
            print("Using cross entropy loss without class weights and label smoothing")
            loss = nn.CrossEntropyLoss(label_smoothing=ce_label_smoothing)

    elif loss_name == "angular":
        print("Using Angular Loss")
        loss = losses.AngularLoss(alpha=40)

    elif loss_name == "arcface":
        print(f"Using ArcFace Loss (margin={arcface_margin}, scale={arcface_scale})")
        loss = losses.ArcFaceLoss(num_classes, embedding_size,
                                  margin=arcface_margin, scale=arcface_scale)

    elif loss_name == "cosface":
        print(f"Using CosFace Loss / AM-Softmax (margin={cosface_margin}, scale={cosface_scale})")
        loss = losses.CosFaceLoss(num_classes, embedding_size,
                                  margin=cosface_margin, scale=cosface_scale)

    elif loss_name == "admsoft":
        print(f"Using Dual-Margin AdMSoftmax Loss "
              f"(s={admsoft_scale}, m_l={admsoft_m_l}, m_s={admsoft_m_s})")
        loss = DualMarginAdMSoftmaxLoss(embedding_size, num_classes,
                                        s=admsoft_scale, m_l=admsoft_m_l, m_s=admsoft_m_s)

    elif loss_name == "custom_anomaly":
        print("Using Custom Anomaly Loss")
        loss = CustomAnomalyLoss()

    else:
        raise ValueError(f"Unknown loss: '{loss_name}'")

    return loss
