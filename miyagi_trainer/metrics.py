import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


softmax = nn.Softmax(dim=1)
METRIC_GOALS = {
    "acc": "max",
    "eer": "min",
    "f1_score": "max",
    "per_class_accuracy": "max",
    "loss": "min"
}

def init_metrics_best(metrics_list):
    best = {}
    phases = ["train", "val"]
    for m in metrics_list:
        if m != "confusion_matrix":           
            for ph in phases:
                if METRIC_GOALS[m] == "max":
                    best[f"{ph}_{m}"] = 0.
                elif METRIC_GOALS[m] == "min":
                    best[f"{ph}_{m}"] = np.inf
    return best


def get_metrics_new_best(metrics_list, phase, epoch_log, curr_best):
    metrics_with_new_bests = []
    for metric in metrics_list:
        if metric == "confusion_matrix" or metric == "per_class_accuracy":
            continue

        if METRIC_GOALS[metric] == "max" and epoch_log[f"{phase}_{metric}"] > curr_best[f"{phase}_{metric}"]:
            curr_best[f"{phase}_{metric}"] = epoch_log[f"{phase}_{metric}"]
            metrics_with_new_bests.append(f"{phase}_{metric}")
        elif METRIC_GOALS[metric] == "min" and epoch_log[f"{phase}_{metric}"] < curr_best[f"{phase}_{metric}"]:
            curr_best[f"{phase}_{metric}"] = epoch_log[f"{phase}_{metric}"]
            metrics_with_new_bests.append(f"{phase}_{metric}")

    return curr_best, metrics_with_new_bests


def compute_metrics(metrics_list, phase, epoch_labels, epoch_preds, epoch_logits = None, class_names=None):
    """
    Compute metrics to be used in the training loop for logging them into WandB 
        and getting the best model

        metrics can be: ["acc", "eer", "f1_score", "confusion_matrix", "per_class_accuracy"]
    """
    #TODO: fix per_class_accuracy
    #TODO: make sure we are putting % (*100) when makes sense
    flat_labels = np.concatenate(epoch_labels)
    flat_preds = np.concatenate(epoch_preds)
    epoch_log = {}

    print("Computeing metrics for", phase)
    print("Number of samples:", len(flat_labels))
    if "acc" in metrics_list:
        epoch_acc = 100.0 * (flat_preds == flat_labels).sum() / len(flat_labels)
        epoch_log[f"{phase}_acc"] = epoch_acc

    if "eer" in metrics_list and epoch_logits is not None and eer_metric is not None:
        flat_logits = np.concatenate(epoch_logits)
        flat_scores = torch.softmax(torch.from_numpy(flat_logits), dim=1).numpy()
        positive_scores = flat_scores[:, 1]  # for binary, use positive class index
        epoch_eer = 100 * eer_metric(flat_labels, positive_scores)
        epoch_log[f"{phase}_eer"] = epoch_eer

    if "f1_score" in metrics_list:
        epoch_f1 = f1_score(flat_labels, flat_preds, average='macro')
        epoch_log[f"{phase}_f1_score"] = epoch_f1

    if "confusion_matrix" in metrics_list or "per_class_accuracy" in metrics_list:
        cm = confusion_matrix(flat_labels, flat_preds)
        # confusion matrix will be done by wandb
        # if "confusion_matrix" in metrics_list:
        #     epoch_log[f"{phase}_confusion_matrix"] = cm
        if "per_class_accuracy" in metrics_list:
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            for idx, acc in enumerate(per_class_acc):
                class_name = class_names[idx] if class_names else str(idx)
                epoch_log[f"{phase}_class_{class_name}_acc"] = 100.0 * acc
            
    return epoch_log
            

def print_metrics(metrics_list, epoch_log, phase, class_names = None):
    parts = []
    for m in metrics_list:
        if m == "confusion_matrix":
            continue
        if m == "per_class_accuracy":
            for cls in class_names:
                k = f"{phase}_class_{cls}_acc"
                val = epoch_log.get(k, None)
                parts.append(f"{k}: {val:.2f}%")
        else:
            key = f"{phase}_{m}"
            val = epoch_log.get(key, None)
            if val is not None:
                suffix = "%" if m in ["acc", "eer"] else ""
                parts.append(f"{m}: {val:.2f}{suffix}")
    print(" | ".join(parts))



class Metrics():

    def __init__(self, TP=None, FP=None, TN=None, FN=None):
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN

    def get_accuracy(self):
        if None in [self.TP, self.TN, self.FP, self.FN]:
            return 0

        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)

    def get_FAR(self):
        if None in [self.FP, self.TN]:
            return 0

        if self.FP+self.TN == 0:
            return 0

        return self.FP/(self.FP+self.TN)

    def get_FRR(self):
        if None in [self.FN, self.TP]:
            return 0

        if self.FN+self.TP == 0:
            return 0

        return self.FN/(self.FN+self.TP)

def eer_metric(labels, scores):
    """Compute the Equal Error Rate (EER) from the predictions and scores.
    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        scores (list[float]): scores from model
    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold
    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.
       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """

    # seems these are logits, lets transform them in probs using Softmax
    
    #import pdb; pdb.set_trace()
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    return eer


def calc_far_frr(results, ground_truths, const:int=100):
    """
    Calculate the FAR / FRR for a liveness result.
    Input:
        results: a list of probabilities of the given image being a real person
        ground_truths: a list of ints, where 0=spoof, 1=live
        const: how many threshold steps to use. 1 would be 100, 10 would be 1000,
            so larger numbers would present a finer result but will be slower to
            compute.
    Returns:
        fars[float]: list of FARs
        frrs[float]: list of FRRs
        EER (int): Equal Error Rate
    """
    results = np.array(results)
    ground_truths = np.array(ground_truths)
    fars, frrs = [], []
    for thr in range(0,100*const+1):
        thr = thr/(const*100)
        passed_thresh = (results>=thr).astype(np.int32)
        # np.dot will sum all the results where passed_thresh==1 and ground_truths==1
        TP = np.dot(passed_thresh, ground_truths)
        # Inverting results and gts will sum passed_thresh==0 and ground_truths==0
        TN = np.dot(1-passed_thresh, 1-ground_tmean_per_class_accruths)
        # Inverting the passed_thresh will sum passed_thresh==0 and ground_truths==1
        FN = np.dot(1-passed_thresh, ground_truths)
        # Inverting the ground_truths will sum passed_thresh==1 and ground_truths==0
        FP = np.dot(passed_thresh, 1-ground_truths)

        metrics = Metrics(TP, FP, TN, FN)
        # if thr<1:
        #     print('{:.3f} - FAR: {:.3f} - FRR: {:.3f} - TP: {} FP: {} TN: {} FN: {}'.format(
        #         thr, metrics.get_FAR(), metrics.get_FRR(), TP, FP, TN, FN))
        fars.append(metrics.get_FAR())
        frrs.append(metrics.get_FRR())

    eer = 100
    for idx in range(len(fars)-1):
        far_l, frr_l = fars[idx], frrs[idx]
        far_r, frr_r = fars[idx+1], frrs[idx+1]
        if far_l >= frr_l and far_r <= frr_r:
            # (gfickel) TODO: perhaps do a linear interpolation instead of
            # just using the raw mean
            eer = ((far_l+far_r)/2+(frr_l+frr_r)/2)/2
            break

    return fars, frrs, eer

