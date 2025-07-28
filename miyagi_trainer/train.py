import os
import argparse
import time
import datetime
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

import models
import dataloaders
import augmentations
import optimizers
import schedulers
import metrics
import losses


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                loss_function,
                n_epochs,
                metrics_list,
                track_experiment,
                track_images, 
                save_best_model,
                save_best_metric):
    """
    Train a model given model params and dataset loaders
    """
    torch.backends.cudnn.benchmark = True

    # the backbone, usually will not restrict the input size of the data
    # e.g.: before the fc of Resnet we have (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # but the input channels are related:
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model.to(device)
    if track_experiment:
        wandb.watch(model)

    since = time.time()

    best_metrics = metrics.init_metrics_best(metrics_list)
    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    print("Dataset classes:")
    print("Train", dataloaders["train"].dataset.join_class_to_idx)
    print("Val", dataloaders["val"].dataset.join_class_to_idx)

    class_names = list(dataloaders["train"].dataset.join_class_to_idx.keys())
    print("Classes:", class_names)
    phases = ["train", "val"]
    # dataset_sizes = {x: len(dataloaders[x].dataset) for x in phases}
    num_epochs = n_epochs
    model_name = wandb.run.name if track_experiment else \
                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    if save_best_model:
        if not os.path.exists("trained_models/"):
            os.makedirs("trained_models/")
    
    model_path = os.path.join("trained_models/", f"{model_name}.pt")
    for epoch in range(num_epochs):
        start_epoch = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_log = {}
        # Each epoch has a training and validation phase
        for phase in phases:
            epoch_preds, epoch_labels, epoch_logits = [], [], []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data in batches
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                # TODO: needs to cast to float.
                inputs = inputs.float().to(device)
                # TODO: a bunch of stupid convertion for label.
                labels = labels.type(torch.LongTensor).flatten().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

                epoch_logits.append(outputs.detach().cpu().numpy())
                epoch_preds.append(preds.detach().cpu().numpy())
                epoch_labels.append(labels.detach().cpu().numpy())


            if phase == 'train':
                scheduler.step()

            # compute and store loss
            epoch_loss = running_loss/len(dataloaders[phase])
            epoch_log[f"{phase}_loss"] = epoch_loss

            # store all other metrics 
            epoch_log.update(metrics.compute_metrics(metrics_list, phase, epoch_labels, epoch_preds, epoch_logits, class_names=class_names))
            print(f'{phase} Loss: {epoch_loss:.4f} |', end=" ")

            # compute best metrics so far
            best_metrics, metrics_w_new_best = metrics.get_metrics_new_best(metrics_list, phase, epoch_log, best_metrics)
            metrics.print_metrics(metrics_list, epoch_log, phase, class_names)            
          
            if save_best_model and phase == "val":
                # if the metric got a new best in val
                if f"{phase}_{save_best_metric}" in metrics_w_new_best: 
                    print(f"Metric {phase}_{save_best_metric} got new best, saving model.")
                    torch.save(model, model_path)

                    with open(f"trained_models/{model_name}_class_to_idx.json", "w") as f:
                        json.dump(dataloaders["train"].dataset.join_class_to_idx, f)
                    
            if track_experiment:
                if phase == "val":
                    for key, val in best_metrics.items():
                        wandb.run.summary[f"best_{key}"] = val

                if track_images and phase == "train":
                    epoch_log.update({"last_train_batch" : wandb.Image(inputs)})


        duration_epoch = time.time() - start_epoch
        if track_experiment:
            if "confusion_matrix" in metrics_list and phase == "val":
                labels = dataloaders["val"].dataset.classes
                epoch_log[f"{phase}_confusion_matrix"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.concatenate(epoch_labels),
                    preds=np.concatenate(epoch_preds),
                    class_names=labels
                )                

            epoch_log.update({"duration_epoch": duration_epoch})
            wandb.log(epoch_log, step=epoch)
        print()


    time_elapsed = time.time() - since
    if track_experiment:
        wandb.run.summary["total_duration"] = time_elapsed

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print("Best metrics:")
    # metrics.print_metrics(metrics_list + ["loss"], best_metrics, "val")

    
def train(args):
    resize_size = int(args.resize_size) if args.resize_size is not None else None

    train_transform, val_transform = augmentations.get_augmentations(resize_size, args.augmentation, args.resize_policy)
    transformers = {
        "train": train_transform,
        "val": val_transform
    }
    print("val_transform", val_transform)


    # NOTE: I'am enabling using + between dataset names because of sweeps which does not work with nargs
    in_datasets_names = {
        "train": args.train_datasets[0].split("+") if "+" in args.train_datasets[0] else args.train_datasets,
        "val": args.val_datasets[0].split("+") if "+" in args.val_datasets[0] else args.val_datasets,
    }

    train_loader, val_loader, train_class_counts = dataloaders.get_dataset_loaders(in_datasets_names,
                                                                transformers,
                                                                int(args.batch_size),
                                                                int(args.num_dataloader_workers),
                                                                args.class_imbalance_strategy)

    model = models.get_model(args.backbone, len(train_loader.dataset.classes),
                                        not args.no_transfer_learning, args.freeze_all_but_last)
    print(f"model {args.backbone}")
    # print(model)

    optimizer = optimizers.get_optimizer(model, args.optimizer, args.weight_decay)
    scheduler = schedulers.get_scheduler(optimizer, args, train_loader)

    if args.class_imbalance_strategy == "loss":
        class_weights = train_class_counts.sum() / (len(train_class_counts) * train_class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        loss_function = losses.get_loss(args.loss, class_weights=class_weights.to(device))
    else:
        loss_function = losses.get_loss(args.loss)
    

    def get_model_size(model):
        return sum(p.numel() for p in model.parameters())
    
    model_size_params = get_model_size(model)
    if args.wandb_sweep_activated:
        wandb.init(project=args.experiment_group, entity=args.wandb_user, config=args, 
                            name=f"{args.backbone}_rs{args.resize_size}")
        wandb.config.model_size_params = model_size_params
    elif args.track_experiment:
        if args.experiment_group == "" or args.experiment_name == "":
            raise Exception("Should define both the experiment group and name.")
        else:
            wandb.init(project=args.experiment_group, name=args.experiment_name, entity=args.wandb_user, config=args)
        wandb.config.model_size_params = model_size_params

    train_model(model, train_loader, val_loader, optimizer, scheduler, loss_function,
                    int(args.n_epochs), args.metrics, args.track_experiment, 
                    args.track_images, args.save_best_model, args.save_best_metric)
    
    if args.wandb_sweep_activated:
        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50")


    parser.add_argument("--no_transfer_learning", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_all_but_last", action=argparse.BooleanOptionalAction)

    # {phase} datasets are hope to have {phase}-named folders inside them
    parser.add_argument("--train_datasets", action='store', type=str, nargs="+", required=True)
    parser.add_argument("--val_datasets", action='store', type=str, nargs="+", required=True)
    parser.add_argument(
        "--class_imbalance_strategy", type=str, default="none",
        choices=["none", "batch", "loss"],
        help="How to handle class imbalance: none, batch, or loss"
    )

    parser.add_argument("--resize_size", default=None)
    parser.add_argument(
        "--resize_policy",
        type=str,
        default="resize_exact",
        choices=["resize_then_center_crop", "resize_exact", "resize_with_padding"],
        help=(
            "How to resize input images: "
            "'resize_then_center_crop' (keep aspect, center crop), "
            "'resize_exact' (force resize, may distort), "
            "'resize_with_padding' (letterbox style, pad to fit)."
        ),
    )
    
    parser.add_argument("--num_dataloader_workers", default=8) # recomends to be 4 x #GPU

    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--n_epochs", default=30)

    parser.add_argument('--track_experiment', action=argparse.BooleanOptionalAction)
    parser.add_argument("--experiment_group", default="miyagi-pytorch-trainer")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--track_images", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_user")
    parser.add_argument("--wandb_sweep_activated", action=argparse.BooleanOptionalAction)

    parser.add_argument("--augmentation", type=str, default="simple",
                             choices=["noaug", "simple", "random_erase"])

    # options for optimizers
    parser.add_argument("--optimizer", default="sgd") # possible adam, adamp and sgd
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # options for model saving
    parser.add_argument("--save_best_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save_best_metric", type=str, default="f1_score", choices=["acc", "loss", "eer", "f1_score", "confusion_matrix", "per_class_accuracy"],
        help="Save model with goal metric"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["acc", "f1_score", "confusion_matrix", "per_class_accuracy"],
        choices=["acc", "eer", "f1_score", "confusion_matrix", "per_class_accuracy"],
        help="List of metrics to compute: eer, f1_score, confusion_matrix, per_class_accuracy"
    )
    
    # options for losses
    parser.add_argument("--loss",  type=str, default="cross_entropy")
    # some choices are "cross_entropy", "angular", "custom_anomaly", "cross_entropy_label_smoothing_0.1", etc.
    parser.add_argument("--ce_loss_label_smoothing",  type=float, required=False, default=0.0)

    parser.add_argument("--scheduler", type=str, choices=["cosine", "onecycle"], default="cosine")

    args = parser.parse_args()
    train(args)
