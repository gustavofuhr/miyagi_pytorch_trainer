import os
import argparse
import time
import datetime
import copy
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


def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                loss_function,
                n_epochs = 100,
                metric_eer = False,
                track_experiment = False,
                track_images = False, 
                save_best_model = False):
    """
    Train a model given model params and dataset loaders
    """
    torch.backends.cudnn.benchmark = True

    # the backbone, usually will not restrict the input size of the data
    # e.g.: before the fc of Resnet we have (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # but the input channels are related:
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    model.to(device)
    if track_experiment:
        wandb.watch(model)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_eer = 100


    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    print("Dataset classes:")
    print("Train", dataloaders["train"].dataset.classes)
    print("Val", dataloaders["val"].dataset.classes)

    # TODO: from where to get
    # cls_nms = val_loader.dataset.classes
    phases = ["train", "val"]
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in phases}
    num_epochs = n_epochs

    start = time.time()


    for epoch in range(num_epochs):
        start_epoch = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        if track_experiment:
            epoch_log = {}

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            running_labels = torch.Tensor()
            running_outputs = torch.Tensor()

            #wrong_epoch_images = deque(maxlen=32)
            #wrong_epoch_attr = deque(maxlen=32)

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                if metric_eer:
                    running_labels = torch.cat((running_labels, labels.detach().cpu()))
                # TODO: needs to cast to float.
                inputs = inputs.float().to(device)
                # TODO: a bunch of stupid convertion for label.
                labels = labels.type(torch.LongTensor).flatten().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
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
                running_corrects += torch.sum(preds == labels.data)
                if metric_eer:
                    running_outputs = torch.cat((running_outputs, outputs.detach().cpu()))

                #if phase == "train":
                #    wrong_epoch_images.extend([x for x in inputs[preds!=labels]])
                    #if track_images:
                    #    wrong_epoch_attr.extend([(labels[i], preds[i])\
                    #                                for i in (preds!=labels).nonzero().flatten()])

            if phase == 'train':
                scheduler.step()

            if metric_eer:
                probs = metrics.softmax(running_outputs)
                scores = probs[:,1]

                epoch_eer = 100 * metrics.eer_metric(running_labels, scores)

            epoch_loss = running_loss/len(dataloaders[phase])
            epoch_acc = 100 * running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                if metric_eer and epoch_eer < best_eer:
                    best_eer = epoch_eer

            if save_best_model:
                save_curr_model = (not metric_eer and epoch_acc > best_acc) or \
                                  (metric_eer and epoch_eer < best_eer)
                if save_curr_model:
                    model_folder = wandb.run.name if track_experiment else \
                                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    if not os.path.exists(model_folder):
                        os.mkdir(model_folder)
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, os.path.join(model_folder, "best_saved_model.pt"))
                    
            if track_experiment:
                if phase == "val":
                    wandb.run.summary["best_val_acc"] = best_acc

                epoch_log.update({
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc,
                })
                if metric_eer:
                    epoch_log.update({
                        f"{phase}_eer": epoch_eer
                    })
                    if phase == "val":
                        wandb.run.summary["best_val_eer"] = best_eer

                if track_images and phase == "train":
                    epoch_log.update({"last_train_batch" : wandb.Image(inputs)})


        duration_epoch = time.time() - start_epoch

        if track_experiment:
            epoch_log.update({"duration_epoch": duration_epoch})
            wandb.log(epoch_log)

        print()


    time_elapsed = time.time() - since
    if track_experiment:
        wandb.run.summary["total_duration"] = time_elapsed

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train(args):
    resize_size = int(args.resize_size) if args.resize_size is not None else None

    train_transform, val_transform = augmentations.get_augmentations(resize_size, args.augmentation)
    transformers = {
        "train": train_transform,
        "val": val_transform
    }

    # NOTE: I'am enabling using + between dataset names because of sweeps which does not work with nargs
    in_datasets_names = {
        "train": args.train_datasets[0].split("+") if "+" in args.train_datasets[0] else args.train_datasets,
        "val": args.val_datasets[0].split("+") if "+" in args.val_datasets[0] else args.val_datasets,
    }

    train_loader, val_loader = dataloaders.get_dataset_loaders(in_datasets_names,
                                                               transformers,
                                                               int(args.batch_size),
                                                               int(args.num_dataloader_workers),
                                                               args.balanced_weights,
                                                               args.multiple_balanced_datasets)

    model = models.get_model(args.backbone, len(train_loader.dataset.classes),
                                        not args.no_transfer_learning, args.freeze_all_but_last)
    print(f"model {args.backbone}")
    if args.weights:
        model.load_state_dict(torch.load(args.weights))
    # print(model)

    optimizer = optimizers.get_optimizer(model, args.optimizer, args.weight_decay)
    scheduler = schedulers.get_scheduler(optimizer, args)
    loss_function = losses.get_loss(args.loss)

    if args.wandb_sweep_activated:
        wandb.init(project=args.experiment_group, entity=args.wandb_user, config=args)
    elif args.track_experiment:
        if args.experiment_group == "" or args.experiment_name == "":
            raise Exception("Should define both the experiment group and name.")
        else:
            wandb.init(project=args.experiment_group, name=args.experiment_name, entity=args.wandb_user, config=args)

    train_model(model, train_loader, val_loader, optimizer, scheduler, loss_function,
                    int(args.n_epochs), args.metric_eer, args.track_experiment, args.track_images, 
                    args.save_best_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50")


    parser.add_argument("--no_transfer_learning", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_all_but_last", action=argparse.BooleanOptionalAction)

    # {phase} datasets are hope to have {phase}-named folders inside them
    parser.add_argument("--train_datasets", action='store', type=str, nargs="+", required=True)
    parser.add_argument("--val_datasets", action='store', type=str, nargs="+", required=True)
    parser.add_argument("--balanced_weights", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multiple_balanced_datasets", action=argparse.BooleanOptionalAction,
        help="Dataset path contains multiple datasets that will be combined, each one "
        "having equal weight")

    parser.add_argument("--resize_size", default=None)
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
        choices=["noaug", "simple", "rand-m9-n3-mstd0.5", "rand-mstd1-w0", "random_erase"])

    # options for optimizers
    parser.add_argument("--optimizer", default="sgd") # possible adam, adamp and sgd
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # options for model saving
    parser.add_argument("--save_best_model", action=argparse.BooleanOptionalAction)

    # options for liveness
    parser.add_argument("--metric_eer", action=argparse.BooleanOptionalAction)

    # options for losses
    parser.add_argument("--loss",  type=str, default="cross_entropy")
    # some choices are "cross_entropy", "angular", "custom_anomaly", "cross_entropy_label_smoothing_0.1", etc.
    parser.add_argument("--ce_loss_label_smoothing",  type=float, required=False, default=0.0)

    args = parser.parse_args()
    train(args)
