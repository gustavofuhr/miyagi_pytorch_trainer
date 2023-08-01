import os
import inspect
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms

from datasets import CUSTOM_DATASETS


def _get_pytorch_dataloders(
        dataset, batch_size, num_workers, balanced_weights=False,
        multiple_datasets_temperature=0.0):
    
    if balanced_weights:
        class_sample_count = torch.tensor([*dataset.class_sample_count.values()])

        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in dataset.all_targets])

        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers, 
            pin_memory=True, sampler=sampler)

    elif multiple_datasets_temperature:
        samples_weight_list = []
        num_datasets = len(dataset.dataset_idx)
        datasets_sizes = [sum(dataset.ds_class_sample_count[idx].values())
                          for idx in range(num_datasets)]
        t = multiple_datasets_temperature # temperature
        sizes_tensor = torch.tensor(datasets_sizes).float()
        weights = torch.softmax(sizes_tensor/(max(datasets_sizes)*t), dim=0)
        dataset_weights = weights.numpy(force=True)
        
        for idx, ds_idx in enumerate(dataset.dataset_idx):
            class_sample_count = torch.tensor([*dataset.ds_class_sample_count[idx].values()])
            weight = (class_sample_count/torch.sum(class_sample_count)).float()
            # degenerate case, when we have only one class
            if len(weight) == 1:
                weight = torch.tensor([0.5]).float()
            weight *= dataset_weights[idx]
            if idx < num_datasets-1:
                next_ds_idx = dataset.dataset_idx[idx+1]
            else:
                next_ds_idx = len(dataset.all_targets)
            samples_weight_list.extend(
                [weight[t] for t in dataset.all_targets[ds_idx:next_ds_idx]])
            
        samples_weight = torch.tensor(samples_weight_list)
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, sampler=sampler)
        
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True)

    return loader


def _get_pytorch_dataset(dataset_name, split, transform):
    dataset = getattr(torchvision.datasets, dataset_name)
    dataset_sig = inspect.signature(dataset)

    # all_transform = _get_pytorch_default_transform(resize_size)

    if "train" in dataset_sig.parameters:
        dataset = dataset(root="../data/", train=(split == "train"),
                                            transform=transform, download=True)
    elif "split" in dataset_sig.parameters:
        dataset = dataset(root="../data/", split=split,
                                            transform=transform, download=True)
    else:
        raise Exception("Don't understand dataset method signature.")

    return dataset

def _get_image_folder_dataset(dataset_name, split, transform):
    dataset = torchvision.datasets.ImageFolder(root=os.path.join(CUSTOM_DATASETS[dataset_name], split),
                                                                            transform=transform)

    return dataset


class DatasetJoin(torch.utils.data.ConcatDataset):

    def __init__(self, imagefolder_dataset_list):
        super(DatasetJoin, self).__init__(imagefolder_dataset_list)
        self.join_classes()

    def join_classes(self):
        join_class_to_idx = None
        class_sample_count = Counter()
        self.ds_class_sample_count = []
        self.all_targets = []
        self.dataset_idx = []
        for ds in self.datasets:
            if join_class_to_idx is None:
                join_class_to_idx = ds.class_to_idx
            else:
                target_mapping = {}
                new_classes = []
                for k in ds.classes:
                    if k in join_class_to_idx.keys() and join_class_to_idx[k] != ds.class_to_idx[k]: # different ids
                        # shoud use a transform to the previously define int
                        target_mapping[ds.class_to_idx[k]] = join_class_to_idx[k]
                    else:
                        new_classes.append(k)
                sub_class_to_idx = {key: ds.class_to_idx[key] for key in new_classes}
                join_class_to_idx.update(sub_class_to_idx)
                ds.target_transform = lambda y: target_mapping.get(y, y)
            this_ds_counts = Counter(ds.targets)
            # TODO: this class_sample_count is not taking into account the target_mapping
            class_sample_count.update(this_ds_counts)
            self.ds_class_sample_count.append(this_ds_counts)
            self.dataset_idx.append(len(self.all_targets))
            self.all_targets.extend(ds.targets)

        # TODO: for this to work the order should be same when Im weighting samples        
        self.all_targets = torch.tensor(self.all_targets)

        self.class_sample_count = class_sample_count

        self.join_class_to_idx = join_class_to_idx
        s = set().union(*[ds.classes for ds in self.datasets])
        self.classes = list(s)


def get_dataset_loaders(dataset_names,
                        transforms,
                        batch_size = 32,
                        num_workers = 4,
                        balanced_weights = False,
                        multiple_datasets_temperature = 0.2):

    """
    Expecting dataset_names and transforms to be dict with "train" and "val" keys
    """
    splits = ["train", "val"]
    combined_datasets = {}
    data_loaders = {}
    for s in splits:
        split_datasets = []
        for ds_name in dataset_names[s]:
            if ds_name in dir(torchvision.datasets):
                this_dataset = _get_pytorch_dataset(ds_name, s, transforms[s])
            elif ds_name in CUSTOM_DATASETS.keys():
                this_dataset = _get_image_folder_dataset(ds_name, s, transforms[s])
            else:
                raise ValueError(f'Invalid dataset: {ds_name}')

            split_datasets.append(this_dataset)

        # TODO: https://stackoverflow.com/questions/71173583/concat-datasets-in-pytorch
        combined_datasets[s] = DatasetJoin(split_datasets)
        if s == 'train':
            data_loaders[s] = _get_pytorch_dataloders(
                combined_datasets[s], batch_size, num_workers,
                balanced_weights,
                multiple_datasets_temperature)
        else:
            data_loaders[s] = _get_pytorch_dataloders(
                combined_datasets[s], batch_size, num_workers)

    return data_loaders["train"], data_loaders["val"]


if __name__ == "__main__":
    dataset_names = {
        "train": ["liveness_simple", "flash_ds"],
        "val": ["liveness_simple"]
    }
    from resnet_exp.augmentations import simple_augmentation
    empty_transf = simple_augmentation(128)

    transforms = {
        "train": empty_transf,
        "val": empty_transf
    }
    train_dataloader, val_dataloader = get_dataset_loaders(dataset_names, transforms)
