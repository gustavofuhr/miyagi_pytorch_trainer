import os
import inspect
import re
from collections import defaultdict, Counter

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from datasets import CUSTOM_DATASETS


VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """ImageFolder that also returns the image file path as a third element."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

def _get_pytorch_dataloders(dataset, batch_size, num_workers, balance_sampling = False):
    if balance_sampling:
        class_sample_count = torch.tensor([*dataset.class_sample_count.values()])

        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in dataset.all_targets])

        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, 
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


def _get_image_folder_dataset_from_path(root_path, split, transform, filter_off_regex=None):
    split_path = os.path.join(root_path, split)
    if not os.path.isdir(split_path):
        raise ValueError(
            f"Dataset path does not exist or is not a directory: {split_path!r}. "
            f"Expected structure: <root>/<split>/class/image.png"
        )

    is_valid_func = None
    if filter_off_regex:
        print(f"Applying regex filter '{filter_off_regex}' to {root_path}/{split}")
        pattern = re.compile(filter_off_regex)

        def file_filter(path):
            if not path.lower().endswith(VALID_IMG_EXTENSIONS):
                return False
            if pattern.search(path):
                return False
            return True

        is_valid_func = file_filter

    return ImageFolderWithPaths(
        root=split_path,
        transform=transform,
        is_valid_file=is_valid_func,
    )


def _is_path(name):
    """Return True if the dataset token should be treated as a filesystem path."""
    return name.startswith("/") or name.startswith("./") or name.startswith("../") or os.sep in name


def _get_image_folder_dataset(dataset_name, split, transform, filter_off_regex=None):
    root_path = os.path.join(CUSTOM_DATASETS[dataset_name], split)

    # Logic: If regex is provided, we build a custom checker. 
    # Otherwise we let ImageFolder use its default (None).
    is_valid_func = None
    
    if filter_off_regex:
        print(f"Applying regex filter '{filter_off_regex}' to {dataset_name}/{split}")
        pattern = re.compile(filter_off_regex)

        def file_filter(path):
            # 1. Check if it's an image file (Case insensitive)
            if not path.lower().endswith(VALID_IMG_EXTENSIONS):
                return False
            
            # 2. Check if path matches the user regex (e.g. "bad_file")
            # We search the full path so you can filter by folder names too if needed
            if pattern.search(path):
                return False
            
            return True
        
        is_valid_func = file_filter

    # Pass is_valid_file. Note: extensions arg is ignored if is_valid_file is not None.
    dataset = ImageFolderWithPaths(
        root=root_path,
        transform=transform,
        is_valid_file=is_valid_func,
    )

    return dataset


class DatasetJoin(torch.utils.data.ConcatDataset):

    def __init__(self, imagefolder_dataset_list):
        super(DatasetJoin, self).__init__(imagefolder_dataset_list)
        self.join_classes()

    def join_classes(self):
        join_class_to_idx = None
        class_sample_count = Counter()
        self.all_targets = []
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
                            class_imbalance_strategy = "none",
                            filter_off_regex = None): 

    splits = ["train", "val"]
    combined_datasets = {}
    data_loaders = {}
    for s in splits:
        split_datasets = []
        for ith_ds, ds_name in enumerate(dataset_names[s]):
            transform = transforms[s]
            # single_eye_crop mode passes a (left_transform, right_transform) tuple;
            # load the dataset twice (once per eye) so each image yields two samples.
            if isinstance(transform, tuple):
                for t in transform:
                    if ds_name in dir(torchvision.datasets):
                        split_datasets.append(_get_pytorch_dataset(ds_name, s, t))
                    elif ds_name in CUSTOM_DATASETS.keys():
                        split_datasets.append(_get_image_folder_dataset(ds_name, s, t, filter_off_regex))
                    elif _is_path(ds_name):
                        resolved = os.path.abspath(ds_name)
                        split_datasets.append(_get_image_folder_dataset_from_path(resolved, s, t, filter_off_regex))
                    else:
                        raise ValueError(
                            f"Dataset {ds_name!r} is not a torchvision built-in, not a key in CUSTOM_DATASETS, "
                            f"and does not look like a filesystem path. "
                            f"Pass an absolute path (/foo/bar) or a relative path (./foo/bar or foo/bar)."
                        )
                continue
            else:
                if ds_name in dir(torchvision.datasets):
                    this_dataset = _get_pytorch_dataset(ds_name, s, transform)
                elif ds_name in CUSTOM_DATASETS.keys():
                    this_dataset = _get_image_folder_dataset(ds_name, s, transform, filter_off_regex)
                elif _is_path(ds_name):
                    resolved = os.path.abspath(ds_name)
                    this_dataset = _get_image_folder_dataset_from_path(resolved, s, transform, filter_off_regex)
                else:
                    raise ValueError(
                        f"Dataset {ds_name!r} is not a torchvision built-in, not a key in CUSTOM_DATASETS, "
                        f"and does not look like a filesystem path. "
                        f"Pass an absolute path (/foo/bar) or a relative path (./foo/bar or foo/bar)."
                    )

            split_datasets.append(this_dataset)

        combined_datasets[s] = DatasetJoin(split_datasets)
        data_loaders[s] = _get_pytorch_dataloders(combined_datasets[s], batch_size, num_workers, class_imbalance_strategy == "batch")

    train_targets = combined_datasets["train"].all_targets
    train_class_counts = np.bincount(train_targets.numpy(), minlength=len(combined_datasets["train"].classes))
    print("train class counts:", train_class_counts)

    return data_loaders["train"], data_loaders["val"], train_class_counts


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
