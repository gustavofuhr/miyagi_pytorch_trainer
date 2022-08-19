# Miyagi's Pytorch Trainer

A pytorch trainer with a range of choice for backbones, losses, augmentation and wandb sweeps.

![Catch that fly!](https://observatoriodocinema.uol.com.br/wp-content/uploads/2021/01/miyagi.jpg)

# Features

- Easy to use for your own datasets, based on [`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html), supports multiple combinations and pre-existing [datasets from Pytorch](https://pytorch.org/vision/stable/datasets.html);
- Several backbones available, based on the awesome [`timm`](https://github.com/rwightman/pytorch-image-models);
- Losses are a mix from what you can find in the official Pytorch and  [Pytorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- Experiments can be tracked with (Weight & Biases)[https://wandb.ai/] and sweeps are integrated, check section on them bellow,

# How to use

## Quickstart

1. Create and activate the conda environment (it should take a few minutes):

```
conda env create --name miyagi --file=environment.yml
conda activate miyagi
```

2. (Optionally) Login in with your [wandb](https://wandb.ai/) account:

```
wandb login
```

3. Run training (ex.: CIFAR10, mobilenet_v3, CE loss):

```
python3 miyagi_trainer/train.py --resize_size 224 --train_datasets CIFAR10 --val_datasets CIFAR10 --backbone mobilenetv3_large_100
```


## Datasets

To have up and running in no time, the trainer support using the [Pytorch official datasets](https://pytorch.org/vision/stable/datasets.html). Anyone in that list should work fine, samples:
```
--train-datasets CIFAR10 --val-datasets CIFAR10
```

Of course you can also use your own data to train. For that we use the `torchvision.datasets.ImageFolder`. As per the documentation, you should put your images for each class inside a folder named with that class, ex:


```
custom_dataset_folder
│
└───train
│   │
│   └───class1
│   |       file001.png
│   |       file002.png
│   |       ...
│   └───class2
│           file003.jpg
│           file004.png
│           ...
└───val
    └───class1
    │       file005.jpeg
    ...
```
For reasons that should become clear, you need to chose a name for your custom dataset and put them into `CUSTOM_DATASETS` in `datasets.py`:

```
CUSTOM_DATASETS = {
    "custom_dataset": "data/custom_dataset_folder/"
}
```
Then you can reference by name. Is's always expected that you have a `train` and `val` folder as the primary subfolders in your path (check tree above). You also can combine multiples datasets using `+`, link this:

```
--train-datasets CIFAR10+CIFAR100 --val-datasets CIFAR10
```
of course, the classes should be in all the datasets included. I implemented using `+` because otherwise sweeps will not work for them.

## Choosing the backbone

The lib uses the 0.5.4 version of the [`timm`](https://github.com/rwightman/pytorch-image-models) package. You can always update to the newest version, if you want it. It also works on [torchvision models], but priority is always given to `timm`. You can control if want a pre-trained model (the default) or don't (`--no_transfer_learning`) which I personally don't recommend. 
You can specify a backbone using the flag:

`--backbone 


## Choosing the loss
## Choosing the augmentation
## Enabling experiment tracking and sweeps


# Samples



