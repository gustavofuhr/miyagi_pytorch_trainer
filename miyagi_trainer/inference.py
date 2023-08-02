import csv
import argparse

import torch
from tqdm import tqdm

import models
import dataloaders
import augmentations


def run_model(model, ds_loader, out_csv, isolate_class):
    """
    Runs a model given model on a dataset loader and save the results on a CSV
    """
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    model.to(device)
    model.eval()   # Set model to evaluate mode

    with open(out_csv, 'w', newline='') as fid:
        csv_writer = csv.writer(fid)
        csv_writer.writerow(['score', 'class_id', 'gt'])
        for inputs, labels in tqdm(ds_loader):
            labels = labels.type(torch.LongTensor).flatten().to(device)
            inputs = inputs.float().to(device)
            outputs = model(inputs)

            if isolate_class != -1:
                scores = torch.softmax(outputs, 1)
                scores = scores[:,isolate_class]
                class_ids = [isolate_class]*len(labels)
            else:
                scores, target_classes = torch.max(outputs, 1)
                class_ids = target_classes.to('cpu').numpy().tolist() 
            scores = scores.detach().to('cpu').numpy().tolist()
            labels = labels.to('cpu').numpy().tolist()
            csv_writer.writerows(list(zip(scores, class_ids, labels)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--isolate_class", type=int, default=-1,
        help='Only save the score of a given class index instead of the highest one')
    parser.add_argument("--datasets", action='store', type=str, nargs="+", required=True)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_dataloader_workers", type=int, default=8) # recomends to be 4 x #GPU
    args = parser.parse_args()

    resize_size = int(args.resize_size) if args.resize_size is not None else None

    _, ds_transform = augmentations.get_augmentations(resize_size, None) 

    datasets_names = args.datasets[0].split("+") if "+" in args.datasets[0] else args.datasets
    ds_loader = dataloaders.get_loader('test',
                                       datasets_names,
                                       ds_transform,
                                       args.batch_size,
                                       args.num_dataloader_workers)

    
    model = models.get_model(args.backbone, len(ds_loader.dataset.classes))
    state_dict = torch.load(args.model_path)['model_state_dict']
    model.load_state_dict(state_dict)

    run_model(model, ds_loader, args.output_csv, args.isolate_class)
