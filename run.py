import torch
from loss import ERMLoss, GDROLoss, UpweightLoss, ERMGDROLoss
import models
import utils.process_data_utils as utils
from train_eval import run_trials
import pandas as pd
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--test_name', default='test')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.dataset == 'waterbirds':
    train_dataloader, val_dataloader, test_dataloader = utils.get_waterbirds_dataloaders(batch_size=8, device=device)
    model_class = models.TransferModel50
    model_args = {'device': device, 'freeze': False}

    # From Distributionally Robust Neural Networks
    eta = 0.01
    epochs = 300
    optimizer_class = torch.optim.SGD
    optimizer_args = {'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.9}
    num_subclasses = 4
    sub_batches = 1

elif args.dataset == 'mnist':
    train_dataloader, val_dataloader, test_dataloader = utils.get_MNIST_dataloaders(batch_size=1024, device=device,
                                                                                    seed=42)
    model_class = models.NeuralNetwork
    model_args = {'layers': [28 * 28, 256, 64, 10]}

    eta = 0.01
    epochs = 10
    optimizer_class = torch.optim.Adam
    optimizer_args = {'lr': 0.0005, 'weight_decay': 0.005}
    num_subclasses = 10
    sub_batches = 1

elif args.dataset == 'civilcomments':
    train_dataloader, val_dataloader, test_dataloader = utils.get_CivilComments_DataLoaders(device=device)
    model_class = models.BertClassifier
    model_args = {}

    # From WILDS
    eta = 0.01
    epochs = 5
    optimizer_class = torch.optim.Adam
    optimizer_args = {'lr': 0.00001, 'weight_decay': 0.01}
    num_subclasses = 18
    sub_batches = 1

trials = 1
split_path = "train_test_splits/LIDC_data_split.csv"
subclass_path = 'subclass_labels/subclasses.csv'
feature_path = 'LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'

results_root_dir = 'test_results/'

test_name = args.test_name

verbose = args.verbose

subtypes = ["Overall"]
subtypes.extend(list(range(num_subclasses)))

erm_class = ERMLoss
erm_name = "ERMLoss"
erm_args = {'loss_fn': torch.nn.CrossEntropyLoss()}
gdro_class = GDROLoss
gdro_name = "GDROLoss"
gdro_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses}
upweight_class = UpweightLoss
upweight_name = "UpweightLoss"
upweight_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'num_subclasses': num_subclasses}
mix25_class = ERMGDROLoss
mix25_name = "MixLoss25"
mix25_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.25}
mix50_class = ERMGDROLoss
mix50_name = "MixLoss50"
mix50_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.50}
mix75_class = ERMGDROLoss
mix75_name = "MixLoss75"
mix75_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.75}

results = {"Accuracies": {}, "q": {}, "ROC": {}}

# for loss_class, fn_name, loss_args in zip(
#         [erm_class, gdro_class, upweight_class, mix25_class, mix50_class, mix75_class],
#         [erm_name,  gdro_name,  upweight_name,  mix25_name,  mix50_name,  mix75_name],
#         [erm_args,  gdro_args,  upweight_args,  mix25_args,  mix50_args,  mix75_args]):
for loss_class, fn_name, loss_args in zip([erm_class, gdro_class], [erm_name, gdro_name], [erm_args, gdro_args]):
    if verbose:
        print(f"Running trials: {fn_name}")

    accuracies, q_data, roc_data = run_trials(
        num_trials=trials,
        epochs=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model_class=model_class,
        model_args=model_args,
        loss_class=loss_class,
        loss_args=loss_args,
        optimizer_class=optimizer_class,
        optimizer_args=optimizer_args,
        device=device,
        scheduler_class=None,
        scheduler_args=None,
        verbose=verbose,
        record=True,
        num_subclasses=num_subclasses,
        sub_batches=sub_batches
    )
    results["Accuracies"][fn_name] = accuracies
    results["q"][fn_name] = q_data
    results["ROC"]["labels"] = roc_data[1].tolist()
    results["ROC"][fn_name] = roc_data[0].tolist()

accuracies_df = pd.DataFrame(
    results["Accuracies"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs + 1), subtypes],
        names=["trial", "epoch", "subtype"]
    )
)
q_df = pd.DataFrame(
    results["q"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs + 1), subtypes[1:]],
        names=["trial", "epoch", "subtype"]
    )
)
roc_df = pd.DataFrame(results["ROC"])

now = datetime.now()

results_dir = results_root_dir + f'{test_name}/'
os.mkdir(results_dir)

accuracies_df.to_csv(results_dir + f'accuracies.csv')
q_df.to_csv(results_dir + f'q.csv')
roc_df.to_csv(results_dir + f'roc.csv', index=False)
