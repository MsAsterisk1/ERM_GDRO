import torch
from loss import ERMLoss, GDROLoss, UpweightLoss, ERMGDROLoss
import models
import utils.process_data_utils as utils
from train_eval import run_trials
import pandas as pd
import os
import argparse
from math import ceil
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--test_name', default='test')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.dataset == 'waterbirds':


    train_dataset, val_dataset, test_dataset = utils.get_waterbirds_datasets(device=device)

    # From Distributionally Robust Neural Networks
    batch_size = (32, 32)
    eta = 0.01


    run_trials_args = {
        'model_class': models.TransferModel50,
        'model_args': {'device': device, 'freeze': False},
        'epochs': 300,
        'optimizer_class': torch.optim.SGD,
        'optimizer_args': {'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.9},
        'num_subclasses': 4,
    }


 


elif args.dataset == 'mnist':
    train_dataset, val_dataset, test_dataset = utils.get_MNIST_datasets(device=device)
    batch_size = (1024, 1024)
    eta = 0.01

    
    run_trials_args = {
        'model_class': models.NeuralNetwork,
        'model_args': {'layers': [28 * 28, 256, 64, 10]},
        'epochs': 10,
        'optimizer_class': torch.optim.Adam,
        'optimizer_args': {'lr': 0.0005, 'weight_decay': 0.005},
        'num_subclasses': 10,
    }


elif args.dataset == 'civilcomments':
    
    print('getting dataset')
    train_dataset, val_dataset, test_dataset = utils.get_CivilComments_Datasets(device=device)
    print('got dataset')
    batch_size = (16, 32)

    # From WILDS
    epochs = 5
    num_training_steps = ceil(len(train_dataset) / batch_size[0]) * epochs
    eta = 0.01
   

    run_trials_args = {
        'model_class': models.BertClassifier,
        'model_args': {'device':device},
        'epochs': 5,
        'optimizer_class': torch.optim.AdamW,
        'optimizer_args': {'lr': 0.00001, 'weight_decay': 0.01},
        'num_subclasses': 18,
        'scheduler_class': transformers.get_linear_schedule_with_warmup,
        'scheduler_args': {'num_warmup_steps':0, 'num_training_steps':num_training_steps},
        'gradient_clip': 1,
        'vector_subclass':True
    }



trials = 5
run_trials_args['num_trials'] = trials
split_path = "train_test_splits/LIDC_data_split.csv"
subclass_path = 'subclass_labels/subclasses.csv'
feature_path = 'LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'

results_root_dir = 'test_results/'
test_name = args.test_name
results_dir = results_root_dir + f'{test_name}/'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

verbose = args.verbose

num_subclasses = run_trials_args['num_subclasses']

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

for loss_class, fn_name, loss_args in zip(
        [erm_class, gdro_class, upweight_class, mix25_class, mix50_class, mix75_class],
        [erm_name,  gdro_name,  upweight_name,  mix25_name,  mix50_name,  mix75_name],
        [erm_args,  gdro_args,  upweight_args,  mix25_args,  mix50_args,  mix75_args]):
    if verbose:
        print(f"Running trials: {fn_name}")

    reweight_train = loss_class != erm_class
    train_dataloader, val_dataloader, test_dataloader, = utils.get_dataloaders(
        (train_dataset, val_dataset, test_dataset),
        batch_size=batch_size,
        reweight_train=reweight_train
    )

    run_trials_args['loss_class'] = loss_class
    run_trials_args['loss_args'] = loss_args

    run_trials_args['train_dataloader'] = train_dataloader
    run_trials_args['val_dataloader'] = val_dataloader
    run_trials_args['test_dataloader'] = test_dataloader



    accuracies, q_data, roc_data = run_trials(**run_trials_args)
    results["Accuracies"][fn_name] = accuracies
    results["q"][fn_name] = q_data
    # results["ROC"]["labels"] = roc_data[1].tolist()
    # results["ROC"][fn_name] = roc_data[0].tolist()

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
    # roc_df = pd.DataFrame(results["ROC"])

    accuracies_df.to_csv(results_dir + f'accuracies.csv')
    q_df.to_csv(results_dir + f'q.csv')
    # roc_df.to_csv(results_dir + f'roc.csv', index=False)
