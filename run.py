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
parser.add_argument('loss', nargs='+')
parser.add_argument('--trials', default=5)
parser.add_argument('--test_name', default='test')
parser.add_argument('--device', default="0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--verbose', action='store_true')



args = parser.parse_args()

# hyperparameters

if args.device == 'cpu':
    device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch.cuda.set_device(int(args.device))
    device = 'cuda'

if args.dataset == 'waterbirds':
    print('getting data')
    train_dataset, val_dataset, test_dataset = utils.get_waterbirds_datasets(device=device)
    print('got data')
    # From Distributionally Robust Neural Networks
    batch_size = (256, 256)
    eta = 0.01
    num_subclasses = 4

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
    num_subclasses = 10

    run_trials_args = {
        'model_class': models.NeuralNetwork,
        'model_args': {'layers': [28 * 28, 256, 64, 10], 'device': device},
        'epochs': 10,
        'optimizer_class': torch.optim.Adam,
        'optimizer_args': {'lr': 0.0005, 'weight_decay': 0.005},
        'num_subclasses': 10,
    }
elif args.dataset == 'civilcomments':
    train_dataset, val_dataset, test_dataset = utils.get_CivilComments_Datasets(device=device)
    batch_size = (16, 128)

    # for gdro only train on labels as classes
    num_subclasses = 4

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

trials = int(args.trials)
run_trials_args['num_trials'] = trials

run_trials_args['verbose'] = args.verbose
run_trials_args['record'] = True

results_root_dir = 'test_results/'
test_name = args.test_name
results_dir = results_root_dir + f'{test_name}/'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

verbose = args.verbose


subtypes = ["Overall"]
subtypes.extend(list(range(run_trials_args['num_subclasses'])))

erm_class = ERMLoss
erm_args = {'loss_fn': torch.nn.CrossEntropyLoss()}
gdro_class = GDROLoss
gdro_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses}
upweight_class = ERMLoss
upweight_args = {'loss_fn': torch.nn.CrossEntropyLoss()}
mix25_class = ERMGDROLoss
mix25_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.25, 'partitioned': True}
mix50_class = ERMGDROLoss
mix50_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.50, 'partitioned': True}
mix75_class = ERMGDROLoss
mix75_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.75, 'partitioned': True}

losses = {'erm': (erm_class, erm_args), 'gdro': (gdro_class, gdro_args), 'upweight': (upweight_class, upweight_args), 'mix25': (mix25_class, mix25_args), 'mix50': (mix50_class, mix50_args), 'mix75': (mix75_class, mix75_args)}

accuracies = {}

print('got loss functions')
for loss_fn in args.loss:
    if verbose:
        print(f"Running trials: {loss_fn}")

    reweight_train = loss_fn != 'erm'
    train_dataloader, val_dataloader, test_dataloader, = utils.get_dataloaders(
        (train_dataset, val_dataset, test_dataset),
        batch_size=batch_size,
        reweight_train=reweight_train,
        split=loss_fn.startswith('mix'),
        proportion=0.7
    )

    run_trials_args['loss_class'], run_trials_args['loss_args'] = losses[loss_fn]

    run_trials_args['train_dataloader'] = train_dataloader
    run_trials_args['val_dataloader'] = val_dataloader
    run_trials_args['test_dataloader'] = test_dataloader

    print('run trials')
    accuracies[loss_fn] = run_trials(**run_trials_args)[0]

    accuracies_df = pd.DataFrame(
        accuracies,
        index=pd.MultiIndex.from_product(
            [range(run_trials_args['num_trials']), range(run_trials_args['epochs'] + 1), subtypes],
            names=["trial", "epoch", "subtype"]
        )
    )

    accuracies_df.to_csv(results_dir + f'accuracies.csv')
