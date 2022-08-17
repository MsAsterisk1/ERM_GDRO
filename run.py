import torch
from loss import ERMLoss, GDROLoss, CRISLoss
import models
import utils.process_data_utils as utils
from train_eval import run_trials
import pandas as pd
import os
import argparse
from math import ceil
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='The dataset to use ex. waterbirds, civilcomments, celeba')
parser.add_argument('loss', help='The loss scheme to use ex. erm, gdro, cris')
parser.add_argument('--trials', default=5, help='The number of trials to run, default is 5')
parser.add_argument('--test_name', default='test', help='The name of the test to run, the file will be saved at ./test_results/[name].csv')
parser.add_argument('--device', default="0" if torch.cuda.is_available() else "cpu", help='The GPU to use. If left blank, will default to GPU 0 or CPU if cuda is unavailable')
parser.add_argument('--verbose', action='store_true', help='Whether to print the progress and epoch results')
parser.add_argument('--cris_prop', default=0.7, help='The proportion of data to be dedicated to training the featurizer when using CRIS (ignored if not using CRIS)')
parser.add_argument('--subclass_labels', action='store_true', help='Trains the model using the subclass labels (evalutaion is still done using the superclass labels, but is done with respect to each subclass)')
parser.add_argument('--val', default=None, help='If equal to 0, best overall validation accuracy is tracked and printed at the end of each trial. If another integer, best worst-group accuracy is used instead. If left empty, validation accuracy is not tracked')


args = parser.parse_args()

# hyperparameters

if args.device == 'cpu':
    device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch.cuda.set_device(int(args.device))
    device = 'cuda'

if args.dataset == 'waterbirds':

    train_dataset, val_dataset, test_dataset = utils.get_waterbirds_datasets(device=device, subclass_label=args.subclass_labels)

    # From Distributionally Robust Neural Networks
    batch_size = (128, 512)
    eta = 0.01
    num_subclasses = 4

    run_trials_args = {
        'model_class': models.TransferModel50,
        'model_args': {'device': device, 'freeze': False, 'num_labels': 4 if args.subclass_labels else 2},
        'epochs': 300,
        'optimizer_class': torch.optim.SGD,
        'optimizer_args': {'lr': 0.0001, 'weight_decay': 0.0001},  # , 'momentum': 0.9},
        'num_subclasses': 4,
    }
elif args.dataset == 'mnist':
    train_dataset, val_dataset, test_dataset = utils.get_MNIST_datasets(device=device, subclass_label=args.subclass_labels)
    batch_size = (1024, 1024)
    eta = 0.01
    num_subclasses = 10

    run_trials_args = {
        'model_class': models.NeuralNetwork,
        'model_args': {'layers': [28 * 28, 256, 64, 10 if args.subclass_labels else 2], 'device': device},
        'epochs': 10,
        'optimizer_class': torch.optim.Adam,
        'optimizer_args': {'lr': 0.0005, 'weight_decay': 0.005},
        'num_subclasses': 10,
    }
elif args.dataset == 'civilcomments':
    train_dataset, val_dataset, test_dataset = utils.get_CivilComments_Datasets(device=device, subclass_label=args.subclass_labels)
    batch_size = (16, 128)

    # for gdro only train on labels as classes
    num_subclasses = 4

    # From WILDS
    epochs = 5
    num_training_steps = ceil(len(train_dataset) / batch_size[0]) * epochs
    eta = 0.01

    run_trials_args = {
        'model_class': models.BertClassifier,
        'model_args': {'device': device, 'num_labels': 4 if args.subclass_labels else 2},
        'epochs': 5,
        'optimizer_class': torch.optim.AdamW,
        'optimizer_args': {'lr': 0.00001, 'weight_decay': 0.01},
        'num_subclasses': 18,
        'scheduler_class': transformers.get_linear_schedule_with_warmup,
        'scheduler_args': {'num_warmup_steps': 0, 'num_training_steps': num_training_steps},
        'gradient_clip': 1,
        'vector_subclass': True
    }
elif args.dataset == 'celeba':
    train_dataset, val_dataset, test_dataset = utils.get_celeba_datasets(device=device, subclass_label=args.subclass_labels)
    batch_size = (128, 128)

    num_subclasses = 4

    run_trials_args = {
        'model_class': models.TransferModel50,
        'model_args': {'device': device, 'freeze': False, 'num_labels': 4 if args.subclass_labels else 2},
        'epochs': 50,
        'optimizer_class': torch.optim.SGD,
        'optimizer_args': {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9},
        'num_subclasses': num_subclasses,
    }


trials = int(args.trials)
run_trials_args['num_trials'] = trials

run_trials_args['verbose'] = args.verbose
run_trials_args['record'] = True
run_trials_args['validation'] = int(args.val)

run_trials_args['subclass_labels'] = args.subclass_labels

results_dir = 'test_results/'
test_name = args.test_name
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
cris_class = CRISLoss
cris_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses}
losses = {'erm': (erm_class, erm_args), 'gdro': (gdro_class, gdro_args), 'upweight': (upweight_class, upweight_args), 'cris': (cris_class, cris_args), 'rwcris': (cris_class, cris_args)}

accuracies = {}

# Number of epochs to use in the index of the results DataFrame (not necessarily the number actually trained)
epoch_index_length = run_trials_args['epochs']
if args.loss in ['cris', 'rwcris']:
    epoch_index_length *= 2

if verbose:
    print(f"Running trials: {args.loss}")

train_dataloader, val_dataloader, test_dataloader, = utils.get_dataloaders(
    (train_dataset, val_dataset, test_dataset),
    batch_size=batch_size,
    reweight_train=args.loss not in ['erm', 'cris'],
    split=args.loss in ['cris', 'rwcris'],
    proportion=float(args.cris_prop)
)

run_trials_args['loss_class'], run_trials_args['loss_args'] = losses[args.loss]

run_trials_args['train_dataloader'] = train_dataloader
run_trials_args['val_dataloader'] = val_dataloader
run_trials_args['test_dataloader'] = test_dataloader

accuracies[args.loss] = run_trials(**run_trials_args)

accuracies_df = pd.DataFrame(
    accuracies,
    index=pd.MultiIndex.from_product(
        [range(run_trials_args['num_trials']), range(epoch_index_length + 1), subtypes],
        names=["trial", "epoch", "subtype"]
    )
)

accuracies_df.to_csv(results_dir + f'{args.test_name}.csv')
