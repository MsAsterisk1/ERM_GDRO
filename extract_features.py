import pandas as pd
from utils.process_data_utils import  get_dataloaders, get_waterbirds_datasets, split_dataset, get_partitioned_dataloader
from loss import ERMLoss, GDROLoss, ERMGDROLoss, UpweightLoss
from train_eval import train, evaluate, train_epochs
import torch
from models import BertClassifier
from math import ceil
import transformers


device = 'cuda' if torch.cuda.is_available() else 'cpu'




train_dataset, val_dataset, test_dataset = get_waterbirds_datasets(device=device, subclass_label=True)

# From Distributionally Robust Neural Networks
batch_size = (128, 128)
eta = 0.01
num_subclasses = 4

run_trials_args = {
    'model_class': models.TransferModel50,
    'model_args': {'device': device, 'freeze': False, 'num_labels': 4 if args.multiclass else 2},
    'epochs': 300,
    'optimizer_class': torch.optim.SGD,
    'optimizer_args': {'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.9},
    'num_subclasses': 4,
}


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.model.avgpool.register_forward_hook(get_activation('avgpool'))