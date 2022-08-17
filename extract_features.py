import pandas as pd
import numpy as np

import argparse
from tqdm import tqdm

import torch
from torch.optim import SGD, AdamW
import torch.nn as nn

from utils.process_data_utils import  get_dataloaders, get_waterbirds_datasets, get_CivilComments_Datasets
from loss import ERMLoss, GDROLoss
from train_eval import  train_epochs
from models import TransferModel50, BertClassifier


from umap import UMAP
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('-f', '--file_name', default='features')
parser.add_argument('-d', '--device', default='0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-r', '--reweight_train', action='store_true')
parser.add_argument('-s', '--subclass_label', action='store_true')
parser.add_argument('-l', '--loss_function', default='ERM')
parser.add_argument('-sc', '--silhouette_score', action='store_true')


args = parser.parse_args()

if args.device == 'cpu':
    device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch.cuda.set_device(int(args.device))
    device = 'cuda'

#For GDRO, we must reweight
if args.loss_function == 'GDRO':
    args.reweight_train =True

if args.dataset == 'waterbirds':

    batch_size = (128, 256)
    eta = 0.01
    num_subclasses = 4
    num_labels = 4 if args.subclass_label else 2
    epochs = 131


    model_class = TransferModel50

    opt_class = SGD
    opt_args = {'lr':0.0001, 'weight_decay':0.0001, 'momentum':0.9}

    train_dataset, val_dataset, test_dataset = get_waterbirds_datasets(device=device, subclass_label=args.subclass_label)
elif args.dataset == 'civilcomments':
     batch_size = (16, 128)

    # for gdro traon on labels x identity (identity or others)
     num_subclasses = 4
     num_labels = 4 if args.subclass_label else 2


     # From WILDS
     epochs = 2
     eta = 0.01


     model_class = BertClassifier
     opt_class = AdamW
     opt_args = {'lr': 0.00001, 'weight_decay': 0.01}

     train_dataset, val_dataset, test_dataset = get_CivilComments_Datasets(device=device, subclass_label=args.subclass_label)




train_dataloader, val_dataloader, test_dataloader = get_dataloaders((train_dataset, val_dataset, test_dataset), batch_size=batch_size, reweight_train=args.reweight_train)
trials = 1

if args.silhouette_score:
    trials = 5
    sc = []
    sc_b = []

for trial in tqdm(range(trials)):

    model = model_class(device=device, num_labels=num_labels)


    if args.loss_function == 'ERM':
        loss_fn = ERMLoss(model, nn.CrossEntropyLoss())
    else:
        loss_fn = GDROLoss(model, nn.CrossEntropyLoss(), eta=eta, num_subclasses=num_subclasses)

    optimizer = opt_class(model.parameters(), **opt_args)

    train_epochs(epochs, train_dataloader, val_dataloader, test_dataloader, model, loss_fn, optimizer, verbose=args.verbose, num_subclasses=num_subclasses)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if args.dataset == 'waterbirds':
        model.model.avgpool.register_forward_hook(get_activation('featurizer'))
    else:
        model.bert.distilbert.register_forward_hook(get_activation('featurizer'))

    model.eval()

    features = []
    labels = []
    pred_labels = []

    for _ in range(test_dataloader.batches_per_epoch()):
        X,y,c = next(test_dataloader)

        preds = torch.argmax(model(X), 1)
        img_features = activation['featurizer'].squeeze()
        features.extend(torch.unbind(img_features))
        pred_labels.extend(preds.int().tolist())
        labels.extend(c.int().tolist())


    for _ in range(val_dataloader.batches_per_epoch()):
        X,y,c = next(val_dataloader)

        preds = torch.argmax(model(X), 1)
        img_features = activation['avgpool'].squeeze()
        features.extend(torch.unbind(img_features))
        pred_labels.extend(preds.int().tolist())
        labels.extend(c.int().tolist())

    cols = []
    for idx,label in enumerate(labels):
        cols.append([label] +  [pred_labels[idx]] + features[idx].cpu().numpy().tolist())

    df_feats = pd.DataFrame(cols).rename({0:'label', 1:'pred_labels'}, axis=1)

    if args.silhouette_score:
        feats = df_feats.drop(['label', 'pred_labels'], axis=1).values
        reducer = UMAP(random_state=8, n_components=2)
        embeds = reducer.fit_transform(feats)

        groups =  df_feats['label'].values
        groups_b = np.where(groups>1,1,0)

        silhouette_avg = silhouette_score(embeds, groups)
        silhouette_avg_b = silhouette_score(embeds, groups_b)

        
        print(f'Trial {trial+1}/{trials} sc average: {silhouette_avg} sc binary average: {silhouette_avg_b}')
        sc.append(silhouette_avg)
        sc_b.append(silhouette_avg_b)



if args.silhouette_score:
    print(f'Silhouette Scores across trials {sc}')
    print(f'Average SC acoss trials {np.mean(sc)}')

    print(f'Binary Silhouette Scores across trials {sc_b}')
    print(f'Average Binary SC acoss trials {np.mean(sc_b)}')

else:
    df_feats.to_csv(f'{args.file_name}.csv')


