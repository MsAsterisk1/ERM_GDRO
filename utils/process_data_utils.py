import torch
import pandas as pd
import numpy as np

from datasets import SubclassedDataset
from dataloaders import InfiniteDataLoader
from transformers import DistilBertTokenizer



url_CivilComments='https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/all_data_with_identities.csv'

CC_subgroup_cols = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religion', 'black', 'white']
split_rename = {'train':0, 'val':1, 'test':2}


def get_CivilComments_df(csv_file_path=url_CivilComments):

    CC_df = pd.read_csv(csv_file_path, index_col=0)

    CC_df.sort_values('id', inplace=True)
    CC_df.reset_index(drop=True, inplace=True)

    CC_df['split'] = tuple(map(lambda x:split_rename[x], CC_df['split'].values))

    CC_df[CC_subgroup_cols] = (CC_df[CC_subgroup_cols] >= 0.5).astype(int)
    CC_df['toxicity']=(CC_df['toxicity'] >= 0.5).astype(int)
    CC_df['others'] = (CC_df[CC_subgroup_cols] == 0).all(axis=1).astype(int)

    return CC_df


def get_CivilComments_Datasets(CC_df=None, device='cpu'):

    if CC_df is None:
        CC_df = get_CivilComments_df()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

    datasets = []
    for split in (0,1,2):

        sub_df = CC_df[CC_df['split']==split]

        # features, labels, subclasses
        tokens = tokenizer(list(sub_df['comment_text'].values),padding='max_length', max_length = 300, 
                       truncation=True, return_tensors="pt")

        labels = torch.from_numpy(sub_df['toxicity'].values)
        subclasses = torch.from_numpy(sub_df[CC_subgroup_cols + ['others']].values)

        datasets.append(SubclassedDataset(tokens['input_ids'], tokens['attention_mask'], labels, subclasses, device=device))

    return datasets


def get_CivilComments_DataLoaders(CC_df=None, datasets=None, device='cpu'):

    if datasets is None:
        datasets = get_CivilComments_Datasets(CC_df=CC_df, device=device)

    dataloaders = []

    train = InfiniteDataLoader(datasets[0], batch_size=16)
    cv = InfiniteDataLoader(datasets[1], batch_size=len(datasets[1]))
    test = InfiniteDataLoader(datasets[2], batch_size=len(datasets[2]))

    return train, cv, test


def get_subclassed_MNIST_datasets():
    root = 'data/'

    train_images = np.fromfile(root + 'mnist/train-images.idx3-ubyte', dtype='>u1')[16:]
    train_labels = np.fromfile(root + 'mnist/train-labels.idx1-ubyte', dtype='>u1')[8:]
    test_images = np.fromfile(root + 'mnist/t10k-images.idx3-ubyte', dtype='>u1')[16:]
    test_labels = np.fromfile(root + 'mnist/t10k-labels.idx1-ubyte', dtype='>u1')[8:]

    train_subclass_labels = train_labels.copy()
    train_labels = (train_labels >= 5).astype(int)
    test_subclass_labels = test_labels.copy()
    test_labels = (test_labels >= 5).astype(int)

    train_dataset = SubclassedDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels),
        torch.from_numpy(train_subclass_labels)
    )
    test_dataset = SubclassedDataset(
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels),
        torch.from_numpy(test_subclass_labels)
    )

    return train_dataset, test_dataset


def get_subclassed_MNIST_dataloaders():
    train_dataset, test_dataset = get_subclassed_MNIST_datasets()
    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=256)
    test_dataloader = InfiniteDataLoader(test_dataset, replacement=False, batch_size=len(test_dataset))

    return train_dataloader, test_dataloader
