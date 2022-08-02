import random

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
        num_groups = len(CC_subgroup_cols) + 1 #also need the others 'subgroup'

        super_subclasses = torch.from_numpy(sub_df[CC_subgroup_cols + ['others']].values)
        repreat_labels = labels.unsqueeze(1).repeat(1,num_groups)
        
        toxic_subclasses = torch.logical_and(super_subclasses, repreat_labels)
        nontoxic_subclasses = torch.logical_and(super_subclasses, torch.logical_not(repreat_labels))
        subclasses = torch.cat((toxic_subclasses, nontoxic_subclasses),dim=1).long()


        datasets.append(SubclassedDataset(tokens['input_ids'], tokens['attention_mask'], labels, subclasses, device=device))

    return datasets


def get_CivilComments_DataLoaders(CC_df=None, datasets=None, device='cpu'):

    if datasets is None:
        datasets = get_CivilComments_Datasets(CC_df=CC_df, device=device)

    dataloaders = []

    train = InfiniteDataLoader(datasets[0], batch_size=16, replacement=False, drop_last=False)
    cv = InfiniteDataLoader(datasets[1], batch_size=32, replacement=False, drop_last=False)
    test = InfiniteDataLoader(datasets[2], batch_size=32, replacement=False, drop_last=False)

    return train, cv, test


def color_MNIST(images, labels, flip_prob=0.25, seed=None):
    dtype = images.dtype
    # number of images, number of pixels in image
    n, size = images.shape
    images = np.reshape(images, (n, size, 1))

    rng = np.random.default_rng(seed)
    flip_array = rng.random(n) < flip_prob

    labels = np.logical_xor(labels, flip_array)

    images = np.concatenate(
        [
            (labels * images.T).T,
            ((1-labels) * images.T).T,
            np.zeros((n, size, 1), dtype=dtype)
        ],
        axis=2
    )

    return images


def get_colored_MNIST_datasets(device='cpu', seed=None):
    train_images = np.fromfile('data/mnist/train-images.idx3-ubyte', dtype='>u1')[16:]
    train_labels = np.fromfile('data/mnist/train-labels.idx1-ubyte', dtype='>u1')[8:]
    test_images = np.fromfile('data/mnist/t10k-images.idx3-ubyte', dtype='>u1')[16:]
    test_labels = np.fromfile('data/mnist/t10k-labels.idx1-ubyte', dtype='>u1')[8:]

    train_images = np.reshape(train_images, (-1, 28 * 28))
    test_images = np.reshape(test_images, (-1, 28 * 28))

    train_subclass_labels = train_labels.copy()
    train_labels = (train_labels >= 5)
    test_subclass_labels = test_labels.copy()
    test_labels = (test_labels >= 5)

    train_images = color_MNIST(train_images, train_labels, flip_prob=0.1, seed=seed)
    test_images = color_MNIST(test_images, test_labels, flip_prob=0.9, seed=seed)

    train_dataset = SubclassedDataset(
        torch.from_numpy(train_images).to(device=device, dtype=torch.float),
        torch.from_numpy(train_labels).to(device=device, dtype=torch.long),
        torch.from_numpy(train_subclass_labels).to(device=device, dtype=torch.long)
    )
    test_dataset = SubclassedDataset(
        torch.from_numpy(test_images).to(device=device, dtype=torch.float),
        torch.from_numpy(test_labels).to(device=device, dtype=torch.long),
        torch.from_numpy(test_subclass_labels).to(device=device, dtype=torch.long)
    )

    return train_dataset, test_dataset


def get_colored_MNIST_dataloaders(batch_size, device='cpu', seed=None):
    train_dataset, test_dataset = get_colored_MNIST_datasets(device=device)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000],
        generator=torch.Generator().manual_seed(seed)
    )

    # randomly remove 95% of 8s

    #TODO use Subset instead of sample weights because I think this may not work for the test dataloader

    rng = np.random.default_rng(42)
    train_weights = (train_dataset.subclasses != 8) & (rng.random(len(train_dataset)) > 0.95)
    val_weights = (val_dataset.subclasses != 8) & (rng.random(len(val_dataset)) > 0.95)
    test_weights = (test_dataset.subclasses != 8) & (rng.random(len(test_dataset)) > 0.95)

    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, weights=train_weights)
    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, weights=val_weights)
    test_dataloader = InfiniteDataLoader(test_dataset, replacement=False, batch_size=len(test_dataset), weights=test_weights)

    return train_dataloader, val_dataloader, test_dataloader
