import torch
import pandas as pd
import numpy as np

import wilds

from datasets import SubclassedDataset
from dataloaders import InfiniteDataLoader
from transformers import DistilBertTokenizer

url_CivilComments = 'https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/all_data_with_identities.csv'

CC_subgroup_cols = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religion', 'black', 'white']
split_rename = {'train': 0, 'val': 1, 'test': 2}


def split_stratified(dataset, sizes, rng):
    """
    Shuffle and split a SubclassedDataset into two, stratified so the subclass proportions of each new dataset are as close as
    possible to the original
    """
    assert len(sizes) == 2

    # Shuffle dataset
    shuffle_idx = np.arange(len(dataset))
    rng.shuffle(shuffle_idx)
    dataset = SubclassedDataset(*dataset[shuffle_idx])

    subclasses = torch.unique(dataset.subclasses)

    total_subclass_sizes = np.array([sum(dataset.subclasses == c).item() for c in subclasses])
    subset_subclass_sizes = np.array(
        [np.round(total_subclass_sizes * sizes[d] / len(dataset)).astype(int) for d in range(len(sizes))])

    idxs = [[], []]
    for c in subclasses:
        subclass_idx = np.where((dataset.subclasses == c).tolist())[0]

        idxs[0].extend((subclass_idx[:subset_subclass_sizes[0][c]]).tolist())
        idxs[1].extend((subclass_idx[subset_subclass_sizes[0][c]:]).tolist())

    return SubclassedDataset(*(dataset[idxs[0]])), SubclassedDataset(*(dataset[idxs[1]]))


def get_CivilComments_df(csv_file_path=url_CivilComments):
    CC_df = pd.read_csv(csv_file_path, index_col=0)

    CC_df.sort_values('id', inplace=True)
    CC_df.reset_index(drop=True, inplace=True)

    CC_df['split'] = tuple(map(lambda x: split_rename[x], CC_df['split'].values))

    CC_df[CC_subgroup_cols] = (CC_df[CC_subgroup_cols] >= 0.5).astype(int)
    CC_df['toxicity'] = (CC_df['toxicity'] >= 0.5).astype(int)
    CC_df['others'] = (CC_df[CC_subgroup_cols] == 0).all(axis=1).astype(int)

    return CC_df


def get_CivilComments_Datasets(CC_df=None, device='cpu'):
    if CC_df is None:
        CC_df = get_CivilComments_df()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

    datasets = []
    for split in (0, 1, 2):
        sub_df = CC_df[CC_df['split'] == split]

        # features, labels, subclasses
        tokens = tokenizer(list(sub_df['comment_text'].values), padding='max_length', max_length=300,
                           truncation=True, return_tensors="pt")

        labels = torch.from_numpy(sub_df['toxicity'].values)
        features = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=1)

        num_groups = len(CC_subgroup_cols) + 1  # also need the others 'subgroup'

        super_subclasses = torch.from_numpy(sub_df[CC_subgroup_cols + ['others']].values)
        repeat_labels = labels.unsqueeze(1).repeat(1, num_groups)

        toxic_subclasses = torch.logical_and(super_subclasses, repeat_labels)
        nontoxic_subclasses = torch.logical_and(super_subclasses, torch.logical_not(repeat_labels))
        subclasses = torch.cat((toxic_subclasses, nontoxic_subclasses), dim=1).long()

        datasets.append(SubclassedDataset(features, labels, subclasses, device=device))

    return datasets


def get_CivilComments_DataLoaders(CC_df=None, datasets=None, device='cpu'):
    if datasets is None:
        datasets = get_CivilComments_Datasets(CC_df=CC_df, device=device)

    dataloaders = []

    train = InfiniteDataLoader(datasets[0], batch_size=16, replacement=False, drop_last=False)
    cv = InfiniteDataLoader(datasets[1], batch_size=32, replacement=False, drop_last=False)
    test = InfiniteDataLoader(datasets[2], batch_size=32, replacement=False, drop_last=False)

    return train, cv, test


def get_MNIST_datasets(device='cpu', rng=np.random.default_rng()):
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

    train_dataset, val_dataset = split_stratified(dataset=train_dataset, sizes=[50000, 10000], rng=rng)
    # randomly remove 95% of 8s from the training set
    # this happens after the train/val split so the split can be done with nice round numbers 50k and 10k
    remove_digit = 8
    remove_frac = 0.95

    train_dataset = SubclassedDataset(*train_dataset[
        (train_dataset.subclasses != remove_digit) |
        torch.tensor(rng.random(len(train_dataset)) > remove_frac, device=device)
        ])

    return train_dataset, val_dataset, test_dataset


def get_MNIST_dataloaders(batch_size, device='cpu', seed=None):
    train_dataset, val_dataset, test_dataset = get_MNIST_datasets(device=device,
                                                                  rng=np.random.default_rng(seed))

    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = InfiniteDataLoader(test_dataset, replacement=False, batch_size=len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader


def get_waterbirds_datasets():
    return None
