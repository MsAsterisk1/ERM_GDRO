import re
from re import sub
import torch
import pandas as pd
import numpy as np

# from wilds import get_dataset
from PIL import Image
from torchvision import transforms

from datasets import SubclassedDataset, SubDataset
from dataloaders import InfiniteDataLoader, PartitionedDataLoader
from transformers import DistilBertTokenizer

url_CivilComments = 'https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/all_data_with_identities.csv'

CC_subgroup_cols = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']
split_rename = {'train': 0, 'val': 1, 'test': 2}


def get_sampler_weights(subclass_labels):
    '''
    Returns a list of weights that allows uniform sampling of dataset
    by subclasses
    '''

    subclasses = torch.unique(subclass_labels)
    subclass_freqs = []

    for subclass in subclasses:
        subclass_counts = sum(subclass_labels == subclass)
        subclass_freqs.append(1 / subclass_counts)

    subclass_weights = torch.zeros_like(subclass_labels).float()

    for idx, label in enumerate(subclass_labels):
        subclass_weights[idx] = subclass_freqs[int(label)]

    return subclass_weights


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
        # CC_df = CC_df[:100]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

    datasets = []
    for split in (0, 1, 2):
        sub_df = CC_df[CC_df['split'] == split]

        # features, labels, subclasses
        tokens = tokenizer(list(sub_df['comment_text'].values), padding='max_length', max_length=300,
                           truncation=True, return_tensors="pt")

        labels = torch.from_numpy(sub_df['toxicity'].values)
        features = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=1)

        # for train, we only group by labels
        if split == 0:
            others = torch.from_numpy(sub_df['others'].values)
            subclasses = labels * 2 + others

        else:
            num_groups = len(CC_subgroup_cols) + 1  # also need the others 'subgroup'
            super_subclasses = torch.from_numpy(sub_df[CC_subgroup_cols + ['others']].values)
            repeat_labels = labels.unsqueeze(1).repeat(1, num_groups)

            toxic_subclasses = torch.logical_and(super_subclasses, repeat_labels)
            nontoxic_subclasses = torch.logical_and(super_subclasses, torch.logical_not(repeat_labels))
            subclasses = torch.cat((toxic_subclasses, nontoxic_subclasses), dim=1)

        labels = labels.to(device).long()
        subclasses = subclasses.to(device).long()
        datasets.append(SubclassedDataset(features, labels, subclasses))

    return datasets


def get_MNIST_datasets(device='cpu', path='data/mnist/', rng=np.random.default_rng()):
    train_images = np.fromfile(path + 'train-images.idx3-ubyte', dtype='>u1')[16:]
    train_labels = np.fromfile(path + 'train-labels.idx1-ubyte', dtype='>u1')[8:]
    test_images = np.fromfile(path + 't10k-images.idx3-ubyte', dtype='>u1')[16:]
    test_labels = np.fromfile(path + 't10k-labels.idx1-ubyte', dtype='>u1')[8:]

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


def get_images(root, paths, transform=transforms.ToTensor()):
    img_tensors = []
    for img_path in paths:
        # image tensors go on CPU RAM until the model needs to move them to GPU
        img = Image.open(root + img_path)
        img_tensors.append(transform(img).to('cpu'))
        img.close()
    return torch.stack(img_tensors)


def get_waterbirds_datasets(device='cpu', path='data/waterbirds_v1.0/'):
    metadata_df = pd.read_csv(path + 'metadata.csv')

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    features = get_images(root=path, paths=metadata_df['img_filename'].values, transform=transform)

    # column 2: image label (bird type)
    labels = torch.LongTensor(metadata_df.iloc[:, 2].values).squeeze().to(device)
    # column 4 contains the confounding label (place), which is combined with column 2 to get the subclass
    subclasses = torch.LongTensor(2 * metadata_df.iloc[:, 2].values + metadata_df.iloc[:, 4].values).squeeze().to(
        device)

    train_idx = metadata_df['split'] == 0
    val_idx = metadata_df['split'] == 1
    test_idx = metadata_df['split'] == 2

    train_dataset = SubclassedDataset(features[train_idx], labels[train_idx], subclasses[train_idx])
    val_dataset = SubclassedDataset(features[val_idx], labels[val_idx], subclasses[val_idx])
    test_dataset = SubclassedDataset(features[test_idx], labels[test_idx], subclasses[test_idx])

    return train_dataset, val_dataset, test_dataset


def get_celeba_datasets(device='cpu', path='data/celeba/'):

    with open(path + 'list_attr_celeba.txt') as f:
        lines = f.readlines()
        n = int(lines[0])
        attr_names = re.split(" +", lines[1])[:-1]
        anno_df = pd.DataFrame(columns=attr_names, index=range(n))
        filenames = []
        for i in range(n):
            line_data = re.split(" +", lines[i + 2])
            anno_df.loc[i] = line_data[1:]
            filenames.append(line_data[0])

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    features = get_images(root=path, paths=filenames, transform=transform)

    # Using Blond_Hair as the label and Male as the confounding attribute
    labels = torch.LongTensor(anno_df['Blond_Hair'].values.astype(int) > 0, device=device)
    subclasses = torch.LongTensor(
        2 * (anno_df['Blond_Hair'].values.astype(int) > 0) +
        (anno_df['Male'].values.astype(int) > 0),
        device=device
    )

    train_idx = []
    val_idx = []
    test_idx = []
    with open(path + 'list_eval_partition.txt') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            # No header line(s)
            # Evaluation status is at index 1 of each line
            split = int(re.split(" +", lines[i])[1])
            # Store status as binary values in three lists for easy indexing later
            train_idx.append(split == 0)
            val_idx.append(split == 1)
            test_idx.append(split == 2)

    train_dataset = SubclassedDataset(features[train_idx], labels[train_idx], subclasses[train_idx])
    val_dataset = SubclassedDataset(features[val_idx], labels[val_idx], subclasses[val_idx])
    test_dataset = SubclassedDataset(features[test_idx], labels[test_idx], subclasses[test_idx])

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(datasets, batch_size, reweight_train=False, split=False, proportion=0.5, seed=None):
    train_dataset, val_dataset, test_dataset = datasets

    if split:
        train_dataloader = get_partitioned_dataloader(train_dataset, batch_size[0], proportion=proportion, seed=seed)
    else:
        train_dataloader = InfiniteDataLoader(
            train_dataset,
            batch_size=batch_size[0],
            weights=get_sampler_weights(train_dataset.subclasses) if reweight_train else None,
            replacement=reweight_train,
            drop_last=reweight_train
        )

    val_dataloader = InfiniteDataLoader(
        val_dataset,
        batch_size=batch_size[1],
        replacement=False,
        drop_last=False
    )
    test_dataloader = InfiniteDataLoader(
        test_dataset,
        batch_size=batch_size[1],
        replacement=False,
        drop_last=False
    )

    return train_dataloader, val_dataloader, test_dataloader


def split_dataset(dataset, proportion=0.5, seed=None):
    indices = np.arange(len(dataset))

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(indices)

    split_point = int(proportion * len(dataset))

    indices_1 = indices[:split_point]
    indices_2 = indices[split_point:]

    return SubDataset(indices_1, dataset), SubDataset(indices_2, dataset)


def get_partitioned_dataloader(dataset, batch_size, proportion=0.5, seed=None):
    dataset0, dataset1 = split_dataset(dataset, proportion=proportion, seed=seed)

    dataloader = PartitionedDataLoader(dataset0, int(batch_size * proportion),
                                       dataset1, int(batch_size * (1 - proportion)),
                                       replacement0=False, drop_last0=False,
                                       replacement1=True, drop_last1=True,
                                       weights1=get_sampler_weights(dataset1.subclasses))

    return dataloader
