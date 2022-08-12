import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

root = './data/'


class SubclassedDataset(Dataset):
    """
    Dataset for subclassed data
    Similar to a normal dataset, but getitem returns the subclass label as well as the feature tensor and superclass label
    """

    def __init__(self, features, labels, subclasses):
        '''
        INPUTS:
        args: of the form X, y, c
              where X is the model inputs
              y is the labels
              and c is the subclass labels
        '''
        self.features = features
        self.labels = labels
        self.subclasses = subclasses

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.features[idx], self.labels[idx], self.subclasses[idx]


class OnDemandImageDataset(Dataset):
    """
    This class was originally created to load data from waterbirds, so it may be necessary to modify the code below to refer to the correct columns in the metadata dataframe
    """

    def __init__(self, metadata, root_dir, transform, device):
        """
        :param metadata: dataframe storing the image paths, labels, and subclasses
        :param root_dir: the directory where the image files are stored
        :param transform: the transform to apply to the image when it is loaded
        :param device to move tensors to as they are loaded
        """

        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        img_tensors = []
        for i in range(len(metadata)):
            img_path = metadata.iloc[i, 1]
            # image tensors go on CPU RAM until the model needs to move them to GPU
            img = Image.open(self.root_dir + img_path)
            img_tensors.append(transform(img).to('cpu'))
            img.close()
        self.features = torch.stack(img_tensors)

        # column 2: image label
        self.labels = torch.LongTensor(self.metadata.iloc[:, 2].values).squeeze().to(self.device)
        # column 4 contains the confounding label, which is combined with column 2 to get the subclass
        self.subclasses = torch.LongTensor(2 * self.metadata.iloc[:, 2].values + self.metadata.iloc[:, 4].values).squeeze().to(
            self.device)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        subclass = self.subclasses[idx]

        return self.features[idx], label, subclass


class SubDataset(Dataset):

    def __init__(self, indices, dataset):
        self.indices = indices
        self.subclasses = torch.index_select(dataset.subclasses, 0, torch.tensor(indices, device=dataset.subclasses.device))
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])

