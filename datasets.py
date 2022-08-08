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
    A dataset for loading images from the disk as they are needed rather than storing them all in GPU RAM
    This is very, very slow but if RAM is limited or the images are too large this may be necessary
    This class was originally created to load data from waterbirds, so it may be necessary to modify the code below to refer to the correct columns in the metadata dataframe
    """

    def __init__(self, metadata, root_dir, transform, device):
        """
        :param metadata: dataframe storing the image paths, labels, and subclasses
        :param root_dir: the directory where the image files are stored
        :param transform: the transform to apply to the image when it is loaded
        :param device to move tensors to as they are loaded
        """

        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # column 1: image path
        img_path = self.metadata.iloc[idx, 1]
        img_tensor = self.transform(Image.open(self.root_dir + img_path)).squeeze().to(self.device)

        # column 2: image label
        label = torch.LongTensor([self.metadata.iloc[idx, 2]]).squeeze().to(self.device)

        # column 4 contains the confounding label, which is combined with column 2 to get the subclass
        subclass = torch.LongTensor([2 * self.metadata.iloc[idx, 2] + self.metadata.iloc[idx, 4]]).squeeze().to(
            self.device)

        return img_tensor, label, subclass
