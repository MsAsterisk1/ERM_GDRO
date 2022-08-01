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

    def __init__(self, data):
        '''
        INPUTS:
        features: list of features (as Pytorch tensors)
        labels:   list of corresponding lables
        subclasses: list of corresponding subclasses

        '''

        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vals = [d[idx] for d in self.data]
        return vals


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


class SubclassedMNISTDataset(Dataset):
    """
    MNIST handwritten digits dataset, where the classification task is to classify digits as lt or geq 5, with each digit as a subclass
    Similar to http://stanford.edu/~nims/no_subclass_left_behind.pdf
    """

    def __init__(self, test=False):
        if test:
            self.images = np.fromfile(root + 'mnist/t10k-images.idx3-ubyte', dtype='>u1')[16:]
            self.labels = np.fromfile(root + 'mnist/t10k-labels.idx1-ubyte', dtype='>u1')[8:]
        else:
            self.images = np.fromfile(root + 'mnist/train-images.idx3-ubyte', dtype='>u1')[16:]
            self.labels = np.fromfile(root + 'mnist/train-labels.idx1-ubyte', dtype='>u1')[8:]

        self.subclass_labels = self.labels.copy()
        self.labels = (self.labels >= 5).astype(int)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = torch.tensor(self.images[28 * 28 * idx: 28 * 28 * idx + 28 * 28])
        label_tensor = torch.tensor(self.labels[idx])
        subclass_tensor = torch.tensor(self.subclass_labels[idx])

        return img_tensor, label_tensor, subclass_tensor

