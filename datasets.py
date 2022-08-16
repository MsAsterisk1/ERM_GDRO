import torch
from torch.utils.data import Dataset

root = './data/'


class SubclassedDataset(Dataset):
    """
    Dataset for subclassed data
    Similar to a normal dataset, but getitem returns the subclass label as well as the feature tensor and superclass label
    """

    def __init__(self, features, labels, subclasses, subclass_label=False):
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
        self.subclass_label = subclass_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.features[idx], self.subclasses[idx] if self.subclass_label else self.labels[idx], self.subclasses[idx]


class SubDataset(Dataset):
    """
    Used to get a consistent subset of a dataset
    Torch's built-in Subset is unsuitable because it does not allow for subclass data
    """

    def __init__(self, indices, dataset):
        self.indices = indices
        self.subclasses = torch.index_select(dataset.subclasses, 0, torch.tensor(indices, device=dataset.subclasses.device))
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])

