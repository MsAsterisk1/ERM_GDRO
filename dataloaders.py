"""
Code directly adapted from DomainBeds paper
"""


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import math


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    """
    Uses an infinite sampler to create a dataloader that never becomes empty
    """
    def __init__(self, dataset, batch_size, replacement=True, drop_last=True, weights=None, num_workers=0):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(
                dataset,
                replacement=replacement)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __next__(self):
        return next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

    def batches_per_epoch(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class PartitionedDataLoader:
    """
    Wraps two dataloaders for training on independent splits
    """
    def __init__(self, dataset0, batch_size0,
                       dataset1, batch_size1,
                       replacement0=True, drop_last0=True, weights0=None, num_workers0=0,
                       replacement1=True, drop_last1=True, weights1=None, num_workers1=0):
        self.dataloader0 = InfiniteDataLoader(
            dataset0,
            batch_size0,
            replacement=replacement0,
            drop_last=drop_last0,
            weights=weights0,
            num_workers=num_workers0
        )
        self.dataloader1 = InfiniteDataLoader(
            dataset1,
            batch_size1,
            replacement=replacement1,
            drop_last=drop_last1,
            weights=weights1,
            num_workers=num_workers1
        )

    def __next__(self):
        return tuple(zip(next(self.dataloader0), next(self.dataloader1)))

    def __len__(self):
        raise ValueError

    def batches_per_epoch(self):
        return self.dataloader0.batches_per_epoch()
