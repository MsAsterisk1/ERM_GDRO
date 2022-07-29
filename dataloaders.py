"""
Code directly adapted from DomainBeds paper
"""


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


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
    def __init__(self, dataset, batch_size, weights=None, num_workers=0):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(
                dataset,
                replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

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
        return len(self.dataset) // self.batch_size
