import collections
import itertools
from typing import Iterable

import torch
from torch.utils.data import DataLoader


class CombinedDataLoaders:
    def __init__(self, loaders:  Iterable[DataLoader]) -> None:
        self.iterators = [iter(loader) for loader in loaders]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.iterators) == 1:
            return self.iterators[0].next()
        next_data = [iterator.next() for iterator in self.iterators]
        return concat_collate(next_data)


def concat_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, dim=0, out=out)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: concat_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        return list(itertools.chain.from_iterable(batch))

    raise TypeError(f"concat_collate: batch must contain tensors, dicts or lists; found {type(elem)}")
