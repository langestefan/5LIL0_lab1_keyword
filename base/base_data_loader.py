import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler, SequentialSampler


class SubsetSeedSampler(Sampler):
    """Samples elements based on seed value from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        seed (integer): seed for deterministic sequence
    """

    def __init__(self, indices, seed):
        self.seed = seed % 4294967295 # otherwise seed gets too large
        self.indices = indices

    def __iter__(self):
        # every iterator call the seed gets incremented (once every epoch)
        self.seed = (self.seed + 1) % 4294967295
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, seed, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        
        self.batch_idx = 0
        self.n_samples = len(dataset)
        
        if not (hasattr(self, 'sampler') and hasattr(self, 'valid_sampler')): # overwrite if not exist yet
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        idx_full = np.arange(self.n_samples)
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        valid_sampler = SequentialSampler(valid_idx)
        
        if self.shuffle:
            train_sampler = SubsetSeedSampler(train_idx, self.seed) # shuffle every epoch
        else:
            train_sampler = SequentialSampler(train_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
