import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader, SubsetSeedSampler
from data.KeywordSpotting import KeywordsDataset
from torch.utils.data.sampler import SequentialSampler

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, seed=None, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(), # normalize from 0..255 to 0..1
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, seed, validation_split, num_workers)

class KeywordDataLoader(BaseDataLoader):
    """
    Keyword Spotting dataset loader (preprocessed MFCC feature vectors)
    
    Note: validation_split is ignored for this dataset; train/val/test splits are fixed to 80/10/10.
    """
    def __init__(self, data_dir, batch_size, shuffle=True, seed=None, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)) # convert to FloatTensor
        ])
        self.data_dir = data_dir
        self.dataset = KeywordsDataset(self.data_dir, train=training, transform=trsfm)
        if training:
            validation_split = len(self.dataset.data_val) / len(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, seed, validation_split, num_workers)
    
