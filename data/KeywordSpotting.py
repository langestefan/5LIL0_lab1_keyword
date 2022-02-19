import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

file_dir = os.path.dirname(os.path.abspath(__file__)) + '/KeywordsDataset'

class KeywordsDataset(Dataset):
    """
    Keyword Spotting dataset (preprocessed MFCC feature vectors)
    """
    classes = ['0 - unknown', '1 - one', '2 - two', '3 - three']
    onehot_to_int = lambda _,x: np.argmax(x, axis=1)
    
    def __init__(self, root, train=True, transform=None, target_transform=None):
        """
        Args:
            dataset_pkl (string): Path to the pkl file with data and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_file = os.path.join(root, 'IA_DATASET')     
        self.train = train # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        with open(self.dataset_file, 'rb') as f:
            raw_dict = pickle.load(f)
        
        if self.train:
            self.data_train = raw_dict['x_train']
            self.targets_train = self.onehot_to_int(raw_dict['y_train'])
            self.data_val = raw_dict['x_val']
            self.targets_val = self.onehot_to_int(raw_dict['y_val'])
            self.data = np.concatenate([self.data_val, self.data_train], axis=0)
            self.targets = np.concatenate([self.targets_val, self.targets_train], axis=0)
        else:
            self.data = raw_dict['x_test']
            self.targets = self.onehot_to_int(raw_dict['y_test'])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def _check_exists(self):
        return os.path.exists(self.dataset_file) 

if __name__ == "__main__":
    # some small sanity checks
    kw_dataset = KeywordsDataset(file_dir, train=True)
    assert len(kw_dataset) == 33046
    kw_dataset = KeywordsDataset(file_dir, train=False)
    assert len(kw_dataset) == 3672
    try: # insert wrong path
        kw_dataset = KeywordsDataset(os.path.dirname(file_dir), train=False)
    except RuntimeError:
        pass # Dataset not found.
    
