import os
import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from pathlib import Path
from parse_config import ConfigParser
from utils import write_yaml
from optimization.pruning import KernelPruner

def main(config):
    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    val_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    print(model)
    
    # get function handles of loss and metrics
    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # setup pruner
    error_margin = config['optimization']['pruner']['error_margin']
    pruner = KernelPruner(model, config=config, 
                    val_data_loader=val_data_loader,
                    error_margin=error_margin)
    
    # prune model
    ddict = pruner.PruneNet()

    # write results to file
    fpath = os.path.join(str(config.cfg_fname.parent), 'pruning.yml')
    index, i = '', 0
    while os.path.isfile(fpath+index):
        pad = int(np.floor(int(np.log10(i)))) if i > 0 else 0
        index = '.{}'.format(int(index[-1-pad:])+1) if index else '.1'+'0'*pad  # increment count
        i += 1
    fpath += index # append index if file already exists...
    write_yaml(ddict, Path(fpath))
    print("Saving config: {} ...".format(fpath))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default=None, type=str, required=True,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args, log_results=False)
    ret = main(config)
