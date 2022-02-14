import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from os.path import realpath
from utils.util import test_dataset
from optimization.utility.helpers import apply_quantization, apply_pruning
from parse_config import ConfigParser

def main(config):
    # retrieve test_data_loader
    test_data_loader = getattr(module_data, 
        config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.initialize('arch', module_arch)
    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False) # load weights
    
    # optional: apply quantization solution
    if hasattr(config, 'quantization'):
        print('Apply quantization: {} ...'.format(config.quantization))
        apply_quantization(model, config.cfg_quantization)
    
    # optional: apply pruning solution
    if hasattr(config, 'pruning'):
        print('Apply pruning: {} ...'.format(config.pruning))
        apply_pruning(model, config.cfg_pruning)
    
    # get function handles of loss and metrics and test model
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    log = test_dataset(model, test_data_loader, loss_fn, metric_fns)
    print("Metrics: {}".format(log))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default=None, type=str, required=True,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-q', '--quantization', default=None, type=str,
                      help='path to quantization configuration (.yml) (default: None)')
    args.add_argument('-p', '--pruning', default=None, type=str,
                      help='path to pruning configuration (.yml) (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args, log_results=False)
    main(config)
