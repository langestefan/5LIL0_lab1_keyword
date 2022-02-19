import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from os.path import realpath
from optimization.utility.helpers import apply_quantization, apply_pruning

def main(config):
    logger = config.get_logger('train')

    # make training procedure reproducible
    if config['deterministic']:
        torch.backends.cudnn.deterministic = True # slightly reduces throughput
        seed = 101 # for model weights initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        logger.warning("Warning: Deterministic flag is disabled. Training procedures might not be reproduceable!")

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # optional: apply quantization solution
    if config.resume and hasattr(config, 'quantization'):
        logger.info('Apply quantization: {} ...'.format(config.quantization))
        apply_quantization(model, config.cfg_quantization)
    
    # optional: apply pruning solution
    if config.resume and hasattr(config, 'pruning'):
        logger.info('Apply pruning: {} ...'.format(config.pruning))
        apply_pruning(model, config.cfg_pruning)
    
    # modify training parameters a bit for finetuning
    if config.resume and (hasattr(config, 'quantization') or hasattr(config, 'pruning')):
        logger.info('Perform fixed-point finetuning for {} additional epochs!'\
            .format(config['optimization']['finetuning']['epochs']))
        config['trainer']['epochs'] += config['optimization']['finetuning']['epochs'] # increment max_epochs
        config['optimizer']['args']['lr'] /= 10 # reduce learning rate
        logger.info(model)
    
    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    group = args.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    group.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-q', '--quantization', default=None, type=str,
                      help='path to quantization configuration (.yml) (default: None)')
    args.add_argument('-p', '--pruning', default=None, type=str,
                      help='path to pruning configuration (.yml) (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
