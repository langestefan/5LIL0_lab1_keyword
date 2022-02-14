import copy
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
from torch.nn import Conv2d, Linear
from optimization.utility.layers import QLinear, QConv2d

def count_conv2d(m, x, y):
    x = x[0]

    # Layer dimensions
    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # MACs per output element
    kernel_macs = kh * kw * cin

    # total ops
    num_out_elements = y.numel()
    if type(m) == QConv2d: 
        batch, cout, hin, win = y.shape
        num_out_elements -= batch * torch.sum(m.mask != 1) * hin * win # channel pruning
    total_macs = kernel_macs * num_out_elements

    # incase same conv is used multiple times
    m.total_macs += torch.Tensor([int(total_macs)])
    m.total_activations = torch.Tensor([int(num_out_elements)])

def count_linear(m, x, y):
    # MACs per output element
    kernel_macs = m.in_features
    
    # total ops
    num_out_elements = y.numel()
    if type(m) == QLinear: 
        batch, cin = y.shape
        num_out_elements -= torch.sum(m.mask != 1) * cin # channel pruning
    total_macs = kernel_macs * num_out_elements

    # incase same linear is used multiple times
    m.total_macs += torch.Tensor([int(total_macs)])
    m.total_activations = torch.Tensor([int(num_out_elements)])

def profile(model, input_size, custom_ops = {}):
    model = copy.deepcopy(model) # attach hooks to local copy
    model.eval()
    
    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_macs', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))
        m.register_buffer('total_activations', torch.zeros(1))
        
        # determine model size
        for p in m.parameters():
            total_params = torch.sum(p != 0) # to account for pruning
            m.total_params += torch.Tensor([total_params])
        
        if isinstance(m, Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, Linear):
            m.register_forward_hook(count_linear)

    # attach hooks to model
    model.apply(add_hooks)

    # run dummy input through model (hooks will be executed for every layer as well)
    x = torch.zeros(input_size)
    model(x)
    
    # retrieve values that were collected by the hook of every layer
    total_macs = 0
    total_params = 0
    total_param_bits = 0
    total_activation_bits = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_macs += m.total_macs
        total_params += m.total_params
        bits = m.bits_weights if type(m) in (QConv2d,QLinear) and m.bits_weights > 0 else 32
        total_param_bits += m.total_params * bits
        bits = m.bits_outputs if type(m) in (QConv2d,QLinear) and m.bits_outputs > 0 else 32
        total_activation_bits += m.total_activations * bits
    return total_macs, total_params, total_param_bits, total_activation_bits

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
    
    # profile model
    input_size = next(iter(test_data_loader))[0].shape
    batch_size = int(test_data_loader.batch_size)
    total_macs, total_params, total_param_bits, total_activation_bits = profile(model, input_size)
    print("MACs/classification: %i"%(total_macs//batch_size))
    print("Number of Parameters: %i"%total_params)
    print("Model Size:")
    print("  - Weights: %ikB"%(total_param_bits//1024))
    print("  - Activations: %ikB"%(total_activation_bits//(1024*batch_size)))
    
    # get function handles of loss and metrics and test model
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    log = test_dataset(model, test_data_loader, loss_fn, metric_fns, print_cm=False)
    print("Accuracy: {:.2f}%".format(log['top1']*100))


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
