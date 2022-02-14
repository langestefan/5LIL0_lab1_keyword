import torch
import numpy as np
from torch.nn import Linear, Conv2d
from .layers import QLinear, QConv2d

def apply_quantization(model, q_config):
    for name,s in q_config['solutions'].items():
        layer = model.get_layer(name)
        
        # Step 1: define new layer
        if isinstance(layer, (Conv2d, QConv2d)):
            module = QConv2d(
                  in_channels      = layer.in_channels
                , out_channels     = layer.out_channels
                , kernel_size      = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                , padding          = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                , stride           = layer.stride
                , groups           = layer.groups
                , bits_inputs      = s['bits_inputs']
                , bits_weights     = s['bits_weights']
                , bits_accumulator = s['bits_accumulator']
                , bits_outputs     = s['bits_outputs']
                , frac_len_inputs  = s['bits_inputs'] - s['il_inputs'] - 1 
                , frac_len_weights = s['bits_weights'] - s['il_weights'] - 1 
                , frac_len_outputs = s['bits_outputs'] - s['il_outputs'] - 1
                , name             = name
                , bias             = True if layer.bias is not None else False
            )
        elif isinstance(layer, (Linear, QLinear)):
            module = QLinear(
                  in_features      = layer.in_features
                , out_features     = layer.out_features
                , bits_inputs      = s['bits_inputs']
                , bits_weights     = s['bits_weights']
                , bits_accumulator = s['bits_accumulator']
                , bits_outputs     = s['bits_outputs']
                , frac_len_inputs  = s['bits_inputs'] - s['il_inputs'] - 1 
                , frac_len_weights = s['bits_weights'] - s['il_weights'] - 1 
                , frac_len_outputs = s['bits_outputs'] - s['il_outputs'] - 1
                , name             = name
                , bias             = True if layer.bias is not None else False
            )
        else:
            raise ValueError("Unknown layer type: '{}'"\
            .format(type(getattr(model, container_name))))
        
        # Step 2: construct final container and update model.
        module._parameters = model._modules[name]._parameters # share parameters
        model._modules[name] = module # overwrite original layer in model

def apply_pruning(model, p_config=None):
    for name in model.get_names(ltypes=(Conv2d, Linear)):
        layer = model.get_layer(name)
        
        # Step 1: define new layer
        if type(layer) is Conv2d:
            module = QConv2d(
                  in_channels      = layer.in_channels
                , out_channels     = layer.out_channels
                , kernel_size      = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                , padding          = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                , stride           = layer.stride
                , groups           = layer.groups
                , name             = name
                , bias             = True if layer.bias is not None else False
            )
        elif type(layer) is Linear:
            module = QLinear(
                  in_features      = layer.in_features
                , out_features     = layer.out_features
                , name             = name
                , bias             = True if layer.bias is not None else False
            )
        else:
            raise ValueError("Unknown layer type: '{}'"\
            .format(type(getattr(model, container_name))))
        
        # Step 2: construct final container and update model.
        module._parameters = model._modules[name]._parameters # share parameters
        model._modules[name] = module # overwrite original layer in model
    
    # Step 3: Optional: disable filters according to existing solution
    if p_config is not None:
        with torch.no_grad():
            for name,prune_ind in p_config['pruned_filters'].items():
                layer = model.get_layer(name)
                layer.mask[prune_ind] = 0.
                layer.weight[prune_ind] = 0. # zero weights (optional; masking is sufficient)

def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0.
    nb_zero_param = 0.
    layer_names = model.get_names(ltypes=(Conv2d, QConv2d, Linear, QLinear))
    
    for name in layer_names:
        weights = model.get_layer(name).weight
        
        # count weights in layer
        param_this_layer = 1
        for dim in weights.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if weights.ndimension() != 1:
            zero_param_this_layer = float(torch.sum(weights == 0))
            nb_zero_param += zero_param_this_layer
            
            if verbose:
                num_kernels = weights.shape[0]
                num_zero_kernels = sum([int(torch.sum(x.data.view(-1) != 0)) == 0 for x in weights])
                print("Layer {:8} | {:<8} | {:<20} | {:.2f}% parameters pruned" \
                    .format(
                        name,
                        'Conv' if weights.ndimension() == 4 else 'Linear',
                        "{}/{} kernels pruned".format(num_zero_kernels, num_kernels),
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Total pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc
