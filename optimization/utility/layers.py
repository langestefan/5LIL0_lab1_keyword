import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizers import quantize, wraparound, masking

class QConv2d(nn.Conv2d):
    """ 
    QConv2d simulates the reduced-precision forward pass of nn.Conv2d by quantizing 
    inputs, weights and outputs. Also the weights are extended with masks to allow pruning. 
    
    Biases are not pruned...
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 verbose=False, **kwargs):
        
        # assign keyword attributes to self.__dict__
        for attr,val in kwargs.items():
            setattr(self, attr, kwargs.get(attr, val))
        
        # set quantization parameters to default value (0)
        for attr in ['bits_inputs', 'bits_weights', 'bits_accumulator', 'bits_outputs',
                     'frac_len_inputs', 'frac_len_weights', 'frac_len_outputs']:
            setattr(self, attr, kwargs.get(attr, 0))
        
        self.fp_groups = ['inputs', 'weights', 'accumulator', 'outputs']
        self.quantized_list = [g for g in self.fp_groups if getattr(self, 'bits_' + g) > 0]
        if verbose and self.quantized_list:
            print("{}: {{{}}} are quantized!"\
            .format(self.name, ", ".join(self.quantized_list)))
    
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride, padding, dilation, groups, bias)
        
        # weight mask. mask is needed to disable weights during inference and training
        self.register_buffer('mask', torch.ones(out_channels))
    
    def forward(self, x):
        q_input = quantize(x, self.bits_inputs, self.frac_len_inputs)
        q_weights = quantize(masking(self.weight, self.mask), self.bits_weights, self.frac_len_weights)
        q_bias = quantize(self.bias, self.bits_accumulator, self.frac_len_inputs + self.frac_len_weights)
        acc = F.conv2d(q_input, q_weights, q_bias, self.stride, self.padding, self.dilation, self.groups)
        out = wraparound(acc, self.bits_accumulator, self.frac_len_inputs + self.frac_len_weights)
        return quantize(out, self.bits_outputs, self.frac_len_outputs, inplace=True)
    
    def __repr__(self):
        s = "{layer_type} ({in_channels}, {out_channels}"
        s += ", kernel_size={kernel_size}"
        s += ", stride={stride}"
        s += ", padding={padding}"
        s += ", bits_{quantized_list}={quantized_bits})"
        return s.format(layer_type=self.__class__.__name__, 
                        quantized_bits=[int(getattr(self,'bits_'+g)) for g in self.fp_groups \
                            if getattr(self,'bits_'+g) > 0], **self.__dict__)


class QLinear(nn.Linear):
    """ 
    QLinear simulates the reduced-precision forward pass of nn.Linear by quantizing 
    inputs, weights and outputs. Also the weights are extended with masks to allow pruning. 
    
    Biases are not pruned...
    """
    def __init__(self, in_features, out_features, bias=True,
                 verbose=False, **kwargs):
        
        # assign keyword attributes to self.__dict__
        for attr,val in kwargs.items():
            setattr(self, attr, kwargs.get(attr, val))
    
        # set quantization parameters to default value (0)
        for attr in ['bits_inputs', 'bits_weights', 'bits_accumulator', 'bits_outputs',
                     'frac_len_inputs', 'frac_len_weights', 'frac_len_outputs']:
            setattr(self, attr, kwargs.get(attr, 0))
        
        self.fp_groups = ['inputs', 'weights', 'accumulator', 'outputs']
        self.quantized_list = [g for g in self.fp_groups if getattr(self, 'bits_' + g) > 0]
        if verbose and self.quantized_list:
            print("{}: {{{}}} are quantized!"\
            .format(self.name, ", ".join(self.quantized_list)))
        
        super().__init__(in_features, out_features, bias)
        
        # weight mask. mask is needed to disable weights during inference and training
        self.register_buffer('mask', torch.ones(out_features))
    
    def forward(self, x):
        q_input = quantize(x, self.bits_inputs, self.frac_len_inputs)
        q_weights = quantize(masking(self.weight, self.mask), self.bits_weights, self.frac_len_weights)
        q_bias = quantize(self.bias, self.bits_accumulator, self.frac_len_inputs + self.frac_len_weights)
        acc = F.linear(q_input, q_weights, q_bias)
        out = wraparound(acc, self.bits_accumulator, self.frac_len_inputs + self.frac_len_weights)
        return quantize(out, self.bits_outputs, self.frac_len_outputs, inplace=True)

    def __repr__(self):
        s = "{layer_type} ({in_features}, {out_features}"
        s += ", bits_{quantized_list}={quantized_bits})"
        return s.format(layer_type=self.__class__.__name__, 
                        quantized_bits=[int(getattr(self,'bits_'+g)) for g in self.fp_groups \
                            if getattr(self,'bits_'+g) > 0], **self.__dict__)
