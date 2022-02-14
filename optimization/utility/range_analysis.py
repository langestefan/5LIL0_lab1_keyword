#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.nn import Linear, Conv2d
from .layers import QLinear, QConv2d

class RangeLogger: 
    """ 
    Log statistics of Convolutional and Fully-connected layers.
    """
    def __init__(self, model, do_hist=False):
        self.do_hist = do_hist
        self.model = model
        layer_names = model.get_names()
        groups = ['inputs', 'weights', 'outputs']
        
        # Collect different running stats
        stats = ['min', 'max', 'absmax', 'nsamples']
        indices = pd.MultiIndex.from_product([groups, stats], names=['group', 'statistic'])
        self.df = pd.DataFrame(0., index=indices, columns=layer_names)
        
        # Collect data distribution (histogram)
        self.num_bins = 32
        self.offset = -16
        self.bins = np.linspace(self.offset, self.offset + self.num_bins, self.num_bins+1)
        indices = pd.MultiIndex.from_product([groups, self.bins[:-1]])
        self.df_hist = pd.DataFrame(0, index=indices, columns=layer_names)
    
    def _hook(self, module, input, output):
        """ 
        Hook to extract layer-wise range statistics. 
        
            This hook is executed after a batch is forwarded through the layer. 
        """
        name = self.model.get_module_name(module)
        assert len(input) == 1
        for g,x in zip(['inputs', 'weights', 'outputs'], [input[0], module.weight, output]):
            # update running stats
            new_min = min(self.df.at[(g,'min'),name], float(torch.min(x)))
            new_max = max(self.df.at[(g,'max'),name], float(torch.max(x)))
            new_absmax = max(self.df.at[(g,'absmax'),name], max(abs(new_min), abs(new_max)))
            self.df.at[(g,'min'),name] = new_min
            self.df.at[(g,'max'),name] = new_max
            self.df.at[(g,'absmax'),name] = new_absmax
            
            # update histogram
            if self.do_hist and (g != 'weights' or self.df.loc['weights','nsamples'][name] == 0.0): # count weights only once!
                flat_x = x.data.cpu().numpy().flatten()
                log2_data = np.clip(np.log2(np.abs(flat_x[flat_x>0])), self.offset, self.offset + self.num_bins)
                hist, _ = np.histogram(log2_data, bins=self.bins)
                self.df_hist.loc[pd.IndexSlice[g,:], name] += hist
                self.df.at[(g,'nsamples'),name] += len(flat_x)
    
    def __enter__(self):
        self._layer_hooks = [] 
        for module in self.model.modules(): # attach hooks to layers
            if isinstance(module, (Conv2d, Linear, QConv2d, QLinear)):
                self._layer_hooks.append(module\
                  .register_forward_hook(self._hook))
        return self
    
    def __exit__(self, *args):
        for hook in self._layer_hooks: # remove all hooks
            hook.remove()
