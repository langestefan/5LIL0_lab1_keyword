import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import Linear, Conv2d
from .utility.layers import QConv2d, QLinear
from .utility.helpers import prune_rate, apply_pruning

class KernelPruner:
    """
    KernelPruner class
    """
    def __init__(self, model, config, val_data_loader, error_margin=3):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) # transfer model to gpu if available
        self.layer_names = model.get_names(ltypes=(Conv2d, Linear))
        self.val_data_loader = val_data_loader
        self.error_margin = error_margin # pruning parameter
        self.prune_dict = OrderedDict((k,[]) for k in self.layer_names) # store which kernels (indices) are pruned until now
    
    def PruneNet(self):
        """
        Full pruning logic; based on minimum weight method from Molchanov et al.[1]
        
        [1] Pruning Convolutional Neural Netowrks for Resource Efficient Inference (ICLR 2017)
        """
        print("\n[PRUNING] Step 1: Collect baseline accuracy.\n") 
        val_top1_ref = self._RunForwardBatches(self.model, self.val_data_loader)
        print("baseline float32 validation accuracy: {:.5f}" .format(val_top1_ref))
        
        print("\n[PRUNING] Step 2: Start pruning until error margin is violated\n") 
        model = copy.deepcopy(self.model)
        apply_pruning(model)
        model = model.to(self.device)
        while True:            
            # prune the least important filter (according to rank)
            name,filter_idx = self._prune_one_filter(model)
            prune_rate(model)
            val_top1 = self._RunForwardBatches(model, self.val_data_loader)
            print("\nvalidation accuracy: {:.5f}\n".format(val_top1))
            
            # check if maximum accuracy penalty is exceeded
            if val_top1 + self.error_margin/100 < val_top1_ref:
                print("Error margin exceeded; save previous solution!")
                break
            self.prune_dict[name].append(filter_idx)
        
        print("\n[PRUNING] Step 3: Evaluate final solution on validation set.\n")
        # initialize new model and set layers to minimum number of kernels just above margin
        model = copy.deepcopy(self.model)
        apply_pruning(model)
        model = model.to(self.device)
        with torch.no_grad():
            for name,prune_ind in self.prune_dict.items():
                layer = model.get_layer(name)
                layer.mask[prune_ind] = 0.
                layer.weight[prune_ind] = 0. # zero weights (optional; masking is sufficient)
        final_val_top1 = self._RunForwardBatches(model, self.val_data_loader)
        print("final validation accuracy: {:.5f}" .format(final_val_top1))
        
        print("\n---------------------------------------------")
        print("Network pruning results.")
        print("Baseline float32: {:.5f}".format(val_top1_ref))
        print("Kernel pruning results:")
        prune_rate(model)
        print("Accuracy: {:.5f}".format(final_val_top1))
        print("---------------------------------------------")
        return OrderedDict([
            ("config", OrderedDict([
                    ("arch", self.config['arch']['type']),
                    ("model_path", str(self.config.resume)), 
                    ("error_margin", self.error_margin),
                ])),
            ("results", OrderedDict([
                    ("baseline_top1", val_top1_ref),
                    ("accuracy_top1", final_val_top1),
                ])),
            ("pruned_filters", 
                    OrderedDict((k,sorted(v)) for k,v in self.prune_dict.items())
                ),
            ])
    
    def _prune_one_filter(self, model):
        '''
        This method prunes one least important feature map (ranked by the scaled l2norm of kernel weights)
        
        Pruning by magnitude of kernel weights is perhaps the simplest possible criterion, and it does 
        not require any additional computation during the fine-tuning process.
        
        Pruning complete convolution filters instead of individual weights has the advantage that it is 
        more hardware friendly. 
        
        Source: Minimum weight method from 'Pruning Convolutional Neural Netowrks for Resource Efficient 
        Inference' [Molchanov et al, ICLR 2017]
        '''
        layer_names = model.get_names(ltypes=(Conv2d, QConv2d, Linear, QLinear))
        
        values = []
        for name in layer_names:
            weights = model.get_layer(name).weight
            
            # find the scaled l2 norm for each filter in this layer
            if weights.ndimension() == 4: # select conv layer 
                C,K,H,W = weights.shape
                rdims = [1,2,3] # K,H,W
                rank_per_kernel = torch.sum(torch.mul(weights,weights),rdims).data / (K * H * W)
            elif weights.ndimension() == 2: # select fc layer 
                C,K = weights.shape
                rdims = [1] # K
                rank_per_kernel = torch.sum(torch.mul(weights,weights),rdims).data / K
            else:
                raise Exception()
            
            ## find kernel that can be removed (exclude already removed kernels)
            sort_ind = np.argsort(rank_per_kernel.cpu()) # low to high index sort
            first_nonzero_idx = (rank_per_kernel[sort_ind] <= 0).sum()
            if first_nonzero_idx < len(sort_ind):
                min_idx = sort_ind[first_nonzero_idx]
                min_value = rank_per_kernel[min_idx]
            else: # all weights int this layer are pruned!
                min_idx = np.inf
                min_value = np.inf
            values.append([min_value, min_idx])
            
        values = np.array(values)

        # set mask corresponding to the filter to prune
        to_prune_layer_ind = np.argmin(values[:,0])
        to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
        
        # remove weights of one kernel
        layer = model.get_layer(layer_names[to_prune_layer_ind])
        layer.mask[to_prune_filter_ind] = 0.
        layer.weight[to_prune_filter_ind] = 0. # zero weights
        
        name = layer_names[to_prune_layer_ind]
        print('Prune filter #{} in layer {}'.format(to_prune_filter_ind, name))
        return name, to_prune_filter_ind
    
    def _top1(self, output, target):
        """
        Compute the top-1 accuracy for a prediction vector for a batch
        """
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)
    
    def _RunForwardBatches(self, model, data_loader):
        """ 
        Run through validation or test dataset and return the top-1 classification accuracy
        """ 
        model.eval()
        total_top1 = 0.
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                #print("batch {}: top1: {}".format(batch_idx, self._top1(output, target)))
                total_top1 += self._top1(output, target)
        return total_top1 / len(data_loader)

