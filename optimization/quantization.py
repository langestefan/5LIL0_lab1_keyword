import copy
import torch
import functools
import collections
import numpy as np
from collections import OrderedDict
from torch.nn import Linear, Conv2d
from .utility.layers import QLinear, QConv2d
from .utility.range_analysis import RangeLogger

class Quantizer:
    """
    Quantizer class
    """
    def __init__(self, model, config, val_data_loader, error_margin=5, decrement=lambda x:x-2):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) # transfer model to gpu if available
        self.val_data_loader = val_data_loader
        self.max_bitwidth = 16
        self.error_margin = error_margin # quantization parameter
        self.decrement = decrement
    
    def QuantizeNet2DynamicFixedPoint(self):
        """
        Full quantization logic; similar to dynamic fixed-point quantization procedure in Caffe-Ristretto[1].
        
        [1] Ristretto: A Framework for Empirical Study of Resource-Efficient Inference in Convolutional Neural Networks (2018)
        """
        print("\n[QUANTIZATION] Step 1: Collect network statistics and baseline accuracy.\n") 
        with RangeLogger(self.model) as stats:
            val_top1_ref = self._RunForwardBatches(self.model, self.val_data_loader)
        print("baseline float32 validation accuracy: {:.5f}" .format(val_top1_ref))
        
        print("\n[QUANTIZATION] Step 2: Determine integer length.\n")
        # use the absolute max value of every layer for range estimates.
        df_amax = stats.df.xs('absmax',level=1)
        self.il = df_amax.applymap(lambda x: int(np.floor(np.log2(x))+1))
        print(self.il)
        
        print("\n[QUANTIZATION] Step 3: Test quantization solutions.\n")
        print("*** stage 1: determine minimum weight bitwidth in convolutional layers ***")
        conv_val_top1 = []
        conv_w_bits = []
        bitwidth = self.max_bitwidth
        while bitwidth > 0:
            # initialize new model with quantized convolutional layers and validate
            model = copy.deepcopy(self.model)
            for name in model.get_names(ltypes=(Conv2d, QConv2d)):
                self._EditLayerDescription2FixedPoint(model, name, bits_weights=bitwidth)
            val_top1 = self._RunForwardBatches(model, self.val_data_loader)
            print("\n{}\nvalidation accuracy: {:.5f}".format(model, val_top1))
            
            # store results and check if maximum accuracy penalty is exceeded
            conv_val_top1.append(val_top1)
            conv_w_bits.append(bitwidth)
            if val_top1 + self.error_margin/100 < val_top1_ref:
                break
            bitwidth = self.decrement(bitwidth)
        
        print("\n*** stage 2: determine minimum weight bitwidth in fully-connected layers ***")
        fc_val_top1 = []
        fc_w_bits = []
        bitwidth = self.max_bitwidth
        while bitwidth > 0:
            # initialize new model with quantized convolutional layers and validate
            model = copy.deepcopy(self.model)
            for name in model.get_names(ltypes=(Linear, QLinear)):
                self._EditLayerDescription2FixedPoint(model, name, bits_weights=bitwidth)
            val_top1 = self._RunForwardBatches(model, self.val_data_loader)
            print("\n{}\nvalidation accuracy: {:.5f}".format(model, val_top1))
            
            # store results and check if maximum accuracy penalty is exceeded
            fc_val_top1.append(val_top1)
            fc_w_bits.append(bitwidth)
            if val_top1 + self.error_margin/100 < val_top1_ref:
                break
            bitwidth = self.decrement(bitwidth)
        
        print("\n*** stage 3: determine minimum data/activation bitwidth in network ***")
        data_val_top1 = []
        data_bits = []
        bitwidth = self.max_bitwidth
        while bitwidth > 0:
            # initialize new model with quantized convolutional layers and validate
            model = copy.deepcopy(self.model)
            for name in model.get_names(ltypes=(Linear, QLinear, Conv2d, QConv2d)):
                self._EditLayerDescription2FixedPoint(model, name, bits_data=bitwidth)
            val_top1 = self._RunForwardBatches(model, self.val_data_loader)
            print("\n{}\nvalidation accuracy: {:.5f}".format(model, val_top1))
            
            # store results and check if maximum accuracy penalty is exceeded
            data_val_top1.append(val_top1)
            data_bits.append(bitwidth)
            if val_top1 + self.error_margin/100 < val_top1_ref:
                break
            bitwidth = self.decrement(bitwidth)
        
        print("\n[QUANTIZATION] Step 4: Evaluate final solution on validation set.\n")
        best_data_bits = data_bits[-2]
        best_conv_w_bits = conv_w_bits[-2]
        best_fc_w_bits = fc_w_bits[-2]
        
        # initialize new model and set layers to minimum bitwidth just above margin
        model = copy.deepcopy(self.model)
        for name in model.get_names(ltypes=(Conv2d, QConv2d)):
            self._EditLayerDescription2FixedPoint(model, name, best_data_bits, best_conv_w_bits)
        for name in model.get_names(ltypes=(Linear, QLinear)):
            self._EditLayerDescription2FixedPoint(model, name, best_data_bits, best_fc_w_bits)
        final_val_top1 = self._RunForwardBatches(model, self.val_data_loader)
        print("{}\nvalidation accuracy: {:.5f}".format(model, final_val_top1))
        
        print("\n---------------------------------------------")
        print("Network quantization results.")
        print("Baseline float32: {:.5f}".format(val_top1_ref))
        print("Conv layer weights:")
        for bits,val_top1 in zip(conv_w_bits,conv_val_top1):
            print("    bitwidth: {:2} => top1: {:.5f}".format(bits, val_top1))
        print("FC layer weights:")
        for bits,val_top1 in zip(fc_w_bits,fc_val_top1):
            print("    bitwidth: {:2} => top1: {:.5f}".format(bits, val_top1))
        print("Conv/FC inputs/activations:")
        for bits,val_top1 in zip(data_bits,data_val_top1):
            print("    bitwidth: {:2} => top1: {:.5f}".format(bits, val_top1))
        print("Dynamic fixed point net:")
        print("Conv weights bitwidth: {}".format(best_conv_w_bits))
        print("FC weights bitwidth: {}".format(best_fc_w_bits))
        print("Activations bitwidth: {}".format(best_data_bits))
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
            ("solutions", OrderedDict([
                (name, OrderedDict([
                    ("bits_inputs", best_data_bits),
                    ("bits_weights", best_conv_w_bits if isinstance(model.get_layer(name),(Conv2d, QConv2d)) else best_fc_w_bits),
                    ("bits_accumulator", model.get_layer(name).bits_accumulator),
                    ("bits_outputs", best_data_bits),
                    ("il_inputs", int(self.il.loc['inputs',name])),
                    ("il_weights", int(self.il.loc['weights',name])),
                    ("il_outputs", int(self.il.loc['outputs',name])),
                    ])) for name in model.get_names(ltypes=(Linear, QLinear, Conv2d, QConv2d))
                ])),
            ])
    
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
    
    def _EditLayerDescription2FixedPoint(self, model, name, bits_data=0, bits_weights=0):
        """ 
        Edit specific layer in model (only Convolution and Fully-connected layers are supported) 
        """
        layer = model.get_layer(name)
        
        # Step 1: determine accumulator bitwidth (pessimistic)
        if bits_data == 0 or bits_weights == 0: # consider worst-case accumulator
            bits_accumulator = (2*self.max_bitwidth - 1) + int(np.log2(self._GetKernelSize(layer)))
        else:
            bits_accumulator = (bits_data + bits_weights - 1) + int(np.log2(self._GetKernelSize(layer))) # can we do better?
        
        # Step 2: define new layer
        if isinstance(layer, (Conv2d, QConv2d)):
            module = QConv2d(
                  in_channels      = layer.in_channels
                , out_channels     = layer.out_channels
                , kernel_size      = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                , padding          = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                , stride           = layer.stride
                , groups           = layer.groups
                , bits_inputs      = bits_data
                , bits_weights     = bits_weights
                , bits_accumulator = bits_accumulator
                , bits_outputs     = bits_data
                , frac_len_inputs  = bits_data - self.il.loc['inputs',name]- 1 
                , frac_len_weights = bits_weights - self.il.loc['weights',name] - 1 
                , frac_len_outputs = bits_data - self.il.loc['outputs',name] - 1
                , name             = name
                , bias             = True if layer.bias is not None else False
            )
        elif isinstance(layer, (Linear, QLinear)):
            module = QLinear(
                  in_features      = layer.in_features
                , out_features     = layer.out_features
                , bits_inputs      = bits_data
                , bits_weights     = bits_weights
                , bits_accumulator = bits_accumulator
                , bits_outputs     = bits_data
                , frac_len_inputs  = bits_data - self.il.loc['inputs',name]- 1 
                , frac_len_weights = bits_weights - self.il.loc['weights',name] - 1 
                , frac_len_outputs = bits_data - self.il.loc['outputs',name] - 1
                , name             = name
                , bias             = True if layer.bias is not None else False
            )
        else:
            raise ValueError("Unknown layer type: '{}'"\
            .format(type(getattr(model, container_name))))
        
        # Step 3: construct final container and update model.
        module._parameters = model._modules[name]._parameters # share parameters
        model._modules[name] = module.to(self.device) # overwrite original layer in model
    
    def _GetKernelSize(self, layer):
        """ 
        Return kernel size of Convolutional or Fully-connected layer. 
        """
        kernel_size = np.prod(layer.weight.size()[1:]) # prod([output_channels, input_channels, kernel_y, kernel_x][:1])
        if layer.bias is not None:
            kernel_size += 1
        return kernel_size
    
    @functools.lru_cache(maxsize=8192) # number of kernels is generally smaller than this
    def _GetMaxAbsKernelSum(self, layer, frac_len_weights='float', frac_len_inputs='float'):
        """ 
        Get the maximum absolute kernel value after quantization. 
        """
        
        # quantize weights
        p = layer.weight.clone() # do not modify the original weights
        p = p.view(p.size()[0], -1) # shape (out_channels, kernel)
        if frac_len_weights != 'float':
            p.data.mul_(pow(2, frac_len_weights))
            p.data.round_() 
            p.data.mul_(pow(2, -frac_len_weights)) 
        p.data.abs_()
        abs_kernel_sums = torch.sum(p, dim=1)
    
        # quantize bias
        if layer.bias is not None:
            b = layer.bias.clone() # do not modify the original weights
            if frac_len_weights != 'float' and frac_len_inputs != 'float':
                b.data.mul_(pow(2, frac_len_weights + frac_len_inputs))
                b.data.round_() 
                b.data.mul_(pow(2, -(frac_len_weights + frac_len_inputs))) 
            b.data.abs_()
            abs_kernel_sums += b
        
        result = torch.max(abs_kernel_sums).data[0] # maximum of all output channels
        return result 
