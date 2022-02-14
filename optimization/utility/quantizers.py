import math
import torch

pow2 = lambda x: math.pow(2,x)

class Quantize(torch.autograd.Function):
    """ 
    Symmetric Linear Quantizer. The forward pass reduces the precision and clamps the range.
    The backward pass computes the unclipped Straight-Through Estimator (STE) i.e. the gradients
    are not affected by this node.
    """
    @staticmethod
    def forward(ctx, input, bits, frac_len, inplace):
        if bits == 0 or input is None: # no quantization
            return input

        # reduce precision
        y = input if inplace else input.clone()
        y.mul_(pow2(frac_len))
        y.round_() # policy: Round half away from zero (ex: 23.5 becomes 24, -23.5 becomes -24)
        y.mul_(pow2(-frac_len))
        
        # clamp range
        y_min = -pow2(bits-1) * pow2(-frac_len)
        y_max = (pow2(bits-1)-1) * pow2(-frac_len)
        y.clamp_(y_min, y_max)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None # Straight-through estimator (unclipped). Only input needs gradient.

class Wrap(torch.autograd.Function):
    """
    Wraparound operator. The forward pass simulates a 2's complement integer accumulator.
    The backward pass computes the unclipped Straight-Through Estimator (STE) i.e. the gradients
    are not affected by this node.
    """
    @staticmethod
    def forward(ctx, input, bits, frac_len, inplace):
        if bits == 0 or input is None: # no quantization
            return input
        
        # reduce precision
        y = input if inplace else input.clone()
        y.mul_(pow2(frac_len))
        y.round_() # Policy: Round half away from zero (ex: 23.5 becomes 24, -23.5 becomes -24)
        y.mul_(pow2(-frac_len))
        
        # wrap range: [-pow2(IL), pow2(IL)-1]
        y_min = -pow2(bits-1) * pow2(-frac_len)
        y_range = pow2(bits) * pow2(-frac_len)
        # equivalent to y = (y_min + y) % y_range + y_min
        # simplify modulo using definition: x%y = x - floor(x/y)*y (Knuth's way)
        y -= (torch.floor((y_min + y) / y_range) + 1) * y_range
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None # Straight-through estimator (unclipped). Only input needs gradient.

class Masking(torch.autograd.Function):
    """
    Masking operator. Used for simulation pruning where multiple kernels are excluded.
    The gradient is blocked to prevent zero-ed kernels from learning.
    """
    @staticmethod
    def forward(ctx, weights, mask):
        # mask weights
        if weights.ndimension() == 2: # fc
            ctx.mask = mask.view(-1,1)
            return torch.mul(weights, ctx.mask)
        elif weights.ndimension() == 4: # conv
            ctx.mask = mask.view(-1,1,1,1)
            return torch.mul(weights, ctx.mask)
        else:
            raise Exception()
    
    @staticmethod
    def backward(ctx, grad_output):
        return torch.mul(grad_output, ctx.mask), None, None # Block gradients to pruned channels

def quantize(input, bits, frac_len, inplace=False):
    return Quantize.apply(input, bits, frac_len, inplace)

def wraparound(input, bits, frac_len, inplace=True):
    return Wrap.apply(input, bits, frac_len, inplace)

def masking(weights, mask):
    return Masking.apply(weights, mask)
