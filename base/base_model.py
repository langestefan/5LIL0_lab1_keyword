import torch.nn as nn
import numpy as np
from copy import deepcopy
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def get_names(self, ltypes=None):
        """ 
        Returns list with layer names
        """
        layer_names = []
        for idx,(container_name, module) in enumerate(self.named_modules()):
            is_container = len(list(module.modules())) > 1 # exclude containers or modules with multiple layers
            is_eligible = True if ltypes is None else type(module) in tuple(ltypes)
            if isinstance(module, nn.Module) and not is_container and is_eligible:
                layer_names.append("{}".format(container_name)) # example: features.0 <- index
        return layer_names
    
    def get_layer(self, name):
        """ 
        Returns module of layer with requested name
        """
        module = self
        for name_or_idx in name.split('.'):
            if name_or_idx.isdigit():
                module = module[int(name_or_idx)] # example: features.0 <- index
            else:
                module = getattr(module, name_or_idx)
        return module
    
    def get_module_name(self, module):
        """ 
        Returns name of module or None if module not exists in model (comparison by identity)
        """
        for idx,(name, m) in enumerate(self.named_modules()):
            is_container = len(list(module.modules())) > 1 # exclude containers or modules with multiple layers
            is_same_object = module is m
            if isinstance(module, nn.Module) and not is_container and is_same_object:
                return name
        return None
    
    def get_model_size(self):
        """ 
        Returns model dimensions TODO: Work-In-Progress
        """
        for name in self.get_names():
            module = self.get_layer(name)
            print("Name", name)
            for name, param in module.named_parameters():
                print(name, type(param.data), param.size())
            for name, buf in module.named_buffers():
                print(name, type(buf.data), buf.size())
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
