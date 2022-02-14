import torch
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())

def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

yaml.add_representer(OrderedDict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)

def read_yaml(fname):
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, default_flow_style=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

def test_dataset(model, dataset_loader, loss_fn, metric_fns, print_cm=True):
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval() # disable all training-specific layer configurations (e.g. dropout)
    
    # runt dataset loader through model
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    pred_labels,true_labels = [],[]
    with torch.no_grad():
        for i, (data, target) in enumerate(dataset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if print_cm: # store results for confusion matrix
                pred_labels.extend(torch.argmax(output, dim=1).tolist())
                true_labels.extend(target.tolist())

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    
    if print_cm: # print confusion matrix
        cm = pd.crosstab(np.asarray(true_labels), np.asarray(pred_labels), 
                        rownames=['True'], colnames=['Predicted'], margins=True)
        print(cm)
    
    # collect metrics
    n_samples = len(dataset_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    return log
