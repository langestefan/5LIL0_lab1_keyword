import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_yaml, write_yaml

class ConfigParser:
    def __init__(self, args, options='', log_results=True, timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
        if args.resume is None:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yml', for example."
            assert args.config is not None, msg_no_cfg
            msg_no_pq = "Quantization and Pruning only supported for post-training finetuning. Add '-r model.pth', for example." 
            assert args.quantization is None and args.pruning is None, msg_no_pq
            self.resume = None
            self.cfg_fname = Path(args.config)
        else:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.yml'
            
        if hasattr(args, 'quantization') and args.quantization is not None:
            self.quantization = Path(args.quantization)
            self.cfg_quantization = read_yaml(self.quantization)
        if hasattr(args, 'pruning') and args.pruning is not None:
            self.pruning = Path(args.pruning)
            self.cfg_pruning = read_yaml(self.pruning)

        # load config file and apply custom cli options
        config = read_yaml(self.cfg_fname)
        self._config = _update_config(config, options, args)
        
        if log_results:
            # set save_dir where trained model and log will be saved.
            save_dir = Path(self.config['trainer']['save_dir'])
            timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''

            exper_name = self.config['name']
            self._save_dir = save_dir / 'models' / exper_name / timestamp
            self._log_dir = save_dir / 'log' / exper_name / timestamp

            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save updated config file to the checkpoint dir
            write_yaml(self.config, self.save_dir / 'config.yml')

            # configure logging module
            setup_logging(self.log_dir)
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }
        else:
            self._save_dir = None
            self._log_dir = None
            self.log_levels = None
    
    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        module_args = self.handle_exceptions(name)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
    
    def handle_exceptions(self, name):
        if self[name]['type'] == 'LambdaLR' and 'lr_lambda' in self[name]['args']:
            self[name]['args']['lr_lambda'] = eval(self[name]['args']['lr_lambda']) # cannot store lambda functions in JSON/YML(?)
        return dict(self[name]['args'])

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
