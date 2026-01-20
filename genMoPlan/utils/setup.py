import copy
import json
import os
import importlib
import random
import sys
import time
from typing import List, Optional
import warnings

import numpy as np
import torch
from tap import Tap

from .paths import mkdir


def recursive_update(d, u):
    for k, v in u.items():
        # First, check if the key exists in the base dictionary
        if k in d:
            # If both the new value and the existing value are dictionaries, recurse
            if isinstance(v, dict) and isinstance(d[k], dict):
                recursive_update(d[k], v)
            # If the new value is a dict but the old one isn't, raise the type error
            elif isinstance(v, dict) and not isinstance(d[k], dict):
                raise ValueError(f"[ utils/setup ] Type mismatch: cannot overwrite non-dict with dict for key '{k}'")
            # Otherwise, just update the value
            else:
                d[k] = v
        # If the key is new, just add it
        else:
            d[k] = v
    return d


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def watch_dict(args_to_watch):
    def _fn(args, method_name: str = None):
        timestamp = f"{time.strftime('%y_%m_%d-%H_%M_%S')}"

        actual_args_to_watch = [(method_name, "METHOD"), *args_to_watch]

        exp_name = [
            timestamp
        ]

        for key, label in actual_args_to_watch:
            if key not in args:
                continue
            val = args[key]
            if type(val) == dict:
                val = "_".join(f"{k}-{v}" for k, v in val.items())
            if type(val) == bool:
                val = "T" if val else "F"
            if val is None:
                val = "F"
            if type(val) == float:
                val = str(val).replace(".", "p")
            if type(val) == int:
                val = str(val)

            exp_name.append(f"{label}-{val}")

        return "_".join(exp_name)
    return _fn


def watch(args_to_watch):
    def _fn(args):
        timestamp = f"{time.strftime('%y_%m_%d-%H_%M_%S')}"
        exp_name = [
            f"{args.prefix}{timestamp}"
        ]
        
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = "_".join(f"{k}-{v}" for k, v in val.items())
            if type(val) == bool:
                val = "T" if val else "F"
            if val is None:
                val = "F"
            if type(val) == float:
                val = str(val).replace(".", "p")

            exp_name.append(f"{label}-{val}")
                

        exp_name = "_".join(exp_name) + ("" if not args.used_variations else f"_{'_'.join(args.used_variations)}")
        exp_name = exp_name.replace("/_", "/")
        exp_name = exp_name.replace("(", "").replace(")", "")
        exp_name = exp_name.replace(", ", "-")

        return exp_name

    return _fn


def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Args:
    def __init__(self, args):
        object.__setattr__(self, '_args', args)

    def __getattr__(self, key):
        if hasattr(self._args, key):
            return getattr(self._args, key)
        else:
            warnings.warn(f"'Args' object has no attribute '{key}'")
            return None
        
    def __setattr__(self, key, value):
        if key == '_args':
            object.__setattr__(self, key, value)
        else:
            setattr(self._args, key, value)

    def __delattr__(self, key):
        delattr(self._args, key)

    def __contains__(self, key):
        return hasattr(self._args, key)
    
    def __len__(self):
        return len(vars(self._args))
    
    def __iter__(self):
        return iter(vars(self._args))
    
    def __getitem__(self, key):
        return getattr(self._args, key)
    
    def __setitem__(self, key, value):
        setattr(self._args, key, value)
    
    def __delitem__(self, key):
        delattr(self._args, key)

    def __repr__(self):
        return f"Args({vars(self._args)})"
    
    def __str__(self):
        return str(vars(self._args))
    
    def __copy__(self):
        return Args(copy.copy(self._args))
    
    def __deepcopy__(self, memo):
        return Args(copy.deepcopy(self._args, memo))
    
    def copy(self):
        return self.__copy__()
    
    def to_dict(self):
        return vars(self._args)
    
    def to_json(self):
        return json.dumps(vars(self._args))
    
    def safe_get(self, key, default):
        if not hasattr(self._args, key):
            return default
        return getattr(self._args, key)

class Parser(Tap):
    first_save = True
    suffix: Optional[str] = None
    dataset: str = None
    method: Optional[str] = None
    variations: List[str] = []
    no_inference: bool = False

    def __init__(self, **kwargs):
        super().__init__()
        self._args = []

        for k, v in kwargs.items():
            self._args.append(f"--{str(k)}")
            if isinstance(v, list):
                for item in v:
                    self._args.append(str(item))
            else:
                self._args.append(str(v))
        

    def save(self, args):
        if self.first_save:
            self.first_save = False
            self.mkdir(args)

        fullpath = os.path.join(args.savepath, "args.json")
        print(f"[ utils/setup ] Saved args to {fullpath}")
        super().save(fullpath, skip_unpicklable=True)

    def get_cmd_args(self, ignore_sys_argv=False):
        if ignore_sys_argv:
            return self._args
        elif len(self._args) == 0:
            return sys.argv[1:]
        elif len(sys.argv) == 1:
            return self._args
        elif len(sys.argv) % 2 != 0:
            return sys.argv[1:] + self._args
        else:
            return sys.argv + self._args

    def parse_args(self, ignore_sys_argv=False, is_for_exp=True):
        """
        Parse arguments and set up experiment

        First, parse arguments from command line
        Read config file and override parameters of experiment if necessary
        Add extras from command line to override parameters from config file
        """
        cmd_args = self.get_cmd_args(ignore_sys_argv)
        args = super().parse_args(known_only=True, args=cmd_args)
        args.config = f"config.{args.dataset}"
        args = self.read_config(args, method=args.method, variations=args.variations)
        self.apply_config_overrides(args)
        self.eval_fstrings(args)

        if is_for_exp:
            self.set_seed(args)
            self.set_loadbase(args)
            self.generate_exp_name(args)

            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)

        return Args(args)

    def read_config(self, args, method, variations):
        """
        Load parameters from config file

        If the experiment is in the config file, override the base parameters

        """
        dataset = args.dataset.replace("-", "_")
        print(f"[ utils/setup ] Reading config: {args.config}:{dataset}")
        module = importlib.import_module(args.config)
        params = getattr(module, "base")["base"].copy()

        if method is not None:
            if method not in getattr(module, "base"):
                raise ValueError(f"[ utils/setup ] Method {method} not found in config: {args.config}")

            print(f"[ utils/setup ] Using method: {method}")

            params.update(getattr(module, "base")[method])

        args.used_variations = []

        if variations:
            valid_variations = []
            for variation in variations:
                if hasattr(module, variation):
                    valid_variations.append(variation)

            if len(valid_variations) != len(variations):
                print(f"[ utils/setup ] Warning: below variations not found in config: {args.config}:")
                for variation in variations:
                    if variation not in valid_variations:
                        print(f"    - {variation}")

                input("Press Enter to continue with the remaining variations...")

            variations = valid_variations


            for variation in variations:
                print(
                    f"[ utils/setup ] Using overrides | config: {args.config} | variation: {variation}"
                )
                overrides = getattr(module, variation)
                recursive_update(params, overrides)
                args.used_variations.append(variation)
        else:
            print(
                f"[ utils/setup ] Not using overrides | config: {args.config} | variation: base"
            )

        # Create system instance
        if hasattr(module, 'get_system'):
            print(f"[ utils/setup ] Creating system instance")

            # Get use_manifold flag from method config if it exists
            use_manifold = params.get('use_manifold', False)

            # Create system instance with training params (stride, history_length, horizon_length)
            system = module.get_system(
                config=getattr(module, "base"),
                use_manifold=use_manifold,
                stride=params.get('stride', 1),
                history_length=params.get('history_length', 1),
                horizon_length=params.get('horizon_length', 31),
            )

            # Store the system instance - dataset and gen_model will extract what they need
            params['system'] = system

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def apply_config_overrides(self, args):
        """
        Override config parameters with command-line arguments
        """
        extras = args.extra_args
        if not len(extras):
            return

        print(f"[ utils/setup ] Found extras: {extras}")
        if len(extras) % 2 != 0:
            print(f"[ utils/setup ] Found odd number ({len(extras)}) of extras: {extras}. Ignoring extras.")
            return
        
        for i in range(0, len(extras), 2):
            key = extras[i].replace("--", "")
            val = extras[i + 1]
            assert hasattr(
                args, key
            ), f"[ utils/setup ] {key} not found in config: {args.config}"
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f"[ utils/setup ] Overriding config | {key} : {old_val} --> {val}")
            if val == "None":
                val = None
            elif val == "latest":
                val = "latest"
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(
                        f"[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str"
                    )
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == "f:":
                val = old.replace("{", "{args.").replace("f:", "")
                new = lazy_fstring(val, args)
                print(f"[ utils/setup ] Lazy fstring | {key} : {old} --> {new}")
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not hasattr(args, "seed") or args.seed is None:
            return
        print(f"[ utils/setup ] Setting seed: {args.seed}")
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, "loadbase") and args.loadbase is None:
            print(f"[ utils/setup ] Setting loadbase: {args.logbase}")
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not "exp_name" in dir(args):
            return
        exp_name = getattr(args, "exp_name")
        if not callable(exp_name):
            exp_name_string = exp_name
        else:
            exp_name_string = exp_name(args)

        if self.suffix is not None:
            exp_name_string = f"{exp_name_string}_{self.suffix}"

        print(f"[ utils/setup ] Setting exp_name to: {exp_name_string}")
        setattr(args, "exp_name", exp_name_string)
        self._dict["exp_name"] = exp_name_string

    def mkdir(self, args):
        if (
            ("logbase" in dir(args) or "logbase" in self._dict or "logbase" in dir(args._args))
            and ("dataset" in dir(args) or "dataset" in self._dict or "dataset" in dir(args._args))
            and ("exp_name" in dir(args) or "exp_name" in self._dict or "exp_name" in dir(args._args))
        ):
            self._dict["savepath"] = args.savepath
            if "suffix" in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f"[ utils/setup ] Made savepath: {args.savepath}")


def get_dataset_config(
    dataset: str,
    *,
    method: Optional[str] = None,
    variations: List[str] = None,
):
    """
    Get dataset config from config file. Meant for inference or other non-training tasks.
    """

    if method is not None and variations is not None:
        parser = Parser(
            dataset=dataset, 
            method=method, 
            variations=variations
        )

    elif method is not None:
        parser = Parser(
            dataset=dataset, 
            method=method
        )

    elif variations is not None:
        parser = Parser(
            dataset=dataset, 
            variations=variations
        )

    else:
        parser = Parser(
            dataset=dataset
        )

    return parser.parse_args(is_for_exp=False, ignore_sys_argv=True)

