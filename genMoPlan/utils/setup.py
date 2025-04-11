import os
import importlib
import random
import sys
import time
from typing import List

import numpy as np
import torch
from tap import Tap

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


class Parser(Tap):
    first_save = True

    def __init__(self, **kwargs):
        super().__init__()
        self._args = []

        for k, v in kwargs.items():
            self._args.append(f"--{str(k)}")
            self._args.append(str(v))
        

    def save(self, args):
        if self.first_save:
            self.first_save = False
            self.mkdir(args)
            self.save_diff(args)

        fullpath = os.path.join(args.savepath, "args.json")
        print(f"[ utils/setup ] Saved args to {fullpath}")
        super().save(fullpath, skip_unpicklable=True)

    def get_args(self, ignore_sys_argv=False):
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

    def parse_args(self, ignore_sys_argv=False):
        """
        Parse arguments and set up experiment

        First, parse arguments from command line
        Read config file and override parameters of experiment if necessary
        Add extras from command line to override parameters from config file
        """
        cmd_args = self.get_args(ignore_sys_argv)
        args = super().parse_args(known_only=True, args=cmd_args)
        args.config = f"config.{args.dataset}"
        args = self.read_config(args, method=args.method, variations=args.variations)
        self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.get_commit(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)

        args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)

        return args

    def read_config(self, args, method, variations):
        """
        Load parameters from config file

        If the experiment is in the config file, override the base parameters

        """
        dataset = args.dataset.replace("-", "_")
        print(f"[ utils/setup ] Reading config: {args.config}:{dataset}")
        module = importlib.import_module(args.config)
        params = getattr(module, "base")["base"].copy()

        if method not in getattr(module, "base"):
            raise ValueError(f"[ utils/setup ] Method {method} not found in config: {args.config}")
        
        print(f"[ utils/setup ] Using method: {method}")

        params.update(getattr(module, "base")[method])

        args.used_variations = []
        
        if variations:
            for variation in variations:
                if hasattr(module, variation):
                    print(
                        f"[ utils/setup ] Using overrides | config: {args.config} | variation: {variation}"
                    )
                    overrides = getattr(module, variation)
                    params.update(overrides)
                    args.used_variations.append(variation)
                else:
                    print(f"[ utils/setup ] Warning: variation {variation} not found in config: {args.config}")
        else:
            print(
                f"[ utils/setup ] Not using overrides | config: {args.config} | variation: base"
            )

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        """
        Override config parameters with command-line arguments
        """
        extras = args.extra_args
        if not len(extras):
            return

        print(f"[ utils/setup ] Found extras: {extras}")
        assert (
            len(extras) % 2 == 0
        ), f"Found odd number ({len(extras)}) of extras: {extras}"
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
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f"[ utils/setup ] Setting exp_name to: {exp_name_string}")
            setattr(args, "exp_name", exp_name_string)
            self._dict["exp_name"] = exp_name_string

    def mkdir(self, args):
        if (
            "logbase" in dir(args)
            and "dataset" in dir(args)
            and "exp_name" in dir(args)
        ):
            self._dict["savepath"] = args.savepath
            if "suffix" in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f"[ utils/setup ] Made savepath: {args.savepath}")

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, "diff.txt"))
        except:
            print("[ utils/setup ] WARNING: did not save git diff")


class TrainingParser(Parser):
    dataset: str
    method: str
    variations: List[str] = []
