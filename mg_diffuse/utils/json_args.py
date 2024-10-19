import collections
import copy
import json

class JSONArgs(collections.Mapping):
    def __init__(self, json_file, verbose=False):
        with open(json_file, 'r') as f:
            self._data = json.load(f)

        self._process_data_structures(verbose)

    def _process_data_structures(self, verbose):
        for key, value in self._data.items():
            if type(value) is dict and '_value' in value:
                if '_string' in value:
                    if verbose: print(' [ utils/json_args ] Ignoring complex data structure:', key)
                    continue
                keyType = eval(value['_type'])
                self._data[key] = keyType(value['_value'])

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    def __contains__(self, key):
        return key in self._data

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'JSONArgs' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key):
        if key == '_data':
            super().__delattr__(key)
        else:
            del self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __eq__(self, other):
        return self._data == other

    def __ne__(self, other):
        return self._data != other

    def __hash__(self):
        return hash(tuple(sorted(self._data.items())))

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._data = self._data.copy()
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        new_instance._data = copy.deepcopy(self._data, memo)
        return new_instance

    def copy(self):
        return self.__copy__()

    def deepcopy(self, memo=None):
        if memo is None:
            memo = {}
        return self.__deepcopy__(memo)

    def to_dict(self):
        return self._data

    def to_json(self):
        return json.dumps(self._data)

    def to_file(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(self._data, f)

    def __call__(self, key):
        return self._data[key]
