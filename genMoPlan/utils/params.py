import importlib


def load_roa_estimation_params(dataset):
    config = f'config.{dataset}'
    module = importlib.import_module(config)
    params = getattr(module, "base")["roa_estimation"]

    return params