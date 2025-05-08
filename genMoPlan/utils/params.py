import importlib


def load_inference_params(dataset):
    config = f'config.{dataset}'
    module = importlib.import_module(config)
    params = getattr(module, "base")["inference"]

    return params