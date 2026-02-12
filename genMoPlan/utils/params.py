import importlib


def load_inference_params(dataset, *, system):
    config = f'config.{dataset}'
    module = importlib.import_module(config)
    params = dict(getattr(module, "base")["inference"])

    if system is None:
        raise ValueError("load_inference_params requires a `system` instance.")

    # Merge system config into params (system provides defaults, config overrides)
    system_inference_config = system.get_inference_config()
    for key, value in system_inference_config.items():
        if key not in params:
            params[key] = value

    return params