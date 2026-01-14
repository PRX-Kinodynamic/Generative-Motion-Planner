import importlib


def load_inference_params(dataset):
    config = f'config.{dataset}'
    module = importlib.import_module(config)
    params = dict(getattr(module, "base")["inference"])

    # Also load system-provided inference config if available
    if hasattr(module, "get_system"):
        try:
            system = module.get_system()
            system_inference_config = system.get_inference_config()
            # Merge system config into params (system provides defaults, config overrides)
            for key, value in system_inference_config.items():
                if key not in params:
                    params[key] = value
        except Exception:
            # If system creation fails, continue with just config-provided params
            pass

    return params