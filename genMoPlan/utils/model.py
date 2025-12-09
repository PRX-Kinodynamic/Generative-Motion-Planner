from os import path

from typing import Tuple
import torch
from torch import nn

from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.utils import JSONArgs, import_class

def get_method_name(model_args: JSONArgs) -> str:
    method_name = None

    if hasattr(model_args, "method_name"):
        method_name = model_args.method_name
    else:
        if model_args.method == "models.generative.Diffusion":
            method_name = "diffusion"
        elif model_args.method == "models.generative.FlowMatching":
            method_name = "flow_matching"
        
    return method_name


def load_model_args(experiments_path):
    model_args_path = path.join(experiments_path, "args.json")
    return JSONArgs(model_args_path)


def load_model(
        experiments_path: str,
        device: str,
        model_state_name: str = 'best.pt',
        verbose: bool = False,
        inference_params: dict = None,
        load_ema: bool = False,
    ) -> Tuple[GenerativeModel, JSONArgs]:
    model_args = load_model_args(experiments_path)
    model_path = path.join(experiments_path, model_state_name)

    if verbose:
        print(f"[ utils/model ] Loading model from {model_path} | {'Loading EMA Weights' if load_ema else 'Loading Raw Weights'}")

    model_state_dict = torch.load(model_path, weights_only=False, map_location=torch.device(device))
    diff_model_state = model_state_dict["model"] if not load_ema else model_state_dict["ema"]

    ml_model_class = import_class(model_args.model, verbose)
    method_class = import_class(model_args.method, verbose)

    if model_args.manifold is not None:
        ml_model_input_dim = model_args.manifold.compute_feature_dim(model_args.observation_dim, n_fourier_features=model_args.method_kwargs.get("n_fourier_features", 1))

        if inference_params is not None and "manifold_unwrap_fns" in inference_params:
            model_args.manifold.manifold_unwrap_fns = inference_params["manifold_unwrap_fns"]
            model_args.manifold.manifold_unwrap_kwargs = inference_params["manifold_unwrap_kwargs"]
    else:
        ml_model_input_dim = model_args.observation_dim

    ml_model = ml_model_class(
        prediction_length=model_args.horizon_length + model_args.history_length,
        input_dim=ml_model_input_dim,
        output_dim=model_args.observation_dim,
        query_dim=0 if model_args.is_history_conditioned else model_args.observation_dim,
        verbose=verbose,
        **model_args.model_kwargs,
    ).to(device)

    method_model: GenerativeModel = method_class(
        model=ml_model,
        input_dim=model_args.observation_dim,
        output_dim=model_args.observation_dim,
        prediction_length=model_args.horizon_length + model_args.history_length,
        history_length=model_args.history_length,
        clip_denoised=model_args.clip_denoised,
        loss_type=model_args.loss_type,
        action_indices=model_args.action_indices,
        has_local_query=model_args.has_local_query,
        has_global_query=model_args.has_global_query,
        manifold=model_args.manifold,
        verbose=verbose,
        **model_args.method_kwargs,
    ).to(device)

    # Load model state dict
    method_model.load_state_dict(diff_model_state, strict=False)

    return method_model, model_args


def get_normalizer_params(model_args, normalizer_type: str = "trajectory"):
    normalizer_params = None

    if hasattr(model_args, "normalizer_params"):
        normalizer_params = model_args.normalizer_params[normalizer_type]
    elif hasattr(model_args, "normalization_params"):
        normalizer_params = model_args.normalization_params
    else:
        raise ValueError("Normalizer params not found in model_args")

    return normalizer_params


def get_parameter_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            # --- module-type rules ---
            if isinstance(module, whitelist_weight_modules) and param_name == "weight":
                decay.add(full_name); continue
            if isinstance(module, blacklist_weight_modules):
                no_decay.add(full_name); continue

            # --- generic rules ---
            if param_name.endswith("bias"):
                no_decay.add(full_name); continue

            # --- special cases in your model ---
            # learned absolute PE
            if "positional_encoding" in full_name:
                no_decay.add(full_name); continue
            # LayerScale scalars
            if full_name.endswith("alpha_attn") or full_name.endswith("alpha_ff"):
                no_decay.add(full_name); continue

            # PyTorch MultiheadAttention internals
            # in_proj_weight should decay; in_proj_bias should not
            if full_name.endswith("in_proj_weight") or full_name.endswith("out_proj.weight"):
                decay.add(full_name); continue
            if full_name.endswith("in_proj_bias") or full_name.endswith("out_proj.bias"):
                no_decay.add(full_name); continue

            # default: decay
            decay.add(full_name)

    # sanity: no overlaps
    intersect = decay & no_decay
    if len(intersect) > 0:
        raise ValueError(f"param group conflict: {intersect}")

    # map names to tensors
    param_dict = {n: p for n, p in model.named_parameters()}
    decay_params = [param_dict[n] for n in sorted(list(decay))]
    no_decay_params = [param_dict[n] for n in sorted(list(no_decay))]

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]