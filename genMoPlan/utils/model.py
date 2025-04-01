from os import path

from typing import Tuple
import torch

from genMoPlan.models.generative.abs_gen_model import GenerativeModel
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


def load_model(experiments_path: str, model_state_name: str, verbose: bool = False) -> Tuple[GenerativeModel, JSONArgs]:
    model_args = load_model_args(experiments_path)
    model_path = path.join(experiments_path, model_state_name)

    if verbose:
        print(f"[ utils/model ] Loading model from {model_path}")

    model_state_dict = torch.load(model_path, weights_only=False)
    diff_model_state = model_state_dict["model"]

    ml_model_class = import_class(model_args.model)
    method_class = import_class(model_args.method)

    ml_model = ml_model_class(
        prediction_length=model_args.horizon_length + model_args.history_length,
        input_dim=model_args.observation_dim,
        output_dim=model_args.observation_dim,
        query_dim=0 if model_args.is_history_conditioned else model_args.observation_dim,
        **model_args.model_kwargs,
    ).to(model_args.device)

    method_model = method_class(
        model=ml_model,
        input_dim=model_args.observation_dim,
        output_dim=model_args.observation_dim,
        prediction_length=model_args.horizon_length + model_args.history_length,
        history_length=model_args.history_length,
        clip_denoised=model_args.clip_denoised,
        loss_type=model_args.loss_type,
        loss_weights=model_args.loss_weights,
        loss_discount=model_args.loss_discount,
        action_indices=model_args.action_indices,
        has_query=model_args.has_query,
        **model_args.method_kwargs,
    ).to(model_args.device)

    # Load model state dict
    method_model.load_state_dict(diff_model_state)

    return method_model, model_args

