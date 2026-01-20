import torch
import genMoPlan.utils as utils
from genMoPlan.adaptive_training import AdaptiveTrainer
from genMoPlan.models import GenerativeModel, TemporalModel

parser = utils.Parser()
args = parser.parse_args()

if 'adaptive_training' not in args.variations:
    raise ValueError("Variations must include 'adaptive_training'")

utils.set_device(args.device)
print(f"Using device: {utils.DEVICE}\n")

# Get system object - single source of truth for system-specific parameters
system = args.system

if system.manifold is not None:
    ml_model_input_dim = system.manifold.compute_feature_dim(
        system.state_dim,
        n_fourier_features=args.method_kwargs.get("n_fourier_features", 1)
    )
else:
    ml_model_input_dim = system.state_dim

ml_model_class_loader = utils.ClassLoader(
    args.model,
    savepath=(args.savepath, "ml_model_config.pkl"),
    prediction_length=system.history_length + system.horizon_length,
    input_dim=ml_model_input_dim,
    output_dim=system.state_dim,
    query_dim=0 if args.is_history_conditioned else system.state_dim,
    **args.model_kwargs,
    device=args.device,
)

gen_model_class_loader = utils.ClassLoader(
    args.method,
    savepath=(args.savepath, "gen_model_config.pkl"),
    system=system,  # Pass system instead of individual attributes
    prediction_length=system.history_length + system.horizon_length,
    history_length=system.history_length,
    clip_denoised=args.clip_denoised,
    loss_type=args.loss_type,
    has_local_query=args.has_local_query,
    has_global_query=args.has_global_query,
    val_seed=args.val_seed,
    use_history_mask=args.use_history_mask,
    **args.method_kwargs,
    device=args.device,
)


# # -----------------------------------------------------------------------------#
# # -------------------------------- instantiate --------------------------------#
# # -----------------------------------------------------------------------------#

ml_model: TemporalModel = ml_model_class_loader()

gen_model: GenerativeModel = gen_model_class_loader(ml_model)

trainer_class_loader = utils.ClassLoader(
    AdaptiveTrainer,
    model=gen_model,
    args=args,
    **args.adaptive_training_kwargs,
)

trainer: AdaptiveTrainer = trainer_class_loader()

# # -----------------------------------------------------------------------------#
# # ------------------------ test forward & backward pass -----------------------#
# # -----------------------------------------------------------------------------#

utils.report_parameters(ml_model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(trainer.val_dataset[0])
loss, _ = gen_model.loss(*batch)
loss.backward()
print("✓")

# -----------------------------------------------------------------------------#
# ------------------------------ save configs ---------------------------------#
# -----------------------------------------------------------------------------#

parser.save(args)
ml_model_class_loader.save()
gen_model_class_loader.save()
trainer_class_loader.save()


# # -----------------------------------------------------------------------------#
# # --------------------------------- main loop ---------------------------------#
# # -----------------------------------------------------------------------------#
torch.set_num_threads(args.num_workers)
trainer.run()
