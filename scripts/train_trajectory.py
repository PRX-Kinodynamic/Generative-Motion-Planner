import timeit
from math import ceil

import torch
import genMoPlan.utils as utils
from genMoPlan.models import GenerativeModel, TemporalModel
from scripts.estimate_roa import estimate_roa
from scripts.viz_model import visualize_generated_trajectories

parser = utils.TrainingParser()
args = parser.parse_args()

utils.set_device(args.device)
print(f"Using device: {utils.DEVICE}\n")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

train_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    dataset=args.dataset,
    horizon_length=args.horizon_length,
    history_length=args.history_length,
    stride=args.stride,
    observation_dim=args.observation_dim,
    trajectory_normalizer=args.trajectory_normalizer,
    plan_normalizer=args.plan_normalizer,
    normalizer_params=args.normalizer_params,
    trajectory_preprocess_fns=args.trajectory_preprocess_fns,
    plan_preprocess_fns=args.plan_preprocess_fns,
    preprocess_kwargs=args.preprocess_kwargs,
    dataset_size=args.train_dataset_size,
    use_history_padding=args.use_history_padding,
    use_horizon_padding=args.use_horizon_padding,
    use_plan=args.use_plan,
    is_history_conditioned=args.is_history_conditioned,
    **args.safe_get("dataset_kwargs", {}),
)

args.val_dataset_size = None if not hasattr(args, 'val_dataset_size') else args.val_dataset_size

if args.val_dataset_size is not None:
    print("Loading validation dataset...\n")
    val_dataset_config = utils.Config(
        args.loader,
        dataset=args.dataset,
        horizon_length=args.horizon_length,
        history_length=args.history_length,
        stride=args.stride,
        observation_dim=args.observation_dim,
        trajectory_normalizer=args.trajectory_normalizer,
        plan_normalizer=args.plan_normalizer,
        normalizer_params=args.normalizer_params,
        trajectory_preprocess_fns=args.trajectory_preprocess_fns,
        plan_preprocess_fns=args.plan_preprocess_fns,
        preprocess_kwargs=args.preprocess_kwargs,
        dataset_size=args.val_dataset_size,
        use_history_padding=args.use_history_padding,
        use_horizon_padding=args.use_horizon_padding,
        use_plan=args.use_plan,
        is_history_conditioned=args.is_history_conditioned,
        is_validation=True,
        **args.safe_get("dataset_kwargs", {}),
    )

print(f"[ scripts/train_trajectory ] Loading dataset")
train_dataset = train_dataset_config()
print(f"[ scripts/train_trajectory ] Training Data Size: {len(train_dataset)}")

if args.val_dataset_size is not None:
    print(f"[ scripts/train_trajectory ] Loading validation dataset")
    val_dataset = val_dataset_config()
    print(f"[ scripts/train_trajectory ] Validation Data Size: {len(val_dataset)}")

observation_dim = args.observation_dim


# -----------------------------------------------------------------------------#
# ------------------------------ manifold -------------------------------------#
# -----------------------------------------------------------------------------#

if args.manifold is not None:
    manifold = utils.ManifoldWrapper(args.manifold)
    args.manifold = manifold
    ml_model_input_dim = manifold.compute_feature_dim(observation_dim, n_fourier_features=args.model_kwargs.get("n_fourier_features", 1))
else:
    manifold = None
    ml_model_input_dim = observation_dim

# # -----------------------------------------------------------------------------#
# # ------------------------------ model & trainer ------------------------------#
# # -----------------------------------------------------------------------------#

ml_model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "ml_model_config.pkl"),
    prediction_length=args.horizon_length + args.history_length,
    input_dim=ml_model_input_dim,
    output_dim=observation_dim,
    query_dim=0 if args.is_history_conditioned else observation_dim,
    **args.model_kwargs,
    device=args.device,
)

gen_model_config = utils.Config(
    args.method,
    savepath=(args.savepath, "gen_model_config.pkl"),
    input_dim=observation_dim,
    output_dim=observation_dim,
    prediction_length=args.horizon_length + args.history_length,
    history_length=args.history_length,
    clip_denoised=args.clip_denoised,
    loss_type=args.loss_type,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    action_indices=args.action_indices,
    has_local_query=args.has_local_query,
    has_global_query=args.has_global_query,
    manifold=manifold,
    val_seed=args.val_seed,
    **args.method_kwargs,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    batch_size=args.batch_size,
    min_num_batches_per_epoch=args.min_num_batches_per_epoch,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    validation_kwargs=args.validation_kwargs,
    val_batch_size=args.val_batch_size,
    num_epochs=args.num_epochs,
    patience=args.patience,
    min_delta=args.min_delta,
    early_stopping=args.early_stopping,
    ema_decay=args.ema_decay,
    save_freq=args.save_freq,
    log_freq=args.log_freq,
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    n_reference=args.n_reference,
    method=args.method,
    exp_name=args.exp_name,
    num_workers=args.num_workers,
    device=args.device,
    seed=args.seed,
)

# # -----------------------------------------------------------------------------#
# # -------------------------------- instantiate --------------------------------#
# # -----------------------------------------------------------------------------#

ml_model: TemporalModel = ml_model_config()

gen_model: GenerativeModel = gen_model_config(ml_model)

trainer: utils.Trainer = trainer_config(gen_model, train_dataset, val_dataset)

# # -----------------------------------------------------------------------------#
# # ---------------------------- update and save args ---------------------------#
# # -----------------------------------------------------------------------------#

args.observation_dim = observation_dim
args.dataset_size = len(train_dataset)
args.num_batches_per_epoch = trainer.num_batches_per_epoch
args.num_steps_per_epoch = ceil(trainer.num_batches_per_epoch / trainer.gradient_accumulate_every)

# # -----------------------------------------------------------------------------#
# # ------------------------ test forward & backward pass -----------------------#
# # -----------------------------------------------------------------------------#

utils.report_parameters(ml_model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(train_dataset[0])
loss, _ = gen_model.loss(*batch)
loss.backward()
print("✓")

# -----------------------------------------------------------------------------#
# ------------------------------ profile model --------------------------------#
# -----------------------------------------------------------------------------#


gen_model.eval()
sample = train_dataset[0]
print(f"[ scripts/train_trajectory ] Forward pass time: {timeit.timeit(lambda: gen_model.forward(cond=sample.conditions, global_query=sample.global_query, local_query=sample.local_query), number=10) / 10} seconds")
gen_model.train()

# -----------------------------------------------------------------------------#
# ------------------------------ save configs ---------------------------------#
# -----------------------------------------------------------------------------#

parser.save(args)
train_dataset_config.save()
ml_model_config.save()
gen_model_config.save()
trainer_config.save()


# # -----------------------------------------------------------------------------#
# # --------------------------------- main loop ---------------------------------#
# # -----------------------------------------------------------------------------#
torch.set_num_threads(args.num_workers)
trainer.train()

if args.no_inference:
    exit()

# -----------------------------------------------------------------------------#
# ------------------------------visualize trajectories-------------------------#
# -----------------------------------------------------------------------------#

try:
    visualize_generated_trajectories(
        args.dataset,
        num_trajs=1000,
        model_paths=args.savepath,
        model_state_name="best.pt",
        observation_dim=observation_dim,
    )
except Exception as e:
    print(f"Error visualizing trajectories: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error traceback:")
    import traceback
    traceback.print_exc()


# -----------------------------------------------------------------------------#
# ---------------------------------- estimate roa -----------------------------#
# -----------------------------------------------------------------------------#

try:
    estimate_roa(
        dataset=args.dataset,
        model_state_name="best.pt",
        model_path=args.savepath,
        num_batches=3,
    )
except Exception as e:
    print(f"Error estimating ROA: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error traceback:")
    import traceback
    traceback.print_exc()
