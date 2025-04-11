from typing import List
from math import ceil
import genMoPlan.utils as utils
from genMoPlan.models import GenerativeModel, TemporalModel
from scripts.estimate_roa import estimate_roa
from scripts.viz_model import visualize_generated_trajectories


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

if __name__ == '__main__':
    parser = utils.TrainingParser()
    args = parser.parse_args()





    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset=args.dataset,
        horizon_length=args.horizon_length,
        history_length=args.history_length,
        stride=args.stride,
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
        dt=args.dt if hasattr(args, 'dt') else None,
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
            dt=args.dt if hasattr(args, 'dt') else None,
            is_validation=True,
        )

    print(f"[ scripts/train_trajectory ] Loading dataset")
    dataset = dataset_config()
    print(f"[ scripts/train_trajectory ] Training Data Size: {len(dataset)}")

    if args.val_dataset_size is not None:
        print(f"[ scripts/train_trajectory ] Loading validation dataset")
        val_dataset = val_dataset_config()
        print(f"[ scripts/train_trajectory ] Validation Data Size: {len(val_dataset)}")

    observation_dim = dataset.observation_dim


    # # -----------------------------------------------------------------------------#
    # # ------------------------------ model & trainer ------------------------------#
    # # -----------------------------------------------------------------------------#

    ml_model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "ml_model_config.pkl"),
        prediction_length=args.horizon_length + args.history_length,
        input_dim=observation_dim,
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
        has_query=args.has_query,
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
        val_num_batches=args.val_num_batches,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stopping=args.early_stopping,
        ema_decay=args.ema_decay,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
        method=args.method,
        exp_name=args.exp_name,
    )

    # # -----------------------------------------------------------------------------#
    # # -------------------------------- instantiate --------------------------------#
    # # -----------------------------------------------------------------------------#

    ml_model: TemporalModel = ml_model_config()

    gen_model: GenerativeModel = gen_model_config(ml_model)

    trainer: utils.Trainer = trainer_config(gen_model, dataset, val_dataset)

    # # -----------------------------------------------------------------------------#
    # # ---------------------------- update and save args ---------------------------#
    # # -----------------------------------------------------------------------------#

    args.observation_dim = observation_dim
    args.dataset_size = len(dataset)
    args.num_batches_per_epoch = trainer.num_batches_per_epoch
    args.num_steps_per_epoch = ceil(trainer.num_batches_per_epoch / trainer.gradient_accumulate_every)

    # # -----------------------------------------------------------------------------#
    # # ------------------------ test forward & backward pass -----------------------#
    # # -----------------------------------------------------------------------------#

    utils.report_parameters(ml_model)

    print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = gen_model.loss(*batch)
    loss.backward()
    print("✓")

    # -----------------------------------------------------------------------------#
    # ------------------------------ save configs --------------------------------#
    # -----------------------------------------------------------------------------#

    parser.save(args)
    dataset_config.save()
    ml_model_config.save()
    gen_model_config.save()
    trainer_config.save()


    # # -----------------------------------------------------------------------------#
    # # --------------------------------- main loop ---------------------------------#
    # # -----------------------------------------------------------------------------#

    trainer.train()

    # -----------------------------------------------------------------------------#
    # ------------------------------visualize trajectories-------------------------#
    # -----------------------------------------------------------------------------#

    try:
        visualize_generated_trajectories(
            args.dataset,
            num_trajs=1000,
            compare=False,
            show_traj_ends=False,
            model_paths=args.savepath,
            model_state_name="best.pt",
            only_execute_next_step=True,
        )
    except Exception as e:
        print(f"Error visualizing trajectories: {e}")


    # -----------------------------------------------------------------------------#
    # ---------------------------------- estimate roa -----------------------------#
    # -----------------------------------------------------------------------------#

    try:
        estimate_roa(
            dataset,
            model_state_name="best.pt",
            model_paths=args.savepath,
        )
    except Exception as e:
        print(f"Error estimating ROA: {e}")
