import mg_diffuse.utils as utils
from viz_model import visualize_generated_trajectories
from estimate_roa import generate_and_analyze_runs


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

if __name__ == '__main__':
    class Parser(utils.Parser):
        dataset: str = "acrobot"
        config: str = "config.acrobot_traj"
        variation: str = ""

    parser = Parser()
    args = parser.parse_args("flow_matching")




    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset=args.dataset,
        horizon_length=args.horizon_length,
        history_length=args.history_length,
        horizon_stride=args.horizon_stride,
        history_stride=args.history_stride,
        trajectory_normalizer=args.trajectory_normalizer,
        normalizer_params=args.normalizer_params,
        trajectory_preprocess_fns=args.trajectory_preprocess_fns,
        preprocess_kwargs=args.preprocess_kwargs,
        dataset_size=args.train_set_limit,
        dt=args.dt,
        use_horizon_padding=args.use_horizon_padding,
        use_history_padding=args.use_history_padding,
        use_plan=args.use_plan,
    )

    dataset = dataset_config()

    observation_dim = dataset.observation_dim

    print('Dataset size:', len(dataset))

    # # -----------------------------------------------------------------------------#
    # # ---------------------------- update and save args ---------------------------#
    # # -----------------------------------------------------------------------------#

    args.observation_dim = observation_dim
    args.dataset_size = len(dataset)
    args.normalization_params = dataset.trajectory_normalizer.params
    parser.save()

    # raise ValueError("Stop here")

    # # -----------------------------------------------------------------------------#
    # # ------------------------------ model & trainer ------------------------------#
    # # -----------------------------------------------------------------------------#

    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        horizon_length=args.horizon_length + args.history_length,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        device=args.device,
        **args.model_kwargs,
    )

    flow_matching_config = utils.Config(
        args.flow_matching,
        savepath=(args.savepath, "flow_matching_config.pkl"),
        observation_dim=observation_dim,
        history_length=args.history_length,
        horizon_length=args.horizon_length,
        clip_denoised=args.clip_denoised,
        loss_type=args.loss_type,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        action_indices=args.action_indices,
        device=args.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, "trainer_config.pkl"),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        save_freq=args.save_freq,
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
    )

    # # -----------------------------------------------------------------------------#
    # # -------------------------------- instantiate --------------------------------#
    # # -----------------------------------------------------------------------------#

    model = model_config()

    flow_matching = flow_matching_config(model)

    trainer = trainer_config(flow_matching, dataset)


    # # -----------------------------------------------------------------------------#
    # # ------------------------ test forward & backward pass -----------------------#
    # # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = flow_matching.loss(*batch)
    loss.backward()
    print("✓")


    # # -----------------------------------------------------------------------------#
    # # --------------------------------- main loop ---------------------------------#
    # # -----------------------------------------------------------------------------#

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    best_loss = float('inf')

    try:
        for i in range(n_epochs):
            print(f"Epoch {i} / {n_epochs} | {args.savepath}")
            best_loss = trainer.train(n_train_steps=args.n_steps_per_epoch, best_loss=best_loss)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    trainer.save('final')