import mg_diffuse.utils as utils
from viz_model import visualize_generated_trajectories
from estimate_roa import generate_and_analyze_runs


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

if __name__ == '__main__':
    class Parser(utils.Parser):
        dataset: str = "pendulum_lqr_5k"
        config: str = "config.pendulum_lqr_5k"
        variation: str = ""

    parser = Parser()
    args = parser.parse_args("diffusion")




    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        preprocess_kwargs=args.preprocess_kwargs,
        dataset_size=args.train_set_limit,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
    )

    dataset = dataset_config()

    print('Dataset size:', len(dataset))

    observation_dim = dataset.observation_dim


    # # -----------------------------------------------------------------------------#
    # # ---------------------------- update and save args ---------------------------#
    # # -----------------------------------------------------------------------------#

    args.observation_dim = observation_dim
    args.dataset_size = len(dataset)
    args.normalization_params = dataset.normalizer.params
    parser.save()

    # raise ValueError("Stop here")

    # # -----------------------------------------------------------------------------#
    # # ------------------------------ model & trainer ------------------------------#
    # # -----------------------------------------------------------------------------#

    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        horizon=args.horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        dim_mults=args.dim_mults,
        attention=args.attention,
        device=args.device,
    )

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, "diffusion_config.pkl"),
        horizon=args.horizon,
        observation_dim=observation_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        ## loss weighting
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=args.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, "trainer_config.pkl"),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=args.save_freq,
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
    )

    # # -----------------------------------------------------------------------------#
    # # -------------------------------- instantiate --------------------------------#
    # # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset)


    # # -----------------------------------------------------------------------------#
    # # ------------------------ test forward & backward pass -----------------------#
    # # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print("✓")


    # # -----------------------------------------------------------------------------#
    # # --------------------------------- main loop ---------------------------------#
    # # -----------------------------------------------------------------------------#

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    min_delta = 1e-4  # Minimum improvement to qualify as progress
    no_improve_counter = 0

    best_loss = float('inf')

    try:
        for i in range(n_epochs):
            print(f"Epoch {i} / {n_epochs} | {args.savepath}")
            best_loss = trainer.train(n_train_steps=args.n_steps_per_epoch, best_loss=best_loss)

            # Evaluate on validation set
            current_val_loss = trainer.validate()

            if current_val_loss is not None:
                print(f"Validation Loss: {current_val_loss:.6f}")

                # Check if validation loss has improved sufficiently
                if best_val_loss - current_val_loss > min_delta:
                    best_val_loss = current_val_loss
                    no_improve_counter = 0
                    trainer.save('best_state')  # Save the best model
                else:
                    no_improve_counter += 1
                    print(f"No improvement for {no_improve_counter} epoch(s).")

                if no_improve_counter >= patience:
                    print("Early stopping triggered due to convergence.")
                    break

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        
    trainer.save('final')

    # # -----------------------------------------------------------------------------#
    # # --------------------------- save visualizations -----------------------------#
    # # -----------------------------------------------------------------------------#

    print("Saving visualizations...")
    
    visualize_generated_trajectories(
        dataset=args.dataset,
        num_trajs=1000,
        compare=False,
        show_traj_ends=False,
        exp_name=args.exp_name,
        model_state_name='best.pt',
        only_execute_next_step=False,
    )
    
    # # -----------------------------------------------------------------------------#
    # # --------------------------- generate runs for RoA estimation ----------------#
    # # -----------------------------------------------------------------------------#
    
    print("Generating runs for RoA estimation...")
    path_prefix, actual_exp_name = args.exp_name.split("/")
    generate_and_analyze_runs(
        dataset=args.dataset,
        exp_name=actual_exp_name,
        model_state_name='best.pt',
        generate_img=True,
        path_prefix=path_prefix,
    )