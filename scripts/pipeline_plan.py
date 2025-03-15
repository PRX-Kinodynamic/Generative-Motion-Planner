import mg_diffuse.utils as utils
from analize_model import analize_model
from estimate_roa import generate_and_analyze_runs
import sys
import argparse
from flow_matching.utils.manifolds import Euclidean, Sphere, FlatTorus, Product


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument("--exp_name", type=str, required=False, help="Experiment name")
    # parser.add_argument("--method_type", type=str, required=True, help="Method_type: diffussion or flowmatching")
    parser.add_argument("--dataset", type=str, required=False, help="dataset name")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--variation", type=str, required=False, help="Variation parameter")
    parser.add_argument("--run_traj", action="store_true", help="Run trajectory flag")
    parser.add_argument("--run_train", action="store_true", help="Run trainning flag")
    parser.add_argument("--run_roa", action="store_true", help="Run RoA flag")
    
    
    args = parser.parse_args()

    run_train = "--run_train" in sys.argv and not sys.argv.remove("--run_train")
    run_traj = "--run_traj" in sys.argv and not sys.argv.remove("--run_traj")
    run_roa = "--run_roa" in sys.argv and not sys.argv.remove("--run_roa")

    # save and remove --exp_name and --method
    # TODO make Parser more general
    if "--exp_name" in sys.argv:
        exp_name = args.exp_name
        sys.argv.remove("--exp_name")
        sys.argv.remove(f"{args.exp_name}")
    # method = args.method 
    # sys.argv.remove("--method")
    # sys.argv.remove(f"{args.method}")
    





    if run_train:
        # -----------------------------------------------------------------------------#
        # ---------------------------------- Train Config -----------------------------#
        # -----------------------------------------------------------------------------#

        class Parser(utils.Parser):
            dataset: str = "pendulum_lqr_5k"
            config: str = "config.pendulum_lqr_5k"
            variation: str = ""
            # method_type: str = "flowmatching"
        
        parser = Parser()
        args = parser.parse_args("method")

        # -----------------------------------------------------------------------------#
        # ---------------------------------- dataset ----------------------------------#
        # -----------------------------------------------------------------------------#

        manifold=Product(sphere_dim = args.sphere_dim, torus_dim = args.torus_dim, euclidean_dim = args.euclidean_dim)

        # dataset_config = utils.Config(
        #     args.loader,
        #     savepath=(args.savepath, "dataset_config.pkl"),
        #     dataset=args.dataset,
        #     horizon=args.horizon,
        #     normalizer=args.normalizer,
        #     preprocess_fns=args.trajectory_preprocess_fns,
        #     preprocess_kwargs=args.preprocess_kwargs,
        #     dataset_size=args.train_set_limit,
        #     use_padding=args.use_padding,
        #     max_path_length=args.max_path_length,
        #     stride=args.stride,
        #     manifold=manifold
        # )

        dataset_config = utils.Config(
            args.loader,
            savepath=(args.savepath, "dataset_config.pkl"),
            dataset=args.dataset,
            horizon=args.horizon,
            normalizer=args.normalizer,
            preprocess_fns=args.trajectory_preprocess_fns,
            preprocess_kwargs=args.preprocess_kwargs,
            dataset_size=args.train_set_limit,
            use_padding=args.use_padding,
            max_path_length=args.max_path_length,
            stride=args.stride,
            normalizer_params=args.normalizer_params,
            use_plans=args.use_plans,
            manifold=manifold
        )

        

        dataset = dataset_config()

        print('Dataset size:', len(dataset))

        observation_dim = dataset.observation_dim

        # # -----------------------------------------------------------------------------#
        # # ---------------------------- update and save args ---------------------------#
        # # -----------------------------------------------------------------------------#

        args.observation_dim = observation_dim
        args.dataset_size = len(dataset)
        # args.normalization_params = dataset.normalizer.params
        parser.save()

        # breakpoint()

        # raise ValueError("Stop here")

        # # -----------------------------------------------------------------------------#
        # # ------------------------------ model & trainer ------------------------------#
        # # -----------------------------------------------------------------------------#


        # sphere and torus have two features for dimension (cos, sin)
        features_dim = 2*args.sphere_dim + 2*args.torus_dim + args.euclidean_dim
        output_dim = args.sphere_dim + args.torus_dim + args.euclidean_dim  # dimension of the manifold

        assert output_dim == args.observation_dim  # observation_dim is the dim of the manifold

        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, "model_config.pkl"),
            horizon=args.horizon,
            transition_dim=features_dim,
            cond_dim=args.observation_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )

        method_config = utils.Config(
            args.method_type,
            savepath=(args.savepath, "method_config.pkl"),
            horizon=args.horizon,
            observation_dim=observation_dim,
            n_timesteps=args.method_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            ## loss weighting
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            device=args.device,
            manifold=manifold
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

        method = method_config(model)

        trainer = trainer_config(method, dataset)


        # # -----------------------------------------------------------------------------#
        # # ------------------------ test forward & backward pass -----------------------#
        # # -----------------------------------------------------------------------------#

        utils.report_parameters(model)

        print("Testing forward...", end=" ", flush=True)
        batch = utils.batchify(dataset[0])
        loss, _ = method.loss(*batch)
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
        # # --------------------------- end training  -----------------------------------#
        # # -----------------------------------------------------------------------------#

    if "exp_name" not in locals():  # add exp_name for path for visualizations and/or RoA
        exp_name = args.exp_name


    if run_traj:
        # # -----------------------------------------------------------------------------#
        # # --------------------------- save visualizations -----------------------------#
        # # -----------------------------------------------------------------------------#

        print("Saving visualizations...")

        analize_model(
            dataset=args.dataset,
            num_trajs=13387908,
            compare=False,
            show_traj_ends=True,
            exp_name=exp_name,
            model_state_name='best.pt',
            only_execute_next_step=False,
        )

    if run_roa:
        # # -----------------------------------------------------------------------------#
        # # --------------------------- generate runs for RoA estimation ----------------#
        # # -----------------------------------------------------------------------------#

        print("Generating runs for RoA estimation...")

        generate_and_analyze_runs(
            dataset=args.dataset,
            exp_name=args.exp_name,
            model_state_name='best.pt',
            generate_img=True,
        )