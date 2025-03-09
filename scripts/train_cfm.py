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
    args = parser.parse_args("flow_matching")




    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset=args.dataset,
        horizon=args.horizon,
        history_length=args.history_length,
        trajectory_normalizer=args.trajectory_normalizer,
        plan_normalizer=args.plan_normalizer,
        normalizer_params=args.normalizer_params,
        trajectory_preprocess_fns=args.trajectory_preprocess_fns,
        plan_preprocess_fns=args.plan_preprocess_fns,
        preprocess_kwargs=args.preprocess_kwargs,
        dataset_size=args.train_set_limit,
        dt=args.dt,
        use_history_padding=args.use_history_padding,
    )

    dataset = dataset_config()