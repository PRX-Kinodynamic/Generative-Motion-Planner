import argparse
import os
import re
import pickle
from typing import Optional
import torch
import genMoPlan.utils as utils


def _select_checkpoint_name(experiments_path: str, model_state_name: Optional[str]) -> str:
    if model_state_name and model_state_name != "latest":
        return model_state_name

    interrupted_path = os.path.join(experiments_path, "interrupted.pt")
    if os.path.exists(interrupted_path):
        return "interrupted.pt"

    state_pattern = re.compile(r"state_(\d+)_epochs\\.pt$")
    max_state = -1
    chosen_state = None
    for fname in os.listdir(experiments_path):
        m = state_pattern.match(fname)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num > max_state:
                max_state = epoch_num
                chosen_state = fname
    if chosen_state is not None:
        return chosen_state

    best_path = os.path.join(experiments_path, "best.pt")
    if os.path.exists(best_path):
        return "best.pt"

    final_path = os.path.join(experiments_path, "final.pt")
    if os.path.exists(final_path):
        return "final.pt"

    raise FileNotFoundError(
        f"No checkpoint found in {experiments_path}. Expected one of: interrupted.pt, state_*_epochs.pt, best.pt, final.pt"
    )


def _load_classloader(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing config at {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description="Resume training from a saved experiment")
    ap.add_argument("--savepath", required=True, help="Path to existing experiment directory (contains args.json)")
    ap.add_argument("--model_state_name", default="latest", help="Checkpoint filename to resume from, or 'latest'")
    ap.add_argument("--device", default=None, help="Override device (e.g., cuda or cpu)")
    ap.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs to train for this resume run")
    args_cli = ap.parse_args()

    experiments_path = args_cli.savepath

    checkpoint_name = _select_checkpoint_name(experiments_path, args_cli.model_state_name)
    print(f"[ scripts/resume_train_trajectory ] Resuming from checkpoint: {checkpoint_name}")

    model_args = utils.load_model_args(experiments_path)

    device = args_cli.device or getattr(model_args, "device", "cuda")
    utils.set_device(device)
    print(f"Using device: {utils.DEVICE}\n")

    dataset_loader_pkl = os.path.join(experiments_path, "dataset_config.pkl")
    ml_model_loader_pkl = os.path.join(experiments_path, "ml_model_config.pkl")
    gen_model_loader_pkl = os.path.join(experiments_path, "gen_model_config.pkl")
    trainer_loader_pkl = os.path.join(experiments_path, "trainer_config.pkl")

    train_dataset_class_loader = _load_classloader(dataset_loader_pkl)
    ml_model_class_loader = _load_classloader(ml_model_loader_pkl)
    gen_model_class_loader = _load_classloader(gen_model_loader_pkl)
    trainer_class_loader = _load_classloader(trainer_loader_pkl)

    print(f"[ scripts/resume_train_trajectory ] Loading dataset")
    train_dataset = train_dataset_class_loader()
    print(f"[ scripts/resume_train_trajectory ] Training Data Size: {len(train_dataset)}")

    val_dataset = None
    val_dataset_size = getattr(model_args, "val_dataset_size", None)
    if val_dataset_size is not None:
        print(f"[ scripts/resume_train_trajectory ] Loading validation dataset")
        # Reuse the saved train dataset loader's config to build a validation loader with minimal duplication
        val_loader_kwargs = dict(train_dataset_class_loader._dict)
        val_loader_kwargs["dataset_size"] = val_dataset_size
        val_loader_kwargs["is_validation"] = True
        val_dataset_class_loader = utils.ClassLoader(
            train_dataset_class_loader._class,
            device=device,
            **val_loader_kwargs,
        )
        val_dataset = val_dataset_class_loader()
        print(f"[ scripts/resume_train_trajectory ] Validation Data Size: {len(val_dataset)}")

    ml_model = ml_model_class_loader()
    gen_model = gen_model_class_loader(ml_model)

    trainer: utils.Trainer = trainer_class_loader(gen_model, model_args, train_dataset, val_dataset)

    if args_cli.num_epochs is not None:
        trainer.num_epochs = int(args_cli.num_epochs)



    trainer.load(model_state_name=checkpoint_name)

    checkpoint_path = os.path.join(experiments_path, checkpoint_name)
    resume_blob = torch.load(checkpoint_path, map_location=torch.device(device))
    if "optimizer" in resume_blob:
        try:
            trainer.optimizer.load_state_dict(resume_blob["optimizer"])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    if trainer.lr_scheduler is not None and "lr_scheduler" in resume_blob:
        lr_s = resume_blob["lr_scheduler"]
        try:
            if "current_step" in lr_s:
                trainer.lr_scheduler.current_step = lr_s["current_step"]
            # If num_epochs was overridden, adjust total planned steps to align schedule with resumed run
            if args_cli.num_epochs is not None:
                trainer.lr_scheduler.total_steps = (
                    trainer.lr_scheduler.current_step + trainer.num_epochs * trainer.num_steps_per_epoch
                )
        except Exception as e:
            print(f"Warning: Could not restore LR scheduler state: {e}")

    torch.set_num_threads(getattr(model_args, "num_workers", 4))
    trainer.train()


if __name__ == "__main__":
    main()


