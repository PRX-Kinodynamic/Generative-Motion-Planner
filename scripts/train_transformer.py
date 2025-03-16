import mg_diffuse.utils as utils
from viz_model import visualize_generated_trajectories
from estimate_roa import generate_and_analyze_runs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

if __name__ == '__main__':
    class Parser(utils.Parser):
        dataset: str = "acrobot"
        config: str = "config.transformer_cfm"
        variation: str = ""

    parser = Parser()
    args = parser.parse_args("direct_prediction")

    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset=args.dataset,
        horizon=args.horizon,
        horizon_stride=args.horizon_stride,
        history_length=args.history_length,
        history_stride=args.history_stride,
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

    print('Dataset size:', len(dataset))

    plan_dim = dataset.plan_dim
    observation_dim = dataset.observation_dim

    # Create dataloader
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )


    # -----------------------------------------------------------------------------#
    # ---------------------------- update and save args ---------------------------#
    # -----------------------------------------------------------------------------#

    args.observation_dim = observation_dim
    args.plan_dim = plan_dim
    args.dataset_size = len(dataset)
    parser.save()


    # -----------------------------------------------------------------------------#
    # ------------------------------ model setup ----------------------------------#
    # -----------------------------------------------------------------------------#

    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        horizon=args.horizon,
        output_dim=plan_dim,
        cond_dim=observation_dim,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        cross_attention=args.cross_attention,
        device=args.device,
    )

    model = model_config()
    model = model.to(args.device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Setup loss function
    if args.loss_type == 'l1':
        criterion = nn.L1Loss()
    elif args.loss_type == 'l2' or args.loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    # EMA model setup
    ema_model = model_config()
    ema_model = ema_model.to(args.device)
    
    # Copy initial weights
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.copy_(param.data)
    
    # Disable gradient calculation for EMA model
    for param in ema_model.parameters():
        param.requires_grad = False


    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    print("Testing forward...", end=" ", flush=True)
    batch = next(iter(train_loader))
    plans, history = batch.plans, batch.history
    plans = plans.to(args.device)
    history = history.to(args.device)
    
    # Forward pass - direct prediction from history only
    outputs = model(cond=history)
    loss = criterion(outputs, plans)
    loss.backward()
    print("✓")


    # -----------------------------------------------------------------------------#
    # --------------------------------- training ----------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    
    best_val_loss = float('inf')
    patience = args.patience  # Number of epochs to wait for improvement
    min_delta = args.min_delta  # Minimum improvement to qualify as progress
    no_improve_counter = 0
    
    # Create checkpoint directory
    os.makedirs(os.path.join(args.savepath, 'checkpoints'), exist_ok=True)
    
    try:
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            
            # Training loop
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}") as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    plans, history = batch.plans, batch.history
                    plans = plans.to(args.device)
                    history = history.to(args.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass - direct prediction from history only
                    outputs = model(cond=history)
                    
                    # Calculate loss
                    loss = criterion(outputs, plans)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    # Update EMA model
                    with torch.no_grad():
                        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                            ema_param.data = args.ema_decay * ema_param.data + (1 - args.ema_decay) * param.data
                    
                    # Update progress bar
                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=epoch_loss/(batch_idx+1))
                    
                    # Save model periodically
                    if (batch_idx + 1) % args.save_freq == 0:
                        model_path = os.path.join(args.savepath, 'checkpoints', f'model_epoch{epoch+1}_batch{batch_idx+1}.pt')
                        torch.save({
                            'epoch': epoch,
                            'batch': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'ema_model_state_dict': ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                        }, model_path)
                        print(f"Model saved at {model_path}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    plans, history = batch.plans, batch.history
                    plans = plans.to(args.device)
                    history = history.to(args.device)
                    
                    # Forward pass - direct prediction from history only
                    outputs = model(cond=history)
                    
                    # Calculate loss
                    loss = criterion(outputs, plans)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Check for improvement
            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                no_improve_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(args.savepath, 'best.pt'))
                print(f"Saved best model with val_loss: {val_loss:.6f}")
            else:
                no_improve_counter += 1
                print(f"No improvement for {no_improve_counter} epoch(s).")
            
            # Early stopping
            if no_improve_counter >= patience:
                print("Early stopping triggered due to convergence.")
                break
    
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': best_val_loss,
    }, os.path.join(args.savepath, 'final.pt'))
    print("Saved final model.")

