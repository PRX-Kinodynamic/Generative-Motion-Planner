from collections import defaultdict
from math import ceil
import os
import copy
from typing import List
import numpy as np
import torch
import multiprocessing
import matplotlib.pyplot as plt

from .arrays import batch_to_device
from .timer import Timer
from .model import GenerativeModel

def cycle(dl):
    while True:
        for data in dl:
            yield data

# --- Plotting Function ---
def plot_losses(losses: dict, filepath: str, title: str):
    """Plots losses in both linear and log scale and saves the figure."""
    try:
        # Check if there are any losses to plot
        if not losses or not any(losses.values()):
            print(f"No data provided for plotting {title}. Skipping plot generation.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Determine x-axis label based on title
        xlabel = 'Step' if 'Training' in title else 'Epoch'
        
        # Linear scale plot
        for key, loss_list in losses.items():
            if loss_list:  # Check if the list is not empty
                steps = range(len(loss_list))
                ax1.plot(steps, loss_list, label=key)  # Plot each loss type
        ax1.set_title(f"{title} (Linear Scale)")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Loss')
        ax1.legend()  # Add legend to identify lines
        ax1.grid(True)
        
        # Log scale plot
        for key, loss_list in losses.items():
            if loss_list:  # Check if the list is not empty
                steps = range(len(loss_list))
                # Filter out non-positive values for log scale if necessary
                positive_steps = [s for s, l in zip(steps, loss_list) if l > 0]
                positive_losses = [l for l in loss_list if l > 0]
                if positive_losses:
                    ax2.plot(positive_steps, positive_losses, label=key)  # Plot each loss type
        ax2.set_yscale('log')
        ax2.set_title(f"{title} (Log Scale)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Loss (log)')
        ax2.legend()  # Add legend to identify lines
        ax2.grid(True)
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(filepath)
        plt.close()  # Close the plot to free memory
    except Exception as e:
        print(f"Error plotting {title}: {e}")

class EMA:
    """
        empirical moving average
    """
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    latest_model_state_name = None

    def __init__(
        self,
        model: GenerativeModel,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset = None,
        validation_kwargs: dict = {},
        ema_decay: float = 0.995,
        batch_size: int = 32,
        min_num_batches_per_epoch: int = 10000,
        train_lr: float = 2e-5,
        gradient_accumulate_every: int = 2,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        log_freq: int = 100,
        save_freq: int = 1000,
        save_parallel: bool = False,
        results_folder: str = './results',
        n_reference: int = 8,
        val_batch_size: int = 32,
        num_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        early_stopping: bool = False,
        method: str = "",
        exp_name: str = "",
        num_workers: int = 4,
        device: str = 'cuda',
        seed: int = None,
    ):
        super().__init__()
        self.model = model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_parallel = save_parallel
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = early_stopping
        self.validation_kwargs = validation_kwargs
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logdir = results_folder
        self.n_reference = n_reference
        self.method = method
        self.exp_name = exp_name
        self.device = device


        self.num_batches_per_epoch = max(ceil(len(train_dataset) / (batch_size * gradient_accumulate_every)), 
        min_num_batches_per_epoch)

        print(f"[ utils/training ] Number of batches per epoch: {self.num_batches_per_epoch}")

        self.val_num_batches = ceil(len(val_dataset) / (val_batch_size))

        print(f"[ utils/training ] Number of validation batches: {self.val_num_batches}")

        # Create CUDA generator for DataLoader
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        self.dataloader_train = cycle(torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=True, 
            pin_memory=True,
            generator=generator
        ))
        if val_dataset is not None:
            self.dataloader_val = cycle(torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                shuffle=True, 
                pin_memory=True,
                generator=generator
            ))
        else:
            self.dataloader_val = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        self.reset_parameters()
        self.step = 0

        self.train_losses = defaultdict(lambda : [])
        self.val_losses = defaultdict(lambda : [])

        os.makedirs(self.logdir, exist_ok=True)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def _plot_in_background(self, losses: dict, filepath: str, title: str):
        """Helper to launch plotting in a background process."""
        # Pass a copy of the losses to avoid potential shared state issues
        losses_copy = {k: list(v) for k, v in losses.items()}  # Ensure lists are copied
        process = multiprocessing.Process(target=plot_losses, args=(losses_copy, filepath, title))
        process.start()
        # We don't join, let it run in the background

    def train_one_epoch(self, store_losses: bool = True):
        timer = Timer()
        batch_count = 0

        self.model.train()
        while batch_count < self.num_batches_per_epoch:
            train_losses = defaultdict(lambda : 0)
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch, device=self.device)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                train_losses['train_loss'] += loss.item()

                for key, val in infos.items():
                    train_losses[key] += val

                batch_count += 1

            if store_losses:
                self.add_train_losses(train_losses)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                info_items = infos.items()
                if info_items:
                    infos_str = ' | ' + ' | '.join([f'{key}: {val:8.4f}' for key, val in info_items])
                else:
                    infos_str = ''
                print(f'{self.step}: {train_losses["train_loss"]:8.6f}{infos_str} | t: {timer():8.4f}', flush=True)

            self.step += 1

    def save_model(self, label, save_path=None):
        '''
            saves model and ema to disk;
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if save_path is None:
            save_path = self.logdir
        model_state_name = f'{label}.pt'
        savepath = os.path.join(save_path, model_state_name)
        torch.save(data, savepath)
        self.latest_model_state_name = model_state_name

    def add_train_losses(self, losses):
        for key, loss in losses.items():
            # Ensure loss is a scalar float or int before appending
            if isinstance(loss, (float, int)):
                self.train_losses[key].append(loss)
            elif hasattr(loss, 'item'):  # Handle tensors
                self.train_losses[key].append(loss.item())
            else:
                print(f"Warning: Skipping non-scalar loss value for key '{key}' in train_losses: {loss}")

        if self.step % self.log_freq == 0 and self.step > 0:  # Avoid plotting at step 0 if empty
            filepath = os.path.join(self.logdir, 'train_loss_plot.png')
            self._plot_in_background(self.train_losses, filepath, 'Training Loss')

    def add_val_losses(self, losses):
        for key, loss in losses.items():
            # Ensure loss is a scalar float or int before appending
            if isinstance(loss, (float, int)):
                self.val_losses[key].append(loss)
            elif hasattr(loss, 'item'):  # Handle tensors
                self.val_losses[key].append(loss.item())
            else:
                print(f"Warning: Skipping non-scalar loss value for key '{key}' in val_losses: {loss}")

        filepath = os.path.join(self.logdir, 'val_loss_plot.png')
        self._plot_in_background(self.val_losses, filepath, 'Validation Loss')

    def save_losses(self):
        '''
            saves train and validation losses to disk
        '''
        for key in self.train_losses.keys():
            train_losses = np.array(self.train_losses[key])
            savepath = os.path.join(self.logdir, f'train_losses_{key}.txt')
            with open(savepath, 'w') as f:
                f.write(f'Step\tLoss\n')
                for i, loss in enumerate(train_losses):
                    f.write(f'{i}\t{loss:8.8f}\n')

        for key in self.val_losses.keys():
            val_losses = np.array(self.val_losses[key])
            savepath = os.path.join(self.logdir, f'val_losses_{key}.txt')
            with open(savepath, 'w') as f:
                f.write(f'Epoch\tLoss\n')
                for i, loss in enumerate(val_losses):
                    f.write(f'{i}\t{loss:8.8f}\n')

        print(f'[ utils/training ] Saved losses to {self.logdir}', flush=True)

    def load(self, model_state_name: str = 'best.pt'):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, model_state_name)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self, reraise_keyboard_interrupt: bool = False):
        best_val_loss = float('inf')
        no_improve_counter = 0

        print(f"\nTraining for {self.num_epochs} epochs\n")

        try:
            for i in range(self.num_epochs):
                print(f"Epoch {i} / {self.num_epochs} | {self.train_dataset} | {self.method} | {self.exp_name}")
                self.train_one_epoch()  

                val_loss = self.validate()

                # Save model if it's the best so far (any improvement, no matter how small)
                if val_loss < best_val_loss:
                    old_best = best_val_loss
                    best_val_loss = val_loss
                    self.save_model('best')
                    
                    # Check if this improvement is meaningful for early stopping
                    if old_best - val_loss > self.min_delta:
                        no_improve_counter = 0
                        print(f"Validation Loss: {val_loss:8.6f} | New best validation loss!")
                    else:
                        no_improve_counter += 1
                        print(f"Validation Loss: {val_loss:8.6f} | New best validation loss! (but minimal improvement, counter: {no_improve_counter})")
                else:
                    no_improve_counter += 1
                    print(f"Validation Loss: {val_loss:8.6f} | Current best: {best_val_loss:8.6f} | No improvement for {no_improve_counter} epoch(s).")

                if self.early_stopping and no_improve_counter >= self.patience:
                    print("Early stopping triggered due to convergence.")
                    break
            

                if i % self.save_freq == 0:
                    self.save_model(f'state_{i}_epochs')

        except KeyboardInterrupt:
            print("Keyboard interrupt. Training stopped.")
            if reraise_keyboard_interrupt:
                raise KeyboardInterrupt

        self.save_model('final')
        self.save_losses()

        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    def validate(self, store_losses: bool = True):
        if self.dataloader_val is None:
            return None
        val_losses = defaultdict(lambda : 0.0)  # Initialize with float 0.0
        self.model.eval()
        with torch.no_grad():
            for i in range(self.val_num_batches):
                try:
                    batch = next(self.dataloader_val)
                    batch = batch_to_device(batch, device=self.device)
            
                    loss, infos = self.model.validation_loss(*batch, **self.validation_kwargs)
                    
                    # Check if loss is valid
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid validation loss detected (NaN/Inf) at batch {i}. Skipping.")
                        continue

                    val_losses['val_loss'] += loss.item() / self.val_num_batches

                    for key, val in infos.items():
                        # Handle potential tensor values in infos
                        value_item = val.item() if hasattr(val, 'item') else val
                        if isinstance(value_item, (float, int)):
                            val_losses[key] += value_item / self.val_num_batches
                        else:
                            print(f"Warning: Skipping non-scalar info value for key '{key}' in validation: {value_item}")
                            
                except StopIteration:
                    print("Warning: Validation dataloader exhausted before reaching val_num_batches.")
                    # Adjust val_num_batches if it happened early, to normalize correctly
                    if i == 0: return float('inf')  # Avoid division by zero if no batches ran
                    break  # Exit loop

        if store_losses:
            self.add_val_losses(val_losses)

        return val_losses.get('val_loss', float('inf'))  # Use .get for safety

            
