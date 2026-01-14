from collections import defaultdict
from collections import deque
from contextlib import contextmanager
import math
from math import ceil
import os
import copy
from typing import List, Optional

import numpy as np
import torch
import multiprocessing
import matplotlib.pyplot as plt


from .arrays import batch_to_device
from .timer import Timer
from .model import GenerativeModel, get_parameter_groups


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

class WarmupCosineDecayScheduler:
    """Learning rate scheduler with linear warmup followed by cosine decay based on steps."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=None, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr'] if base_lr is None else base_lr
        self.current_step = 0
            
    def step(self, step=None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        if self.current_step <= self.warmup_steps:
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = max(0.0, min(progress, 1.0))
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

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
        min_num_steps_per_epoch: int = 10000,
        train_lr: float = 2e-5,
        gradient_accumulate_every: int = 2,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        log_freq: int = 100,
        save_freq: int = 20,
        save_parallel: bool = False,
        results_folder: str = './results',
        val_batch_size: int = 32,
        num_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        warmup_epochs: int = 0,
        early_stopping: bool = False,
        method: str = "",
        exp_name: str = "",
        num_workers: int = 4,
        device: str = 'cuda',
        seed: int = None,
        use_lr_scheduler: bool = False,
        lr_scheduler_warmup_steps: int = 0,
        lr_scheduler_min_lr: float = 0.0,
        useAdamW: bool = False,
        optimizer_kwargs: dict = None,
        clip_grad_norm: Optional[float] = None,
        eval_freq: int = 10,
        eval_batch_size: int = 32,
        eval_seed: int = 42,
        perform_final_state_evaluation: bool = False,
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
        self.warmup_epochs = warmup_epochs
        self.validation_kwargs = validation_kwargs
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logdir = results_folder
        self.method = method
        self.exp_name = exp_name
        self.device = device
        self.val_batch_size = val_batch_size
        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size
        self.eval_seed = eval_seed
        self.perform_final_state_evaluation = perform_final_state_evaluation


        self.num_steps_per_epoch = max(ceil(len(train_dataset) / (batch_size * gradient_accumulate_every)), 
        min_num_steps_per_epoch)

        print(f"[ utils/training ] Number of steps per epoch: {self.num_steps_per_epoch}")

        if val_dataset is not None:
            self.val_num_batches = ceil(len(val_dataset) / (val_batch_size))
            print(f"[ utils/training ] Number of validation batches: {self.val_num_batches}")
        else:
            self.val_num_batches = 0

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
            self.dataloader_val = torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=val_batch_size, 
                num_workers=num_workers, 
                shuffle=False, 
                pin_memory=True,
                generator=generator
            )
        else:
            self.dataloader_val = None

        self.optimizer = self.get_optimizer(model, train_lr, optimizer_kwargs, useAdamW)

        self.clip_grad_norm = clip_grad_norm
        
        # Initialize learning rate scheduler
        self.use_lr_scheduler = use_lr_scheduler
        if self.use_lr_scheduler:
            self.lr_scheduler = WarmupCosineDecayScheduler(
                self.optimizer,
                warmup_steps=lr_scheduler_warmup_steps,
                total_steps=num_epochs * self.num_steps_per_epoch,
                base_lr=train_lr,
                min_lr=lr_scheduler_min_lr
            )
        else:
            self.lr_scheduler = None

        self.reset_parameters()
        self.step = 0

        self.train_losses = defaultdict(lambda : [])
        self.val_losses = defaultdict(lambda : [])
        self.eval_losses = defaultdict(lambda : [])
        self._train_losses_saved_counts = {}
        self._val_losses_saved_counts = {}
        self._eval_losses_saved_counts = {}

        os.makedirs(self.logdir, exist_ok=True)
        
    def get_optimizer(self, model, train_lr, optimizer_kwargs, useAdamW=False):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        # remove global weight_decay so per-group values take effect
        optimizer_kwargs = dict(optimizer_kwargs)  # shallow copy
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)

        param_groups = get_parameter_groups(model, weight_decay=weight_decay)

        if useAdamW:
            optimizer = torch.optim.AdamW(param_groups, lr=train_lr, **optimizer_kwargs)
        else:
            optimizer = torch.optim.Adam(param_groups, lr=train_lr, **optimizer_kwargs)

        return optimizer

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

        # sync buffers too
        for b, mb in zip(self.model.buffers(), self.ema_model.buffers()):
            mb.data.copy_(b.data)

    @contextmanager
    def swap_to_ema(self):
        backup = copy.deepcopy(self.model.state_dict())
        try:
            self.model.load_state_dict(self.ema_model.state_dict())
            yield
        finally:
            self.model.load_state_dict(backup)

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

    def save_model(self, label, save_path=None):
        '''
            saves model and ema to disk;
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_epochs': self.num_epochs,
            'num_steps_per_epoch': self.num_steps_per_epoch,
            'epoch': self.step // max(1, self.num_steps_per_epoch),
            'total_planned_steps': self.num_epochs * self.num_steps_per_epoch,
        }
        if self.lr_scheduler is not None:
            data['lr_scheduler'] = {
                'warmup_steps': self.lr_scheduler.warmup_steps,
                'total_steps': self.lr_scheduler.total_steps,
                'base_lr': self.lr_scheduler.base_lr,
                'min_lr': self.lr_scheduler.min_lr,
                'current_step': self.lr_scheduler.current_step,
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

    def add_eval_losses(self, losses):
        for key, loss in losses.items():
            # Ensure loss is a scalar float or int before appending
            if isinstance(loss, (float, int)):
                self.eval_losses[key].append(loss)
            elif hasattr(loss, 'item'):  # Handle tensors
                self.eval_losses[key].append(loss.item())
            else:
                print(f"Warning: Skipping non-scalar loss value for key '{key}' in eval_losses: {loss}")

        filepath = os.path.join(self.logdir, 'eval_loss_plot.png')
        self._plot_in_background(self.eval_losses, filepath, 'Evaluation Loss')

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
        # Train losses: append only new entries
        for key in self.train_losses.keys():
            losses = self.train_losses[key]
            savepath = os.path.join(self.logdir, f'train_losses_{key}.txt')
            os.makedirs(self.logdir, exist_ok=True)
            file_exists = os.path.exists(savepath)

            start_idx = self._train_losses_saved_counts.get(key, 0)

            mode = 'a' if file_exists else 'w'
            with open(savepath, mode) as f:
                if not file_exists:
                    f.write('Step\tLoss\n')
                for i in range(start_idx, len(losses)):
                    f.write(f'{i}\t{float(losses[i]):8.8f}\n')
            self._train_losses_saved_counts[key] = len(losses)

        # Val losses: append only new entries
        for key in self.val_losses.keys():
            losses = self.val_losses[key]
            savepath = os.path.join(self.logdir, f'val_losses_{key}.txt')
            os.makedirs(self.logdir, exist_ok=True)
            file_exists = os.path.exists(savepath)

            start_idx = self._val_losses_saved_counts.get(key, 0)

            mode = 'a' if file_exists else 'w'
            with open(savepath, mode) as f:
                if not file_exists:
                    f.write('Epoch\tLoss\n')
                for i in range(start_idx, len(losses)):
                    f.write(f'{i}\t{float(losses[i]):8.8f}\n')
            self._val_losses_saved_counts[key] = len(losses)

        # Eval losses: append only new entries
        for key in self.eval_losses.keys():
            losses = self.eval_losses[key]
            savepath = os.path.join(self.logdir, f'eval_losses_{key}.txt')
            os.makedirs(self.logdir, exist_ok=True)
            file_exists = os.path.exists(savepath)
            start_idx = self._eval_losses_saved_counts.get(key, 0)
            mode = 'a' if file_exists else 'w'
            with open(savepath, mode) as f:
                if not file_exists:
                    f.write('Epoch\tLoss\n')
                for i in range(start_idx, len(losses)):
                    f.write(f'{i}\t{float(losses[i]):8.8f}\n')
            self._eval_losses_saved_counts[key] = len(losses)

        print(f'[ utils/training ] Saved losses to {self.logdir}', flush=True)

    def load_losses(self):
        """Load previously saved loss curves so plots continue on resume."""
        try:
            # Train losses
            for fname in os.listdir(self.logdir):
                if fname.startswith('train_losses_') and fname.endswith('.txt'):
                    key = fname[len('train_losses_'):-4]
                    values = []
                    with open(os.path.join(self.logdir, fname), 'r') as f:
                        lines = f.readlines()[1:]
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    values.append(float(parts[1]))
                                except Exception:
                                    pass
                    self.train_losses[key] = values
                    self._train_losses_saved_counts[key] = len(values)

            # Val losses
            for fname in os.listdir(self.logdir):
                if fname.startswith('val_losses_') and fname.endswith('.txt'):
                    key = fname[len('val_losses_'):-4]
                    values = []
                    with open(os.path.join(self.logdir, fname), 'r') as f:
                        lines = f.readlines()[1:]
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    values.append(float(parts[1]))
                                except Exception:
                                    pass
                    self.val_losses[key] = values
                    self._val_losses_saved_counts[key] = len(values)

            # Eval losses
            for fname in os.listdir(self.logdir):
                if fname.startswith('eval_losses_') and fname.endswith('.txt'):
                    key = fname[len('eval_losses_'):-4]
                    values = []
                    with open(os.path.join(self.logdir, fname), 'r') as f:
                        lines = f.readlines()[1:]
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    values.append(float(parts[1]))
                                except Exception:
                                    pass
                    self.eval_losses[key] = values
                    self._eval_losses_saved_counts[key] = len(values)

            # Re-render current plots in background (if any data present)
            if any(len(v) > 0 for v in self.train_losses.values()):
                filepath = os.path.join(self.logdir, 'train_loss_plot.png')
                self._plot_in_background(self.train_losses, filepath, 'Training Loss')
            if any(len(v) > 0 for v in self.val_losses.values()):
                filepath = os.path.join(self.logdir, 'val_loss_plot.png')
                self._plot_in_background(self.val_losses, filepath, 'Validation Loss')
            if any(len(v) > 0 for v in self.eval_losses.values()):
                filepath = os.path.join(self.logdir, 'eval_loss_plot.png')
                self._plot_in_background(self.eval_losses, filepath, 'Evaluation Loss')
        except Exception as e:
            print(f"Warning: Could not load prior losses: {e}")

    def load(self, model_state_name: str = 'best.pt'):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, model_state_name)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        # Restore epoch metadata if present
        self.num_epochs = data.get('num_epochs', self.num_epochs)
        # Do not forcibly override num_steps_per_epoch if it changed due to config, but use if present
        self.num_steps_per_epoch = data.get('num_steps_per_epoch', self.num_steps_per_epoch)
        # Preload prior losses to keep plots cumulative across resumes
        self.load_losses()

    def train_one_epoch(self, store_losses: bool = True):
        timer = Timer()
        steps_start = self.step

        self.model.train()
        while (self.step - steps_start) < self.num_steps_per_epoch:
            train_losses = defaultdict(lambda : 0)
            infos = defaultdict(lambda : 0)
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch, device=self.device)

                loss, batch_infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                train_losses['train_loss'] += loss.item()

                for key, val in batch_infos.items():
                    infos[key] += val / self.gradient_accumulate_every
                    train_losses[key] += val / self.gradient_accumulate_every

            if store_losses:
                self.add_train_losses(train_losses)

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.step)

    def train(self, reraise_keyboard_interrupt: bool = False):
        best_val = float('inf')
        best_epoch = -1

        recent_improvements = deque(maxlen=self.patience)

        print(f"\nTraining for {self.num_epochs} epochs\n")

        try:
            for epoch in range(1, self.num_epochs + 1):
                # Stop if we've already reached the planned total steps (especially on resume)
                planned_total_steps = (
                    self.lr_scheduler.total_steps if self.lr_scheduler is not None else self.num_epochs * self.num_steps_per_epoch
                )
                if self.step >= planned_total_steps:
                    print(f"Reached planned total steps ({planned_total_steps}); stopping.")
                    break
                # Update learning rate if scheduler is enabled
                if self.lr_scheduler is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}/{self.num_epochs} | LR: {current_lr:.6e} | {self.exp_name}")
                else:
                    print(f"Epoch {epoch}/{self.num_epochs} | {self.exp_name}")

                self.train_one_epoch()

                val, final_horizon_step_loss = self.validate()

                if self.perform_final_state_evaluation and (epoch % self.eval_freq) == 0:
                    final_rollout_step_loss, final_rollout_infos = self.evaluate_final_states()

                    if final_rollout_infos:
                        infos_str = ' | ' + ' | '.join([f'{key}: {val:8.4f}' for key, val in final_rollout_infos.items()])
                    else:
                        infos_str = ''

                    print(f"Final Rollout Evaluation:\n{infos_str}")

                    

                if not (val == val) or val in (float('inf'), -float('inf')):
                    print(f"Validation returned non-finite value ({val}); stopping early.")
                    break

                improvement = 0.0
                if best_val != float('inf') and abs(best_val) > 1e-12:
                    improvement = (best_val - val) / abs(best_val)

                if val < best_val:
                    pos_impr = max(improvement, 0.0) if best_val != float('inf') else 0.0
                    recent_improvements.append(pos_impr)

                    old_best = best_val
                    best_val = val
                    best_epoch = epoch
                    self.save_model('best')

                    window_gain = sum(recent_improvements)
                    print(
                        f"Validation: {val:8.6f} | New val best (Δ={old_best - val:+.6g}) | Final Horizon Step Loss: {final_horizon_step_loss:8.6f} | "
                        f"Val window_gain={window_gain:.3g} over last ≤{self.patience} epoch(s); "
                        f"target ≥ {self.min_delta})"
                    )
                else:
                    recent_improvements.append(0.0)
                    window_gain = sum(recent_improvements)
                    print(
                        f"Validation: {val:8.6f} | Best: {best_val:8.6f} (epoch {best_epoch}) | Final Horizon Step Loss: {final_horizon_step_loss:8.6f} | "
                        f"Val window_gain={window_gain:.3g} over last ≤{self.patience} epoch(s); target ≥ {self.min_delta}"
                    )

                if (epoch % self.save_freq) == 0:
                    self.save_model(f"state_{epoch}_epochs")

                if self.early_stopping and epoch > self.warmup_epochs:
                    if len(recent_improvements) == self.patience:
                        window_gain = sum(recent_improvements)
                        if window_gain < self.min_delta:
                            print(
                                f"Early stopping: cumulative gain over the last {self.patience} epochs "
                                f"is {window_gain:.3g} < required {self.min_delta}. "
                                f"Best @ epoch {best_epoch} with val={best_val:8.6f}"
                            )
                            break
            else:
                print(f"Finished all epochs. Best @ epoch {best_epoch} with val={best_val:8.6f}")

            self.save_model("final")

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            self.save_model("interrupted")
            if reraise_keyboard_interrupt:
                raise
        
        self.save_losses()
        print(f"\nTraining complete. Best validation loss: {best_val:.6f}")

    def evaluate_final_states(self, store_losses: bool = True):
        eval_data = batch_to_device(self.val_dataset.eval_data, device=self.device)
        num_evals = eval_data.final_states.shape[0]

        eval_losses = defaultdict(lambda : 0.0)
        infos = defaultdict(lambda : 0.0)
        num_batches = 0

        # Get max_path_length from eval_data or validation_kwargs
        max_path_length = eval_data.max_path_length
        if max_path_length is None:
            max_path_length = self.validation_kwargs.get('max_path_length')

        for i in range(0, num_evals, self.val_batch_size):
            batch_start_states = eval_data.histories[i:i+self.val_batch_size]
            batch_final_states = eval_data.final_states[i:i+self.val_batch_size]
            loss, infos = self.model.evaluate_final_states(
                batch_start_states,
                batch_final_states,
                get_conditions=self.val_dataset.get_conditions,
                max_path_length=max_path_length,
                **self.validation_kwargs
            )
            eval_losses['final_rollout_step_loss'] += loss.item()
            for key, val in infos.items():
                infos[key] += val
            num_batches += 1

        final_infos = {}

        for key in infos.keys():
            final_infos['final_rollout_' + key] = infos[key] / num_batches
        eval_losses['final_rollout_step_loss'] /= num_batches
        eval_losses.update(final_infos)

        if store_losses:
            self.add_eval_losses(eval_losses)

        return eval_losses.get('final_rollout_step_loss', float('inf')), eval_losses
            
    def validate(self, store_losses: bool = True):
        if self.dataloader_val is None:
            return None
        val_losses = defaultdict(lambda : 0.0)  # Initialize with float 0.0
        num_batches = 0
        self.model.eval()
        with torch.no_grad(), self.swap_to_ema():
            for i, batch in enumerate(self.dataloader_val):
                batch = batch_to_device(batch, device=self.device)

                loss, infos = self.model.validation_loss(*batch, **self.validation_kwargs)

                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid validation loss detected (NaN/Inf) at batch {i}. Skipping.")
                    continue

                val_losses['val_loss'] += loss.item()
                num_batches += 1

                for key, val in infos.items():
                    # Handle potential tensor values in infos
                    value_item = val.item() if hasattr(val, 'item') else val
                    if isinstance(value_item, (float, int)):
                        val_losses[key] += value_item
                    else:
                        print(f"Warning: Skipping non-scalar info value for key '{key}' in validation: {value_item}")

        for key in val_losses.keys():
            val_losses[key] /= num_batches

        if store_losses:
            self.add_val_losses(val_losses)

        return val_losses.get('val_loss', float('inf')), val_losses.get('final_horizon_step_loss', float('inf'))  # Use .get for safety

            
