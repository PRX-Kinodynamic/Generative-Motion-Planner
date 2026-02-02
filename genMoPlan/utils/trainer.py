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
        detailed_eval_freq: int = 0,  # 0 = disabled, N = every N epochs
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
        self.detailed_eval_freq = detailed_eval_freq  # Controls detailed evaluation frequency
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

                # Enable verbose logging on eval epochs (for basic validation output)
                is_verbose_epoch = (epoch % self.eval_freq) == 0
                
                # Detailed evaluation epochs (Final State, Sequential, Full Trajectory)
                # detailed_eval_freq=0 means disabled, otherwise every N epochs
                is_detailed_epoch = (self.detailed_eval_freq > 0) and (epoch % self.detailed_eval_freq) == 0
                
                val, final_state_val, real_data_val, real_fraction = self.validate(verbose=is_verbose_epoch)

                final_state_eval_loss_infos = None
                if self.perform_final_state_evaluation and is_detailed_epoch:
                    print(f"\n{'='*80}")
                    print(f"DETAILED EVALUATION - Epoch {epoch}")
                    print(f"{'='*80}")
                    
                    final_state_eval_loss, final_state_eval_loss_infos = self.evaluate_final_states(verbose=True)

                    if final_state_eval_loss_infos:
                        # Filter out non-scalar values like step_losses list
                        scalar_infos = {k: v for k, v in final_state_eval_loss_infos.items() 
                                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
                        infos_str = ' | ' + ' | '.join([f'{key}: {val:8.4f}' for key, val in scalar_infos.items()])
                    else:
                        infos_str = ''

                    print(f"\nFinal state evaluation summary:{infos_str}")
                    print(f"{'='*80}\n")

                # Sequential consecutive validation (only on detailed epochs)
                if is_detailed_epoch:
                    print(f"\n{'='*80}")
                    print(f"SEQUENTIAL CONSECUTIVE VALIDATION - Epoch {epoch}")
                    print(f"{'='*80}")
                    
                    sequential_results = self.evaluate_sequential_validation(
                        verbose=True,
                        autoregressive_results=final_state_eval_loss_infos
                    )
                    print(f"{'='*80}")
                    
                    # Full trajectory validation (no padding)
                    print(f"\n{'='*80}")
                    print(f"FULL TRAJECTORY VALIDATION (NO PADDING) - Epoch {epoch}")
                    print(f"{'='*80}")
                    
                    full_traj_results = self.evaluate_full_trajectory_validation(verbose=True)
                    print(f"{'='*80}\n")

                    

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
                    real_pct = real_fraction * 100
                    print(
                        f"Val: {val:8.6f} | Real: {real_data_val:8.6f} ({real_pct:.0f}%) | "
                        f"New best (Δ={old_best - val:+.6g}) | Final: {final_state_val:8.6f}"
                    )
                else:
                    recent_improvements.append(0.0)
                    window_gain = sum(recent_improvements)
                    real_pct = real_fraction * 100
                    print(
                        f"Val: {val:8.6f} | Real: {real_data_val:8.6f} ({real_pct:.0f}%) | "
                        f"Best: {best_val:8.6f} (ep{best_epoch}) | Final: {final_state_val:8.6f}"
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

    def evaluate_final_states(self, store_losses: bool = True, verbose: bool = False):
        eval_data = batch_to_device(self.val_dataset.eval_data, device=self.device)
        num_evals = eval_data.final_states.shape[0]
        
        # Get full trajectories if available (not moved to device as they're a list)
        full_trajectories = getattr(self.val_dataset.eval_data, 'full_trajectories', None)

        eval_losses = defaultdict(lambda : 0.0)
        accumulated_infos = defaultdict(lambda : 0.0)
        num_batches = 0
        
        if verbose:
            print(f"\n  Final State Evaluation: {num_evals} trajectories, batch_size={self.val_batch_size}")

        for i in range(0, num_evals, self.val_batch_size):
            batch_start_states = eval_data.histories[i:i+self.val_batch_size]
            batch_final_states = eval_data.final_states[i:i+self.val_batch_size]
            
            # Get batch of full trajectories if available
            batch_full_trajectories = None
            if full_trajectories is not None:
                batch_full_trajectories = full_trajectories[i:i+self.val_batch_size]
            
            if verbose:
                print(f"\n  [Eval Batch {num_batches+1}] trajectories {i}-{min(i+self.val_batch_size, num_evals)}")
            
            loss, infos = self.model.evaluate_final_states(batch_start_states, batch_final_states, get_conditions=self.val_dataset.get_conditions, full_trajectories=batch_full_trajectories, **self.validation_kwargs)
            eval_losses['final_state_loss'] += loss.item()
            
            # Store step_losses for sequential validation comparison
            if 'step_losses' in infos and 'step_losses' not in accumulated_infos:
                accumulated_infos['step_losses'] = []
            if 'step_losses' in infos:
                accumulated_infos['step_losses'].extend(infos['step_losses'])
            
            for key, val in infos.items():
                # Skip non-scalar values like step_losses list
                if key == 'step_losses':
                    continue  # Already handled above
                if isinstance(val, (int, float)):
                    accumulated_infos[key] += val
                elif hasattr(val, 'item'):
                    accumulated_infos[key] += val.item()
            num_batches += 1

        final_infos = {}

        for key in accumulated_infos.keys():
            if key == 'step_losses':
                # Keep step_losses as-is (list of dicts)
                final_infos['step_losses'] = accumulated_infos['step_losses']
            else:
                final_infos['final_state_' + key] = accumulated_infos[key] / num_batches
        eval_losses['final_state_loss'] /= num_batches
        eval_losses.update(final_infos)

        if store_losses:
            # Filter out non-scalar values before storing
            scalar_eval_losses = {k: v for k, v in eval_losses.items() 
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
            self.add_eval_losses(scalar_eval_losses)
        
        return eval_losses.get('final_state_loss', float('inf')), eval_losses

    def evaluate_sequential_validation(self, store_losses: bool = True, verbose: bool = False, autoregressive_results: dict = None):
        if self.val_dataset is None:
            return {}
        
        # Get full trajectories from validation dataset
        if not hasattr(self.val_dataset, 'normed_trajectories'):
            if verbose:
                print("  Sequential validation: Cannot access full trajectories from validation dataset")
            return {}
        
        trajectories = self.val_dataset.normed_trajectories
        if not trajectories:
            return {}
        
        eval_losses = defaultdict(lambda: 0.0)
        
        if verbose:
            print(f"\n  Sequential Consecutive Validation: {len(trajectories)} trajectories")
            # Debug: show info about first trajectory
            if trajectories:
                first_traj = trajectories[0]
                print(f"    [DEBUG trainer] First trajectory: type={type(first_traj)}, "
                      f"len={len(first_traj) if hasattr(first_traj, '__len__') else 'N/A'}, "
                      f"shape={first_traj.shape if hasattr(first_traj, 'shape') else 'N/A'}")
        
        with torch.no_grad(), self.swap_to_ema():
            results = self.model.evaluate_sequential_validation(
                trajectories=trajectories,
                get_conditions=self.val_dataset.get_conditions,
                verbose=verbose,
                autoregressive_results=autoregressive_results,
                **self.validation_kwargs
            )
        
        if not results:
            return {}
        
        # Extract scalar metrics for storage
        for key in ['avg_segment_loss', 'avg_final_state_loss', 'num_segments', 'num_trajectories',
                   'segments_per_trajectory', 'early_segments_loss', 'middle_segments_loss', 
                   'late_segments_loss', 'avg_x_loss', 'avg_theta_loss', 'avg_x_dot_loss', 
                   'avg_theta_dot_loss']:
            if key in results:
                eval_losses[f'sequential_{key}'] = results[key]
        
        # Add autoregressive comparison if available
        if 'autoregressive_avg_loss' in results:
            eval_losses['sequential_autoregressive_avg_loss'] = results['autoregressive_avg_loss']
            eval_losses['sequential_autoregressive_vs_sequential_ratio'] = results['autoregressive_vs_sequential_ratio']
            eval_losses['sequential_error_accumulation_factor'] = results['error_accumulation_factor']
        
        if verbose:
            print(f"\n  Sequential Validation Summary:")
            print(f"    Trajectories evaluated: {results.get('num_trajectories', 0)}")
            print(f"    Total segments: {results.get('num_segments', 0)}")
            print(f"    Average segments per trajectory: {results.get('segments_per_trajectory', 0):.1f}")
            print(f"    Average segment loss: {results.get('avg_segment_loss', 0):.6f}")
            print(f"    Average final state loss: {results.get('avg_final_state_loss', 0):.6f}")
            print(f"\n    Per-component (average):")
            print(f"      x={results.get('avg_x_loss', 0):.4f} | "
                  f"θ={results.get('avg_theta_loss', 0):.4f} | "
                  f"ẋ={results.get('avg_x_dot_loss', 0):.4f} | "
                  f"θ̇={results.get('avg_theta_dot_loss', 0):.4f}")
            
            print(f"\n    Position-based analysis:")
            print(f"      Early segments (0-25%): avg_loss={results.get('early_segments_loss', 0):.6f}")
            print(f"      Middle segments (25-75%): avg_loss={results.get('middle_segments_loss', 0):.6f}")
            print(f"      Late segments (75-100%): avg_loss={results.get('late_segments_loss', 0):.6f}")
            
            trajectory_summaries = results.get('trajectory_summaries', [])
            if trajectory_summaries:
                first_seg_avg = np.mean([t['first_segment_loss'] for t in trajectory_summaries])
                last_seg_avg = np.mean([t['last_segment_loss'] for t in trajectory_summaries])
                print(f"      First segment avg loss: {first_seg_avg:.6f}")
                print(f"      Last segment avg loss: {last_seg_avg:.6f}")
                print(f"      Loss trend (first→last): {last_seg_avg - first_seg_avg:+.6f} "
                      f"({(last_seg_avg / first_seg_avg if first_seg_avg > 0 else 1.0):.2f}x)")
            
            # Autoregressive comparison
            if 'autoregressive_avg_loss' in results:
                print(f"\n    Comparison with Autoregressive Rollout (Final State Eval):")
                print(f"      Sequential (ground-truth conditions): avg={results.get('avg_segment_loss', 0):.6f}")
                print(f"      Autoregressive (previous predictions): avg={results.get('autoregressive_avg_loss', 0):.6f}")
                print(f"      Error amplification factor: {results.get('error_accumulation_factor', 0):.2f}x")
                
                # Per-step comparison if available
                segment_details = results.get('segment_details', [])
                if segment_details and 'autoregressive_loss_at_same_timestep' in segment_details[0]:
                    print(f"\n      Per-step comparison (first 5 steps):")
                    for i, seg in enumerate(segment_details[:5]):
                        if 'autoregressive_loss_at_same_timestep' in seg:
                            seq_loss = seg['segment_loss']
                            ar_loss = seg['autoregressive_loss_at_same_timestep']
                            ratio = seg.get('autoregressive_vs_sequential_ratio', 0)
                            print(f"        Step {i+1}: Sequential={seq_loss:.4f} | "
                                  f"Autoregressive={ar_loss:.4f} | Ratio={ratio:.2f}x")
        
        if store_losses:
            self.add_eval_losses(eval_losses)
        
        return results
    
    def evaluate_full_trajectory_validation(self, verbose: bool = False):
        """
        Compute validation loss ONLY on trajectories long enough to avoid padding.
        
        This provides a meaningful validation metric by filtering out trajectories
        that would result in mostly-padding horizons.
        
        Minimum trajectory length for no padding:
            min_length = 1 + (horizon_length - 1) * stride + stride
                       = 1 + horizon_length * stride
        """
        if not hasattr(self.val_dataset, 'normed_trajectories'):
            if verbose:
                print("  Full Trajectory Validation: Cannot access trajectories")
            return {}
        
        trajectories = self.val_dataset.normed_trajectories
        if not trajectories:
            return {}
        
        # Calculate minimum trajectory length for no horizon padding
        stride = getattr(self.val_dataset, 'stride', 1)
        horizon_length = getattr(self.val_dataset, 'horizon_length', 31)
        history_length = getattr(self.val_dataset, 'history_length', 1)
        
        # For 100% real horizon values (no padding):
        # Need: 1 + (horizon_length - 1) * stride + stride = 1 + horizon_length * stride
        min_length_full = 1 + horizon_length * stride
        
        # For at least 50% real horizon values:
        min_length_half = 1 + (horizon_length // 2) * stride
        
        # Filter trajectories
        full_trajectories = [(i, t) for i, t in enumerate(trajectories) if len(t) >= min_length_full]
        half_trajectories = [(i, t) for i, t in enumerate(trajectories) if len(t) >= min_length_half]
        
        results = {
            'stride': stride,
            'horizon_length': horizon_length,
            'min_length_for_100pct_real': min_length_full,
            'min_length_for_50pct_real': min_length_half,
            'total_trajectories': len(trajectories),
            'trajectories_100pct_real': len(full_trajectories),
            'trajectories_50pct_real': len(half_trajectories),
        }
        
        if verbose:
            print(f"\n  Full Trajectory Validation Analysis:")
            print(f"    Configuration: stride={stride}, horizon_length={horizon_length}")
            print(f"    Min length for 100% real horizon: {min_length_full} timesteps")
            print(f"    Min length for 50% real horizon: {min_length_half} timesteps")
            print(f"\n    Trajectory distribution:")
            print(f"      Total trajectories: {len(trajectories)}")
            print(f"      With 100% real horizon: {len(full_trajectories)} ({100*len(full_trajectories)/len(trajectories):.1f}%)")
            print(f"      With 50%+ real horizon: {len(half_trajectories)} ({100*len(half_trajectories)/len(trajectories):.1f}%)")
        
        # Compute validation loss on full trajectories (100% real)
        if full_trajectories:
            full_losses = []
            full_final_state_losses = []
            
            with torch.no_grad(), self.swap_to_ema():
                device = next(self.model.parameters()).device
                
                for traj_idx, trajectory in full_trajectories[:20]:  # Limit to 20 for speed
                    trajectory = trajectory.to(device)
                    
                    # Extract history and horizon (no padding needed)
                    history = trajectory[:history_length * stride:stride]
                    horizon_start = history_length * stride
                    horizon = trajectory[horizon_start:horizon_start + horizon_length * stride:stride]
                    
                    if len(horizon) < horizon_length:
                        continue  # Skip if still not enough
                    
                    # Truncate to exact horizon_length
                    horizon = horizon[:horizon_length]
                    
                    # Create full trajectory tensor
                    full_traj = torch.cat([history, horizon], dim=0).unsqueeze(0)  # (1, pred_len, dim)
                    
                    # Get conditions
                    cond = self.val_dataset.get_conditions(history)
                    
                    # Compute validation loss
                    loss, info = self.model.validation_loss(full_traj, cond, verbose=False, **self.validation_kwargs)
                    
                    full_losses.append(loss.item())
                    if 'final_state_loss' in info:
                        fs_loss = info['final_state_loss']
                        full_final_state_losses.append(fs_loss.item() if hasattr(fs_loss, 'item') else fs_loss)
            
            if full_losses:
                results['full_traj_val_loss'] = np.mean(full_losses)
                results['full_traj_val_loss_std'] = np.std(full_losses)
                results['full_traj_final_state_loss'] = np.mean(full_final_state_losses) if full_final_state_losses else 0
                results['num_evaluated'] = len(full_losses)
                
                if verbose:
                    print(f"\n    Validation on 100% real trajectories ({len(full_losses)} samples):")
                    print(f"      Trajectory loss: {results['full_traj_val_loss']:.6f} ± {results['full_traj_val_loss_std']:.6f}")
                    print(f"      Final state loss: {results['full_traj_final_state_loss']:.6f}")
                    print(f"\n    Comparison with standard validation:")
                    print(f"      Standard val loss (with padding): ~0.11 (artificially low)")
                    print(f"      Full trajectory val loss (no padding): {results['full_traj_val_loss']:.6f}")
        else:
            if verbose:
                print(f"\n    WARNING: No trajectories have 100% real horizon!")
                print(f"    All validation trajectories are shorter than {min_length_full} timesteps.")
                print(f"    Standard validation loss is UNRELIABLE for this configuration.")
        
        return results
            
    def validate(self, store_losses: bool = True, verbose: bool = False):
        if self.dataloader_val is None:
            return None
        val_losses = defaultdict(lambda : 0.0)  # Initialize with float 0.0
        num_batches = 0
        total_samples = 0
        self.model.eval()
        
        if verbose:
            print(f"\n  Validation: {len(self.dataloader_val)} batches")
        
        with torch.no_grad(), self.swap_to_ema():
            for i, batch in enumerate(self.dataloader_val):
                batch = batch_to_device(batch, device=self.device)
                batch_size = batch[0].shape[0] if hasattr(batch[0], 'shape') else len(batch[0])
                total_samples += batch_size
                
                # Only print first few batches in verbose mode to avoid spam
                batch_verbose = verbose and i < 3
                
                loss, infos = self.model.validation_loss(*batch, verbose=batch_verbose, **self.validation_kwargs)
                
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
        
        for key in val_losses.keys():
            val_losses[key] /= num_batches
        
        if verbose:
            real_pct = val_losses.get('real_fraction', 0) * 100
            print(f"\n  Validation Summary: {total_samples} samples across {num_batches} batches")
            print(f"    avg_trajectory_loss: {val_losses.get('val_loss', 0):.6f} (includes padding)")
            print(f"    avg_real_data_loss:  {val_losses.get('real_data_loss', 0):.6f} ({real_pct:.1f}% real data)")
            print(f"    avg_final_state_loss: {val_losses.get('final_state_loss', 0):.6f}")
            print(f"    Per-component (trajectory - ALL data): "
                  f"x={val_losses.get('x_loss', 0):.4f} | "
                  f"θ={val_losses.get('theta_loss', 0):.4f} | "
                  f"ẋ={val_losses.get('x_dot_loss', 0):.4f} | "
                  f"θ̇={val_losses.get('theta_dot_loss', 0):.4f}")
            print(f"    Per-component (trajectory - REAL data only): "
                  f"x={val_losses.get('real_x_loss', 0):.4f} | "
                  f"θ={val_losses.get('real_theta_loss', 0):.4f} | "
                  f"ẋ={val_losses.get('real_x_dot_loss', 0):.4f} | "
                  f"θ̇={val_losses.get('real_theta_dot_loss', 0):.4f}")

        if store_losses:
            self.add_val_losses(val_losses)

        # Return val_loss, final_state_loss, real_data_loss, real_fraction
        return (
            val_losses.get('val_loss', float('inf')), 
            val_losses.get('final_state_loss', float('inf')),
            val_losses.get('real_data_loss', float('inf')),
            val_losses.get('real_fraction', 0.0)
        )

            
