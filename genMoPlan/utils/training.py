from math import ceil
import os
import copy
import numpy as np
import torch
import multiprocessing
import matplotlib.pyplot as plt

from .arrays import batch_to_device
from .timer import Timer
from .cloud import sync_logs
from .model import GenerativeModel

def cycle(dl):
    while True:
        for data in dl:
            yield data

# --- Plotting Function ---
def plot_losses(losses, filepath, title):
    """Plots losses and saves the figure."""
    try:
        plt.figure()
        plt.plot(losses)
        plt.title(title)
        plt.xlabel('Step' if 'Training' in title else 'Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(filepath)
        plt.close() # Close the plot to free memory
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
        dataset: torch.utils.data.Dataset,
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
        bucket: str = None,
        val_num_batches: int = 10,
        num_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        early_stopping: bool = False,
        method: str = "",
        exp_name: str = "",
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
        self.val_num_batches = val_num_batches
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.method = method
        self.exp_name = exp_name


        self.num_batches_per_epoch = max(ceil(len(dataset) / (batch_size * gradient_accumulate_every)), 
        min_num_batches_per_epoch)

        print(f"[ utils/training ] Number of batches per epoch: {self.num_batches_per_epoch}")

        self.dataloader_train = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        if val_dataset is not None:
            self.dataloader_val = cycle(torch.utils.data.DataLoader(
                self.val_dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
            ))
        else:
            self.dataloader_val = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        self.reset_parameters()
        self.step = 0

        self.train_losses = []
        self.val_losses = []

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

    def _plot_in_background(self, losses, filepath, title):
        """Helper to launch plotting in a background process."""
        # Pass a copy of the losses to avoid potential shared state issues
        losses_copy = list(losses) 
        process = multiprocessing.Process(target=plot_losses, args=(losses_copy, filepath, title))
        process.start()
        # We don't join, let it run in the background

    def add_train_loss(self, loss):
        self.train_losses.append(loss)

        if self.step % self.log_freq == 0:
            filepath = os.path.join(self.logdir, 'train_loss_plot.png')
            self._plot_in_background(self.train_losses, filepath, 'Training Loss')

    def add_val_loss(self, loss):
        self.val_losses.append(loss)
        filepath = os.path.join(self.logdir, 'val_loss_plot.png')
        self._plot_in_background(self.val_losses, filepath, 'Validation Loss')

    def train_one_epoch(self):
        timer = Timer()
        total_loss = 0.0
        batch_count = 0

        self.model.train()

        while batch_count < self.num_batches_per_epoch:
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                total_loss += loss.item()
                batch_count += 1

            actual_loss = loss.item() * self.gradient_accumulate_every

            self.add_train_loss(actual_loss)

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
                print(f'{batch_count}: {actual_loss:8.6f}{infos_str} | t: {timer():8.4f}', flush=True)

            self.step += 1

    def save_model(self, label):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        model_state_name = f'{label}.pt'
        savepath = os.path.join(self.logdir, model_state_name)
        torch.save(data, savepath)
        self.latest_model_state_name = model_state_name
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def save_losses(self):
        '''
            saves train and val losses to disk
        '''
        train_losses = np.array(self.train_losses)
        val_losses = np.array(self.val_losses)

        train_savepath = os.path.join(self.logdir, 'train_losses.txt')
        val_savepath = os.path.join(self.logdir, 'val_losses.txt')

        with open(train_savepath, 'w') as f:
            f.write(f'Step\tLoss\n')
            for i, loss in enumerate(train_losses):
                f.write(f'{i}\t{loss:8.8f}\n')

        with open(val_savepath, 'w') as f:
            f.write(f'Epoch\tLoss\n')
            for i, loss in enumerate(val_losses):
                f.write(f'{i}\t{loss:8.8f}\n')

        print(f'[ utils/training ] Saved losses to {train_savepath} and {val_savepath}', flush=True)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def validate(self):
        '''
            runs validation on the model
        '''
        if self.dataloader_val is None:
            return None
        
        total_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for i in range(self.val_num_batches):
                batch = next(self.dataloader_val)
                batch = batch_to_device(batch)
        
                loss, infos = self.model.validation_loss(*batch, **self.validation_kwargs)
                total_loss += loss.item()

        avg_val_loss = total_loss / self.val_num_batches
        self.add_val_loss(avg_val_loss)

        return avg_val_loss

    def train(self):
        best_val_loss = float('inf')
        no_improve_counter = 0

        print(f"\nTraining for {self.num_epochs} epochs\n")

        try:
            for i in range(self.num_epochs):
                print(f"Epoch {i} / {self.num_epochs} | {self.dataset} | {self.method} | {self.exp_name}")
                self.train_one_epoch()  

                val_loss = self.validate()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_counter = 0
                    self.save_model('best')
                elif val_loss > best_val_loss + self.min_delta:
                    no_improve_counter += 1
                    print(f"No improvement for {no_improve_counter} epoch(s).")
                    
                    if self.early_stopping and no_improve_counter >= self.patience:
                        print("Early stopping triggered due to convergence.")
                        break

                if val_loss == best_val_loss:
                    print(f"Validation Loss: {val_loss:8.6f} | New best validation loss!")
                else:
                    print(f"Validation Loss: {val_loss:8.6f} | Current best: {best_val_loss:8.6f}")
            

                if i % self.save_freq == 0:
                    self.save_model(f'state_{i}_epochs')

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")

        self.save_model('final')
        self.save_losses()

        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")


            
