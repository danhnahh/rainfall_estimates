import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import CNN
from utils import RainfallLoss, calculate_metrics
from lr_schedule import MyScheduler

class Trainer:
    def __init__(self, model, train_loader, val_loader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_step=50, val_step=200,
                 model_save_path='best_model.pth'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_step = log_step
        self.val_step = val_step
        self.model_save_path = model_save_path

        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (sat, met, target) in enumerate(self.train_loader):
            self.global_step += 1
            sat, met, target = sat.to(self.device), met.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(sat, met)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % self.log_step == 0:
                avg_loss = running_loss / (batch_idx + 1)
                tqdm.write(f"Step {self.global_step} | Train Loss: {avg_loss:.4f}")

        return running_loss / len(self.train_loader)

    def validate(self):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        all_metrics = {'RMSE': 0, 'MAE': 0, 'Correlation': 0}
        with torch.no_grad():
            for sat, met, target in tqdm(self.val_loader, desc="Validation"):
                sat, met, target = sat.to(self.device), met.to(self.device), target.to(self.device)
                output = self.model(sat, met)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                metrics = calculate_metrics(output, target)
                for k, v in metrics.items():
                    all_metrics[k] += v

        val_loss = total_loss / len(self.val_loader)
        for k in all_metrics:
            all_metrics[k] /= len(self.val_loader)

        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': all_metrics
            }, self.model_save_path)
            tqdm.write(f"✓ New best model saved at step {self.global_step} with val_loss {val_loss:.4f}")

        self.model.train()
        return val_loss, all_metrics

    def train(self, num_epochs=10):
        history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []}
        for epoch in range(num_epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            train_loss = self.train_one_epoch()
            val_loss, metrics = self.validate() if self.val_loader else (None, None)

            history['train_loss'].append(train_loss)
            if val_loss is not None:
                history['val_loss'].append(val_loss)
                history['val_rmse'].append(metrics['RMSE'])
                history['val_mae'].append(metrics['MAE'])

            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")

            if self.scheduler is not None and val_loss is not None:
                self.scheduler.step(val_loss)

        return history

def plot_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    if 'val_rmse' in history and history['val_rmse']:
        axes[0, 1].plot(history['val_rmse'])
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].grid(True)

    if 'val_mae' in history and history['val_mae']:
        axes[1, 0].plot(history['val_mae'])
        axes[1, 0].set_title('Validation MAE')
        axes[1, 0].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ History plot saved to {save_path}")
