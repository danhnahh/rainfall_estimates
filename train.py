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
    def __init__(self, model, train_dataloader, val_dataloader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_step=50, val_step=200,
                 model_save_path='best_model.pth'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_step = log_step
        self.val_step = val_step
        self.model_save_path = model_save_path

        self.best_val_loss = float('inf')
        self.global_step = 0

        self.model.to(self.device)

        # Tổng số tham số
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_loss(self, batch):
        """Tính loss cho một batch"""
        sat, met, target = batch
        # DEBUG: In shape trước khi .to(device)
        print(f"DEBUG - sat shape before .to: {sat.shape}")
        print(f"DEBUG - met shape before .to: {met.shape}")
        print(f"DEBUG - target shape before .to: {target.shape}")

        sat = sat.to(self.device)
        met = met.to(self.device)
        target = target.to(self.device)

        # DEBUG: In shape sau khi .to(device)
        print(f"DEBUG - sat shape after .to: {sat.shape}")
        print(f"DEBUG - met shape after .to: {met.shape}")

        outputs = self.model(sat, met)
        loss = self.criterion(outputs, target)

        return loss

    def train(self, num_epochs=10, accumulate_steps=1, save_by='loss'):
        """
        Training loop với gradient accumulation
        accumulate_steps: số batch gom lại trước khi update optimizer
        save_by: 'loss', 'rmse', hoặc 'mae' - metric để save best model
        """
        total_batches = len(self.train_dataloader)
        total_updates = (total_batches + accumulate_steps - 1) // accumulate_steps  # ceil

        history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_corr': []}

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            loop = tqdm(total=total_updates, desc=f"Epoch {epoch + 1}/{num_epochs}",
                        unit="it", ncols=100)
            update_count = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.global_step += 1

                loss = self.get_loss(batch)
                loss = loss / accumulate_steps  # chia loss cho accumulate_steps
                loss.backward()

                # Cập nhật optimizer sau mỗi accumulate_steps batch
                if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == total_batches:
                    # --- grad norm ---
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += (p.grad.data.norm(2).item()) ** 2
                    total_norm = total_norm ** 0.5

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    running_loss += loss.item() * batch[0].size(0) * accumulate_steps  # nhân lại
                    update_count += 1
                    loop.update(1)  # update tqdm theo số update, không phải batch

                    # --- logging ---
                    if update_count % self.log_step == 0:
                        avg_loss = running_loss / ((batch_idx + 1) * self.train_dataloader.batch_size)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        tqdm.write(f"Step {self.global_step}\tTrain Loss: {avg_loss:.7f}\t"
                                   f"Grad Norm: {total_norm:.7f}\tLR: {current_lr:.6f}")

                    # --- validation ---
                    if self.val_dataloader is not None and update_count % self.val_step == 0:
                        val_loss, metrics = self.validate_and_save(save_by=save_by)
                        tqdm.write(f"Val Loss: {val_loss:.4f} | RMSE: {metrics['RMSE']:.4f} | "
                                   f"MAE: {metrics['MAE']:.4f}")
                else:
                    running_loss += loss.item() * batch[0].size(0)

            # --- cuối epoch ---
            epoch_train_loss = running_loss / len(self.train_dataloader.dataset)
            history['train_loss'].append(epoch_train_loss)

            if self.val_dataloader is not None:
                epoch_val_loss, metrics = self.validate_and_save(save_by=save_by)
                history['val_loss'].append(epoch_val_loss)
                history['val_rmse'].append(metrics['RMSE'])
                history['val_mae'].append(metrics['MAE'])
                history['val_corr'].append(metrics['Correlation'])

                print(f"Epoch {epoch + 1}/{num_epochs} finished, "
                      f"Train Loss: {epoch_train_loss:.7f}, Val Loss: {epoch_val_loss:.7f}")
                print(f"  RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, Corr: {metrics['Correlation']:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} finished, Train Loss: {epoch_train_loss:.7f}")

            loop.close()

        return self.model, history

    def validate_and_save(self, save_by='loss'):
        """
        Validation và save model
        save_by: 'loss' (default), 'rmse', hoặc 'mae' - metric để quyết định best model
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_metrics = {'RMSE': 0, 'MAE': 0, 'Correlation': 0}

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                loss = self.get_loss(batch)

                batch_size = batch[0].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Tính metrics
                sat, met, target = batch
                sat = sat.to(self.device)
                met = met.to(self.device)
                target = target.to(self.device)
                outputs = self.model(sat, met)

                metrics = calculate_metrics(outputs, target)
                for k, v in metrics.items():
                    all_metrics[k] += v * batch_size

        # loss mean trên dataset
        val_loss = total_loss / total_samples
        for k in all_metrics:
            all_metrics[k] /= total_samples

        # ----- Save theo best metric -----
        # Chọn metric để save
        if save_by == 'loss':
            current_metric = val_loss
            is_better = current_metric < self.best_val_loss
            metric_name = 'val_loss'
        elif save_by == 'rmse':
            current_metric = all_metrics['RMSE']
            is_better = current_metric < self.best_val_loss
            metric_name = 'RMSE'
        elif save_by == 'mae':
            current_metric = all_metrics['MAE']
            is_better = current_metric < self.best_val_loss
            metric_name = 'MAE'
        else:
            current_metric = val_loss
            is_better = current_metric < self.best_val_loss
            metric_name = 'val_loss'

        if is_better:
            self.best_val_loss = current_metric
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': all_metrics,
                'epoch': self.global_step
            }, self.model_save_path)
            tqdm.write(f"Step {self.global_step}: New best model saved with {metric_name} {current_metric:.6f}")
            tqdm.write(f"  Metrics - RMSE: {all_metrics['RMSE']:.4f}, MAE: {all_metrics['MAE']:.4f}, "
                       f"Corr: {all_metrics['Correlation']:.4f}")

        self.model.train()
        return val_loss, all_metrics

if __name__ == "__main__":
    from model import CNN
    from utils import RainfallDataset
    from torch.utils.data import random_split

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GRADIENT_ACCUMULATION_STEPS = 4
    VAL_BATCH_SIZE = 32
    BATCH_SIZE = 8
    VAL_RATIO = 0.2
    EPOCHS = 10
    log_step = 50
    val_step = 200
    learning_rate = 1e-4
    weight_decay = 1e-5
    model_save_path = 'best_model.pth'
    SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.1
    GAMMA = 0.95

    model = CNN()
    criterion = RainfallLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    dataset = RainfallDataset()

    N = len(dataset)
    train_size = int(N * (1 - VAL_RATIO))

    # Index tách theo thứ tự không random
    indices = list(range(N))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Tính total updates cho scheduler
    total_updates = (len(train_dataset) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS * EPOCHS

    scheduler = MyScheduler(
        optimizer,
        total_steps=total_updates,
        scheduler_type=SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        gamma=GAMMA
    )

    # Khởi tạo trainer
    trainer = Trainer(
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        log_step=log_step,
        val_step=val_step,
        model_save_path=model_save_path
    )

    # Train model
    trained_model, history = trainer.train(
        num_epochs=EPOCHS,
        accumulate_steps=GRADIENT_ACCUMULATION_STEPS,
        save_by='rmse'  # Có thể chọn 'loss', 'rmse', hoặc 'mae'
    )
