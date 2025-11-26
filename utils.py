import torch
import torch.nn as nn


class RainfallLoss(nn.Module):
    """Custom loss cho rainfall estimation"""

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)


def calculate_metrics(pred, target):
    """Tính RMSE, MAE, Correlation"""
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(pred - target))

        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Correlation
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()

        numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
        denominator = torch.sqrt(((pred_flat - pred_mean) ** 2).sum() *
                                 ((target_flat - target_mean) ** 2).sum())

        corr = numerator / (denominator + 1e-8)

    return {
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'Correlation': corr.item()
    }