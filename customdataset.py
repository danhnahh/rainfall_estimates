import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import glob


class RainfallDataset(Dataset):
    """
    Dataset khi dữ liệu đã load sẵn vào memory (numpy arrays)
    Dùng cho dataset nhỏ
    """

    def __init__(self, sat_data, met_data, rainfall_data,
                 transform=None, normalize=True):
        """
        Args:
            sat_data: numpy array (N, 13, H, W)
            met_data: numpy array (N, 8, H, W)
            rainfall_data: numpy array (N, 1, H, W) hoặc (N, H, W)
        """
        assert len(sat_data) == len(met_data) == len(rainfall_data), \
            "All arrays must have same length"

        self.sat_data = sat_data
        self.met_data = met_data

        # Ensure rainfall has shape (N, 1, H, W)
        if rainfall_data.ndim == 3:
            self.rainfall_data = rainfall_data[:, np.newaxis, :, :]
        else:
            self.rainfall_data = rainfall_data

        self.transform = transform
        self.normalize = normalize

        # Compute statistics
        if self.normalize:
            self.sat_mean = sat_data.mean(axis=(0, 2, 3), keepdims=True)
            self.sat_std = sat_data.std(axis=(0, 2, 3), keepdims=True)
            self.met_mean = met_data.mean(axis=(0, 2, 3), keepdims=True)
            self.met_std = met_data.std(axis=(0, 2, 3), keepdims=True)

        print(f"RainfallDataset initialized with {len(sat_data)} samples")

    def __len__(self):
        return len(self.sat_data)

    def __getitem__(self, idx):
        sat = self.sat_data[idx].copy()
        met = self.met_data[idx].copy()
        rainfall = self.rainfall_data[idx].copy()

        # Normalize
        if self.normalize:
            sat = (sat - self.sat_mean) / (self.sat_std + 1e-8)
            met = (met - self.met_mean) / (self.met_std + 1e-8)

        # Convert to tensors
        sat = torch.from_numpy(sat).float()
        met = torch.from_numpy(met).float()
        rainfall = torch.from_numpy(rainfall).float()

        # Apply transforms
        if self.transform:
            combined = torch.cat([sat, met, rainfall], dim=0)
            combined = self.transform(combined)
            sat = combined[:13]
            met = combined[13:21]
            rainfall = combined[21:]

        return sat, met, rainfall
