import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Phiên bản nhẹ hơn cho tài nguyên hạn chế
    """

    def __init__(self, in_channels_sat=13, in_channels_met=8, base_filters=32):
        super(CNN, self).__init__()

        # Kết hợp input ngay từ đầu
        total_channels = in_channels_sat + in_channels_met

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(total_channels, base_filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),

            # Output
            nn.Conv2d(base_filters, 1, kernel_size=1),
            nn.ReLU()
        )


    def forward(self, sat_data, met_data):
        # Kết hợp inputs
        x = torch.cat([sat_data, met_data], dim=1)

        # Encode và decode
        encoded = self.encoder(x)
        rainfall = self.decoder(encoded)

        return rainfall