import torch
import torch.nn as nn

class PrototypicalNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        return x
