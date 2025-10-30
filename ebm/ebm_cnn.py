import torch
import torch.nn as nn

class EBMCNN(nn.Module):
    """Very small energy model for 32x32 RGB images."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.features(x)       # (B, 256, 1, 1)
        h = h.view(h.size(0), -1)  # (B, 256)
        e = self.head(h)           # (B, 1)
        return e.squeeze(1)        # (B,)
