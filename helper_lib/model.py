# helper_lib/model.py
import torch
import torch.nn as nn

__all__ = ["Generator", "Discriminator"]  # helps avoid weird imports

def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 28x28
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                 # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)      # (B,128,7,7)
        x = self.deconv(x)             # (B,1,28,28) in [-1,1]
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.apply(_init_weights)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # raw logit
