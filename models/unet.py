# models/unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    """
    Tiny UNet for 32x32 images.
    We ignore time embeddings to keep it dead simple.
    """
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        # encoder
        self.down1 = DoubleConv(in_channels, base)           # 32x32 -> 32x32
        self.pool1 = nn.MaxPool2d(2)                         # 32x32 -> 16x16
        self.down2 = DoubleConv(base, base * 2)              # 16x16 -> 16x16
        self.pool2 = nn.MaxPool2d(2)                         # 16x16 -> 8x8
        self.down3 = DoubleConv(base * 2, base * 4)          # 8x8 -> 8x8

        # decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 8x8 -> 16x16
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 16x16 -> 32x32
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t=None):
        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))

        # decoder
        u2 = self.up2(x3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv1(u1)

        out = self.out(u1)
        return out
