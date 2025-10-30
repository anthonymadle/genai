# ebm/utils.py
import torch

def make_noisy(x, sigma: float = 0.1):
    """
    Add small Gaussian noise to a batch of images.
    x: (B, C, H, W)
    returns: x + noise
    """
    return x + sigma * torch.randn_like(x)
