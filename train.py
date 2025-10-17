import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.model import Generator, Discriminator
from helper_lib.trainer import train_gan

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MNIST -> tensor in [0,1] -> normalize to [-1,1] (pairs with G's Tanh)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    G = Generator(z_dim=100)
    D = Discriminator()

    os.makedirs("checkpoints", exist_ok=True)
    train_gan(G, D, loader,
              device=device,
              epochs=5,            # start small; bump to 10+ if you want better digits
              z_dim=100,
              lr=2e-4,
              save_dir="checkpoints",
              save_name="generator.pth")

    print("Training done. Generator saved to checkpoints/generator.pth")

if __name__ == "__main__":
    main()
