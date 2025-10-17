# helper_lib/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_gan(generator,
              discriminator,
              dataloader,
              device="cpu",
              epochs=5,
              z_dim=100,
              lr=2e-4,
              save_dir="checkpoints",
              save_name="generator.pth"):
    """
    Train a simple GAN on the MNIST dataset.
    """
    os.makedirs(save_dir, exist_ok=True)

    G = generator.to(device)
    D = discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()
    g_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            b = real_imgs.size(0)

            # Train Discriminator
            d_opt.zero_grad()
            real_logits = D(real_imgs)
            real_targets = torch.ones(b, 1, device=device)
            loss_real = criterion(real_logits, real_targets)

            z = torch.randn(b, z_dim, device=device)
            fake_imgs = G(z).detach()
            fake_logits = D(fake_imgs)
            fake_targets = torch.zeros(b, 1, device=device)
            loss_fake = criterion(fake_logits, fake_targets)

            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            d_opt.step()

            # Train Generator
            g_opt.zero_grad()
            z = torch.randn(b, z_dim, device=device)
            gen_imgs = G(z)
            gen_logits = D(gen_imgs)
            g_targets = torch.ones(b, 1, device=device)
            g_loss = criterion(gen_logits, g_targets)
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix({"d_loss": d_loss.item(), "g_loss": g_loss.item()})

        # Save generator checkpoint
        torch.save(G.state_dict(), os.path.join(save_dir, save_name))

    print("✅ Training finished successfully.")
    return G, D
