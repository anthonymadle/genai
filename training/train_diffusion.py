# training/train_diffusion.py
import os, sys

print(">>> train_diffusion.py: file loaded")

# make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
print(f">>> added ROOT to sys.path: {ROOT}")

# try imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision as tv
    from torch.utils.data import DataLoader
    print(">>> torch & torchvision imported")
except Exception as e:
    print("!!! import error:", e)
    raise

# try project imports
try:
    from models.unet import UNet
    print(">>> imported UNet from models.unet")
except Exception as e:
    print("!!! could not import UNet:", e)
    raise

try:
    from diffusion.diffusion import Diffusion
    print(">>> imported Diffusion from diffusion.diffusion")
except Exception as e:
    print("!!! could not import Diffusion:", e)
    raise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 1
TIMESTEPS = 300

def get_dataloader():
    print(">>> building dataloader (CIFAR10)")
    tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    ds = tv.datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def main():
    print(">>> main() starting...")
    os.makedirs("checkpoints", exist_ok=True)

    model = UNet().to(DEVICE)
    diffusion = Diffusion(img_size=32, device=DEVICE, timesteps=TIMESTEPS)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    loader = get_dataloader()

    model.train()
    for epoch in range(EPOCHS):
        for i, (x, _) in enumerate(loader):
            x = x.to(DEVICE)
            b = x.size(0)

            t = torch.randint(0, TIMESTEPS, (b,), device=DEVICE).long()
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x, t, noise)

            pred = model(x_noisy, t.float().view(b,1,1,1))
            loss = nn.MSELoss()(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i + 1) % 100 == 0:
                print(f"[DIFFUSION] epoch {epoch+1}/{EPOCHS} step {i+1}/{len(loader)} loss={loss.item():.4f}")

    torch.save({"model_state_dict": model.state_dict()}, "checkpoints/diffusion.pt")
    print("[DIFFUSION] saved checkpoint → checkpoints/diffusion.pt")

if __name__ == "__main__":
    print(">>> __main__ reached, calling main()")
    main()
