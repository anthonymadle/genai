# inference/sample_diffusion.py
import os, sys
import torch
from torchvision.utils import make_grid, save_image

# make root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from models.unet import UNet
from diffusion.diffusion import Diffusion

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load model
    model = UNet().to(device)
    ckpt_path = os.path.join(ROOT, "checkpoints", "diffusion.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2) diffusion helper (same params as training)
    diffusion = Diffusion(img_size=32, device=device, timesteps=300)

    # 3) sample
    with torch.no_grad():
        imgs = diffusion.sample(model, n=16, steps=300)
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2.0

    out_dir = os.path.join(ROOT, "demo", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "diff_grid.png")

    grid = make_grid(imgs, nrow=4)
    save_image(grid, out_path)
    print(f"[OK] saved diffusion samples to {out_path}")

if __name__ == "__main__":
    main()
