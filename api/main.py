from fastapi import FastAPI
import os, sys, torch
from torchvision.utils import make_grid, save_image

# --- make project importable ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

app = FastAPI(title="GenAI Assignment 4 – Diffusion + EBM")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- IMPORT YOUR MODELS ----
from models.unet import UNet
from diffusion.diffusion import Diffusion
from ebm.ebm_cnn import EBMCNN

# ---- LOAD DIFFUSION CHECKPOINT ----
diff_ckpt_path = os.path.join(ROOT, "checkpoints", "diffusion.pt")
diff_unet = UNet().to(DEVICE)
diff_ckpt = torch.load(diff_ckpt_path, map_location=DEVICE)
diff_unet.load_state_dict(diff_ckpt["model_state_dict"])
diff_unet.eval()

diff_runner = Diffusion(img_size=32, device=DEVICE, timesteps=300)

# ---- LOAD EBM CHECKPOINT ----
ebm_ckpt_path = os.path.join(ROOT, "checkpoints", "ebm.pt")
ebm_model = EBMCNN().to(DEVICE)
ebm_ckpt = torch.load(ebm_ckpt_path, map_location=DEVICE)
ebm_model.load_state_dict(ebm_ckpt["model_state_dict"])
ebm_model.eval()

# make sure output dir exists
OUT_DIR = os.path.join(ROOT, "demo", "api")
os.makedirs(OUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"msg": "GenAI Assignment 4 API is running"}


@app.post("/generate/diffusion")
def generate_diffusion(num_images: int = 4, steps: int = 200):
    with torch.no_grad():
        imgs = diff_runner.sample(diff_unet, n=num_images, steps=steps)
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2.0

    grid = make_grid(imgs, nrow=4)
    out_path = os.path.join(OUT_DIR, f"diffusion_{num_images}.png")
    save_image(grid, out_path)
    return {"model": "diffusion", "saved_to": out_path}


@app.post("/generate/ebm")
def generate_ebm(num_images: int = 4, steps: int = 200, step_size: float = 1e-3):
    x = torch.randn(num_images, 3, 32, 32, device=DEVICE).requires_grad_(True)
    for _ in range(steps):
        energy = ebm_model(x).sum()
        grad, = torch.autograd.grad(energy, x, create_graph=False)
        x = x - step_size * grad + (2 * step_size) ** 0.5 * torch.randn_like(x)
        x = x.clamp_(-1, 1).detach().requires_grad_(True)

    imgs = (x + 1) / 2.0 if x.min() < 0 else x
    grid = make_grid(imgs, nrow=4)
    out_path = os.path.join(OUT_DIR, f"ebm_{num_images}.png")
    save_image(grid, out_path)
    return {"model": "ebm", "saved_to": out_path}
