# api/main.py
from fastapi import FastAPI, Response, Query
import os
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import Generator  # this import is solid

app = FastAPI(title="MNIST GAN API (Self-contained)")

GEN_CKPT = "checkpoints/generator.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _grid_png_bytes(g: torch.nn.Module, num_samples: int = 16, z_dim: int = 100) -> bytes:
    """Generate an n x n grid of digits and return PNG bytes (no external imports)."""
    g.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=DEVICE)
        imgs = g(z)                         # [-1,1]
        imgs = (imgs + 1.0) / 2.0           # [0,1] for viewing
        grid = make_grid(imgs, nrow=int(np.sqrt(num_samples)))
        arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # If single-channel, save as 'L'
        if arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)
            pil = Image.fromarray(arr, mode="L")
        else:
            pil = Image.fromarray(arr)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Use /generate?n=16 to get a PNG grid of digits.",
        "checkpoint_exists": os.path.exists(GEN_CKPT),
        "device": DEVICE,
    }

@app.get("/generate")
def generate(n: int = Query(16, ge=1, le=64)):
    if not os.path.exists(GEN_CKPT):
        return {"ok": False, "error": f"Checkpoint not found at {GEN_CKPT}. Run: python train.py"}

    G = Generator(z_dim=100)
    G.load_state_dict(torch.load(GEN_CKPT, map_location=DEVICE))
    G.to(DEVICE)

    png_bytes = _grid_png_bytes(G, num_samples=n, z_dim=100)
    return Response(content=png_bytes, media_type="image/png")
