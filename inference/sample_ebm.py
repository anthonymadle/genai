# inference/sample_ebm.py
import os, sys, torch
from torchvision.utils import make_grid, save_image

print(">>> sample_ebm.py: starting")

# project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
print(f">>> ROOT = {ROOT}")

# use the ebm model you just trained
from ebm.ebm_cnn import EBMCNN
from ebm.utils import make_noisy  # just to prove import works

def langevin_sample(model, n=16, steps=200, step_size=1e-3, device="cpu"):
    print(f">>> Langevin: n={n}, steps={steps}, step_size={step_size}, device={device}")
    x = torch.randn(n, 3, 32, 32, device=device)
    x.requires_grad_(True)
    for i in range(steps):
        energy = model(x).sum()
        grad, = torch.autograd.grad(energy, x, create_graph=False)
        x = x - step_size * grad + (2 * step_size) ** 0.5 * torch.randn_like(x)
        x = x.clamp_(-1, 1).detach().requires_grad_(True)
        if (i + 1) % 50 == 0:
            print(f"   step {i+1}/{steps}")
    return x.detach()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load your trained EBM
    ckpt_path = os.path.join(ROOT, "checkpoints", "ebm.pt")
    print(f">>> loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print("!!! ERROR: checkpoint not found")
        return

    ckpt = torch.load(ckpt_path, map_location=device)

    model = EBMCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(">>> model loaded")

    # 2) sample
    imgs = langevin_sample(model, n=16, steps=200, step_size=1e-3, device=device)

    # 3) scale to [0,1] for saving
    if imgs.min() < 0:
        imgs = (imgs + 1) / 2.0

    # 4) save to demo/figures
    out_dir = os.path.join(ROOT, "demo", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ebm_grid.png")

    grid = make_grid(imgs, nrow=4)
    save_image(grid, out_path)
    print(f"[OK] saved EBM samples → {out_path}")

if __name__ == "__main__":
    main()
