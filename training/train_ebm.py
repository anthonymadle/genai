# training/train_ebm.py
import os, sys

print(">>> train_ebm.py: file loaded")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
print(f">>> added ROOT to sys.path: {ROOT}")

import torch
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader

print(">>> torch/torchvision imported in EBM")

# ✅ Use the correct package that actually exists
from ebm.ebm_cnn import EBMCNN
from ebm.utils import make_noisy
print(">>> imported EBMCNN from ebm.ebm_cnn")
print(">>> imported make_noisy from ebm.utils")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 1

def get_dataloader():
    print(">>> EBM: building CIFAR10 dataloader")
    tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    ds = tv.datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def main():
    print(">>> EBM main() starting")
    os.makedirs("checkpoints", exist_ok=True)

    model = EBMCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    loader = get_dataloader()

    step = 0
    for epoch in range(EPOCHS):
        for x, _ in loader:
            step += 1
            x = x.to(DEVICE)
            x_fake = make_noisy(x, sigma=0.1).to(DEVICE)

            e_real = model(x).mean()
            e_fake = model(x_fake).mean()
            loss = e_real - e_fake

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"[EBM] step {step} loss={loss.item():.4f}")

            if step == 500:
                break

    torch.save({"model_state_dict": model.state_dict()}, "checkpoints/ebm.pt")
    print("[EBM] saved checkpoint → checkpoints/ebm.pt")

if __name__ == "__main__":
    print(">>> __main__ in train_ebm.py, calling main()")
    main()
