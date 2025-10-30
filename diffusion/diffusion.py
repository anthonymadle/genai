import torch

class Diffusion:
    def __init__(self, img_size=32, device="cpu", timesteps=300):
        self.img_size = img_size
        self.device = device
        self.T = timesteps

        self.betas = torch.linspace(1e-4, 0.02, self.T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ac = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        om = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return ac * x0 + om * noise

    @torch.no_grad()
    def sample(self, model, n=16, steps=None):
        if steps is None:
            steps = self.T
        x = torch.randn(n, 3, self.img_size, self.img_size, device=self.device)
        for i in reversed(range(steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t.float().view(n,1,1,1))
            beta_t = self.betas[t].view(-1,1,1,1)
            sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
            sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1,1,1,1)
            model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ac * eps)
            if i > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t].view(-1,1,1,1)
                x = model_mean + torch.sqrt(var) * noise
            else:
                x = model_mean
        return x
