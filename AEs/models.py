# Configuration
import torch
import torch.nn as nn

# AE definition
class AE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(),
            nn.Linear(128, 28*28), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z).view(-1, 1, 28, 28)
        return x_hat

# VAE definition
class VAE(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU())
        self.mu    = nn.Linear(256, latent)
        self.logvar= nn.Linear(256, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization

    def forward(self, x):
        h   = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z   = self.reparam(mu, logvar)
        x_hat = self.dec(z).view(-1, 1, 28, 28)
        return x_hat, mu, logvar

    @staticmethod
    def vae_loss(x_hat, x, mu, logvar):
        recon = nn.functional.mse_loss(x_hat, x, reduction='sum')  # using MSE in sake of responsiveness (use bce_loss for better results)
        kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon + kld) / x.size(0)
