# Configuration
import torch
import torch.nn as nn

# AE definition
class AE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )
        self.classifier = nn.Linear(latent, 10)  # 10クラス分類用

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z).view(-1, 1, 28, 28)
        logits = self.classifier(z)
        return x_hat, logits

# VAE definition
class VAE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.mu    = nn.Linear(256, latent)
        self.logvar= nn.Linear(256, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28*28), nn.Sigmoid()
        )
        self.classifier = nn.Linear(latent, 10)  # 10クラス分類用

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization

    def forward(self, x):
        h   = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z   = self.reparam(mu, logvar)
        x_hat = self.dec(z).view(-1, 1, 28, 28)
        logits = self.classifier(z)
        return x_hat, mu, logvar, logits

    @staticmethod
    def vae_loss(x_hat, x, mu, logvar):
        recon = nn.functional.mse_loss(x_hat, x, reduction='sum')  # using MSE in sake of responsiveness (use bce_loss for better results)
        kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon + kld) / x.size(0)

# Conditional AE definition (decoder conditioned on class label)
class ConditionalAE(nn.Module):
    def __init__(self, latent=32, num_classes=10):
        super().__init__()
        self.latent = latent
        self.num_classes = num_classes
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent)
        )
        # Decoder expects concatenated [z, y_onehot]
        self.dec = nn.Sequential(
            nn.Linear(latent + num_classes, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Sigmoid()
        )
        self.classifier = nn.Linear(latent, num_classes)

    def _one_hot(self, y):
        if y.dim() == 1:
            # Long tensor of class indices
            oh = nn.functional.one_hot(y, num_classes=self.num_classes).float()
        else:
            oh = y.float()
        return oh

    def forward(self, x, y):
        z = self.enc(x)
        y_oh = self._one_hot(y)
        zy = torch.cat([z, y_oh], dim=1)
        x_hat = self.dec(zy).view(-1, 1, 28, 28)
        logits = self.classifier(z)
        return x_hat, logits

    @torch.no_grad()
    def generate(self, y, n=1):
        if isinstance(y, int):
            y = torch.tensor([y]*n, dtype=torch.long, device=next(self.parameters()).device)
        elif isinstance(y, (list, tuple)):
            y = torch.tensor(list(y), dtype=torch.long, device=next(self.parameters()).device)
        elif not torch.is_tensor(y):
            raise ValueError("y must be int, list/tuple of ints, or a tensor")
        if y.dim() == 1:
            batch = y.size(0)
        else:
            batch = y.size(0)
        z = torch.randn(batch, self.latent, device=next(self.parameters()).device)
        y_oh = self._one_hot(y)
        zy = torch.cat([z, y_oh], dim=1)
        imgs = self.dec(zy).view(-1, 1, 28, 28)
        return imgs

    def decode(self, z, y):
        y_oh = self._one_hot(y)
        zy = torch.cat([z, y_oh], dim=1)
        return self.dec(zy).view(-1, 1, 28, 28)

# Conditional VAE definition (encoder and decoder conditioned on class label)
class ConditionalVAE(nn.Module):
    def __init__(self, latent=32, num_classes=10):
        super().__init__()
        self.latent = latent
        self.num_classes = num_classes
        # Encoder takes [x_flat, y_onehot]
        self.enc = nn.Sequential(
            nn.Linear(28*28 + num_classes, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)
        # Decoder takes [z, y_onehot]
        self.dec = nn.Sequential(
            nn.Linear(latent + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28*28), nn.Sigmoid()
        )
        self.classifier = nn.Linear(latent, num_classes)

    def _one_hot(self, y):
        if y.dim() == 1:
            return nn.functional.one_hot(y, num_classes=self.num_classes).float()
        return y.float()

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        b = x.size(0)
        x_flat = x.view(b, -1)
        y_oh = self._one_hot(y)
        xy = torch.cat([x_flat, y_oh], dim=1)
        h = self.enc(xy)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        zy = torch.cat([z, y_oh], dim=1)
        x_hat = self.dec(zy).view(-1, 1, 28, 28)
        logits = self.classifier(z)
        return x_hat, mu, logvar, logits

    @staticmethod
    def vae_loss(x_hat, x, mu, logvar):
        recon = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon + kld) / x.size(0)

    @torch.no_grad()
    def generate(self, y, n=1, device=None):
        device = device or next(self.parameters()).device
        if isinstance(y, int):
            y = torch.tensor([y]*n, dtype=torch.long, device=device)
        elif isinstance(y, (list, tuple)):
            y = torch.tensor(list(y), dtype=torch.long, device=device)
        elif torch.is_tensor(y):
            y = y.to(device)
        else:
            raise ValueError("y must be int, list/tuple, or tensor")
        batch = y.size(0)
        z = torch.randn(batch, self.latent, device=device)
        y_oh = self._one_hot(y)
        zy = torch.cat([z, y_oh], dim=1)
        imgs = self.dec(zy).view(-1, 1, 28, 28)
        return imgs

    def encode(self, x, y):
        b = x.size(0)
        x_flat = x.view(b, -1)
        y_oh = self._one_hot(y)
        xy = torch.cat([x_flat, y_oh], dim=1)
        h = self.enc(xy)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        return mu, logvar, z

    def decode(self, z, y):
        y_oh = self._one_hot(y)
        zy = torch.cat([z, y_oh], dim=1)
        return self.dec(zy).view(-1, 1, 28, 28)
