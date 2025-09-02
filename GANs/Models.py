# gan_models.py
import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):  # e.g. (-1, C, H, W)
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)


# Unconditional Generator
class GeneratorDCGAN(nn.Module):
    """Unconditional DCGAN Generator: z(100) -> 1x28x28"""
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.ReLU(True),
            View((-1, 256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        # z: (B, z_dim)
        return self.net(z)

    @torch.no_grad()
    def generate(self, n: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=device)
        return self.forward(z)

# Conditional Generator
class CondGeneratorDCGAN(nn.Module):
    """Conditional DCGAN Generator: z(100) + label -> 1x28x28"""
    def __init__(self, z_dim=100, num_classes=10, embed_dim=10):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.ReLU(True),
            View((-1, 256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        z_cond = torch.cat([z, label_embedding], dim=1)
        return self.net(z_cond)

    @torch.no_grad()
    def generate(self, digit: int, n: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=device)
        labels = torch.full((n,), int(digit), dtype=torch.long, device=device)
        return self.forward(z, labels)


# Unconditional Discriminator
class DiscriminatorDCGAN(nn.Module):
    """Unconditional DCGAN Discriminator: 1x28x28 -> real/fake logit"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head_gan = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 1))

    def forward(self, x):
        feat = self.features(x)
        gan_out = self.head_gan(feat)
        return gan_out

# Conditional Discriminator
class CondDiscriminatorDCGAN(nn.Module):
    """Conditional DCGAN Discriminator: 1x28x28 -> real/fake logit, class logits"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head_gan = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 1))
        self.head_cls = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 10))

    def forward(self, x):
        feat = self.features(x)
        gan_out = self.head_gan(feat)
        cls_out = self.head_cls(feat)
        return gan_out, cls_out

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
