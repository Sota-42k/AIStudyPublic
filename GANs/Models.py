# gan_models.py
import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):  # e.g. (-1, C, H, W)
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class GeneratorDCGAN(nn.Module):
    """ z(100) -> 1x28x28 """
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
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 7->14
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 14->28
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),  # [-1,1]
        )

    def forward(self, z, labels):
        # z: (B, z_dim), labels: (B,)
        label_embedding = self.label_emb(labels)  # (B, embed_dim)
        z_cond = torch.cat([z, label_embedding], dim=1)  # (B, z_dim + embed_dim)
        return self.net(z_cond)

    @torch.no_grad()
    def generate(self, digit: int, n: int = 1, device=None):
        """Generate n images conditioned on a single digit label."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=device)
        labels = torch.full((n,), int(digit), dtype=torch.long, device=device)
        return self.forward(z, labels)

class DiscriminatorDCGAN(nn.Module):
    """ 1x28x28 -> real/fake logit """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),     # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),   # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),  # 7->4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head_gan = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 1))  # real/fake
        self.head_cls = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 10)) # class logits

    def forward(self, x):
        feat = self.features(x)
        gan_out = self.head_gan(feat)  # (B,1)
        cls_out = self.head_cls(feat)  # (B,10)
        return gan_out, cls_out

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
