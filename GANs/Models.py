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
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256*7*7),
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
    def forward(self, z):
        return self.net(z)

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
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 1))  # Sigmoidしない(BCEWithLogitsLoss)
    def forward(self, x):
        return self.head(self.features(x))

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
