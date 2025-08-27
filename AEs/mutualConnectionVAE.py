# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from simpleVAE import vae_training

device = torch.device("mps")
train_loaders, test_loader = get_mnist_loaders(num_groups=2)

def mutual_train(loops=5, device=device, train_loaders=train_loaders, save=True):
    vae1 = vae_training(device=device, train_loader=train_loaders[0], epochs=5, save=False)
    vae2 = vae_training(device=device, train_loader=train_loaders[1], epochs=5, save=False)

    loader1 = iter(train_loaders[0])
    loader2 = iter(train_loaders[1])

    opt1 = optim.Adam(vae1.parameters(), lr=1e-3)
    opt2 = optim.Adam(vae2.parameters(), lr=1e-3)

    recon_loss = nn.MSELoss(reduction="mean")
    beta = 1.0
    def kl_div(mu, logvar):
        # mean over batch of KL(q||N(0, I))
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    for loop in range(loops):
        try:
            imgs1, _ = next(loader1)
        except StopIteration:
            loader1 = iter(train_loaders[0])
            imgs1, _ = next(loader1)
        try:
            imgs2, _ = next(loader2)
        except StopIteration:
            loader2 = iter(train_loaders[1])
            imgs2, _ = next(loader2)

        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)

        # ae1 output -> ae2 input
        for p in vae2.parameters():
            p.requires_grad = False
        opt1.zero_grad()
        x1_hat, mu1, logvar1 = vae1(imgs1)
        x1_12_hat, _, _ = vae2(x1_hat)
        loss1 = recon_loss(x1_12_hat, imgs2) + beta * kl_div(mu1, logvar1)
        loss1.backward()
        opt1.step()
        for p in vae2.parameters():
            p.requires_grad = True
        # vae2 output -> vae1 input
        for p in vae1.parameters():
            p.requires_grad = False
        opt2.zero_grad()
        x2_hat, mu2, logvar2 = vae2(imgs2)
        x2_21_hat, _, _ = vae1(x2_hat)
        loss2 = recon_loss(x2_21_hat, imgs1) + beta * kl_div(mu2, logvar2)
        loss2.backward()
        opt2.step()
        for p in vae1.parameters():
            p.requires_grad = True

    if save:
        torch.save(vae1.state_dict(), "/Users/sotafujii/PycharmProjects/AIStudy/AEs/pths/m_vae1.pth")
        torch.save(vae2.state_dict(), "/Users/sotafujii/PycharmProjects/AIStudy/AEs/pths/m_vae2.pth")

    return vae1, vae2

def mutual_testing(vae1, vae2, device=device, test_loader=test_loader):
    vae1.eval()
    vae2.eval()
    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recon_vae1, _, _ = vae1(imgs)
        recon_vae2, _, _ = vae2(imgs)
    fig, axes = plt.subplots(3, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'label: {labels[i].item()}')
        axes[1, i].imshow(recon_vae1[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(recon_vae2[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed VAE1')
    axes[2, 0].set_ylabel('Reconstructed VAE2')
    plt.tight_layout()
    plt.show()

# Test the VAEs and visualize results
if __name__ == "__main__":
	mutual_testing(*mutual_train(loops=5))
