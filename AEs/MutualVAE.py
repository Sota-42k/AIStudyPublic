# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from SimpleVAE import vae_train

def mutual_train(pretrain_epochs=5, loops=5, device=None, train_loaders=None, save=True, scheduler_type=None, scheduler_kwargs=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if train_loaders is None:
        train_loaders, _ = get_mnist_loaders(num_groups=2)

    vae1 = vae_train(device=device, train_loader=train_loaders[0], epochs=pretrain_epochs, save=False, scheduler_type=None)
    vae2 = vae_train(device=device, train_loader=train_loaders[1], epochs=pretrain_epochs, save=False, scheduler_type=None)

    train_loader1 = iter(train_loaders[0])
    train_loader2 = iter(train_loaders[1])

    opt1 = optim.Adam(vae1.parameters(), lr=1e-3)
    opt2 = optim.Adam(vae2.parameters(), lr=1e-3)

    # Scheduler setup for both optimizers
    scheduler1 = None
    scheduler2 = None
    if scheduler_type is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        if scheduler_type == 'StepLR':
            scheduler1 = optim.lr_scheduler.StepLR(opt1, **scheduler_kwargs)
            scheduler2 = optim.lr_scheduler.StepLR(opt2, **scheduler_kwargs)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, **scheduler_kwargs)
            scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, **scheduler_kwargs)
        elif scheduler_type == 'ExponentialLR':
            scheduler1 = optim.lr_scheduler.ExponentialLR(opt1, **scheduler_kwargs)
            scheduler2 = optim.lr_scheduler.ExponentialLR(opt2, **scheduler_kwargs)
        # Add more schedulers as needed

    recon_loss = nn.MSELoss(reduction="mean")
    beta = 1.0
    def kl_div(mu, logvar):
        # mean over batch of KL(q||N(0, I))
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    for loop in range(loops):
        try:
            imgs1, _ = next(train_loader1)
        except StopIteration:
            train_loader1 = iter(train_loaders[0])
            imgs1, _ = next(train_loader1)
        try:
            imgs2, _ = next(train_loader2)
        except StopIteration:
            train_loader2 = iter(train_loaders[1])
            imgs2, _ = next(train_loader2)

        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)

        # vae1 output -> vae2 input
        opt1.zero_grad()
        x1_hat, mu1, logvar1 = vae1(imgs1)
        x1_12_hat, _, _ = vae2(x1_hat)
        loss1 = recon_loss(x1_12_hat, imgs2) + beta * kl_div(mu1, logvar1)
        loss1.backward()
        # Zero gradients for vae2 so only vae1 is updated
        for p in vae2.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        opt1.step()
        if scheduler1 is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler1.step(loss1.item())
            else:
                scheduler1.step()
        # vae2 output -> vae1 input
        opt2.zero_grad()
        x2_hat, mu2, logvar2 = vae2(imgs2)
        x2_21_hat, _, _ = vae1(x2_hat)
        loss2 = recon_loss(x2_21_hat, imgs1) + beta * kl_div(mu2, logvar2)
        loss2.backward()
        # Zero gradients for vae1 so only vae2 is updated
        for p in vae1.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        opt2.step()
        if scheduler2 is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler2.step(loss2.item())
            else:
                scheduler2.step()

    if save:
        torch.save(vae1.state_dict(), "/Volumes/Buffalo-SSD/AIStudy/AEs/pths/m_vae1.pth")
        torch.save(vae2.state_dict(), "/Volumes/Buffalo-SSD/AIStudy/AEs/pths/m_vae2.pth")

    return vae1, vae2

def mutual_test(vae1, vae2, device=None, test_loader=None, save_fig=False):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders()

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
    if save_fig:
        plt.savefig("AEs/samples/mutualVAE_test.png")
    plt.show()

# Test the VAEs and visualize results
if __name__ == "__main__":
	mutual_test(*mutual_train(loops=5))
