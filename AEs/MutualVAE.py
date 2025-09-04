# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from SimpleVAE import vae_train

# base directory for saving relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def mutual_train(pretrain_epochs=5, loops=1000, device=None, train_loaders=None, save=True, scheduler_type=None, scheduler_kwargs=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders()
    vae1 = vae_train(device=device, train_loader=train_loader, epochs=pretrain_epochs, save=False, scheduler_type=None)
    vae2 = vae_train(device=device, train_loader=train_loader, epochs=pretrain_epochs, save=False, scheduler_type=None)
    train_iter1 = iter(train_loader)
    train_iter2 = iter(train_loader)
    opt1 = optim.Adam(vae1.parameters(), lr=1e-3)
    opt2 = optim.Adam(vae2.parameters(), lr=1e-3)
    cls_loss_fn = nn.CrossEntropyLoss()
    scheduler1 = scheduler2 = None
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
    recon_loss = nn.MSELoss(reduction="mean")
    beta = 1.0
    def kl_div(mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    for loop in range(loops):
        try:
            imgs1, labels1 = next(train_iter1)
        except StopIteration:
            train_iter1 = iter(train_loader)
            imgs1, labels1 = next(train_iter1)
        try:
            imgs2, labels2 = next(train_iter2)
        except StopIteration:
            train_iter2 = iter(train_loader)
            imgs2, labels2 = next(train_iter2)
        imgs1 = imgs1.to(device); labels1 = labels1.to(device)
        imgs2 = imgs2.to(device); labels2 = labels2.to(device)
        opt1.zero_grad()
        x1_hat, mu1, logvar1, logits1 = vae1(imgs1, labels1)
        with torch.no_grad():
            x1_12_hat, _, _, logits2 = vae2(x1_hat, labels2)
        loss1 = recon_loss(x1_12_hat, imgs2) + beta * kl_div(mu1, logvar1) + cls_loss_fn(logits1, labels1) + cls_loss_fn(logits2, labels2)
        loss1.backward(); opt1.step()
        if scheduler1 is not None:
            if scheduler_type == 'ReduceLROnPlateau': scheduler1.step(loss1.item())
            else: scheduler1.step()
        opt2.zero_grad()
        x2_hat, mu2, logvar2, logits2b = vae2(imgs2, labels2)
        with torch.no_grad():
            x2_21_hat, _, _, logits1b = vae1(x2_hat, labels1)
        loss2 = recon_loss(x2_21_hat, imgs1) + beta * kl_div(mu2, logvar2) + cls_loss_fn(logits2b, labels2) + cls_loss_fn(logits1b, labels1)
        loss2.backward(); opt2.step()
        if scheduler2 is not None:
            if scheduler_type == 'ReduceLROnPlateau': scheduler2.step(loss2.item())
            else: scheduler2.step()
    if save:
        base = os.path.join(os.path.dirname(__file__), "pths")
        torch.save(vae1.state_dict(), os.path.join(base, "m_vae1.pth"))
        torch.save(vae2.state_dict(), os.path.join(base, "m_vae2.pth"))
        print("Saved MutualVAE models")
    return (vae1, vae2)

def mutual_test(vae1, vae2, device=None, test_loader=None, save_fig=False):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders()

    vae1.eval()
    vae2.eval()
    
    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        recon_vae1, _, _, logits1 = vae1(imgs, labels)
        recon_vae2, _, _, logits2 = vae2(imgs, labels)
        preds1 = logits1.argmax(dim=1)
        preds2 = logits2.argmax(dim=1)

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'label: {labels[i].item()}')
        axes[1, i].imshow(recon_vae1[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(recon_vae2[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[3, i].text(0.5, 0.5, f'pred1:{preds1[i].item()}\npred2:{preds2[i].item()}', fontsize=10, ha='center')
        axes[3, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed VAE1')
    axes[2, 0].set_ylabel('Reconstructed VAE2')
    axes[3, 0].set_ylabel('Predicted')
    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)
        plt.savefig(os.path.join(BASE_DIR, "samples", "mutualVAE_test.png"))
    plt.show()

# Test the VAEs and visualize results
if __name__ == "__main__":
	mutual_test(*mutual_train(loops=5))
