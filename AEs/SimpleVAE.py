# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders, get_mnist_digit_loader
from Models import ConditionalVAE as VAE

# base directory for saving relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# VAE training
def vae_train(device=None, train_loader=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
    if device is None: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # If no loader is provided, use full MNIST and return a single model
    if train_loader is None:
        train_loader, _ = get_mnist_loaders()

    # Otherwise, train a single VAE on the given loader
    vae = VAE().to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-3)
    cls_loss_fn = nn.CrossEntropyLoss()

    scheduler = None
    if scheduler_type is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(opt, **scheduler_kwargs)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, **scheduler_kwargs)
        elif scheduler_type == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(opt, **scheduler_kwargs)

    vae.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            x_hat, mu, logvar, logits = vae(imgs, labels)
            loss_vae = vae.vae_loss(x_hat, imgs, mu, logvar)
            loss_cls = cls_loss_fn(logits, labels)
            loss = loss_vae + loss_cls
            loss.backward()
            opt.step()
            if scheduler is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
    if save:
        save_path = os.path.join(BASE_DIR, "pths", "vae.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(vae.state_dict(), save_path)
    return vae

# VAE testing
def vae_test(vae, device=None, test_loader=None, save_fig=False):
    if device is None: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders()

    vae.eval()

    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        recon_vae, _, _, logits = vae(imgs, labels)
        preds = logits.argmax(dim=1)

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'label: {labels[i].item()}')
        axes[1, i].imshow(recon_vae[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].text(0.5, 0.5, f'pred: {preds[i].item()}', fontsize=12, ha='center')
        axes[2, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed VAE')
    axes[2, 0].set_ylabel('Predicted')
    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)
        plt.savefig(os.path.join(BASE_DIR, "samples", "simpleVAE_test.png"))
    plt.show()

# Test the VAE and visualize results
if __name__ == "__main__":
    model = vae_train()
    vae_test(model, save_fig=True)