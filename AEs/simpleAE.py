# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from models import AE

device = torch.device("mps")
train_loader, test_loader = get_mnist_loaders()

# AE training
def ae_train(device=device, train_loader=train_loader, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
    ae = AE().to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Scheduler setup
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
        # Add more schedulers as needed

    ae.train()
    for epoch in range(epochs):
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            x_hat = ae(imgs)
            loss = loss_fn(x_hat, imgs)
            loss.backward()
            opt.step()
            # Step scheduler if used (should be after optimizer.step())
            if scheduler is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
    if save:
        torch.save(ae.state_dict(), "/Volumes/Buffalo-SSD/AIStudy/AEs/pths/ae.pth")
    return ae

# AE testing
def ae_test(ae, device=device, test_loader=test_loader):
    ae.eval()

    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        recon_ae = ae(imgs)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'label: {labels[i].item()}')
        axes[1, i].imshow(recon_ae[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed AE')
    plt.tight_layout()
    plt.show()

# Test the VAE and visualize results
if __name__ == "__main__":
    ae_test(ae_train())