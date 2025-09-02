# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders, get_mnist_digit_loader
from Models import ConditionalAE as AE

# AE training
def ae_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None, train_loader=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Use full MNIST when no loader provided
    if train_loader is None:
        train_loader, _ = get_mnist_loaders()
    ae = AE().to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    cls_loss_fn = nn.CrossEntropyLoss()
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
    ae.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            x_hat, logits = ae(imgs, labels)
            loss_recon = loss_fn(x_hat, imgs)
            loss_cls = cls_loss_fn(logits, labels)
            loss = loss_recon + loss_cls
            loss.backward()
            opt.step()
            if scheduler is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
    if save:
        save_path = f"/Volumes/Buffalo-SSD/AIStudy/AEs/pths/ae.pth"
        torch.save(ae.state_dict(), save_path)
        print(f"Saved AE model to {save_path}")
    return ae

# AE testing
def ae_test(ae, device=None, test_loader=None, save_fig=False):
    if device is None: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders()
    ae.eval()

    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        recon_ae, logits = ae(imgs, labels)
        preds = logits.argmax(dim=1)

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'label: {labels[i].item()}')
        axes[1, i].imshow(recon_ae[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].text(0.5, 0.5, f'pred: {preds[i].item()}', fontsize=12, ha='center')
        axes[2, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed AE')
    axes[2, 0].set_ylabel('Predicted')
    plt.tight_layout()
    if save_fig:
        plt.savefig("AEs/samples/simpleAE_test.png")
    plt.show()

# Test the AE and visualize results
if __name__ == "__main__":
    ae_train()
