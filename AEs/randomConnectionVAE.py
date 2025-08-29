# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from simpleVAE import vae_train
from models import VAE

device = torch.device("mps")

def random_train(num_vaes=3, pretrain_epochs=5, loops=10, device=device, scheduler_type=None, scheduler_kwargs=None, save=True, beta=1.0):
    train_loaders, test_loader = get_mnist_loaders(num_groups=num_vaes)
    vaes = [vae_train(device=device, train_loader=train_loaders[i], epochs=pretrain_epochs, save=False, scheduler_type=None) for i in range(num_vaes)]
    opts = [optim.Adam(vae.parameters(), lr=1e-3) for vae in vaes]
    recon_loss = nn.MSELoss(reduction="mean")

    # Scheduler setup for each optimizer
    schedulers = [None] * num_vaes
    if scheduler_type is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        for i in range(num_vaes):
            if scheduler_type == 'StepLR':
                schedulers[i] = optim.lr_scheduler.StepLR(opts[i], **scheduler_kwargs)
            elif scheduler_type == 'ReduceLROnPlateau':
                schedulers[i] = optim.lr_scheduler.ReduceLROnPlateau(opts[i], **scheduler_kwargs)
            elif scheduler_type == 'ExponentialLR':
                schedulers[i] = optim.lr_scheduler.ExponentialLR(opts[i], **scheduler_kwargs)
            # Add more schedulers as needed

    def kl_div(mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    loaders = [iter(loader) for loader in train_loaders]

    for loop in range(loops):
        # Pick a random VAE to start
        idx = random.randint(0, num_vaes-1)
        try:
            imgs, _ = next(loaders[idx])
        except StopIteration:
            loaders[idx] = iter(train_loaders[idx])
            imgs, _ = next(loaders[idx])
        imgs = imgs.to(device)
        # Forward through a random sequence of VAEs
        order = list(range(num_vaes))
        random.shuffle(order)
        # Optionally, ensure the first VAE is idx
        if order[0] != idx:
            order.remove(idx)
            order = [idx] + order
        # Forward pass through the chain
        input_imgs = imgs
        mu, logvar = None, None
        for i in range(num_vaes):
            opts[order[i]].zero_grad()
            out, mu, logvar = vaes[order[i]](input_imgs)
            # If not last VAE, pass output to next VAE
            if i < num_vaes-1:
                input_imgs = out.view_as(input_imgs)
        # Loss: compare final output to original input, add KL
        loss = recon_loss(out, imgs) + beta * kl_div(mu, logvar)
        loss.backward()
        opts[order[-1]].step()
        # Step scheduler only for the optimizer that was stepped
        scheduler = schedulers[order[-1]]
        if scheduler is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(loss.item())
            else:
                scheduler.step()

    if save:
        for i, vae in enumerate(vaes):
            torch.save(vae.state_dict(), f"/Volumes/Buffalo-SSD/AIStudy/AEs/pths/rand_vae{i+1}.pth")

    return vaes, test_loader

def random_test(vaes, device=device, test_loader=None):
    if test_loader is None:
        _, test_loader = get_mnist_loaders(num_groups=len(vaes))
    for vae in vaes:
        vae.eval()
    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    recon_vaes = []
    with torch.no_grad():
        for vae in vaes:
            recon, _, _ = vae(imgs)
            recon_vaes.append(recon)
    fig, axes = plt.subplots(len(vaes)+1, 8, figsize=(16, 2*(len(vaes)+1)))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'label: {labels[i].item()}')
    axes[0, 0].set_ylabel('Original')
    for j in range(len(vaes)):
        for i in range(8):
            axes[j+1, i].imshow(recon_vaes[j][i].cpu().squeeze(), cmap='gray')
            axes[j+1, i].axis('off')
        axes[j+1, 0].set_ylabel(f'Reconstructed VAE{j+1}')
    plt.tight_layout()
    plt.show()

# Test the random VAEs and visualize results
if __name__ == "__main__":
    vaes, test_loader = random_train(
		num_vaes=16, 
		pretrain_epochs=100, 
		loops=100, 
		scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
	)
    random_test(vaes, test_loader=test_loader)
