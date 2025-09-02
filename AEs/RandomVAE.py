# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from SimpleVAE import vae_train

def random_train(num_vaes=3, pretrain_epochs=5, loops=1000, device=None, scheduler_type=None, scheduler_kwargs=None, save=True, beta=1.0):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders()
    vaes = [vae_train(device=device, train_loader=train_loader, epochs=pretrain_epochs, save=False, scheduler_type=None) for _ in range(num_vaes)]
    opts = [optim.Adam(vae.parameters(), lr=1e-3) for vae in vaes]
    recon_loss = nn.MSELoss(reduction="mean")
    cls_loss_fn = nn.CrossEntropyLoss()
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
    def kl_div(mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    loaders = [iter(train_loader) for _ in range(num_vaes)]
    for loop in range(loops):
        idx = random.randint(0, num_vaes-1)
        try:
            imgs, labels = next(loaders[idx])
        except StopIteration:
            loaders[idx] = iter(train_loader)
            imgs, labels = next(loaders[idx])
        imgs = imgs.to(device); labels = labels.to(device)
        order = list(range(num_vaes))
        random.shuffle(order)
        if order[0] != idx:
            order.remove(idx)
            order = [idx] + order
        input_imgs = imgs
        mu, logvar = None, None
        logits_list = []
        for i in range(num_vaes):
            opts[order[i]].zero_grad()
            out, mu, logvar, logits = vaes[order[i]](input_imgs, labels)
            logits_list.append(logits)
            if i < num_vaes-1:
                input_imgs = out.view_as(input_imgs)
        loss_recon = recon_loss(out, imgs)
        loss_kl = beta * kl_div(mu, logvar)
        loss_cls = sum(cls_loss_fn(l, labels) for l in logits_list)
        loss = loss_recon + loss_kl + loss_cls
        loss.backward()
        for j in range(num_vaes):
            if j != order[-1]:
                for p in vaes[order[j]].parameters():
                    if p.grad is not None:
                        p.grad.detach_(); p.grad.zero_()
        opts[order[-1]].step()
        scheduler = schedulers[order[-1]]
        if scheduler is not None:
            if scheduler_type == 'ReduceLROnPlateau': scheduler.step(loss.item())
            else: scheduler.step()
    if save:
        base = os.path.join(os.path.dirname(__file__), "pths")
        for i, vae in enumerate(vaes):
            torch.save(vae.state_dict(), os.path.join(base, f"rand_vae{i+1}.pth"))
        print("Saved RandomVAE models")
    return vaes

def random_test(vaes, device=None, test_loader=None, save_fig=False):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders(num_groups=len(vaes))
    for vae in vaes:
        vae.eval()
    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    labels = labels.to(device)
    recon_vaes = []
    logits_list = []
    with torch.no_grad():
        for vae in vaes:
            recon, _, _, logits = vae(imgs, labels)
            recon_vaes.append(recon)
            logits_list.append(logits)
    preds_list = [logits.argmax(dim=1) for logits in logits_list]
    fig, axes = plt.subplots(len(vaes)+2, 8, figsize=(16, 2*(len(vaes)+2)))
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
    # 分類結果
    for i in range(8):
        pred_str = '\n'.join([f'{p[i].item()}' for p in preds_list])
        axes[len(vaes)+1, i].text(0.5, 0.5, pred_str, fontsize=10, ha='center')
        axes[len(vaes)+1, i].axis('off')
    axes[len(vaes)+1, 0].set_ylabel('Predicted')
    plt.tight_layout()
    if save_fig:
        plt.savefig("AEs/samples/randomVAE_test.png")
    plt.show()

# Test the random VAEs and visualize results
if __name__ == "__main__":
    num_vaes = 16
    vaes = random_train(
		num_vaes=num_vaes, 
		pretrain_epochs=100, 
		loops=100, 
		scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
	)
    random_test(vaes)
