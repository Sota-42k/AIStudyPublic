# Teacher-Student Variational AutoEncoder Training
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist import get_mnist_loaders
from Models import ConditionalVAE as VAE

def single_teacher_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders()
    vae1 = VAE().to(device)
    vae2 = VAE().to(device)
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
    vae1.train(); vae2.train()
    teacher_epochs = student_epochs = epochs
    for epoch in range(teacher_epochs):
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            opt1.zero_grad()
            x1_hat, mu1, logvar1, logits1 = vae1(imgs, labels)
            loss1 = VAE.vae_loss(x1_hat, imgs, mu1, logvar1) + cls_loss_fn(logits1, labels)
            loss1.backward(); opt1.step()
            if scheduler1 is not None:
                if scheduler_type == 'ReduceLROnPlateau': scheduler1.step(loss1.item())
                else: scheduler1.step()
    for p in vae1.parameters(): p.requires_grad = False
    for epoch in range(student_epochs):
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            with torch.no_grad():
                x1_hat_for_student, _, _, logits1 = vae1(imgs, labels)
            opt2.zero_grad()
            x2_hat, mu2, logvar2, logits2 = vae2(x1_hat_for_student.detach(), labels)
            loss2 = VAE.vae_loss(x2_hat, x1_hat_for_student, mu2, logvar2) + cls_loss_fn(logits2, labels)
            loss2.backward(); opt2.step()
            if scheduler2 is not None:
                if scheduler_type == 'ReduceLROnPlateau': scheduler2.step(loss2.item())
                else: scheduler2.step()
    if save:
        base = os.path.join(os.path.dirname(__file__), 'pths')
        torch.save(vae1.state_dict(), os.path.join(base, 's_teacher_vae.pth'))
        torch.save(vae2.state_dict(), os.path.join(base, 's_student_vae.pth'))
        print('Saved SingleTeacherVAE models')
    return (vae1, vae2)


def single_teacher_test(vae1, vae2, device=None, test_loader=None, save_fig=False):
    import matplotlib.pyplot as plt
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders()

    vae1.eval()
    vae2.eval()

    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        recon_vae1, _, _, logits1 = vae1(imgs, labels)
        recon_vae2, _, _, logits2 = vae2(recon_vae1.detach(), labels)
        preds1 = logits1.argmax(dim=1)
        preds2 = logits2.argmax(dim=1)
    fig, axes = plt.subplots(4, 8, figsize=(16, 6))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_vae1[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(recon_vae2[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[3, i].text(0.5, 0.5, f'T:{preds1[i].item()}\nS:{preds2[i].item()}', fontsize=10, ha='center')
        axes[3, i].axis('off')
    axes[0, 0].set_ylabel('Input')
    axes[1, 0].set_ylabel('Teacher')
    axes[2, 0].set_ylabel('Student')
    axes[3, 0].set_ylabel('Predicted')
    plt.tight_layout()
    if save_fig:
        plt.savefig('single_teacher_vae_test.png')
    plt.show()

if __name__ == "__main__":
    single_teacher_train(epochs=30, save=True, scheduler_type='StepLR', scheduler_kwargs={'step_size': 10, 'gamma': 0.5})
