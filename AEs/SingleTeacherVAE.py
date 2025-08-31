# Teacher-Student Variational AutoEncoder Training
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist import get_mnist_loaders
from Models import VAE

def single_teacher_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 2つのグループでデータローダ取得
    train_loaders, _ = get_mnist_loaders(num_groups=2)
    train_loader1, train_loader2 = train_loaders

    vae1 = VAE().to(device)
    vae2 = VAE().to(device)
    opt1 = optim.Adam(vae1.parameters(), lr=1e-3)
    opt2 = optim.Adam(vae2.parameters(), lr=1e-3)

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

    vae1.train()
    vae2.train()
    # 1. 教師の学習
    teacher_epochs = student_epochs = epochs

    for epoch in range(teacher_epochs):
        for (imgs1, _), _ in zip(train_loader1, train_loader2):
            imgs1 = imgs1.to(device)
            # VAE1: 教師
            opt1.zero_grad()
            x1_hat, mu1, logvar1 = vae1(imgs1)
            loss1 = VAE.vae_loss(x1_hat, imgs1, mu1, logvar1)
            loss1.backward()
            opt1.step()
            if scheduler1 is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler1.step(loss1.item())
                else:
                    scheduler1.step()
    # 2. 教師を凍結
    for param in vae1.parameters():
        param.requires_grad = False
    # 3. 生徒の学習（教師の出力のみで学習）
    for epoch in range(student_epochs):
        for (imgs1, _), _ in zip(train_loader1, train_loader2):
            imgs1 = imgs1.to(device)
            with torch.no_grad():
                x1_hat_for_student, _, _ = vae1(imgs1)
            opt2.zero_grad()
            x2_input = x1_hat_for_student.detach()
            x2_hat, mu2, logvar2 = vae2(x2_input)
            loss2 = VAE.vae_loss(x2_hat, x2_input, mu2, logvar2)
            loss2.backward()
            opt2.step()
            if scheduler2 is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler2.step(loss2.item())
                else:
                    scheduler2.step()
    if save:
        torch.save(vae1.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "s_teacher_vae.pth"))
        torch.save(vae2.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "s_student_vae.pth"))
    return vae1, vae2


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
        recon_vae1, _, _ = vae1(imgs)
        recon_vae2, _, _ = vae2(vae1(imgs)[0].detach())
    fig, axes = plt.subplots(3, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_vae1[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(recon_vae2[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
    axes[0, 0].set_ylabel('Input')
    axes[1, 0].set_ylabel('Teacher')
    axes[2, 0].set_ylabel('Student')
    plt.tight_layout()
    if save_fig:
        plt.savefig('single_teacher_vae_test.png')
    plt.show()

if __name__ == "__main__":
    single_teacher_train(epochs=30, save=True, scheduler_type='StepLR', scheduler_kwargs={'step_size': 10, 'gamma': 0.5})
