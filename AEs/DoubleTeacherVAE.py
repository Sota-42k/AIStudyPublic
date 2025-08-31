import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist import get_mnist_loaders
from Models import VAE

def double_teacher_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
    """
    2つの教師VAE(VAE1, VAE2)の潜在表現（concat）から生徒VAE(VAE3)を学習する。
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 2つのグループでデータローダ取得
    train_loaders, _ = get_mnist_loaders(num_groups=2)
    train_loader1, train_loader2 = train_loaders

    # 教師も生徒もlatent=32
    latent = 32
    vae1 = VAE(latent=latent).to(device)
    vae2 = VAE(latent=latent).to(device)
    vae3 = VAE(latent=latent).to(device)
    # 圧縮用線形層（教師2人のlatentをconcat→32次元に圧縮）
    compress_linear = nn.Linear(64, 32).to(device)

    opt1 = optim.Adam(vae1.parameters(), lr=1e-3)
    opt2 = optim.Adam(vae2.parameters(), lr=1e-3)
    opt3 = optim.Adam(list(vae3.parameters()) + list(compress_linear.parameters()), lr=1e-3)

    scheduler1 = scheduler2 = scheduler3 = None
    if scheduler_type is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        if scheduler_type == 'StepLR':
            scheduler1 = optim.lr_scheduler.StepLR(opt1, **scheduler_kwargs)
            scheduler2 = optim.lr_scheduler.StepLR(opt2, **scheduler_kwargs)
            scheduler3 = optim.lr_scheduler.StepLR(opt3, **scheduler_kwargs)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, **scheduler_kwargs)
            scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, **scheduler_kwargs)
            scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(opt3, **scheduler_kwargs)
        elif scheduler_type == 'ExponentialLR':
            scheduler1 = optim.lr_scheduler.ExponentialLR(opt1, **scheduler_kwargs)
            scheduler2 = optim.lr_scheduler.ExponentialLR(opt2, **scheduler_kwargs)
            scheduler3 = optim.lr_scheduler.ExponentialLR(opt3, **scheduler_kwargs)

    vae1.train()
    vae2.train()
    vae3.train()
    compress_linear.train()

    teacher_epochs = epochs // 2
    student_epochs = epochs
    # 1. 教師の学習
    for epoch in range(teacher_epochs):
        for (imgs1, _), (imgs2, _) in zip(train_loader1, train_loader2):
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            # VAE1, VAE2: 教師として通常学習
            opt1.zero_grad()
            x1_hat, mu1, logvar1 = vae1(imgs1)
            loss1 = VAE.vae_loss(x1_hat, imgs1, mu1, logvar1)
            loss1.backward()
            opt1.step()

            opt2.zero_grad()
            x2_hat, mu2, logvar2 = vae2(imgs2)
            loss2 = VAE.vae_loss(x2_hat, imgs2, mu2, logvar2)
            loss2.backward()
            opt2.step()

            # scheduler
            if scheduler1 is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler1.step(loss1.item())
                else:
                    scheduler1.step()
            if scheduler2 is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler2.step(loss2.item())
                else:
                    scheduler2.step()

    # 2. 教師を凍結
    for param in vae1.parameters():
        param.requires_grad = False
    for param in vae2.parameters():
        param.requires_grad = False


    # 3. 生徒の学習（concatのみ）
    for epoch in range(student_epochs):
        for (imgs1, _), (imgs2, _) in zip(train_loader1, train_loader2):
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)

            with torch.no_grad():
                h1 = vae1.enc(imgs1)
                mu1, logvar1 = vae1.mu(h1), vae1.logvar(h1)
                z1 = vae1.reparam(mu1, logvar1)
                h2 = vae2.enc(imgs2)
                mu2, logvar2 = vae2.mu(h2), vae2.logvar(h2)
                z2 = vae2.reparam(mu2, logvar2)
                x1_hat, _, _ = vae1(imgs1)
                x2_hat, _, _ = vae2(imgs2)
                x_teacher_target = (x1_hat + x2_hat) / 2
            # concat→線形層で圧縮
            z_cat = torch.cat([z1, z2], dim=1)  # (batch, 64)
            z_compressed = compress_linear(z_cat)  # (batch, 32)
            x_hat = vae3.dec(z_compressed).view(-1, 1, 28, 28)
            loss = nn.functional.mse_loss(x_hat, x_teacher_target)
            opt3.zero_grad()
            loss.backward()
            opt3.step()

            # scheduler
            if scheduler3 is not None:
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler3.step(loss.item())
                else:
                    scheduler3.step()

    if save:
        torch.save(vae1.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "d_teacher1_vae.pth"))
        torch.save(vae2.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "d_teacher2_vae.pth"))
        torch.save(vae3.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "d_student_vae.pth"))
    return vae1, vae2, vae3


def double_teacher_test(vae1, vae2, vae3, device=None, test_loader=None, save_fig=False):
    import matplotlib.pyplot as plt
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if test_loader is None:
        _, test_loader = get_mnist_loaders()

    vae1.eval()
    vae2.eval()
    vae3.eval()

    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        h1 = vae1.enc(imgs)
        mu1, logvar1 = vae1.mu(h1), vae1.logvar(h1)
        z1 = vae1.reparam(mu1, logvar1)
        h2 = vae2.enc(imgs)
        mu2, logvar2 = vae2.mu(h2), vae2.logvar(h2)
        z2 = vae2.reparam(mu2, logvar2)
        z_student = torch.cat([z1, z2], dim=1)
        recon_vae1, _, _ = vae1(imgs)
        recon_vae2, _, _ = vae2(imgs)
        recon_vae3 = vae3.dec(z_student).view(-1, 1, 28, 28)
    fig, axes = plt.subplots(4, 8, figsize=(16, 5))
    for i in range(8):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_vae1[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(recon_vae2[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[3, i].imshow(recon_vae3[i].cpu().squeeze(), cmap='gray')
        axes[3, i].axis('off')
    axes[0, 0].set_ylabel('Input')
    axes[1, 0].set_ylabel('Teacher1')
    axes[2, 0].set_ylabel('Teacher2')
    axes[3, 0].set_ylabel('Student')
    plt.tight_layout()
    if save_fig:
        plt.savefig('double_teacher_vae_test.png')
    plt.show()

if __name__ == "__main__":
    double_teacher_train(epochs=100, save=True, scheduler_type='StepLR', scheduler_kwargs={'step_size': 100, 'gamma': 0.5})
