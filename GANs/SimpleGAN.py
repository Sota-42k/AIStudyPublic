# simpleGAN.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the existing MNIST loader (be sure to normalize to [-1,1])
# Example: transforms.Normalize((0.5,), (0.5,))
from mnist import get_mnist_loaders
from Models import GeneratorDCGAN, DiscriminatorDCGAN, weights_init

# 学習
def gan_train(
    device=None,
    train_loader=None,
    epochs=10,
    z_dim=100,
    lr=2e-4,
    beta1=0.5,
    save=True,
    scheduler_type=None,
    scheduler_kwargs=None,
    label_smooth=0.9,  # Change real label from 1.0 to 0.9
    gen_cls_weight=1.0,  # weight for generator's class loss (AC-GAN)
):
    if device is None: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if train_loader is None:
        train_loader, _ = get_mnist_loaders()  # This function normalizes to [-1,1] inside

    G = GeneratorDCGAN(z_dim).to(device).apply(weights_init)
    D = DiscriminatorDCGAN().to(device).apply(weights_init)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    # optional schedulers (if needed)
    def build_sched(opt):
        if scheduler_type is None: return None
        kw = scheduler_kwargs or {}
        if scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(opt, **kw)
        if scheduler_type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(opt, **kw)
        if scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(opt, **kw)
        return None
    sch_G, sch_D = build_sched(opt_G), build_sched(opt_D)

    fixed_z = torch.randn(64, z_dim, device=device)
    # fixed labels for grid: cycle 0..9
    fixed_labels = (torch.arange(64, device=device) % 10).long()

    for ep in range(1, epochs+1):
        print(f"Epoch {ep}/{epochs} - starting")
        G.train(); D.train()
        lossD_sum = 0.0; lossG_sum = 0.0; lossD_cls_sum = 0.0

        for real, labels in train_loader:
            real = real.to(device)
            labels = labels.to(device)
            b = real.size(0)

            # --- D update ---
            z = torch.randn(b, z_dim, device=device)
            fake = G(z, labels).detach()
            real_t = torch.full((b,1), fill_value=label_smooth, device=device)  # 0.9
            fake_t = torch.zeros((b,1), device=device)

            D.zero_grad()
            out_real_gan, out_real_cls = D(real)
            out_fake_gan, _ = D(fake)
            # real/fake loss
            lossD_gan = criterion_gan(out_real_gan, real_t) + criterion_gan(out_fake_gan, fake_t)
            # class loss (only for real images)
            lossD_cls = criterion_cls(out_real_cls, labels)
            lossD = lossD_gan + lossD_cls
            lossD.backward()
            opt_D.step()

            # --- G update ---
            z = torch.randn(b, z_dim, device=device)
            fake = G(z, labels)
            G.zero_grad()
            out_gan, out_cls = D(fake)
            # Want to make fake look real (=1) and be classified as the target label
            lossG_adv = criterion_gan(out_gan, torch.ones((b,1), device=device))
            lossG_cls = criterion_cls(out_cls, labels)
            lossG = lossG_adv + gen_cls_weight * lossG_cls
            lossG.backward()
            opt_G.step()

            lossD_sum += lossD_gan.item()
            lossD_cls_sum += lossD_cls.item()
            lossG_sum += lossG.item()

        avgD = lossD_sum / len(train_loader)
        avgD_cls = lossD_cls_sum / len(train_loader)
        avgG = lossG_sum / len(train_loader)
        print(f"[{ep:02d}/{epochs}] D: {avgD:.3f} | D_cls: {avgD_cls:.3f} | G: {avgG:.3f}")

        # schedulers step
        if sch_D:
            if scheduler_type == "ReduceLROnPlateau": sch_D.step(avgD)
            else: sch_D.step()
        if sch_G:
            if scheduler_type == "ReduceLROnPlateau": sch_G.step(avgG)
            else: sch_G.step()


        # Save samples (only if save is True)
        if save:
            with torch.no_grad():
                G.eval()
                samples = G(fixed_z, fixed_labels).cpu()
            save_dir = "/Volumes/Buffalo-SSD/AIStudy/GANs/samples"
            os.makedirs(os.path.join(save_dir), exist_ok=True)
            grid = utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
            utils.save_image(grid, os.path.join(save_dir, f"epoch_{ep:02d}.png"))

    if save:
        save_dir = "/Volumes/Buffalo-SSD/AIStudy/GANs/pths"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        torch.save(G.state_dict(), os.path.join(save_dir, "g.pth"))
        torch.save(D.state_dict(), os.path.join(save_dir, "d.pth"))
        print(f"Saved to {save_dir}")
    return G, D

# Test: Generate images with trained G (save only)
def gan_test(G, device=None, z_dim=100, n=64):
    if device is None: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    G = G.to(device)
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, device=device)
    labels = (torch.arange(n, device=device) % 10).long()
    imgs = G(z, labels).cpu()
    out_path = "/Volumes/Buffalo-SSD/AIStudy/GANs/samples/gan_samples.png"
    grid = utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1,1))
    utils.save_image(grid, out_path)
    print(f"Saved samples -> {out_path}")

# Test the GAN and visualize results
if __name__ == "__main__":
    G, D = gan_train(epochs=20)
    gan_test(G)
