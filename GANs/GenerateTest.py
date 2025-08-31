import torch
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models import GeneratorDCGAN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") if torch.backends.mps.is_available() else torch.device("cpu")

def get_all_gan_outputs(z_dim=100, n=10):
    outputs = []
    model_names = []
    # simpleGAN
    g_path = "GANs/pths/g.pth"
    if os.path.exists(g_path):
        G = GeneratorDCGAN(z_dim).to(device)
        G.load_state_dict(torch.load(g_path, map_location=device))
        G.eval()
        with torch.no_grad():
            z = torch.randn(n, z_dim, device=device)
            imgs = G(z).cpu().numpy()
        outputs.append(imgs)
        model_names.append('simpleGAN')
    # Add more GANs if you have more generator .pth files
    rand_gan_idx = 1
    while True:
        pth = f"GANs/pths/rand_gan{rand_gan_idx}.pth"
        if not os.path.exists(pth):
            break
        G = GeneratorDCGAN(z_dim).to(device)
        G.load_state_dict(torch.load(pth, map_location=device))
        G.eval()
        with torch.no_grad():
            z = torch.randn(n, z_dim, device=device)
            imgs = G(z).cpu().numpy()
        outputs.append(imgs)
        model_names.append(f'randomGAN{rand_gan_idx}')
        rand_gan_idx += 1
    return outputs, model_names

def show_all_gan_outputs(save_fig=False, z_dim=100, n=10):
    outputs, model_names = get_all_gan_outputs(z_dim=z_dim, n=n)
    for row_idx, imgs in enumerate(outputs):
        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        for col_idx, img in enumerate(imgs):
            axes[col_idx].imshow(img[0], cmap='gray')
            axes[col_idx].axis('off')
            axes[col_idx].set_title(col_idx)
        plt.suptitle(model_names[row_idx])
        if save_fig:
            plt.savefig(f"GANs/tests/{model_names[row_idx]}.png")
        plt.show()
        print(f"Displayed {model_names[row_idx]}")

if __name__ == "__main__":
    show_all_gan_outputs(save_fig=True)
