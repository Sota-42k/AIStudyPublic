# GenerateTest for DDPM: generate and save samples for each digit
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DMs.Models import ConditionalDDPM
from torchvision.utils import save_image, make_grid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_ddpm(pth: str = None, timesteps=1000, num_classes=10):
    if pth is None:
        pth = os.path.join(BASE_DIR, "pths", "ddpm.pth")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"DDPM checkpoint not found: {pth}")
    model = ConditionalDDPM(num_classes=num_classes, timesteps=timesteps, device=device)
    state = torch.load(pth, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def generate_and_save_samples(model, save_dir=None, n_per_digit=8):
    if save_dir is None:
        save_dir = os.path.join(BASE_DIR, "tests")
    os.makedirs(save_dir, exist_ok=True)
    all_samples = []
    for digit in range(10):
        y = torch.full((n_per_digit,), digit, dtype=torch.long, device=device)
        samples = model.sample((n_per_digit, 1, 28, 28), y)
        samples = (samples + 1) / 2
        all_samples.append(samples)
        grid = make_grid(samples, nrow=n_per_digit)
        save_image(grid, os.path.join(save_dir, f"digit_{digit}.png"))
    all_samples = torch.cat(all_samples, dim=0)
    grid = make_grid(all_samples, nrow=n_per_digit)
    save_image(grid, os.path.join(save_dir, "all_digits.png"))

if __name__ == "__main__":
    model = load_ddpm()
    generate_and_save_samples(model)
