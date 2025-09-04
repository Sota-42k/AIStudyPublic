# DetectTest for DDPM: digit recognition using the conditional DDPM model
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DMs.Models import ConditionalDDPM, get_timestep_embedding
from mnist import get_mnist_loaders
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Use the denoising network as a digit classifier by passing each possible label and picking the one with lowest denoising error

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


def predict_digit_with_ddpm(model, img, t=500):
    # img: (1, 28, 28) or (B, 1, 28, 28)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.to(device) * 2 - 1
    B = img.size(0)
    t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
    t_emb = get_timestep_embedding(t_tensor, 32)
    errors = []
    for digit in range(10):
        y = torch.full((B,), digit, dtype=torch.long, device=device)
        y_onehot = F.one_hot(y, 10).float()
        cond = torch.cat([t_emb, y_onehot], dim=1)
        out = model.model(img, t_emb, cond)
        # Use MSE between output and zero noise as a proxy for likelihood
        err = F.mse_loss(out, torch.zeros_like(out), reduction='none').mean(dim=[1,2,3])
        errors.append(err.unsqueeze(1))
    errors = torch.cat(errors, dim=1)  # (B, 10)
    pred = errors.argmin(dim=1).cpu().numpy()
    return pred


def test_ddpm_on_mnist(num_samples=10):
    print("=== test_ddpm_on_mnist() called ===")
    _, test_loader = get_mnist_loaders(batch_size=1, test_batch_size=1, download=True)
    images, labels = [], []
    for img, label in test_loader:
        images.append(img)
        labels.append(label.item())
        if len(images) == num_samples:
            break
    model = load_ddpm()
    correct = 0
    for i in range(num_samples):
        pred = predict_digit_with_ddpm(model, images[i])[0]
        print(f"Sample {i}: True={labels[i]}, Pred={pred}")
        if pred == labels[i]:
            correct += 1
    print(f"Accuracy: {correct}/{num_samples}")

if __name__ == "__main__":
    test_ddpm_on_mnist()
