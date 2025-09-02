# Test for digit recognition using GAN Discriminator on 10 random MNIST test images

import os
import sys
import torch

# add project root to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from Models import CondDiscriminatorDCGAN


device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


def load_discriminator(pth: str = "GANs/pths/d.pth") -> CondDiscriminatorDCGAN:
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Discriminator checkpoint not found: {pth}")
    model = CondDiscriminatorDCGAN().to(device)
    state = torch.load(pth, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_digit_with_discriminator(D: CondDiscriminatorDCGAN, img: torch.Tensor) -> int:
    """Predict digit using the discriminator's class head (argmax over logits)."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() == 4 and img.shape[1] != 1:
        img = img[:, None, :, :]
    img = img.to(device)
    with torch.no_grad():
        _, cls_logits = D(img)
        pred = cls_logits.argmax(dim=1).item()
    return pred


def test_discriminator_on_mnist(num_samples: int = 10):
    print("=== test_discriminator_on_mnist() called ===")
    # batch_size=1 for easy sampling of first N items
    _, test_loader = get_mnist_loaders(
        data_dir="./data", batch_size=1, test_batch_size=1, download=True
    )

    # collect first num_samples images
    images, labels = [], []
    for img, label in test_loader:
        images.append(img)
        labels.append(label.item())
        if len(images) == num_samples:
            break

    D = load_discriminator()
    correct = 0
    for i in range(num_samples):
        pred = predict_digit_with_discriminator(D, images[i])
        if pred == labels[i]:
            correct += 1
    print(f"GAN Discriminator: {correct} / {num_samples} correct")


if __name__ == "__main__":
    test_discriminator_on_mnist(10)
