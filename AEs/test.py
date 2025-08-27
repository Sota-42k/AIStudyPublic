
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AE

# Load trained AE from simpleAEs.py (assume weights are saved as 'ae.pth')
device = torch.device("mps")
ae = AE().to(device)
if os.path.exists("ae.pth"):
	ae.load_state_dict(torch.load("ae.pth", map_location=device))
ae.eval()

# Generate outputs for digits 0-9
with torch.no_grad():
	fig, axes = plt.subplots(1, 10, figsize=(20, 2))
	for digit in range(10):
		# One-hot vector for digit
		latent = torch.zeros((1, 32), device=device)
		latent[0, digit % 32] = 1.0
		# Pass through decoder only
		img = ae.dec(latent).view(28, 28).cpu().numpy()
		axes[digit].imshow(img, cmap='gray')
		axes[digit].axis('off')
		axes[digit].set_title(str(digit))
	plt.tight_layout()
	plt.show()
