import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AE, VAE

device = torch.device("mps")

def get_all_model_outputs():
	# Each row: [simpleAE, simpleVAE, mutualAE1, mutualAE2, mutualVAE1, mutualVAE2]
	outputs = []
	# simpleAE
	ae = AE().to(device)
	if os.path.exists("../AEs/pths/ae.pth"):
		ae.load_state_dict(torch.load("../AEs/pths/ae.pth", map_location=device))
	ae.eval()
	row = []
	with torch.no_grad():
		for digit in range(10):
			latent = torch.zeros((1, 32), device=device)
			latent[0, digit % 32] = 1.0
			img = ae.dec(latent).view(28, 28).cpu().numpy()
			row.append(img)
	outputs.append(row)
	# simpleVAE
	vae = VAE().to(device)
	if os.path.exists("../AEs/pths/vae.pth"):
		vae.load_state_dict(torch.load("../AEs/pths/vae.pth", map_location=device))
	vae.eval()
	row = []
	with torch.no_grad():
		for digit in range(10):
			latent = torch.zeros((1, 16), device=device)
			latent[0, digit % 16] = 1.0
			img = vae.dec(latent).view(28, 28).cpu().numpy()
			row.append(img)
	outputs.append(row)
	# mutualAE
	ae1 = AE().to(device)
	ae2 = AE().to(device)
	pth1 = "../AEs/pths/m_ae1.pth"
	pth2 = "../AEs/pths/m_ae2.pth"
	if os.path.exists(pth1):
		ae1.load_state_dict(torch.load(pth1, map_location=device))
	if os.path.exists(pth2):
		ae2.load_state_dict(torch.load(pth2, map_location=device))
	ae1.eval()
	ae2.eval()
	row1, row2 = [], []
	with torch.no_grad():
		for digit in range(10):
			latent = torch.zeros((1, 32), device=device)
			latent[0, digit % 32] = 1.0
			img1 = ae1.dec(latent).view(28, 28).cpu().numpy()
			img2 = ae2.dec(latent).view(28, 28).cpu().numpy()
			row1.append(img1)
			row2.append(img2)
	outputs.append(row1)
	outputs.append(row2)
	# mutualVAE
	vae1 = VAE().to(device)
	vae2 = VAE().to(device)
	pth1 = "../AEs/pths/m_vae1.pth"
	pth2 = "../AEs/pths/m_vae2.pth"
	if os.path.exists(pth1):
		vae1.load_state_dict(torch.load(pth1, map_location=device))
	if os.path.exists(pth2):
		vae2.load_state_dict(torch.load(pth2, map_location=device))
	vae1.eval()
	vae2.eval()
	row1, row2 = [], []
	with torch.no_grad():
		for digit in range(10):
			latent = torch.zeros((1, 16), device=device)
			latent[0, digit % 16] = 1.0
			img1 = vae1.dec(latent).view(28, 28).cpu().numpy()
			img2 = vae2.dec(latent).view(28, 28).cpu().numpy()
			row1.append(img1)
			row2.append(img2)
	outputs.append(row1)
	outputs.append(row2)
	return outputs

def show_all_model_outputs():
	outputs = get_all_model_outputs()
	model_names = [
		'simpleAE', 'simpleVAE', 'mutualAE1', 'mutualAE2', 'mutualVAE1', 'mutualVAE2'
	]
	fig, axes = plt.subplots(len(outputs), 10, figsize=(20, 2*len(outputs)))
	for row_idx, row in enumerate(outputs):
		for col_idx, img in enumerate(row):
			axes[row_idx, col_idx].imshow(img, cmap='gray')
			axes[row_idx, col_idx].axis('off')
			if row_idx == 0:
				axes[row_idx, col_idx].set_title(str(col_idx))
		axes[row_idx, 0].set_ylabel(model_names[row_idx], rotation=0, labelpad=40, fontsize=12, va='center')
	plt.tight_layout()
	plt.suptitle('All AE/VAE Model Outputs', fontsize=16, y=1.02)
	plt.show()

if __name__ == "__main__":
	show_all_model_outputs()
