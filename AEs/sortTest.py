# Test for digit recognition using AE/VAE models on 10 random MNIST test images
import torch
import os
print('=== Script started ===')
print('CWD:', os.getcwd())
if os.path.exists('pths'):
	print('Files in pths:', os.listdir('pths'))
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AE, VAE
from mnist import get_mnist_loaders
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_models():
	models = {}
	model_names = []
	print("Loading models...")
	# simpleAE
	ae = AE().to(device)
	if os.path.exists("AEs/pths/ae.pth"):
		ae.load_state_dict(torch.load("AEs/pths/ae.pth", map_location=device))
		print("Loaded simpleAE")
	ae.eval()
	models['simpleAE'] = ae
	model_names.append('simpleAE')
	# simpleVAE
	vae = VAE().to(device)
	if os.path.exists("AEs/pths/vae.pth"):
		vae.load_state_dict(torch.load("AEs/pths/vae.pth", map_location=device))
		print("Loaded simpleVAE")
	vae.eval()
	models['simpleVAE'] = vae
	model_names.append('simpleVAE')
	# mutualAE1
	ae1 = AE().to(device)
	if os.path.exists("AEs/pths/m_ae1.pth"):
		ae1.load_state_dict(torch.load("AEs/pths/m_ae1.pth", map_location=device))
		print("Loaded mutualAE1")
	ae1.eval()
	models['mutualAE1'] = ae1
	model_names.append('mutualAE1')
	# mutualAE2
	ae2 = AE().to(device)
	if os.path.exists("AEs/pths/m_ae2.pth"):
		ae2.load_state_dict(torch.load("AEs/pths/m_ae2.pth", map_location=device))
		print("Loaded mutualAE2")
	ae2.eval()
	models['mutualAE2'] = ae2
	model_names.append('mutualAE2')
	# mutualVAE1
	vae1 = VAE().to(device)
	if os.path.exists("AEs/pths/m_vae1.pth"):
		vae1.load_state_dict(torch.load("AEs/pths/m_vae1.pth", map_location=device))
		print("Loaded mutualVAE1")
	vae1.eval()
	models['mutualVAE1'] = vae1
	model_names.append('mutualVAE1')
	# mutualVAE2
	vae2 = VAE().to(device)
	if os.path.exists("AEs/pths/m_vae2.pth"):
		vae2.load_state_dict(torch.load("AEs/pths/m_vae2.pth", map_location=device))
		print("Loaded mutualVAE2")
	vae2.eval()
	models['mutualVAE2'] = vae2
	model_names.append('mutualVAE2')
	# randomAE* (auto-detect count)
	rand_ae_idx = 1
	while True:
		pth = f"AEs/pths/rand_ae{rand_ae_idx}.pth"
		if not os.path.exists(pth):
			print(f'{pth}')
			print('no result found')
			break
		ae_rand = AE().to(device)
		ae_rand.load_state_dict(torch.load(pth, map_location=device))
		ae_rand.eval()
		print(f"Loaded randomAE{rand_ae_idx}")
		models[f'randomAE{rand_ae_idx}'] = ae_rand
		model_names.append(f'randomAE{rand_ae_idx}')
		rand_ae_idx += 1
	# randomVAE* (auto-detect count)
	rand_vae_idx = 1
	while True:
		pth = f"AEs/pths/rand_vae{rand_vae_idx}.pth"
		if not os.path.exists(pth):
			print(f'{pth}')
			print('no result found')
			break
		vae_rand = VAE().to(device)
		vae_rand.load_state_dict(torch.load(pth, map_location=device))
		vae_rand.eval()
		print(f"Loaded randomVAE{rand_vae_idx}")
		models[f'randomVAE{rand_vae_idx}'] = vae_rand
		model_names.append(f'randomVAE{rand_vae_idx}')
		rand_vae_idx += 1
	return models, model_names

def guess_digit(model, img, is_vae=False, latent_dim=32):
	# Try all digits 0-9, set one-hot latent, decode, compare to input
	img = img.to(device)
	# Ensure img shape is [1, 1, 28, 28]
	if img.dim() == 3:
		img = img.unsqueeze(0)
	elif img.dim() == 4 and img.shape[1] != 1:
		img = img[:, None, :, :]
	min_err = float('inf')
	best_digit = -1
	for digit in range(10):
		latent = torch.zeros((1, latent_dim), device=device)
		latent[0, digit % latent_dim] = 1.0
		with torch.no_grad():
			recon = model.dec(latent).view(1, 1, 28, 28)
		err = torch.nn.functional.mse_loss(recon, img).item()
		if err < min_err:
			min_err = err
			best_digit = digit
	return best_digit

def test_models_on_mnist():
	print('=== test_models_on_mnist() called ===')
	# Load test loader with batch size 1 for easy sampling
	_, test_loader = get_mnist_loaders(data_dir="./data", batch_size=1, test_batch_size=1, download=True)
	# Randomly select 10 images
	images, labels = [], []
	for img, label in test_loader:
		images.append(img)
		labels.append(label.item())
		if len(images) == 10:
			break
	models, model_names = load_models()
	results = {name: 0 for name in model_names}
	for idx in range(10):
		img = images[idx]
		label = labels[idx]
		for name in model_names:
			model = models[name]
			is_vae = 'VAE' in name
			latent_dim = 16 if is_vae else 32
			pred = guess_digit(model, img, is_vae=is_vae, latent_dim=latent_dim)
			if pred == label:
				results[name] += 1
	print("Results for 10 random test images:")
	for name in model_names:
		print(f"{name}: {results[name]} / 10 correct")

if __name__ == "__main__":
	test_models_on_mnist()
