# Test for digit recognition using AE/VAE models on 10 random MNIST test images

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models import ConditionalAE as AE, ConditionalVAE as VAE
from mnist import get_mnist_loaders

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") if torch.backends.mps.is_available() else torch.device("cpu")


def load_models():
	models = {}
	model_names = []
	print("Loading models...")
	# モデル情報リスト: (pth, モデルクラス, ラベル)
	model_info = [
		("ae.pth", AE, 'simpleAE'),
		("vae.pth", VAE, 'simpleVAE'),
		("m_ae1.pth", AE, 'mutualAE1'),
		("m_ae2.pth", AE, 'mutualAE2'),
		("m_vae1.pth", VAE, 'mutualVAE1'),
		("m_vae2.pth", VAE, 'mutualVAE2'),
		("s_student_ae.pth", AE, 'singleTeacherAE'),
		("s_student_vae.pth", VAE, 'singleTeacherVAE'),
		("d_student_ae.pth", AE, 'doubleTeacherAE'),
		("d_student_vae.pth", VAE, 'doubleTeacherVAE'),
	]
	# randomAE / randomVAE (自動検出)
	rand_ae_idx = 1
	while True:
		pth = f"rand_ae{rand_ae_idx}.pth"
		full_pth = f"AEs/pths/{pth}"
		if not os.path.exists(full_pth):
			break
		model_info.append((pth, AE, f'randomAE{rand_ae_idx}'))
		rand_ae_idx += 1
	rand_vae_idx = 1
	while True:
		pth = f"rand_vae{rand_vae_idx}.pth"
		full_pth = f"AEs/pths/{pth}"
		if not os.path.exists(full_pth):
			break
		model_info.append((pth, VAE, f'randomVAE{rand_vae_idx}'))
		rand_vae_idx += 1

	for pth_name, model_class, label in model_info:
		pth = f"AEs/pths/{pth_name}"
		if not os.path.exists(pth):
			continue
		model = model_class().to(device)
		state = torch.load(pth, map_location=device)
		model.load_state_dict(state, strict=False)
		model.eval()
		models[label] = model
		model_names.append(label)
	return models, model_names


def guess_digit(model, img):
	"""Predict digit by picking label with smallest reconstruction error using conditional forward."""
	img = img.to(device)
	if img.dim() == 3:
		img = img.unsqueeze(0)
	elif img.dim() == 4 and img.shape[1] != 1:
		img = img[:, None, :, :]
	min_err = float('inf')
	best_digit = -1
	for digit in range(10):
		y = torch.tensor([digit], device=device, dtype=torch.long)
		with torch.no_grad():
			out = model(img, y)
			recon = out[0] if isinstance(out, (tuple, list)) else out
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
			pred = guess_digit(model, img)
			if pred == label:
				results[name] += 1
	print("Results for 10 random test images:")
	for name in model_names:
		print(f"{name}: {results[name]} / 10 correct")

if __name__ == "__main__":
	test_models_on_mnist()
