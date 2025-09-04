# Simple DDPM training and sampling on MNIST
import os
import sys
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# base directory for saving relative to this file's parent (so running from any CWD works)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from Models import ConditionalDDPM, get_timestep_embedding
from ImageGeneration.mnist import get_mnist_loaders


def ddpm_train(
	epochs=10,
	batch_size=128,
	lr=2e-4,
	timesteps=1000,
	device=None,
	save=True
):
	"""Train a conditional DDPM on MNIST and save checkpoints/samples.

	Returns the trained model.
	"""
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

	train_loader, _ = get_mnist_loaders(batch_size=batch_size)
	if isinstance(train_loader, list):
		train_loader = train_loader[0]
	model = ConditionalDDPM(num_classes=10, timesteps=timesteps, device=device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	T = timesteps
	for epoch in range(1, epochs+1):
		model.train()
		for i, (x, y) in enumerate(train_loader):
			x = x.to(device) * 2 - 1  # scale to [-1, 1]
			y = y.to(device)
			B = x.size(0)
			t = torch.randint(0, T, (B,), device=device)
			noise = torch.randn_like(x)
			x_noisy = model.q_sample(x, t, noise)
			t_emb = get_timestep_embedding(t, 32)
			y_onehot = F.one_hot(y, 10).float()
			cond = torch.cat([t_emb, y_onehot], dim=1)
			out = model.model(x_noisy, t_emb, cond)
			loss = F.mse_loss(out, noise)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % 100 == 0:
				print(f"Epoch {epoch} Step {i}: Loss {loss.item():.4f}")
		
		# Save samples for each digit after each epoch
		if save:
			with torch.no_grad():
				model.eval()
				save_dir = os.path.join(BASE_DIR, "samples")
				os.makedirs(save_dir, exist_ok=True)
				all_samples = []
				for digit in range(10):
					y = torch.full((8,), digit, dtype=torch.long, device=device)
					samples = model.sample((8, 1, 28, 28), y)
					samples = (samples + 1) / 2
					all_samples.append(samples)
					grid = make_grid(samples, nrow=8)
					save_image(grid, os.path.join(save_dir, f"ddpm_epoch_{epoch}_digit_{digit}.png"))
				# Optionally, save a grid of all digits
				all_samples = torch.cat(all_samples, dim=0)
				grid = make_grid(all_samples, nrow=8)
				save_image(grid, os.path.join(save_dir, f"ddpm_epoch_{epoch}_all_digits.png"))
	# final checkpoint
	if save:
		save_dir = os.path.join(BASE_DIR, "pths")
		os.makedirs(save_dir, exist_ok=True)
		torch.save(model.state_dict(), os.path.join(save_dir, "ddpm.pth"))
		print(f"Saved to {save_dir}")
	return model


def ddpm_test(model, device=None, timesteps=1000, n_per_digit=8):
	"""Generate and save samples from a trained conditional DDPM.

	If model is None the function will try to load weights from `pths/ddpm.pth` next to this file.
	"""
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	# save relative to this file's directory so it works when run from other CWDs
	save_dir = os.path.join(BASE_DIR, "tests")
	os.makedirs(save_dir, exist_ok=True)
	model = model.to(device)
	model.eval()
	with torch.no_grad():
		all_samples = []
		for digit in range(10):
			y = torch.full((n_per_digit,), digit, dtype=torch.long, device=device)
			samples = model.sample((n_per_digit, 1, 28, 28), y)
			samples = (samples + 1) / 2
			all_samples.append(samples.cpu())
			grid = make_grid(samples.cpu(), nrow=n_per_digit)
			save_image(grid, os.path.join(save_dir, f"ddpm_digit_{digit}.png"))
		all_samples = torch.cat(all_samples, dim=0)
		grid = make_grid(all_samples, nrow=n_per_digit)
		out_path = os.path.join(save_dir, "ddpm_samples.png")
		save_image(grid, out_path)
		print(f"Saved samples -> {out_path}")


if __name__ == "__main__":
	model = ddpm_train()
	ddpm_test(model)
