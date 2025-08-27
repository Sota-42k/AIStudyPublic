# Configuration
from matplotlib import axes
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from models import AE, VAE, vae_loss

# Pre-train each AE separately before mutual connection
def pretrain_autoencoder(ae, train_loader, device, epochs=5):
	opt = optim.Adam(ae.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()
	ae.train()
	for epoch in range(epochs):
		for imgs, _ in train_loader:
			imgs = imgs.to(device)
			opt.zero_grad()
			x_hat = ae(imgs)
			loss = loss_fn(x_hat, imgs)
			loss.backward()
			opt.step()

def test_autoencoders(self, test_loader):
	num_aes = len(self.aes)
	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(self.device)
	recon_aes = []
	for i in range(num_aes):
		self.aes[i].eval()
		with torch.no_grad():
			recon = self.aes[i](imgs)
		recon_aes.append(recon)
	fig, axes = plt.subplots(num_aes+1, 8, figsize=(16, 2*(num_aes+1)))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[0, i].set_title(f'label: {labels[i].item()}')
	axes[0, 0].set_ylabel('Original')
	for j in range(num_aes):
		for i in range(8):
			axes[j+1, i].imshow(recon_aes[j][i].cpu().squeeze(), cmap='gray')
			axes[j+1, i].axis('off')
		axes[j+1, 0].set_ylabel(f'Reconstructed AE{j+1}')
	plt.tight_layout()
	plt.show()

# Random AE connection class
class RandomAutoencoder:
	def __init__(self, pretrain_epochs=5, num_aes=3):
		self.train_loaders, self.test_loader = get_mnist_loaders(num_groups=num_aes)
		self.device = torch.device("mps")

		self.aes = [AE().to(self.device) for _ in range(num_aes)]
		self.opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in self.aes]
		self.loss_fn = nn.MSELoss()

		for i in range(num_aes):
			pretrain_autoencoder(self.aes[i], self.train_loaders[i], self.device, epochs=pretrain_epochs)
		test_autoencoders(self, self.test_loader)

	def random_train(self, loops=10):
		import random
		# Set all AEs to train mode
		for ae in self.aes:
			ae.train()
		# Create iterators for each train loader
		loaders = [iter(loader) for loader in self.train_loaders]
		num_aes = len(self.aes)
		for loop in range(loops):
			# Pick a random AE to start
			idx = random.randint(0, num_aes-1)
			try:
				imgs, _ = next(loaders[idx])
			except StopIteration:
				loaders[idx] = iter(self.train_loaders[idx])
				imgs, _ = next(loaders[idx])
			imgs = imgs.to(self.device)
			# Forward through a random sequence of AEs
			order = list(range(num_aes))
			random.shuffle(order)
			# Optionally, ensure the first AE is idx
			if order[0] != idx:
				order.remove(idx)
				order = [idx] + order
			# Forward pass through the chain
			input_imgs = imgs
			for i in range(num_aes):
				self.opts[order[i]].zero_grad()
				out = self.aes[order[i]](input_imgs)
				# If not last AE, pass output to next AE
				if i < num_aes-1:
					input_imgs = out.view_as(input_imgs)
			# Loss: compare final output to original input
			loss = self.loss_fn(out, imgs)
			loss.backward()
			self.opts[order[-1]].step()
		# Optionally, test/visualize after training
		test_autoencoders(self, self.test_loader)
		
# Example usage:
random_ae = RandomAutoencoder(pretrain_epochs=10)
random_ae.random_train(loops=5)