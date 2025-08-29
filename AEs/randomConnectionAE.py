
# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from simpleAE import ae_train
from models import AE

device = torch.device("mps")

def random_train(num_aes=3, pretrain_epochs=5, loops=10, device=device, scheduler_type=None, scheduler_kwargs=None, save=True):
	train_loaders, test_loader = get_mnist_loaders(num_groups=num_aes)
	aes = [ae_train(device=device, train_loader=train_loaders[i], epochs=pretrain_epochs, save=False, scheduler_type=None) for i in range(num_aes)]
	opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in aes]
	loss_fn = nn.MSELoss()

	# Scheduler setup for each optimizer
	schedulers = [None] * num_aes
	if scheduler_type is not None:
		if scheduler_kwargs is None:
			scheduler_kwargs = {}
		for i in range(num_aes):
			if scheduler_type == 'StepLR':
				schedulers[i] = optim.lr_scheduler.StepLR(opts[i], **scheduler_kwargs)
			elif scheduler_type == 'ReduceLROnPlateau':
				schedulers[i] = optim.lr_scheduler.ReduceLROnPlateau(opts[i], **scheduler_kwargs)
			elif scheduler_type == 'ExponentialLR':
				schedulers[i] = optim.lr_scheduler.ExponentialLR(opts[i], **scheduler_kwargs)
			# Add more schedulers as needed

	loaders = [iter(loader) for loader in train_loaders]

	for loop in range(loops):
		# Pick a random AE to start
		idx = random.randint(0, num_aes-1)
		try:
			imgs, _ = next(loaders[idx])
		except StopIteration:
			loaders[idx] = iter(train_loaders[idx])
			imgs, _ = next(loaders[idx])
		imgs = imgs.to(device)
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
			opts[order[i]].zero_grad()
			out = aes[order[i]](input_imgs)
			# If not last AE, pass output to next AE
			if i < num_aes-1:
				input_imgs = out.view_as(input_imgs)
		# Loss: compare final output to original input
		loss = loss_fn(out, imgs)
		loss.backward()
		opts[order[-1]].step()
		# Step scheduler only for the optimizer that was stepped
		scheduler = schedulers[order[-1]]
		if scheduler is not None:
			if scheduler_type == 'ReduceLROnPlateau':
				scheduler.step(loss.item())
			else:
				scheduler.step()

	if save:
		for i, ae in enumerate(aes):
			torch.save(ae.state_dict(), f"/Volumes/Buffalo-SSD/AIStudy/AEs/pths/rand_ae{i+1}.pth")

	return aes, test_loader

def random_test(aes, device=device, test_loader=None):
	if test_loader is None:
		_, test_loader = get_mnist_loaders(num_groups=len(aes))
	for ae in aes:
		ae.eval()
	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(device)
	recon_aes = []
	with torch.no_grad():
		for ae in aes:
			recon = ae(imgs)
			recon_aes.append(recon)
	fig, axes = plt.subplots(len(aes)+1, 8, figsize=(16, 2*(len(aes)+1)))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[0, i].set_title(f'label: {labels[i].item()}')
	axes[0, 0].set_ylabel('Original')
	for j in range(len(aes)):
		for i in range(8):
			axes[j+1, i].imshow(recon_aes[j][i].cpu().squeeze(), cmap='gray')
			axes[j+1, i].axis('off')
		axes[j+1, 0].set_ylabel(f'Reconstructed AE{j+1}')
	plt.tight_layout()
	plt.show()

# Test the random AEs and visualize results
if __name__ == "__main__":
	aes, test_loader = random_train(
		num_aes=16, 
		pretrain_epochs=100, 
		loops=100, 
		scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
	)
	random_test(aes, test_loader=test_loader)