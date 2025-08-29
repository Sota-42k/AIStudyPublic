# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from simpleAE import ae_train

device = torch.device("mps")
train_loaders, test_loader = get_mnist_loaders(num_groups=2)

def mutual_train(pretrain_epochs=5, loops=5, device=device, train_loaders=train_loaders, save=True, scheduler_type=None, scheduler_kwargs=None):
	ae1 = ae_train(device=device, train_loader=train_loaders[0], epochs=pretrain_epochs, save=False, scheduler_type=None)
	ae2 = ae_train(device=device, train_loader=train_loaders[1], epochs=pretrain_epochs, save=False, scheduler_type=None)
	loader1 = iter(train_loaders[0])
	loader2 = iter(train_loaders[1])
	opt1 = optim.Adam(ae1.parameters(), lr=1e-3)
	opt2 = optim.Adam(ae2.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()

	# Scheduler setup for both optimizers
	scheduler1 = None
	scheduler2 = None
	if scheduler_type is not None:
		if scheduler_kwargs is None:
			scheduler_kwargs = {}
		if scheduler_type == 'StepLR':
			scheduler1 = optim.lr_scheduler.StepLR(opt1, **scheduler_kwargs)
			scheduler2 = optim.lr_scheduler.StepLR(opt2, **scheduler_kwargs)
		elif scheduler_type == 'ReduceLROnPlateau':
			scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, **scheduler_kwargs)
			scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, **scheduler_kwargs)
		elif scheduler_type == 'ExponentialLR':
			scheduler1 = optim.lr_scheduler.ExponentialLR(opt1, **scheduler_kwargs)
			scheduler2 = optim.lr_scheduler.ExponentialLR(opt2, **scheduler_kwargs)
		# Add more schedulers as needed

	for loop in range(loops):
		try:
			imgs1, _ = next(loader1)
		except StopIteration:
			loader1 = iter(train_loaders[0])
			imgs1, _ = next(loader1)
		try:
			imgs2, _ = next(loader2)
		except StopIteration:
			loader2 = iter(train_loaders[1])
			imgs2, _ = next(loader2)
		imgs1 = imgs1.to(device)
		imgs2 = imgs2.to(device)

		# ae1 output -> ae2 input
		for p in ae2.parameters():
			p.requires_grad = False
		opt1.zero_grad()
		out1 = ae1(imgs1)
		out2 = ae2(out1)
		loss1 = loss_fn(out2, imgs2)
		loss1.backward()
		opt1.step()
		for p in ae2.parameters():
			p.requires_grad = True
		if scheduler1 is not None:
			if scheduler_type == 'ReduceLROnPlateau':
				scheduler1.step(loss1.item())
			else:
				scheduler1.step()
		# ae2 output -> ae1 input
		for p in ae1.parameters():
			p.requires_grad = False
		opt2.zero_grad()
		out2b = ae2(imgs2)
		out1b = ae1(out2b)
		loss2 = loss_fn(out1b, imgs1)
		loss2.backward()
		opt2.step()
		for p in ae1.parameters():
			p.requires_grad = True
		if scheduler2 is not None:
			if scheduler_type == 'ReduceLROnPlateau':
				scheduler2.step(loss2.item())
			else:
				scheduler2.step()

	if save:
		torch.save(ae1.state_dict(), "/Volumes/Buffalo-SSD/AIStudy/AEs/pths/m_ae1.pth")
		torch.save(ae2.state_dict(), "/Volumes/Buffalo-SSD/AIStudy/AEs/pths/m_ae2.pth")

	return ae1, ae2

def mutual_test(ae1, ae2, device=device, test_loader=test_loader):
	ae1.eval()
	ae2.eval()
	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(device)
	with torch.no_grad():
		recon_ae1 = ae1(imgs)
		recon_ae2 = ae2(imgs)
	fig, axes = plt.subplots(3, 8, figsize=(16, 4))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[0, i].set_title(f'label: {labels[i].item()}')
		axes[1, i].imshow(recon_ae1[i].cpu().squeeze(), cmap='gray')
		axes[1, i].axis('off')
		axes[2, i].imshow(recon_ae2[i].cpu().squeeze(), cmap='gray')
		axes[2, i].axis('off')
	axes[0, 0].set_ylabel('Original')
	axes[1, 0].set_ylabel('Reconstructed AE1')
	axes[2, 0].set_ylabel('Reconstructed AE2')
	plt.tight_layout()
	plt.show()

# Test the AEs and visualize results
if __name__ == "__main__":
	mutual_test(*mutual_train(loops=5))
