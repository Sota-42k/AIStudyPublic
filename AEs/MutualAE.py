# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from SimpleAE import ae_train

# base directory for saving relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def mutual_train(pretrain_epochs=5, loops=1000, device=None, train_loaders=None, save=True, scheduler_type=None, scheduler_kwargs=None):
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	train_loader, _ = get_mnist_loaders()
	ae1 = ae_train(device=device, train_loader=train_loader, epochs=pretrain_epochs, save=False, scheduler_type=None)
	ae2 = ae_train(device=device, train_loader=train_loader, epochs=pretrain_epochs, save=False, scheduler_type=None)
	train_iter1 = iter(train_loader)
	train_iter2 = iter(train_loader)
	opt1 = optim.Adam(ae1.parameters(), lr=1e-3)
	opt2 = optim.Adam(ae2.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()
	cls_loss_fn = nn.CrossEntropyLoss()
	scheduler1 = scheduler2 = None
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
	for loop in range(loops):
		try:
			imgs1, labels1 = next(train_iter1)
		except StopIteration:
			train_iter1 = iter(train_loader)
			imgs1, labels1 = next(train_iter1)
		try:
			imgs2, labels2 = next(train_iter2)
		except StopIteration:
			train_iter2 = iter(train_loader)
			imgs2, labels2 = next(train_iter2)
		imgs1 = imgs1.to(device); labels1 = labels1.to(device)
		imgs2 = imgs2.to(device); labels2 = labels2.to(device)
		# ae1 -> ae2
		opt1.zero_grad()
		out1, logits1 = ae1(imgs1, labels1)
		out2, logits2 = ae2(out1, labels2)
		loss1 = loss_fn(out2, imgs2) + cls_loss_fn(logits1, labels1) + cls_loss_fn(logits2, labels2)
		loss1.backward()
		for p in ae2.parameters():
			if p.grad is not None:
				p.grad.detach_(); p.grad.zero_()
		opt1.step()
		if scheduler1 is not None:
			if scheduler_type == 'ReduceLROnPlateau': scheduler1.step(loss1.item())
			else: scheduler1.step()
		# ae2 -> ae1
		opt2.zero_grad()
		out2b, logits2b = ae2(imgs2, labels2)
		out1b, logits1b = ae1(out2b, labels1)
		loss2 = loss_fn(out1b, imgs1) + cls_loss_fn(logits2b, labels2) + cls_loss_fn(logits1b, labels1)
		loss2.backward()
		for p in ae1.parameters():
			if p.grad is not None:
				p.grad.detach_(); p.grad.zero_()
		opt2.step()
		if scheduler2 is not None:
			if scheduler_type == 'ReduceLROnPlateau': scheduler2.step(loss2.item())
			else: scheduler2.step()
	if save:
		base = os.path.join(BASE_DIR, "pths")
		os.makedirs(base, exist_ok=True)
		torch.save(ae1.state_dict(), os.path.join(base, "m_ae1.pth"))
		torch.save(ae2.state_dict(), os.path.join(base, "m_ae2.pth"))
		print("Saved MutualAE models")
	return (ae1, ae2)

def mutual_test(ae1, ae2, device=None, test_loader=None, save_fig=False):
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	if test_loader is None:
		_, test_loader = get_mnist_loaders()

	ae1.eval()
	ae2.eval()

	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(device)
	labels = labels.to(device)

	with torch.no_grad():
		recon_ae1, logits1 = ae1(imgs)
		recon_ae2, logits2 = ae2(imgs)
		preds1 = logits1.argmax(dim=1)
		preds2 = logits2.argmax(dim=1)

	fig, axes = plt.subplots(4, 8, figsize=(16, 8))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[0, i].set_title(f'label: {labels[i].item()}')
		axes[1, i].imshow(recon_ae1[i].cpu().squeeze(), cmap='gray')
		axes[1, i].axis('off')
		axes[2, i].imshow(recon_ae2[i].cpu().squeeze(), cmap='gray')
		axes[2, i].axis('off')
		axes[3, i].text(0.5, 0.5, f'pred1:{preds1[i].item()}\npred2:{preds2[i].item()}', fontsize=10, ha='center')
		axes[3, i].axis('off')
	axes[0, 0].set_ylabel('Original')
	axes[1, 0].set_ylabel('Reconstructed AE1')
	axes[2, 0].set_ylabel('Reconstructed AE2')
	axes[3, 0].set_ylabel('Predicted')
	plt.tight_layout()
	if save_fig:
		os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)
		plt.savefig(os.path.join(BASE_DIR, "samples", "mutualAE_test.png"))
	plt.show()

# Test the AEs and visualize results
if __name__ == "__main__":
	mutual_test(*mutual_train(loops=5))
