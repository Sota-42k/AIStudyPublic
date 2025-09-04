
# Configuration
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist import get_mnist_loaders
from SimpleAE import ae_train

# base directory for saving relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def random_train(num_aes=3, pretrain_epochs=5, loops=1000, device=None, scheduler_type=None, scheduler_kwargs=None, save=True):
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	train_loader, _ = get_mnist_loaders()
	aes = [ae_train(device=device, train_loader=train_loader, epochs=pretrain_epochs, save=False, scheduler_type=None) for _ in range(num_aes)]
	opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in aes]
	loss_fn = nn.MSELoss()
	cls_loss_fn = nn.CrossEntropyLoss()
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
	loaders = [iter(train_loader) for _ in range(num_aes)]
	for loop in range(loops):
		idx = random.randint(0, num_aes-1)
		try:
			imgs, labels = next(loaders[idx])
		except StopIteration:
			loaders[idx] = iter(train_loader)
			imgs, labels = next(loaders[idx])
		imgs = imgs.to(device)
		labels = labels.to(device)
		order = list(range(num_aes))
		random.shuffle(order)
		if order[0] != idx:
			order.remove(idx)
			order = [idx] + order
		input_imgs = imgs
		logits_list = []
		for i in range(num_aes):
			opts[order[i]].zero_grad()
			out, logits = aes[order[i]](input_imgs, labels)
			logits_list.append(logits)
			if i < num_aes-1:
				input_imgs = out.view_as(input_imgs)
		# Loss
		loss_recon = loss_fn(out, imgs)
		loss_cls = sum(cls_loss_fn(l, labels) for l in logits_list)
		loss = loss_recon + loss_cls
		loss.backward()
		for j in range(num_aes):
			if j != order[-1]:
				for p in aes[order[j]].parameters():
					if p.grad is not None:
						p.grad.detach_(); p.grad.zero_()
		opts[order[-1]].step()
		scheduler = schedulers[order[-1]]
		if scheduler is not None:
			if scheduler_type == 'ReduceLROnPlateau': scheduler.step(loss.item())
			else: scheduler.step()
	if save:
		base = os.path.join(BASE_DIR, "pths")
		os.makedirs(base, exist_ok=True)
		for i, ae in enumerate(aes):
			torch.save(ae.state_dict(), os.path.join(base, f"rand_ae{i+1}.pth"))
		print("Saved RandomAE models")
	return aes

def random_test(aes, device=None, test_loader=None, save_fig=False):
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	if test_loader is None:
		_, test_loader = get_mnist_loaders(num_groups=len(aes))
	for ae in aes:
		ae.eval()
	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(device)
	labels = labels.to(device)
	recon_aes = []
	logits_list = []
	with torch.no_grad():
		for ae in aes:
			recon, logits = ae(imgs)
			recon_aes.append(recon)
			logits_list.append(logits)
	preds_list = [logits.argmax(dim=1) for logits in logits_list]
	fig, axes = plt.subplots(len(aes)+2, 8, figsize=(16, 2*(len(aes)+2)))
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
	# 分類結果
	for i in range(8):
		pred_str = '\n'.join([f'{p[i].item()}' for p in preds_list])
		axes[len(aes)+1, i].text(0.5, 0.5, pred_str, fontsize=10, ha='center')
		axes[len(aes)+1, i].axis('off')
	axes[len(aes)+1, 0].set_ylabel('Predicted')
	plt.tight_layout()
	if save_fig:
		os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)
		plt.savefig(os.path.join(BASE_DIR, "samples", "randomAE_test.png"))
	plt.show()

# Test the random AEs and visualize results
if __name__ == "__main__":
	num_aes = 16
	aes = random_train(
		num_aes=num_aes, 
		pretrain_epochs=100, 
		loops=100, 
		scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
	)
	random_test(aes)