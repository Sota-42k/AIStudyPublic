import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist import get_mnist_loaders
from Models import ConditionalAE as AE

# base directory for saving relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def double_teacher_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	latent = 32
	train_loader, _ = get_mnist_loaders()
	ae1 = AE(latent=latent).to(device)
	ae2 = AE(latent=latent).to(device)
	ae3 = AE(latent=latent).to(device)
	# Register compress layer on ae3 so it saves with the model
	ae3.compress = nn.Linear(64, 32).to(device)
	opt1 = optim.Adam(ae1.parameters(), lr=1e-3)
	opt2 = optim.Adam(ae2.parameters(), lr=1e-3)
	opt3 = optim.Adam(ae3.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()
	cls_loss_fn = nn.CrossEntropyLoss()
	scheduler1 = scheduler2 = scheduler3 = None
	if scheduler_type is not None:
		if scheduler_kwargs is None: scheduler_kwargs = {}
		if scheduler_type == 'StepLR':
			scheduler1 = optim.lr_scheduler.StepLR(opt1, **scheduler_kwargs)
			scheduler2 = optim.lr_scheduler.StepLR(opt2, **scheduler_kwargs)
			scheduler3 = optim.lr_scheduler.StepLR(opt3, **scheduler_kwargs)
		elif scheduler_type == 'ReduceLROnPlateau':
			scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, **scheduler_kwargs)
			scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, **scheduler_kwargs)
			scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(opt3, **scheduler_kwargs)
		elif scheduler_type == 'ExponentialLR':
			scheduler1 = optim.lr_scheduler.ExponentialLR(opt1, **scheduler_kwargs)
			scheduler2 = optim.lr_scheduler.ExponentialLR(opt2, **scheduler_kwargs)
			scheduler3 = optim.lr_scheduler.ExponentialLR(opt3, **scheduler_kwargs)
	ae1.train(); ae2.train(); ae3.train()
	teacher_epochs = epochs // 2
	student_epochs = epochs
	# Train teachers
	for epoch in range(teacher_epochs):
		for imgs, labels in train_loader:
			imgs = imgs.to(device); labels = labels.to(device)
			opt1.zero_grad(); opt2.zero_grad()
			x1_hat, logits1 = ae1(imgs, labels)
			x2_hat, logits2 = ae2(imgs, labels)
			loss1 = loss_fn(x1_hat, imgs) + cls_loss_fn(logits1, labels)
			loss2 = loss_fn(x2_hat, imgs) + cls_loss_fn(logits2, labels)
			loss1.backward(); opt1.step()
			loss2.backward(); opt2.step()
			if scheduler1 is not None:
				if scheduler_type == 'ReduceLROnPlateau': scheduler1.step(loss1.item())
				else: scheduler1.step()
			if scheduler2 is not None:
				if scheduler_type == 'ReduceLROnPlateau': scheduler2.step(loss2.item())
				else: scheduler2.step()
	for p in ae1.parameters(): p.requires_grad = False
	for p in ae2.parameters(): p.requires_grad = False
	# Train student
	for epoch in range(student_epochs):
		for imgs, labels in train_loader:
			imgs = imgs.to(device); labels = labels.to(device)
			with torch.no_grad():
				z1 = ae1.enc(imgs); z2 = ae2.enc(imgs)
				x1_hat, _ = ae1(imgs, labels)
				x2_hat, _ = ae2(imgs, labels)
				x_teacher_target = (x1_hat + x2_hat) / 2
			z_cat = torch.cat([z1, z2], dim=1)
			z_compressed = ae3.compress(z_cat)
			x_hat = ae3.decode(z_compressed, labels)
			logits3 = ae3.classifier(z_compressed)
			loss = nn.functional.mse_loss(x_hat, x_teacher_target) + cls_loss_fn(logits3, labels)
			opt3.zero_grad(); loss.backward(); opt3.step()
			if scheduler3 is not None:
				if scheduler_type == 'ReduceLROnPlateau': scheduler3.step(loss.item())
				else: scheduler3.step()
	if save:
		base = os.path.join(BASE_DIR, 'pths')
		os.makedirs(base, exist_ok=True)
		torch.save(ae1.state_dict(), os.path.join(base, 'd_teacher1_ae.pth'))
		torch.save(ae2.state_dict(), os.path.join(base, 'd_teacher2_ae.pth'))
		torch.save(ae3.state_dict(), os.path.join(base, 'd_student_ae.pth'))
		print('Saved DoubleTeacherAE models')
	return (ae1, ae2, ae3)


def double_teacher_test(ae1, ae2, ae3, device=None, test_loader=None, save_fig=False):
	import matplotlib.pyplot as plt
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	if test_loader is None:
		_, test_loader = get_mnist_loaders()

	ae1.eval()
	ae2.eval()
	ae3.eval()

	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(device); labels = labels.to(device)

	with torch.no_grad():
		z1 = ae1.enc(imgs)
		z2 = ae2.enc(imgs)
		# 教師の出力
		recon_ae1 = ae1.dec(z1).view(-1, 1, 28, 28)
		logits1 = ae1.classifier(z1)
		recon_ae2 = ae2.dec(z2).view(-1, 1, 28, 28)
		logits2 = ae2.classifier(z2)
		x_teacher_target = (recon_ae1 + recon_ae2) / 2
		# concat→線形層で圧縮
		z_cat = torch.cat([z1, z2], dim=1)
		z_compressed = ae3.compress(z_cat)
		recon_ae3 = ae3.decode(z_compressed, labels)
		logits3 = ae3.classifier(z_compressed)
		preds1 = logits1.argmax(dim=1)
		preds2 = logits2.argmax(dim=1)
		preds3 = logits3.argmax(dim=1)
	fig, axes = plt.subplots(5, 8, figsize=(16, 7))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[1, i].imshow(recon_ae1[i].cpu().squeeze(), cmap='gray')
		axes[1, i].axis('off')
		axes[2, i].imshow(recon_ae2[i].cpu().squeeze(), cmap='gray')
		axes[2, i].axis('off')
		axes[3, i].imshow(recon_ae3[i].cpu().squeeze(), cmap='gray')
		axes[3, i].axis('off')
		axes[4, i].text(0.5, 0.5, f'T1:{preds1[i].item()}\nT2:{preds2[i].item()}\nS:{preds3[i].item()}', fontsize=10, ha='center')
		axes[4, i].axis('off')
	axes[0, 0].set_ylabel('Input')
	axes[1, 0].set_ylabel('Teacher1')
	axes[2, 0].set_ylabel('Teacher2')
	axes[3, 0].set_ylabel('Student')
	axes[4, 0].set_ylabel('Predicted')
	plt.tight_layout()
	if save_fig:
		os.makedirs(os.path.join(BASE_DIR, 'samples'), exist_ok=True)
		plt.savefig(os.path.join(BASE_DIR, 'samples', 'double_teacher_ae_test.png'))
	plt.show()

if __name__ == "__main__":
	double_teacher_train(epochs=100, save=True, scheduler_type='StepLR', scheduler_kwargs={'step_size': 100, 'gamma': 0.5})
