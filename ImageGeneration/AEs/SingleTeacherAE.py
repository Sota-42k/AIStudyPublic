# Teacher-Student AutoEncoder Training
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ImageGeneration.mnist import get_mnist_loaders
from Models import ConditionalAE as AE

# base directory for saving relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def single_teacher_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	train_loader, _ = get_mnist_loaders()
	ae1 = AE().to(device)
	ae2 = AE().to(device)
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
	ae1.train(); ae2.train()
	teacher_epochs = student_epochs = epochs
	for epoch in range(teacher_epochs):
		for imgs1, labels1 in train_loader:
			imgs1 = imgs1.to(device)
			labels1 = labels1.to(device)
			opt1.zero_grad()
			x1_hat, logits1 = ae1(imgs1, labels1)
			loss1_recon = loss_fn(x1_hat, imgs1)
			loss1_cls = cls_loss_fn(logits1, labels1)
			loss1 = loss1_recon + loss1_cls
			loss1.backward(); opt1.step()
			if scheduler1 is not None:
				if scheduler_type == 'ReduceLROnPlateau': scheduler1.step(loss1.item())
				else: scheduler1.step()
	for p in ae1.parameters():
		p.requires_grad = False
	for epoch in range(student_epochs):
		for imgs1, labels1 in train_loader:
			imgs1 = imgs1.to(device)
			labels1 = labels1.to(device)
			with torch.no_grad():
				x1_hat_for_student, logits1 = ae1(imgs1, labels1)
			opt2.zero_grad()
			x2_input = x1_hat_for_student.detach()
			x2_hat, logits2 = ae2(x2_input, labels1)
			loss2_recon = loss_fn(x2_hat, x2_input)
			loss2_cls = cls_loss_fn(logits2, labels1)
			loss2 = loss2_recon + loss2_cls
			loss2.backward(); opt2.step()
			if scheduler2 is not None:
				if scheduler_type == 'ReduceLROnPlateau': scheduler2.step(loss2.item())
				else: scheduler2.step()
	if save:
		base = os.path.join(BASE_DIR, 'pths')
		os.makedirs(base, exist_ok=True)
		torch.save(ae1.state_dict(), os.path.join(base, 's_teacher_ae.pth'))
		torch.save(ae2.state_dict(), os.path.join(base, 's_student_ae.pth'))
		print('Saved SingleTeacherAE models')
	return (ae1, ae2)


def single_teacher_test(ae1, ae2, device=None, test_loader=None, save_fig=False):
	import matplotlib.pyplot as plt
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	if test_loader is None:
		_, test_loader = get_mnist_loaders()

	ae1.eval()
	ae2.eval()

	imgs, labels = next(iter(test_loader))
	imgs = imgs.to(device)

	with torch.no_grad():
		recon_ae1, logits1 = ae1(imgs, labels)
		recon_ae2, logits2 = ae2(recon_ae1.detach(), labels)
		preds1 = logits1.argmax(dim=1)
		preds2 = logits2.argmax(dim=1)
	fig, axes = plt.subplots(4, 8, figsize=(16, 6))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[1, i].imshow(recon_ae1[i].cpu().squeeze(), cmap='gray')
		axes[1, i].axis('off')
		axes[2, i].imshow(recon_ae2[i].cpu().squeeze(), cmap='gray')
		axes[2, i].axis('off')
		axes[3, i].text(0.5, 0.5, f'T:{preds1[i].item()}\nS:{preds2[i].item()}', fontsize=10, ha='center')
		axes[3, i].axis('off')
	axes[0, 0].set_ylabel('Input')
	axes[1, 0].set_ylabel('Teacher')
	axes[2, 0].set_ylabel('Student')
	axes[3, 0].set_ylabel('Predicted')
	plt.tight_layout()
	if save_fig:
		os.makedirs(os.path.join(BASE_DIR, 'samples'), exist_ok=True)
		plt.savefig(os.path.join(BASE_DIR, 'samples', 'single_teacher_ae_test.png'))
	plt.show()

if __name__ == "__main__":
	single_teacher_train(epochs=100, save=True, scheduler_type='StepLR', scheduler_kwargs={'step_size': 100, 'gamma': 0.5})
