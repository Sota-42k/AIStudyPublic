import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist import get_mnist_loaders
from Models import AE

def double_teacher_train(device=None, epochs=10, save=True, scheduler_type=None, scheduler_kwargs=None):
	"""
	2つの教師AE(AE1, AE2)の出力（concat）から生徒AE(AE3)を学習する。
	"""
	if device is None:
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	# 2つのグループでデータローダ取得
	train_loaders, _ = get_mnist_loaders(num_groups=2)
	train_loader1, train_loader2 = train_loaders

	# 教師も生徒もlatent=32
	latent = 32
	ae1 = AE(latent=latent).to(device)
	ae2 = AE(latent=latent).to(device)
	ae3 = AE(latent=latent).to(device)
	# 圧縮用線形層（教師2人のlatentをconcat→32次元に圧縮）
	compress_linear = nn.Linear(64, 32).to(device)

	opt1 = optim.Adam(ae1.parameters(), lr=1e-3)
	opt2 = optim.Adam(ae2.parameters(), lr=1e-3)
	opt3 = optim.Adam(list(ae3.parameters()) + list(compress_linear.parameters()), lr=1e-3)
	loss_fn = nn.MSELoss()

	scheduler1 = scheduler2 = scheduler3 = None
	if scheduler_type is not None:
		if scheduler_kwargs is None:
			scheduler_kwargs = {}
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

	ae1.train()
	ae2.train()
	ae3.train()
	compress_linear.train()

	# 1. 教師の学習
	teacher_epochs = epochs // 2
	student_epochs = epochs

	for epoch in range(teacher_epochs):
		for (imgs1, _), (imgs2, _) in zip(train_loader1, train_loader2):
			imgs1 = imgs1.to(device)
			imgs2 = imgs2.to(device)
			# AE1, AE2: 教師として通常学習
			opt1.zero_grad()
			x1_hat = ae1(imgs1)
			loss1 = loss_fn(x1_hat, imgs1)
			loss1.backward()
			opt1.step()

			opt2.zero_grad()
			x2_hat = ae2(imgs2)
			loss2 = loss_fn(x2_hat, imgs2)
			loss2.backward()
			opt2.step()

			# scheduler
			if scheduler1 is not None:
				if scheduler_type == 'ReduceLROnPlateau':
					scheduler1.step(loss1.item())
				else:
					scheduler1.step()
			if scheduler2 is not None:
				if scheduler_type == 'ReduceLROnPlateau':
					scheduler2.step(loss2.item())
				else:
					scheduler2.step()

	# 2. 教師を凍結
	for param in ae1.parameters():
		param.requires_grad = False
	for param in ae2.parameters():
		param.requires_grad = False

	# 3. 生徒の学習（concatのみ）
	for epoch in range(student_epochs):
		for (imgs1, _), (imgs2, _) in zip(train_loader1, train_loader2):
			imgs1 = imgs1.to(device)
			imgs2 = imgs2.to(device)

			with torch.no_grad():
				z1 = ae1.enc(imgs1)
				z2 = ae2.enc(imgs2)
				x1_hat = ae1.dec(z1).view(-1, 1, 28, 28)
				x2_hat = ae2.dec(z2).view(-1, 1, 28, 28)
				x_teacher_target = (x1_hat + x2_hat) / 2
			# concat→線形層で圧縮
			z_cat = torch.cat([z1, z2], dim=1)  # (batch, 64)
			z_compressed = compress_linear(z_cat)  # (batch, 32)
			x_hat = ae3.dec(z_compressed).view(-1, 1, 28, 28)
			loss = nn.functional.mse_loss(x_hat, x_teacher_target)
			opt3.zero_grad()
			loss.backward()
			opt3.step()

			# scheduler for student optimizer
			if scheduler3 is not None:
				if scheduler_type == 'ReduceLROnPlateau':
					scheduler3.step(loss.item())
				else:
					scheduler3.step()

	if save:
		torch.save(ae1.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "d_teacher1_ae.pth"))
		torch.save(ae2.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "d_teacher2_ae.pth"))
		torch.save(ae3.state_dict(), os.path.join(os.path.dirname(__file__), "pths", "d_student_ae.pth"))
	return ae1, ae2, ae3


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
	imgs = imgs.to(device)

	with torch.no_grad():
		z1 = ae1.enc(imgs)
		z2 = ae2.enc(imgs)
		# 教師の出力
		recon_ae1 = ae1.dec(z1).view(-1, 1, 28, 28)
		recon_ae2 = ae2.dec(z2).view(-1, 1, 28, 28)
		x_teacher_target = (recon_ae1 + recon_ae2) / 2
		# concat→線形層で圧縮
		compress_linear = None
		for m in ae3.modules():
			if isinstance(m, nn.Linear) and m.in_features == 64 and m.out_features == 32:
				compress_linear = m
				break
		if compress_linear is None:
			compress_linear = nn.Linear(64, 32).to(device)  # fallback
		z_cat = torch.cat([z1, z2], dim=1)
		z_compressed = compress_linear(z_cat)
		recon_ae3 = ae3.dec(z_compressed).view(-1, 1, 28, 28)
	fig, axes = plt.subplots(4, 8, figsize=(16, 5))
	for i in range(8):
		axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
		axes[0, i].axis('off')
		axes[1, i].imshow(recon_ae1[i].cpu().squeeze(), cmap='gray')
		axes[1, i].axis('off')
		axes[2, i].imshow(recon_ae2[i].cpu().squeeze(), cmap='gray')
		axes[2, i].axis('off')
		axes[3, i].imshow(recon_ae3[i].cpu().squeeze(), cmap='gray')
		axes[3, i].axis('off')
	axes[0, 0].set_ylabel('Input')
	axes[1, 0].set_ylabel('Teacher1')
	axes[2, 0].set_ylabel('Teacher2')
	axes[3, 0].set_ylabel('Student')
	plt.tight_layout()
	if save_fig:
		plt.savefig('double_teacher_ae_test.png')
	plt.show()

if __name__ == "__main__":
	double_teacher_train(epochs=100, save=True, scheduler_type='StepLR', scheduler_kwargs={'step_size': 100, 'gamma': 0.5})
