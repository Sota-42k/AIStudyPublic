import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # ensure AEs/ is on path
from Models import ConditionalAE as AE, ConditionalVAE as VAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Base directory for pth files (AEs/pths)
PTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pths")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_all_model_outputs():
	outputs = []
	model_names = []
	model_groups = []
	model_types = []  # 'ae' or 'vae'
	# モデル情報リスト: (pth, モデルクラス, ラベル, グループ)
	model_info = [
		# simple
		("ae.pth", AE, 'simpleAE', 'simple'),
		("vae.pth", VAE, 'simpleVAE', 'simple'),
		# mutual
		("m_ae1.pth", AE, 'mutualAE1', 'mutual'),
		("m_ae2.pth", AE, 'mutualAE2', 'mutual'),
		("m_vae1.pth", VAE, 'mutualVAE1', 'mutual'),
		("m_vae2.pth", VAE, 'mutualVAE2', 'mutual'),
		# single teacher
		("s_student_ae.pth", AE, 'singleTeacherAE', 'single_teacher'),
		("s_student_vae.pth", VAE, 'singleTeacherVAE', 'single_teacher'),
		# double teacher
		("d_student_ae.pth", AE, 'doubleTeacherAE', 'double_teacher'),
		("d_student_vae.pth", VAE, 'doubleTeacherVAE', 'double_teacher'),
	]
	# randomAE / randomVAE (自動検出)
	rand_ae_idx = 1
	while True:
		pth = f"rand_ae{rand_ae_idx}.pth"
		full_pth = os.path.join(PTH_DIR, pth)
		if not os.path.exists(full_pth):
			break
		model_info.append((pth, AE, f'randomAE{rand_ae_idx}', 'random'))
		rand_ae_idx += 1
	rand_vae_idx = 1
	while True:
		pth = f"rand_vae{rand_vae_idx}.pth"
		full_pth = os.path.join(PTH_DIR, pth)
		if not os.path.exists(full_pth):
			break
		model_info.append((pth, VAE, f'randomVAE{rand_vae_idx}', 'random'))
		rand_vae_idx += 1

	# 1ループで全モデル処理
	for pth_name, model_class, label, group in model_info:
		pth = os.path.join(PTH_DIR, pth_name)
		if not os.path.exists(pth):
			print(f"Skip {label}: '{pth_name}' not found under {PTH_DIR}")
			continue
		model = model_class().to(device)
		state = torch.load(pth, map_location=device, weights_only=True)
		model.load_state_dict(state, strict=False)
		model.eval()
		row = []
		with torch.no_grad():
			for digit in range(10):
				imgs = model.generate(digit, n=1)
				img = imgs[0].view(28, 28).cpu().numpy()
				row.append(img)
		outputs.append(row)
		model_names.append(label)
		model_groups.append(group)
		model_types.append('ae' if model_class is AE else 'vae')

	return outputs, model_names, model_groups, model_types

def save_per_model_images(outputs, model_names, model_groups):
	"""各モデルごとの1x10グリッド画像をグループ別フォルダに保存します。"""
	base_dir = os.path.join(BASE_DIR, "samples")
	for row, name, group in zip(outputs, model_names, model_groups):
		group_dir = os.path.join(base_dir, group)
		os.makedirs(group_dir, exist_ok=True)
		fig, axes = plt.subplots(1, 10, figsize=(20, 2))
		for col_idx, img in enumerate(row):
			axes[col_idx].imshow(img, cmap='gray')
			axes[col_idx].axis('off')
			axes[col_idx].set_title(str(col_idx))
		plt.suptitle(name, fontsize=12)
		save_path = os.path.join(group_dir, f"{name}.png")
		os.makedirs(group_dir, exist_ok=True)
		plt.savefig(save_path, bbox_inches='tight')
		plt.close(fig)
		print(f"Saved {name} -> {save_path}")

def save_group_combined_images(outputs, model_names, model_groups, model_types):
	"""全てのグループについて、AE と VAE をそれぞれ1つの画像にまとめて保存"""
	base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
	os.makedirs(base_dir, exist_ok=True)
	# 既存の全グループを動的に検出（例: mutual, random, simple, single_teacher, double_teacher）
	groups = sorted(set(model_groups))
	for group in groups:
		for mtype in ['ae', 'vae']:
			indices = [i for i,(g,t) in enumerate(zip(model_groups, model_types)) if g==group and t==mtype]
			label = f"{group}_{mtype}"
			if not indices:
				print(f"Skip saving {label}: no models found")
				continue
			rows = len(indices)
			fig, axes = plt.subplots(rows, 10, figsize=(20, 2*rows))
			# Normalize axes to 2D array shape
			if rows == 1:
				axes = axes.reshape(1, -1)
			for r, idx in enumerate(indices):
				row_imgs = outputs[idx]
				for c, img in enumerate(row_imgs):
					axes[r, c].imshow(img, cmap='gray')
					axes[r, c].axis('off')
					if r == 0:
						axes[r, c].set_title(str(c))
				axes[r, 0].set_ylabel(model_names[idx], rotation=0, labelpad=40, fontsize=10, va='center')
			plt.tight_layout()
			plt.suptitle(label, y=1.02)
			group_dir = os.path.join(base_dir, group)
			os.makedirs(group_dir, exist_ok=True)
			save_path = os.path.join(group_dir, f"{label}.png")
			plt.savefig(save_path)
			plt.close(fig)
			print(f"Saved {label} -> {save_path}")

def show_all_model_outputs(save_fig=True, save_per_model=False, save_group_combined=True):
	outputs, model_names, model_groups, model_types = get_all_model_outputs()
	if not outputs:
		print("No model outputs found. Check pth files under AEs/pths.")
		return
	fig, axes = plt.subplots(len(outputs), 10, figsize=(20, 2*len(outputs)))
	for row_idx, row in enumerate(outputs):
		for col_idx, img in enumerate(row):
			axes[row_idx, col_idx].imshow(img, cmap='gray')
			axes[row_idx, col_idx].axis('off')
			if row_idx == 0:
				axes[row_idx, col_idx].set_title(str(col_idx))
		axes[row_idx, 0].set_ylabel(model_names[row_idx], rotation=0, labelpad=40, fontsize=12, va='center')
		print(f"Displayed {model_names[row_idx]}")
	plt.tight_layout()
	plt.suptitle('All AE/VAE Model Outputs', fontsize=16, y=1.02)
	if save_fig:
		save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "all_models.png")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path)
		print(f"Saved figure to {save_path}")
	if save_per_model:
		save_per_model_images(outputs, model_names, model_groups)
	if save_group_combined:
		save_group_combined_images(outputs, model_names, model_groups, model_types)
	plt.show()

if __name__ == "__main__":
	# グループごとの結合画像（全グループ対象）を保存
	show_all_model_outputs()