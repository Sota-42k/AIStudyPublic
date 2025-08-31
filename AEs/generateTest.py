import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # ensure AEs/ is on path
from Models import AE, VAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Base directory for pth files (AEs/pths)
PTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pths")

def get_all_model_outputs():
	outputs = []
	model_names = []
	# モデル情報リスト: (pth, モデルクラス, latent次元, ラベル)
	model_info = [
		# simple
		("ae.pth", AE, 'simpleAE'),
		("vae.pth", VAE, 'simpleVAE'),
		# mutual
		("m_ae1.pth", AE, 'mutualAE1'),
		("m_ae2.pth", AE, 'mutualAE2'),
		("m_vae1.pth", VAE, 'mutualVAE1'),
		("m_vae2.pth", VAE, 'mutualVAE2'),
		# single teacher
		("s_student_ae.pth", AE, 'singleTeacherAE'),
		("s_student_vae.pth", VAE, 'singleTeacherVAE'),
		# double teacher
		("d_student_ae.pth", AE, 'doubleTeacherAE'),
		("d_student_vae.pth", VAE, 'doubleTeacherVAE'),
	]
	# randomAE / randomVAE (自動検出)
	rand_ae_idx = 1
	while True:
		pth = f"rand_ae{rand_ae_idx}.pth"
		full_pth = os.path.join(PTH_DIR, pth)
		if not os.path.exists(full_pth):
			break
		model_info.append((pth, AE, f'randomAE{rand_ae_idx}'))
		rand_ae_idx += 1
	rand_vae_idx = 1
	while True:
		pth = f"rand_vae{rand_vae_idx}.pth"
		full_pth = os.path.join(PTH_DIR, pth)
		if not os.path.exists(full_pth):
			break
		model_info.append((pth, VAE, f'randomVAE{rand_vae_idx}'))
		rand_vae_idx += 1

	# 1ループで全モデル処理
	for pth_name, model_class, label in model_info:
		pth = os.path.join(PTH_DIR, pth_name)
		if not os.path.exists(pth):
			continue
		model = model_class().to(device)
		model.load_state_dict(torch.load(pth, map_location=device))
		model.eval()
		# 推論に使うlatent次元はデコーダの最初のLinear層のin_featuresから推定
		if isinstance(model.dec[0], torch.nn.Linear):
			latent_dim = model.dec[0].in_features
		else:
			latent_dim = 32
		row = []
		with torch.no_grad():
			for digit in range(10):
				latent = torch.zeros((1, latent_dim), device=device)
				latent[0, digit % latent_dim] = 1.0
				img = model.dec(latent).view(28, 28).cpu().numpy()
				row.append(img)
		outputs.append(row)
		model_names.append(label)

	return outputs, model_names

def show_all_model_outputs(save_fig=False):
	outputs, model_names = get_all_model_outputs()
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
	plt.show()

if __name__ == "__main__":
	show_all_model_outputs(save_fig=True)
