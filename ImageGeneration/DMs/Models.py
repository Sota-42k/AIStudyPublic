# Simple DDPM for MNIST (unconditional and class-conditional)
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simple UNet backbone for 1x28x28 images ---
class SimpleUNet(nn.Module):
	def __init__(self, in_channels=1, cond_dim=0):
		super().__init__()
		self.cond_dim = cond_dim
		self.enc1 = nn.Conv2d(in_channels + cond_dim, 32, 3, 1, 1)
		self.enc2 = nn.Conv2d(32, 64, 3, 2, 1)
		self.enc3 = nn.Conv2d(64, 128, 3, 2, 1)
		self.middle = nn.Conv2d(128, 128, 3, 1, 1)
		self.dec3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
		self.dec2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.out = nn.Conv2d(32, in_channels, 3, 1, 1)

	def forward(self, x, t, cond=None):
		# x: (B, 1, 28, 28), t: (B,), cond: (B, cond_dim) or None
		B = x.size(0)
		if self.cond_dim > 0 and cond is not None:
			cond = cond.view(B, self.cond_dim, 1, 1).expand(-1, -1, 28, 28)
			x = torch.cat([x, cond], dim=1)
		h1 = F.relu(self.enc1(x))
		h2 = F.relu(self.enc2(h1))
		h3 = F.relu(self.enc3(h2))
		m = F.relu(self.middle(h3))
		d3 = F.relu(self.dec3(m) + h2)
		d2 = F.relu(self.dec2(d3) + h1)
		out = self.out(d2)
		return out

# --- Sinusoidal timestep embedding (for t input) ---
def get_timestep_embedding(timesteps, embedding_dim):
	# timesteps: (B,) int64
	half_dim = embedding_dim // 2
	emb = torch.exp(
		torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) *
		-(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
	)
	emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
	emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
	if embedding_dim % 2 == 1:
		emb = F.pad(emb, (0,1))
	return emb

# --- DDPM Wrapper ---

# --- Unconditional DDPM ---
class UnconditionalDDPM(nn.Module):
	def __init__(self, timesteps=1000, device=None):
		super().__init__()
		if device is None:
			device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		self.device = device
		self.T = timesteps
		self.model = SimpleUNet(in_channels=1, cond_dim=32).to(device)
		self.betas = torch.linspace(1e-4, 0.02, self.T, device=device)
		self.alphas = 1. - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt().view(-1,1,1,1)
		sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1,1,1,1)
		return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

	def p_sample(self, x, t):
		t_emb = get_timestep_embedding(t, 32)
		out = self.model(x, t_emb)
		beta = self.betas[t].view(-1,1,1,1)
		alpha = self.alphas[t].view(-1,1,1,1)
		alpha_cumprod = self.alphas_cumprod[t].view(-1,1,1,1)
		pred_x0 = (x - (1-alpha_cumprod).sqrt() * out) / alpha_cumprod.sqrt()
		noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
		mean = alpha.sqrt() * pred_x0 + (1-alpha).sqrt() * noise
		return mean

	@torch.no_grad()
	def sample(self, shape):
		x = torch.randn(shape, device=self.device)
		for t in reversed(range(self.T)):
			t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
			x = self.p_sample(x, t_batch)
		return x

# --- Conditional DDPM ---
class ConditionalDDPM(nn.Module):
	def __init__(self, num_classes=10, timesteps=1000, device=None):
		super().__init__()
		if device is None:
			device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		self.device = device
		self.T = timesteps
		self.num_classes = num_classes
		self.model = SimpleUNet(in_channels=1, cond_dim=32+num_classes).to(device)
		self.betas = torch.linspace(1e-4, 0.02, self.T, device=device)
		self.alphas = 1. - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt().view(-1,1,1,1)
		sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1,1,1,1)
		return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

	def p_sample(self, x, t, y):
		t_emb = get_timestep_embedding(t, 32)
		y_onehot = F.one_hot(y, self.num_classes).float().to(x.device)
		cond = torch.cat([t_emb, y_onehot], dim=1)
		out = self.model(x, t_emb, cond)
		beta = self.betas[t].view(-1,1,1,1)
		alpha = self.alphas[t].view(-1,1,1,1)
		alpha_cumprod = self.alphas_cumprod[t].view(-1,1,1,1)
		pred_x0 = (x - (1-alpha_cumprod).sqrt() * out) / alpha_cumprod.sqrt()
		noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
		mean = alpha.sqrt() * pred_x0 + (1-alpha).sqrt() * noise
		return mean

	@torch.no_grad()
	def sample(self, shape, y):
		x = torch.randn(shape, device=self.device)
		for t in reversed(range(self.T)):
			t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
			x = self.p_sample(x, t_batch, y)
		return x
