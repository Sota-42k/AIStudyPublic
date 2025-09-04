
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(data_dir="./data", batch_size=128, test_batch_size=8, download=True, num_groups=1):
	transform = transforms.ToTensor()
	train_ds = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
	test_ds = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

	def split_dataset(dataset, num_groups):
		if num_groups <= 1:
			return [dataset]
		length = len(dataset)
		sizes = [length // num_groups] * num_groups
		for i in range(length % num_groups):
			sizes[i] += 1
		return torch.utils.data.random_split(dataset, sizes)

	train_splits = split_dataset(train_ds, num_groups)
	if num_groups <= 1:
		train_loaders = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	else:
		train_loaders = [DataLoader(split, batch_size=batch_size, shuffle=True) for split in train_splits]
	test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True)
	return train_loaders, test_loader