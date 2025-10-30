"""
This module provides functions to load datasets.
"""

__format__ = "restructuredtext"
__all__ = ["load_dataset"]

import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        # mimic torchvision.datasets.MNIST interface
        data = np.load(root + ("/train_data.npy" if train else "/test_data.npy"))
        labels = np.load(root + ("/train_labels.npy" if train else "/test_labels.npy"))
        self.images = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], int(self.labels[idx])
        img = torch.tensor(img, dtype=torch.float32)
        # if self.transform:
        #     img = self.transform(img)
        return img, label


def _load_eran_dataset(
    dataset: str, dir_path: str = None, max_samples_num: int = 1000
) -> list[tuple[torch.Tensor, torch.Tensor]]:

    dir_path = "../../datasets" if dir_path is None else dir_path
    print(
        f"Load ERAN dataset {dataset} in {dir_path} "
        f"(It has a different index order for image data)."
    )
    print("**** dir_path:", dir_path)

    if dataset == "mnist":
        fpath = dir_path + "/mnist_test_full.csv"
        shape = (1, 1, 28, 28)
    elif dataset == "cifar10":
        fpath = dir_path + "/cifar10_test_5000.csv"
        shape = (1, 3, 32, 32)
    elif dataset == "deeppoly":
        fpath = dir_path + "/deeppoly_test.csv"
        shape = (2,)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")

    print(f"[DEBUG] Load only first {max_samples_num} samples.")
    data_loader = []
    with open(fpath, "r") as f:
        for _ in range(max_samples_num):
            line = f.readline().strip().split(",")
            label = torch.tensor([int(line[0])])
            sample = np.array([float(x) for x in line[1:]]).reshape(shape, order="A")
            sample = torch.from_numpy(sample)
            data_loader.append((sample, label))

    return data_loader


def load_dataset(
    dataset: str,
    dir_path: str,
    normalize: bool = False,
    means: Tensor | None = None,
    stds: Tensor | None = None,
) -> torch.utils.data.DataLoader:
    """
    Load a dataset from torchvision.

    :param dataset: The name of the dataset.
    :param dir_path: The directory where the dataset will be stored.
    :param normalize: True if the dataset should be normalized.
    :param means: The means for normalization.
    :param stds: The standard deviations for normalization.
    :return:
    """

    # __import__('ipdb').set_trace()

    # Create the directory if it does not exist.
    os.makedirs(dir_path, exist_ok=True)

    print(f"Load dataset {dataset} in {dir_path}")

    transform = transforms.Compose([transforms.ToTensor()])

    if normalize:
        print(f"[DEBUG] Set normalization transformer.")
        if means is not None and stds is not None:
            means = tuple(means.tolist())
            stds = tuple(stds.tolist())
        elif dataset == "cifar10":
            means = (0.4914, 0.4822, 0.4465)
            stds = (0.2023, 0.1994, 0.2010)
        elif dataset == "mnist":
            means = (0.0,)
            stds = (1.0,)

        transform.transforms.append(transforms.Normalize(means, stds))

    print(f"[DEBUG] Set dataset loader {dataset}.")

    kwargs = {"train": False, "download": True, "transform": transform}

    if dataset == "mnist":
        test_set = MNIST(root=dir_path, **kwargs)
    elif dataset == "cifar10":
        test_set = CIFAR10(root=dir_path + "/CIFAR10", **kwargs)
    elif dataset == "deeppoly":
        # __import__('ipdb').set_trace()
        test_set = MyDataset(
            root=dir_path + "/DeepPoly",
            train=False,
            transform=transform,
            download=False,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")

    data_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return data_loader
