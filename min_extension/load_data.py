from __future__ import annotations

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


def get_balanced_initial_set(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int = 2,
    num_classes: int = 10,
):
    init_idx = []
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {cls}: {len(cls_indices)} available, need {samples_per_class}"
            )
        sampled = np.random.choice(cls_indices, size=samples_per_class, replace=False)
        init_idx.extend(sampled)
    init_idx = np.array(init_idx)

    pool_mask = np.ones(len(X), dtype=bool)
    pool_mask[init_idx] = False
    pool_idx = np.where(pool_mask)[0]

    return X[init_idx], y[init_idx], X[pool_idx], y[pool_idx]


class LoadData:
    def __init__(
        self,
        seed: int = 271,
        pretrain_size: int = 1000,
        val_size: int = 100,
        train_size: int = 20,
        root: str = "data",
        initial_per_class: int = 2,
    ) -> None:
        self.seed = seed
        self.pretrain_size = pretrain_size
        self.train_size = train_size
        self.val_size = val_size
        self.root = root
        self.initial_per_class = initial_per_class
        self.mnist_train, self.mnist_test = self.download_dataset()
        self.pool_size = (
            len(self.mnist_train) - self.train_size - self.val_size - self.pretrain_size
        )
        (
            self.X_pretrain_All,
            self.y_pretrain_All,
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        return tensor_data.detach().cpu().numpy()

    def check_mnist_folder(self) -> bool:
        return not os.path.exists(os.path.join(self.root, "MNIST"))

    def download_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        download = self.check_mnist_folder()
        mnist_train = MNIST(
            self.root, train=True, download=download, transform=transform
        )
        mnist_test = MNIST(
            self.root, train=False, download=download, transform=transform
        )
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        generator = torch.Generator().manual_seed(self.seed)
        pretrain_set, train_set, val_set, pool_set = random_split(
            self.mnist_train,
            [self.pretrain_size, self.train_size, self.val_size, self.pool_size],
            generator=generator,
        )
        pretrain_loader = DataLoader(
            dataset=pretrain_set, batch_size=self.pretrain_size, shuffle=True
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=10000, shuffle=True
        )
        X_pretrain_All, y_pretrain_All = next(iter(pretrain_loader))
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        return (
            X_pretrain_All,
            y_pretrain_All,
            X_train_All,
            y_train_All,
            X_val,
            y_val,
            X_pool,
            y_pool,
            X_test,
            y_test,
        )

    def preprocess_training_data(self):
        per_class = self.initial_per_class
        X_concat = (
            torch.cat([self.X_train_All, self.X_pool], dim=0).detach().cpu().numpy()
        )
        y_concat = (
            torch.cat([self.y_train_All, self.y_pool], dim=0).detach().cpu().numpy()
        )
        X_init_np, y_init_np, X_pool_np, y_pool_np = get_balanced_initial_set(
            X_concat,
            y_concat,
            samples_per_class=per_class,
            num_classes=10,
        )
        X_init = torch.from_numpy(X_init_np).float()
        y_init = torch.from_numpy(y_init_np).long()
        self.X_pool = torch.from_numpy(X_pool_np).float()
        self.y_pool = torch.from_numpy(y_pool_np).long()

        print(f"Initial training data points: {X_init.shape[0]}")
        binc = np.bincount(y_init.detach().cpu().numpy(), minlength=10)
        print(f"Data distribution for each class: {binc}")
        return X_init, y_init

    def load_all(self):
        return (
            self.tensor_to_np(self.X_pretrain_All),
            self.tensor_to_np(self.y_pretrain_All),
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )
