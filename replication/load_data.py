# MNIST Data loading utilities

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST


class MNISTRegression(Dataset):
    """Wrapper for MNIST dataset that returns one-hot encoded targets as regression outputs."""

    def __init__(self, mnist_dataset, num_classes: int = 10):
        self.mnist_dataset = mnist_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        # Convert discrete label to one-hot encoded vector (as float for regression)
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[label] = 1.0
        return image, one_hot


class LoadData:
    """Download, split, and prepare MNIST for active learning."""

    def __init__(
        self,
        val_size: int = 100,
        train_size: int = 10000,
        seed: int = 369,
        root: str = "data",
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed
        self.root = root
        self.mnist_train, self.mnist_test = self.download_dataset()
        self.pool_size = len(self.mnist_train) - self.train_size - self.val_size
        (
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
        train_set, val_set, pool_set = random_split(
            self.mnist_train,
            [self.train_size, self.val_size, self.pool_size],
            generator=generator,
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
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        initial_idx: np.ndarray = np.array([], dtype=int)
        for i in range(10):
            candidates = np.where(self.y_train_All.numpy() == i)[0]
            idx = np.random.choice(candidates, size=2, replace=False)
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        y_init = self.y_train_All[initial_idx]
        print(f"Initial training data points: {X_init.shape[0]}")
        print(f"Data distribution for each class: {np.bincount(y_init.numpy())}")
        return X_init, y_init

    def load_all(self):
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

    @staticmethod
    def _labels_to_one_hot(labels: torch.Tensor, num_classes: int = 10) -> np.ndarray:
        """Convert discrete labels to one-hot encoded vectors."""
        one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        one_hot[np.arange(labels.shape[0]), labels.numpy()] = 1.0
        return one_hot

    @staticmethod
    def _labels_to_one_hot_tensor(
        labels: torch.Tensor, num_classes: int = 10
    ) -> torch.Tensor:
        """Convert labels to one-hot torch tensor (float)."""
        return F.one_hot(labels.long(), num_classes=num_classes).float()

    @staticmethod
    def make_regression_loader_from_arrays(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a regression DataLoader from numpy arrays (one-hot targets expected)."""
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def load_all_regression(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Load all data with targets converted to one-hot encoded regression format."""
        y_init_onehot = self._labels_to_one_hot(self.y_init)
        y_val_onehot = self._labels_to_one_hot(self.y_val)
        y_pool_onehot = self._labels_to_one_hot(self.y_pool)
        y_test_onehot = self._labels_to_one_hot(self.y_test)

        return (
            self.tensor_to_np(self.X_init),
            y_init_onehot,
            self.tensor_to_np(self.X_val),
            y_val_onehot,
            self.tensor_to_np(self.X_pool),
            y_pool_onehot,
            self.tensor_to_np(self.X_test),
            y_test_onehot,
        )

    def get_regression_dataloaders(
        self, batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Create dataloaders with one-hot encoded targets for regression."""
        train_regression = MNISTRegression(
            self.mnist_train if hasattr(self, "mnist_train") else self.mnist_test
        )
        test_regression = MNISTRegression(self.mnist_test)

        train_loader = DataLoader(train_regression, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_regression, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, train_loader, test_loader

    def get_regression_split_loaders(
        self, batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Create regression dataloaders for init (labeled), val, pool, and test splits."""

        def _make_loader(
            X_t: torch.Tensor, y_t: torch.Tensor, shuffle: bool
        ) -> DataLoader:
            y_oh = self._labels_to_one_hot_tensor(y_t, num_classes=10)
            dataset = torch.utils.data.TensorDataset(X_t.float(), y_oh)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        init_loader = _make_loader(self.X_init, self.y_init, shuffle=True)
        val_loader = _make_loader(self.X_val, self.y_val, shuffle=False)
        pool_loader = _make_loader(self.X_pool, self.y_pool, shuffle=False)
        test_loader = _make_loader(self.X_test, self.y_test, shuffle=False)

        return init_loader, val_loader, pool_loader, test_loader
