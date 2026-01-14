from __future__ import annotations

from glob import glob
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class BiwiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob(f"{root_dir}/*/*_rgb.jpg"))

    def select_faces(self, face_ids):
        self.image_paths = []
        for face_id in face_ids:
            self.image_paths += sorted(glob(f"{self.root_dir}/{face_id:0>2}/*_rgb.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def convert_matrix_to_euler(self, rotation_matrix):
        R = rotation_matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z]) * (180 / np.pi)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_path = img_path.replace("_rgb.jpg", "_pose.txt")
        with open(label_path, "r") as f:
            lines = f.readlines()
            matrix = np.array(
                [
                    [float(v) for v in lines[0].strip().split()],
                    [float(v) for v in lines[1].strip().split()],
                    [float(v) for v in lines[2].strip().split()],
                ]
            )

        angles = self.convert_matrix_to_euler(matrix).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(angles)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_loader(dataset, indices, batch_size, shuffle):
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def prep_data(
    data_dir: str,
    seed: int,
    pool_ratio: float,
    pool_size: int,
    val_size: int,
    test_size: int,
) -> Tuple:
    transform = get_transform()
    pool_dataset = BiwiDataset(data_dir, transform=transform)
    pool_dataset.select_faces(list(range(1, 21)))
    test_dataset = BiwiDataset(data_dir, transform=transform)
    test_dataset.select_faces(list(range(1, 25)))

    total = len(pool_dataset)
    if total < pool_size + val_size:
        raise ValueError("Not enough points available in dataset")
    init_size = int(pool_ratio * pool_size)

    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    idx_labeled = perm[:init_size]
    idx_pool = perm[init_size:pool_size]
    idx_val = perm[pool_size : pool_size + val_size]
    test_perm = torch.randperm(
        len(test_dataset), generator=torch.Generator().manual_seed(seed)
    )
    idx_test = test_perm[:test_size]

    init_loader = make_loader(pool_dataset, idx_labeled, batch_size=32, shuffle=True)
    pool_loader = make_loader(pool_dataset, idx_pool, batch_size=64, shuffle=False)
    determ_train_loader = make_loader(
        pool_dataset, torch.cat((idx_labeled, idx_pool)), batch_size=128, shuffle=False
    )
    val_loader = make_loader(pool_dataset, idx_val, batch_size=64, shuffle=False)
    test_loader = make_loader(test_dataset, idx_test, batch_size=64, shuffle=True)
    loaders = (init_loader, pool_loader, determ_train_loader, val_loader, test_loader)
    idxs = (idx_labeled, idx_pool)
    return loaders, pool_dataset, idxs


def compute_label_stats(
    loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = []
    with torch.no_grad():
        for _, batch_y in loader:
            ys.append(batch_y)
    y_all = torch.cat(ys, dim=0)
    y_mean = y_all.mean(dim=0)
    y_std = y_all.std(dim=0) + 1e-6
    return y_mean.to(device), y_std.to(device)
