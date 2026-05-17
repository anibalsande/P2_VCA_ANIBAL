import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_oct_data(img_dir, mask_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in img_paths]

    images, masks = [], []
    for ip, mp in zip(img_paths, mask_paths):
        img = np.array(Image.open(ip).convert('L'))
        mask = np.array(Image.open(mp).convert('L'))
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        images.append(img)
        masks.append(mask)

    print(f"Dataset cargado: {len(images)} imágenes en RAM.")
    return images, masks


def make_split(n_total, test_ratio=0.2, seed=42):
    """Reproducible train/test split. Returns (train_idx, test_idx)."""
    n_test = int(n_total * test_ratio)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).tolist()
    return perm[:n_total - n_test], perm[n_total - n_test:]


class OCTDataset(Dataset):
    def __init__(self, images, masks, img_transform, mask_transform,
                 int_transform=None, augment=False):
        self.images = images
        self.masks = masks
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.int_transform = int_transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        if self.augment:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.img_transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            image = self.img_transform(image)
            mask = self.mask_transform(mask)

        if self.int_transform is not None:
            image = self.int_transform(image)

        return image, mask
