import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_oct_data(img_dir, mask_dir):
    """Carga todas las imágenes y máscaras en RAM como arrays uint8 (H, W)."""
    img_paths  = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in img_paths]

    images, masks = [], []
    for ip, mp in zip(img_paths, mask_paths):
        img  = np.array(Image.open(ip).convert('L'))
        mask = np.array(Image.open(mp).convert('L'))
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        images.append(img)
        masks.append(mask)

    print(f"Dataset cargado: {len(images)} imágenes en RAM.")
    return images, masks


class OCTDataset(Dataset):
    """
    Dataset OCT con imágenes precargadas en RAM.

    geo_transform  — transforms geométricos: se aplican con la misma seed
                     a la imagen Y a la máscara.
    int_transform  — transforms de intensidad: solo a la imagen, nunca a la máscara.
    base_transform — resize + ToTensor: se usa cuando geo_transform es None.
    """

    def __init__(self, images, masks, base_transform,
                 geo_transform=None, int_transform=None):
        self.images         = images
        self.masks          = masks
        self.base_transform = base_transform
        self.geo_transform  = geo_transform
        self.int_transform  = int_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask  = self.masks[idx].copy()

        if self.geo_transform is not None:
            # Misma seed → misma transformación geométrica en imagen y máscara
            seed = np.random.randint(2147483647)
            random.seed(seed);  torch.manual_seed(seed)
            image = self.geo_transform(image)
            random.seed(seed);  torch.manual_seed(seed)
            mask  = self.geo_transform(mask)
        else:
            image = self.base_transform(image)
            mask  = self.base_transform(mask)

        if self.int_transform is not None:
            image = self.int_transform(image)

        return image, mask
