import random
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


def make_base_transform(rsize):
    """Resize + ToTensor sin augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(rsize, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def make_geo_transform(rsize):
    """
    Transforms geométricos: aplicar con la misma seed a imagen Y máscara.
    NEAREST para que no aparezcan valores intermedios en la máscara binaria.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(rsize, interpolation=InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.9, 1.1), shear=(-3, 3)),
        transforms.ToTensor(),
    ])


def add_speckle_noise(img_tensor, intensity=0.1):
    """Ruido multiplicativo característico de OCT: I' = clamp(I + I·N(0,σ), 0, 1)."""
    noise = torch.randn_like(img_tensor) * intensity
    return torch.clamp(img_tensor + img_tensor * noise, 0.0, 1.0)


def make_int_transform(speckle=False):
    """
    Transforms de intensidad: solo para la imagen, nunca para la máscara.
    speckle=True añade ruido multiplicativo (E4, E5).
    """
    ops = [
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.6),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.Lambda(
            lambda img: img.pow(random.uniform(0.8, 1.2))
            if random.random() < 0.5 else img),
    ]
    if speckle:
        ops.append(transforms.Lambda(
            lambda img: add_speckle_noise(img, intensity=random.uniform(0.02, 0.10))
            if random.random() < 0.5 else img))
    ops.append(transforms.Lambda(lambda img: torch.clamp(img, 0.0, 1.0)))
    return transforms.Compose(ops)
