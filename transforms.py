import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

TARGET_H = 416
TARGET_W = 640


class ResizePad:
    """Resize preserving aspect ratio, then pad with zeros to (TARGET_H, TARGET_W)."""
    def __init__(self, target_h, target_w, interpolation=Image.BILINEAR):
        self.target_h = target_h
        self.target_w = target_w
        self.interp = interpolation

    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), self.interp)
        pad_l = (self.target_w - new_w) // 2
        pad_t = (self.target_h - new_h) // 2
        pad_r = self.target_w - new_w - pad_l
        pad_b = self.target_h - new_h - pad_t
        return TF.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)


class GammaCorrection:
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.lo, self.hi = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.pow(random.uniform(self.lo, self.hi))
        return img


class SpeckleNoise:
    """Multiplicative OCT speckle: I' = clamp(I + I·N(0,σ), 0, 1)."""
    def __init__(self, intensity_range=(0.02, 0.10), p=0.5):
        self.lo, self.hi = intensity_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.lo, self.hi)
            noise = torch.randn_like(img) * sigma
            return torch.clamp(img + img * noise, 0.0, 1.0)
        return img


# ── Base transforms (test / no augmentation) ─────────────────────────────────
base_img = T.Compose([
    T.ToPILImage(),
    ResizePad(TARGET_H, TARGET_W, Image.BILINEAR),
    T.ToTensor(),
])

base_mask = T.Compose([
    T.ToPILImage(),
    ResizePad(TARGET_H, TARGET_W, Image.NEAREST),
    T.ToTensor(),
])

# ── Geometric augmentation (train) ───────────────────────────────────────────
# aug_img and aug_mask must be called with the same seed to sync transforms.
aug_img = T.Compose([
    T.ToPILImage(),
    ResizePad(TARGET_H, TARGET_W, Image.BILINEAR),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=12),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
    T.ToTensor(),
])

aug_mask = T.Compose([
    T.ToPILImage(),
    ResizePad(TARGET_H, TARGET_W, Image.NEAREST),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=12, interpolation=InterpolationMode.NEAREST),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08),
                   interpolation=InterpolationMode.NEAREST),
    T.ToTensor(),
])

# ── Intensity augmentation (image only, never mask) ───────────────────────────
int_transform = T.Compose([
    T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.6),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
    GammaCorrection(gamma_range=(0.8, 1.2), p=0.5),
    T.Lambda(lambda img: torch.clamp(img, 0.0, 1.0)),
])

int_transform_speckle = T.Compose([
    T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.6),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
    GammaCorrection(gamma_range=(0.8, 1.2), p=0.5),
    SpeckleNoise(intensity_range=(0.02, 0.10), p=0.5),
    T.Lambda(lambda img: torch.clamp(img, 0.0, 1.0)),
])
