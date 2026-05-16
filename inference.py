"""
Inference module — run a trained U-Net on a new image directory.

Usage (no masks, pure prediction):
    python inference.py --model results/E3_BCEDice_Aug/model.pth --img_dir /path/to/images

Usage (with masks, full evaluation):
    python inference.py --model results/E3_BCEDice_Aug/model.pth \
        --img_dir /path/to/images --mask_dir /path/to/masks

Optional flags:
    --output       output directory (default: results_inference)
    --threshold    decision threshold (default: 0.5)
    --opt_metric   metric for optimal threshold: dice | iou (default: dice)
"""
import os
import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage

from model import UNet
from transforms import base_img, base_mask
from evaluate import run_evaluation, compute_metrics, find_optimal_threshold

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageOnlyDataset(Dataset):
    """Images from a directory, no masks."""
    def __init__(self, img_dir, img_transform):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(img_dir, ext))
        self.paths = sorted(paths)
        self.transform = img_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(PILImage.open(self.paths[idx]).convert('L'))
        return self.transform(img), self.paths[idx]


class ImageMaskDataset(Dataset):
    """Images + masks from two directories (same filenames)."""
    def __init__(self, img_dir, mask_dir, img_transform, mask_transform):
        import cv2
        exts = ('*.jpg', '*.jpeg', '*.png')
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(img_dir, ext))
        self.img_paths = sorted(paths)
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self._cv2 = cv2

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(PILImage.open(self.img_paths[idx]).convert('L'))
        mask_path = os.path.join(self.mask_dir, os.path.basename(self.img_paths[idx]))
        mask = np.array(PILImage.open(mask_path).convert('L'))
        _, mask = self._cv2.threshold(mask, 100, 255, self._cv2.THRESH_BINARY)
        return self.img_transform(img), self.mask_transform(mask)


def load_model(model_path, device):
    model = UNet(input_channels=1, n_class=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo cargado: {model_path}")
    return model


def _make_overlay(img, pred_bool, gt_bool):
    rgb = np.stack([img, img, img], axis=2)
    rgb[pred_bool & gt_bool] = [0.0, 0.75, 0.0]
    rgb[pred_bool & ~gt_bool] = [0.75, 0.0, 0.0]
    rgb[~pred_bool & gt_bool] = [0.0, 0.0, 0.75]
    return rgb


def _save_pred_masks(probs_t, paths, threshold, output_dir):
    for i, path in enumerate(paths):
        pred = ((probs_t[i].squeeze().numpy() > threshold) * 255).astype(np.uint8)
        name = os.path.splitext(os.path.basename(path))[0]
        PILImage.fromarray(pred).save(os.path.join(output_dir, f'{name}_pred.png'))


def _plot_no_mask(imgs_t, probs_t, threshold, output_dir):
    n = imgs_t.shape[0]
    col_titles = ['Original', 'Probabilidad', f'Pred (thr={threshold:.2f})']
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8)
    for i in range(n):
        img = imgs_t[i].squeeze().numpy()
        prob = probs_t[i].squeeze().numpy()
        pred = (prob > threshold).astype(float)
        for j, (im, cmap) in enumerate([
            (img, 'gray'), (prob, 'hot'), (pred, 'gray')
        ]):
            axes[i, j].imshow(im, cmap=cmap)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'#{i+1}', fontsize=7, rotation=90, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_results.png'), dpi=120, bbox_inches='tight')
    plt.show()


def _plot_with_mask(images, probs, gt, threshold, opt_thr, output_dir):
    n = images.shape[0]
    col_titles = [
        'Original',
        f'Pred (thr={threshold:.2f})',
        f'Pred (opt={opt_thr:.2f})',
        'Ground Truth',
        'Overlay  TP=G FP=R FN=B',
    ]
    patches = [
        mpatches.Patch(color=[0.0, 0.75, 0.0], label='TP'),
        mpatches.Patch(color=[0.75, 0.0, 0.0], label='FP'),
        mpatches.Patch(color=[0.0, 0.0, 0.75], label='FN'),
    ]
    fig, axes = plt.subplots(n, 5, figsize=(15, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8)
    for i in range(n):
        img = images[i].squeeze().numpy()
        prob = probs[i].squeeze().numpy()
        gt_i = gt[i].squeeze().numpy()
        pred_05 = (prob > threshold).astype(bool)
        pred_opt = (prob > opt_thr).astype(bool)
        overlay = _make_overlay(img, pred_opt, gt_i.astype(bool))
        for j, (im, cmap) in enumerate([
            (img, 'gray'),
            (pred_05.astype(float), 'gray'),
            (pred_opt.astype(float), 'gray'),
            (gt_i, 'gray'),
            (overlay, None),
        ]):
            axes[i, j].imshow(im, cmap=cmap, vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'#{i+1}', fontsize=7, rotation=90, va='center')
    fig.legend(handles=patches, loc='lower right', fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_results.png'), dpi=120, bbox_inches='tight')
    plt.show()


def run_inference(model_path, img_dir, output_dir, threshold=0.5,
                  mask_dir=None, opt_metric='dice', device=None):
    if device is None:
        device = DEVICE
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path, device)

    if mask_dir is not None:
        dataset = ImageMaskDataset(img_dir, mask_dir, base_img, base_mask)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        eval_res = run_evaluation(model, loader, device,
                                  threshold=threshold, opt_metric=opt_metric)

        opt_thr = eval_res['opt_threshold']
        print(f"\nMetricas (thr={threshold:.2f}):")
        for k, v in eval_res['metrics_05'].items():
            print(f"  {k:<12}: {v:.4f}")
        print(f"\nMetricas (thr={opt_thr:.3f} optimo por {opt_metric}):")
        for k, v in eval_res['metrics_opt'].items():
            print(f"  {k:<12}: {v:.4f}")

        _plot_with_mask(eval_res['images'], eval_res['probs'], eval_res['gt'],
                        threshold, opt_thr, output_dir)
    else:
        dataset = ImageOnlyDataset(img_dir, base_img)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        all_imgs, all_probs, all_paths = [], [], []
        model.eval()
        with torch.no_grad():
            for imgs, paths in loader:
                all_probs.append(torch.sigmoid(model(imgs.to(device))).cpu())
                all_imgs.append(imgs.cpu())
                all_paths.extend(paths)

        imgs_t = torch.cat(all_imgs)
        probs_t = torch.cat(all_probs)

        _save_pred_masks(probs_t, all_paths, threshold, output_dir)
        _plot_no_mask(imgs_t, probs_t, threshold, output_dir)

    print(f"\nResultados guardados en: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferencia U-Net OCT')
    parser.add_argument('--model', required=True, help='Ruta al .pth del modelo')
    parser.add_argument('--img_dir', required=True, help='Directorio con imagenes')
    parser.add_argument('--output', default='results_inference')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mask_dir', default=None,
                        help='Directorio con mascaras (opcional, activa evaluacion)')
    parser.add_argument('--opt_metric', default='dice', choices=['dice', 'iou'],
                        help='Metrica para umbral optimo')
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        img_dir=args.img_dir,
        output_dir=args.output,
        threshold=args.threshold,
        mask_dir=args.mask_dir,
        opt_metric=args.opt_metric,
    )
