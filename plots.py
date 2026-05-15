import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from evaluate import get_segmentation_masks


def plot_single_loss(losses, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, scale in zip(axes, ['linear', 'log']):
        ax.plot(losses, label='Train Loss')
        ax.set_yscale(scale)
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        ax.set_title(f'{title} – escala {"lineal" if scale == "linear" else "logarítmica"}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_losses(losses_dict, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, scale in zip(axes, ['linear', 'log']):
        for name, losses in losses_dict.items():
            ax.plot(losses, label=name)
        ax.set_yscale(scale)
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        lbl = 'lineal' if scale == 'linear' else 'logarítmica'
        ax.set_title(f'Todos los experimentos – escala {lbl}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_bar(results, save_path=None):
    """Bar chart: IoU y Dice por experimento."""
    names = list(results.keys())
    ious  = [results[n]['iou']  for n in names]
    dices = [results[n]['dice'] for n in names]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - w/2, ious,  w, label='IoU',  alpha=0.85)
    ax.bar(x + w/2, dices, w, label='Dice', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Comparativa IoU / Dice por experimento')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(roc_data, save_path=None):
    """roc_data: {exp_name: (fpr, tpr, auc)}"""
    plt.figure(figsize=(6, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_analysis(sweep_dict, metric='iou', save_path=None):
    """sweep_dict: {exp_name: list_of_records}"""
    plt.figure(figsize=(9, 4))
    for name, records in sweep_dict.items():
        xs = [r['threshold'] for r in records]
        ys = [r[metric]      for r in records]
        plt.plot(xs, ys, marker='o', markersize=3, label=name)
        best_thr = max(records, key=lambda r: r[metric])['threshold']
        plt.axvline(best_thr, linestyle='--', alpha=0.25)
    plt.xlabel('Umbral')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} vs Umbral de decisión')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def show_qualitative(model, loader, device, threshold=0.5,
                     n_show=3, title_prefix='', save_dir=None):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in loader:
            logits     = model(images.to(device))
            pred_masks = get_segmentation_masks(logits, threshold).cpu()
            for i in range(images.size(0)):
                if shown >= n_show:
                    return
                orig = images[i].squeeze().numpy()
                gt   = masks[i].squeeze().numpy()
                pred = pred_masks[i].squeeze().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, im, ttl in zip(axes,
                                       [orig, gt, pred, orig * pred],
                                       ['Original', 'GT', 'Predicción', 'Solapamiento']):
                    ax.imshow(im, cmap='gray')
                    ax.set_title(ttl)
                    ax.axis('off')
                fig.suptitle(f'{title_prefix} – test {shown + 1}')
                plt.tight_layout()
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'qual_{shown+1:02d}.png'),
                                dpi=120, bbox_inches='tight')
                plt.show()
                shown += 1
