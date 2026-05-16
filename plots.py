import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_single_loss(losses, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, scale in zip(axes, ['linear', 'log']):
        ax.plot(losses, label='Train Loss')
        ax.set_yscale(scale)
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        lbl = 'lineal' if scale == 'linear' else 'logarítmica'
        ax.set_title(f'{title} – escala {lbl}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_single_accuracy(accs, title, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(accs, label='Train Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Pixel Accuracy')
    plt.title(f'{title} – accuracy (lineal)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
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


def plot_all_accuracies(accs_dict, save_path=None):
    plt.figure(figsize=(9, 4))
    for name, accs in accs_dict.items():
        plt.plot(accs, label=name)
    plt.xlabel('Época')
    plt.ylabel('Pixel Accuracy')
    plt.title('Train Accuracy – todos los experimentos')
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_bar(results, save_path=None):
    """Bar chart: IoU y Dice por experimento (threshold=0.5)."""
    names = list(results.keys())
    ious = [results[n]['metrics_05']['iou'] for n in names]
    dices = [results[n]['metrics_05']['dice'] for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - w/2, ious, w, label='IoU', alpha=0.85)
    ax.bar(x + w/2, dices, w, label='Dice', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Comparativa IoU / Dice por experimento (thr=0.5)')
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
    plt.title('Curvas ROC – Test')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_table(results, cols=None, save_path=None):
    """Render metrics for both thr=0.5 and optimal as a matplotlib table."""
    if cols is None:
        cols = ['accuracy', 'precision', 'recall', 'dice', 'iou', 'auc']
    names = list(results.keys())
    rows, row_labels = [], []
    for name in names:
        m05 = results[name]['metrics_05']
        mopt = results[name]['metrics_opt']
        opt_thr = results[name]['opt_threshold']
        rows.append([f"{m05[c]:.4f}" for c in cols])
        rows.append([f"{mopt[c]:.4f}" for c in cols])
        row_labels.append(f"{name}  thr=0.50")
        row_labels.append(f"{name}  thr={opt_thr:.2f}*")

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=[c.upper() for c in cols],
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.6)
    ax.set_title('Tabla Comparativa – Métricas Test  (* umbral óptimo por Dice)',
                 pad=20, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _make_overlay(img, pred_bool, gt_bool):
    """RGB error map: TP=green, FP=red, FN=blue, TN=grey image."""
    rgb = np.stack([img, img, img], axis=2)
    rgb[pred_bool & gt_bool] = [0.0, 0.75, 0.0]
    rgb[pred_bool & ~gt_bool] = [0.75, 0.0, 0.0]
    rgb[~pred_bool & gt_bool] = [0.0, 0.0, 0.75]
    return rgb


def show_qualitative_all(images, probs, gt, threshold=0.5, opt_threshold=0.5,
                          title='', save_path=None):
    """
    Single grid showing all test images.
    Columns: Original | Pred (thr) | Pred (opt) | Ground Truth | Overlay
    """
    n = images.shape[0]
    col_titles = [
        'Original',
        f'Pred (thr={threshold:.2f})',
        f'Pred (opt={opt_threshold:.2f})',
        'Ground Truth',
        'Overlay  TP=G FP=R FN=B',
    ]
    fig, axes = plt.subplots(n, 5, figsize=(15, n * 2.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8, pad=3)

    for i in range(n):
        img = images[i].squeeze().numpy()
        prob = probs[i].squeeze().numpy()
        gt_i = gt[i].squeeze().numpy()
        pred_05 = (prob > threshold)
        pred_opt = (prob > opt_threshold)
        overlay = _make_overlay(img, pred_opt, gt_i.astype(bool))

        for j, (im, cmap) in enumerate([
            (img,                  'gray'),
            (pred_05.astype(float), 'gray'),
            (pred_opt.astype(float), 'gray'),
            (gt_i,                 'gray'),
            (overlay,              None),
        ]):
            axes[i, j].imshow(im, cmap=cmap, vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'Test {i+1}', fontsize=7, rotation=90, va='center')

    patches = [
        mpatches.Patch(color=[0.0, 0.75, 0.0], label='TP'),
        mpatches.Patch(color=[0.75, 0.0, 0.0], label='FP'),
        mpatches.Patch(color=[0.0, 0.0, 0.75], label='FN'),
    ]
    fig.legend(handles=patches, loc='lower right', fontsize=8, ncol=3)
    fig.suptitle(title, fontsize=11, y=1.005)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
