"""
Auxiliar de inspección: carga pesos guardados y produce
  1) tabla de métricas para un barrido de umbrales
  2) visualización cualitativa a un umbral concreto

Uso rápido — edita el bloque CONFIG y ejecuta el script.
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Lista de (etiqueta, ruta_pesos).  Añade tantas entradas como quieras comparar.
MODELS = [
    ("E1_Baseline",   "results/E1_Baseline/model.pth"),
    ("E2_BCE_Aug",    "results/E2_BCE_Aug/model.pth"),
]

IMG_DIR  = "OCT-dataset/images"
MASK_DIR = "OCT-dataset/masks"

RSIZE      = (416, 624)
SEED       = 42
BATCH_SIZE = 4

# Umbral que se usa en las visualizaciones cualitativas
VIS_THRESHOLD = 0.5

# Umbrales que se muestran en la tabla de métricas
#   None → barrido completo linspace(0.05, 0.95, 37)
METRIC_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Cuántos ejemplos del test set mostrar por modelo
N_EXAMPLES = 3

# Directorio de salida para los PNGs (None → solo plt.show, sin guardar)
SAVE_DIR = None
# ─────────────────────────────────────────────────────────────────────────────


# ── Imports locales (misma carpeta) ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model   import UNet
from dataset import load_oct_data, OCTDataset
from transforms import make_base_transform
from evaluate   import compute_metrics, threshold_sweep


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_test_loader(img_dir, mask_dir, rsize, seed, batch_size):
    """Reconstruye el mismo test split que main.py (80/20, seed fija)."""
    all_images, all_masks = load_oct_data(img_dir, mask_dir)
    total  = len(all_images)
    n_test = int(total * 0.2)

    generator = torch.Generator().manual_seed(seed)
    perm      = torch.randperm(total, generator=generator).tolist()
    test_idx  = perm[total - n_test:]

    base_t = make_base_transform(rsize)
    test_ds = OCTDataset(
        [all_images[i] for i in test_idx],
        [all_masks[i]  for i in test_idx],
        base_transform=base_t,
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    print(f"  Test split: {len(test_ds)} imágenes")
    return loader


def load_model(weights_path):
    model = UNet(input_channels=1, n_class=1).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Métricas por umbral ───────────────────────────────────────────────────────

def print_threshold_table(label, records, highlight_thrs):
    """Imprime tabla de métricas. Si highlight_thrs es lista, filtra esas filas."""
    if highlight_thrs is not None:
        records = [r for r in records if any(abs(r['threshold'] - t) < 1e-6 for t in highlight_thrs)]

    header = f"  {'thr':>6}  {'acc':>7}  {'prec':>7}  {'rec':>7}  {'dice':>7}  {'iou':>7}"
    sep    = "  " + "─" * (len(header) - 2)
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(sep)
    print(header)
    print(sep)
    for r in records:
        print(f"  {r['threshold']:>6.2f}  "
              f"{r['accuracy']:>7.4f}  "
              f"{r['precision']:>7.4f}  "
              f"{r['recall']:>7.4f}  "
              f"{r['dice']:>7.4f}  "
              f"{r['iou']:>7.4f}")
    best_dice = max(records, key=lambda r: r['dice'])
    best_iou  = max(records, key=lambda r: r['iou'])
    print(sep)
    print(f"  Mejor Dice: {best_dice['threshold']:.2f}  ({best_dice['dice']:.4f})")
    print(f"  Mejor IoU : {best_iou['threshold']:.2f}  ({best_iou['iou']:.4f})")


def plot_threshold_curves(sweep_records_dict, save_path=None):
    """Dice e IoU vs umbral para todos los modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    metrics = ['dice', 'iou']
    for ax, metric in zip(axes, metrics):
        for label, records in sweep_records_dict.items():
            xs = [r['threshold'] for r in records]
            ys = [r[metric]      for r in records]
            ax.plot(xs, ys, marker='o', markersize=3, label=label)
        ax.set_xlabel('Umbral')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Umbral')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── Visualización cualitativa ─────────────────────────────────────────────────

def show_examples(label, model, loader, threshold, n_show, save_dir):
    shown = 0
    with torch.no_grad():
        for images, masks in loader:
            logits     = model(images.to(DEVICE))
            probs      = torch.sigmoid(logits).cpu()
            pred_masks = (probs > threshold).float()

            for i in range(images.size(0)):
                if shown >= n_show:
                    return
                orig = images[i].squeeze().numpy()
                gt   = masks[i].squeeze().numpy()
                pred = pred_masks[i].squeeze().numpy()
                prob = probs[i].squeeze().numpy()

                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                data = [
                    (orig,          'Original',       'gray'),
                    (gt,            'GT',             'gray'),
                    (prob,          f'Prob map',      'hot'),
                    (pred,          f'Pred (thr={threshold})', 'gray'),
                    (orig * pred,   'Solapamiento',   'gray'),
                ]
                for ax, (im, ttl, cmap) in zip(axes, data):
                    ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
                    ax.set_title(ttl, fontsize=9)
                    ax.axis('off')

                # Métricas individuales en el título
                gt_t   = torch.tensor(gt)
                pred_t = torch.tensor(pred)
                m = compute_metrics(pred_t, gt_t)
                fig.suptitle(
                    f"{label}  —  test {shown+1}   "
                    f"Dice={m['dice']:.3f}  IoU={m['iou']:.3f}  "
                    f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}",
                    fontsize=10,
                )
                plt.tight_layout()
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    fname = os.path.join(save_dir, f"{label}_thr{threshold}_ex{shown+1:02d}.png")
                    plt.savefig(fname, dpi=120, bbox_inches='tight')
                plt.show()
                shown += 1


# ── Main ──────────────────────────────────────────────────────────────────────
# --- MODIFICACIÓN EN inspect_model.py ---

def print_dual_metrics(label, records):
    """
    Imprime las métricas comparando el umbral estándar (0.5) 
    frente al umbral que optimiza el IoU.
    """
    # 1. Buscar métricas a 0.5
    r_05 = next((r for r in records if abs(r['threshold'] - 0.5) < 1e-6), None)
    
    # 2. Buscar el mejor umbral para IoU
    best_iou_rec = max(records, key=lambda r: r['iou'])
    best_thr = best_iou_rec['threshold']

    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE RENDIMIENTO: {label}")
    print(f"{'='*60}")
    
    header = f"  {'Configuración':<20} | {'thr':>5} | {'Dice':>7} | {'IoU':>7} | {'Prec':>7} | {'Rec':>7}"
    sep = "  " + "─" * (len(header) - 2)
    
    print(header)
    print(sep)
    
    # Mostrar fila de 0.5
    if r_05:
        print(f"  {'Umbral Estándar':<20} | {0.5:>5.2f} | {r_05['dice']:>7.4f} | {r_05['iou']:>7.4f} | {r_05['precision']:>7.4f} | {r_05['recall']:>7.4f}")
    
    # Mostrar fila del Mejor IoU
    print(f"  {'Mejor IoU (Óptimo)':<20} | {best_thr:>5.2f} | {best_iou_rec['dice']:>7.4f} | {best_iou_rec['iou']:>7.4f} | {best_iou_rec['precision']:>7.4f} | {best_iou_rec['recall']:>7.4f}")
    print(sep)
    print(f"  Nota: El umbral óptimo mejora el IoU en un {((best_iou_rec['iou'] - r_05['iou'])/r_05['iou']*100):.2f}% respecto al estándar.")


def run(models=None, vis_threshold=None, metric_thresholds=None,
        n_examples=None, save_dir=None):
    
    models_list = models or MODELS
    vis_thr = vis_threshold if vis_threshold is not None else VIS_THRESHOLD
    n_ex = n_examples if n_examples is not None else N_EXAMPLES
    out_dir = save_dir if save_dir is not None else SAVE_DIR

    print(f"Device: {DEVICE}")
    set_seed(SEED)
    loader = build_test_loader(IMG_DIR, MASK_DIR, RSIZE, SEED, BATCH_SIZE)

    sweep_dict = {}
    for label, weights_path in models_list:
        if not os.path.isfile(weights_path):
            continue

        model = load_model(weights_path)
        
        # Ejecutamos el barrido completo para encontrar el óptimo
        sweep = threshold_sweep(model, loader, DEVICE)
        sweep_dict[label] = sweep

        # NUEVA FUNCIÓN: Imprime la comparativa 0.5 vs Óptimo
        print_dual_metrics(label, sweep)
        
        # Visualización cualitativa (puedes elegir usar el óptimo aquí si quieres)
        best_iou_thr = max(sweep, key=lambda r: r['iou'])['threshold']
        
        print(f"\n  Generando visualizaciones con thr={vis_thr} (Estándar)...")
        show_examples(label, model, loader, vis_thr, n_ex, out_dir)

if __name__ == "__main__":
    run()
