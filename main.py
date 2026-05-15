import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import load_oct_data, OCTDataset
from transforms import make_base_transform, make_geo_transform, make_int_transform
from losses import get_weighted_bce, BCEDiceLoss
from train import train_one_experiment
from evaluate import evaluate_model, threshold_sweep
from plots import (plot_single_loss, plot_all_losses, plot_comparison_bar,
                   plot_roc_curves, plot_threshold_analysis, show_qualitative)

SEED        = 42
NUM_EPOCHS  = 150
BATCH_SIZE  = 4
LR          = 1e-4
POS_WEIGHT  = 8.0
RSIZE       = (416, 624)
SMOOTHING   = 0.01
THRESHOLD   = 0.5
NUM_WORKERS = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_DIR    = 'OCT-dataset/images'
MASK_DIR   = 'OCT-dataset/masks'
OUTPUT_DIR = 'results'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    # Carga única en RAM
    all_images, all_masks = load_oct_data(IMG_DIR, MASK_DIR)
    total   = len(all_images)
    n_test  = int(total * 0.2)
    n_train = total - n_test

    # Split único (80/20), hecho UNA sola vez
    generator = torch.Generator().manual_seed(SEED)
    perm      = torch.randperm(total, generator=generator).tolist()
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]
    print(f"Split: {n_train} train / {n_test} test (total {total})\n")

    # Transforms compartidos
    base_t      = make_base_transform(RSIZE)
    geo_t       = make_geo_transform(RSIZE)
    int_no_spk  = make_int_transform(speckle=False)
    int_spk     = make_int_transform(speckle=True)

    # Experimentos
    # (augment, speckle, loss_type, label_smoothing, nombre)
    experiments = [
        (False, False, 'bce',      0.0,       'E1_Baseline'),
        (True,  False, 'bce',      0.0,       'E2_BCE_Aug'),
        (True,  False, 'bce_dice', 0.0,       'E3_BCEDice_Aug'),
        (True,  True,  'bce_dice', 0.0,       'E4_BCEDice_Aug_Speckle'),
        (True,  True,  'bce_dice', SMOOTHING, 'E5_BCEDice_Aug_Speckle_LS'),
    ]

    weighted_bce = get_weighted_bce(POS_WEIGHT, DEVICE)
    bce_dice     = BCEDiceLoss(alpha=0.5)

    all_losses  = {}
    all_results = {}
    all_roc     = {}
    all_sweeps  = {}

    for augment, speckle, loss_type, label_smooth, exp_name in experiments:
        print(f"\n{'='*60}\n  {exp_name}\n{'='*60}")

        # Resetear seed: misma inicialización de pesos y orden de batches
        set_seed(SEED)

        exp_dir = os.path.join(OUTPUT_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Construir datasets a partir de los datos ya en RAM
        int_t    = (int_spk if speckle else int_no_spk) if augment else None
        train_ds = OCTDataset(
            [all_images[i] for i in train_idx],
            [all_masks[i]  for i in train_idx],
            base_transform=base_t,
            geo_transform=geo_t if augment else None,
            int_transform=int_t,
        )
        test_ds = OCTDataset(
            [all_images[i] for i in test_idx],
            [all_masks[i]  for i in test_idx],
            base_transform=base_t,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

        model   = UNet(input_channels=1, n_class=1).to(DEVICE)
        loss_fn = weighted_bce if loss_type == 'bce' else bce_dice

        # Entrenamiento
        losses = train_one_experiment(
            model, train_loader,
            num_epochs=NUM_EPOCHS, lr=LR, device=DEVICE,
            loss_fn=loss_fn, label_smoothing=label_smooth,
        )
        all_losses[exp_name] = losses

        # Persistencia
        torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
        plot_single_loss(losses, title=exp_name,
                         save_path=os.path.join(exp_dir, 'loss_curve.png'))

        # Evaluación
        metrics, fpr, tpr = evaluate_model(model, test_loader, DEVICE, THRESHOLD)
        all_results[exp_name] = metrics
        all_roc[exp_name]     = (fpr, tpr, metrics['auc'])

        # Barrido de umbral
        sweep = threshold_sweep(model, test_loader, DEVICE)
        all_sweeps[exp_name] = sweep
        opt_thr = max(sweep, key=lambda r: r['iou'])['threshold']

        print(f"\n  Métricas Test (thr={THRESHOLD}):")
        for k, v in metrics.items():
            print(f"    {k:<12}: {v:.4f}")
        print(f"  Umbral óptimo (IoU máx): {opt_thr:.2f}")

        # Visualización cualitativa
        show_qualitative(model, test_loader, DEVICE,
                         threshold=THRESHOLD, n_show=3,
                         title_prefix=exp_name, save_dir=exp_dir)

    # ── Comparativa final ─────────────────────────────────────────────────
    print(f"\n{'='*60}\n  RESUMEN FINAL\n{'='*60}")
    cols   = ['accuracy', 'precision', 'recall', 'dice', 'iou', 'auc']
    header = f"{'Experimento':<35}" + "".join(f"{c.upper():>10}" for c in cols)
    print(header)
    print("─" * len(header))
    for name, m in all_results.items():
        print(f"{name:<35}" + "".join(f"{m[c]:>10.4f}" for c in cols))

    plot_all_losses(all_losses,
                    save_path=os.path.join(OUTPUT_DIR, 'all_losses.png'))
    plot_comparison_bar(all_results,
                        save_path=os.path.join(OUTPUT_DIR, 'comparison_bar.png'))
    plot_roc_curves(all_roc,
                    save_path=os.path.join(OUTPUT_DIR, 'roc_curves.png'))
    plot_threshold_analysis(all_sweeps, metric='iou',
                             save_path=os.path.join(OUTPUT_DIR, 'threshold_iou.png'))
    plot_threshold_analysis(all_sweeps, metric='dice',
                             save_path=os.path.join(OUTPUT_DIR, 'threshold_dice.png'))

    print(f"\nResultados en: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
