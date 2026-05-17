import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet
from dataset import load_oct_data, make_split, OCTDataset
from transforms import (base_img, base_mask, aug_img, aug_mask,
                        int_transform, int_transform_speckle)
from losses import get_weighted_bce, BCEDiceLoss
from train import train_experiment
from evaluate import run_evaluation
from plots import (plot_single_loss, plot_all_losses,
                   plot_comparison_bar, plot_roc_curves, show_qualitative_all)

SEED = 42
NUM_EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-4
POS_WEIGHT = 8.0
SMOOTHING = 0.01
THRESHOLD = 0.5
NUM_WORKERS = 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_DIR = 'OCT-dataset/images'
MASK_DIR = 'OCT-dataset/masks'
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

    # Load once into RAM
    all_images, all_masks = load_oct_data(IMG_DIR, MASK_DIR)

    # Single split done once before any experiment
    train_idx, test_idx = make_split(len(all_images), test_ratio=0.2, seed=SEED)
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test\n")

    train_images = [all_images[i] for i in train_idx]
    train_masks = [all_masks[i] for i in train_idx]
    test_images = [all_images[i] for i in test_idx]
    test_masks = [all_masks[i] for i in test_idx]

    weighted_bce = get_weighted_bce(POS_WEIGHT, device=DEVICE)
    bce_dice = BCEDiceLoss(alpha=0.5, pos_weight=POS_WEIGHT, device=DEVICE)

    # (augment, speckle, loss_type, label_smoothing, name)
    experiments = [
        (False, False, 'bce',      0.0,       'E1_Baseline'),
        (True,  False, 'bce',      0.0,       'E2_BCE_Aug'),
        (True,  False, 'bce_dice', 0.0,       'E3_BCEDice_Aug'),
        (True,  True,  'bce_dice', 0.0,       'E4_BCEDice_Aug_Speckle'),
        (True,  True,  'bce_dice', SMOOTHING, 'E5_BCEDice_Aug_Speckle_LS'),
    ]

    all_losses = {}
    all_results = {}

    for augment, speckle, loss_type, label_smooth, exp_name in experiments:
        print(f"\n{'-'*60}\n  {exp_name}\n{'-'*60}")
        set_seed(SEED)

        exp_dir = os.path.join(OUTPUT_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        int_t = (int_transform_speckle if speckle else int_transform) if augment else None
        train_ds = OCTDataset(
            train_images, train_masks,
            img_transform=aug_img if augment else base_img,
            mask_transform=aug_mask if augment else base_mask,
            int_transform=int_t,
            augment=augment,
        )
        test_ds = OCTDataset(
            test_images, test_masks,
            img_transform=base_img,
            mask_transform=base_mask,
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        model = UNet(input_channels=1, n_class=1)
        loss_fn = weighted_bce if loss_type == 'bce' else bce_dice

        losses, accuracies = train_experiment(
            model, train_loader,
            num_epochs=NUM_EPOCHS, lr=LR, device=DEVICE,
            loss_fn=loss_fn, label_smoothing=label_smooth,
        )
        all_losses[exp_name] = losses

        torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))

        plot_single_loss(losses, title=exp_name,
                         save_path=os.path.join(exp_dir, 'loss_curve.png'))

        eval_res = run_evaluation(model, test_loader, DEVICE,
                                  threshold=THRESHOLD, opt_metric='dice')
        all_results[exp_name] = eval_res

        opt_thr = eval_res['opt_threshold']
        print(f"\n  Metricas Test (thr={THRESHOLD}):")
        for k, v in eval_res['metrics_05'].items():
            print(f"    {k:<12}: {v:.4f}")
        print(f"\n  Metricas Test (thr={opt_thr:.3f} optimo por Dice):")
        for k, v in eval_res['metrics_opt'].items():
            print(f"    {k:<12}: {v:.4f}")

        plot_roc_curves(
            {exp_name: (eval_res['fpr'], eval_res['tpr'], eval_res['auc'])},
            save_path=os.path.join(exp_dir, 'roc_curve.png'),
        )

        show_qualitative_all(
            eval_res['images'], eval_res['probs'], eval_res['gt'],
            threshold=THRESHOLD, opt_threshold=opt_thr,
            title=exp_name,
            save_path=os.path.join(exp_dir, 'qualitative.png'),
        )

    # ── Final summary ───────────────────────────────────────────────────────
    cols = ['accuracy', 'precision', 'recall', 'dice', 'iou', 'auc']
    header = f"{'Experimento':<38}" + "".join(f"{c.upper():>10}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{sep}\n  RESUMEN FINAL\n{sep}")
    print(header)
    print(sep)
    for name, res in all_results.items():
        m05 = res['metrics_05']
        mopt = res['metrics_opt']
        opt_thr = res['opt_threshold']
        print(f"{(name + '  thr=0.50'):<38}" + "".join(f"{m05[c]:>10.4f}" for c in cols))
        print(f"{'  → opt thr=' + f'{opt_thr:.2f}':<38}" + "".join(f"{mopt[c]:>10.4f}" for c in cols))
        print()

    plot_all_losses(all_losses,
                    save_path=os.path.join(OUTPUT_DIR, 'all_losses.png'))
    plot_comparison_bar(all_results,
                        save_path=os.path.join(OUTPUT_DIR, 'comparison_bar.png'))
    print(f"Resultados en: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
