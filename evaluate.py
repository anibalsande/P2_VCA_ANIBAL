import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def get_segmentation_masks(outputs, threshold=0.5):
    return (torch.sigmoid(outputs) > threshold).float()


def compute_metrics(pred_masks, true_masks, eps=1e-7):
    pred = pred_masks.view(-1).float()
    true = true_masks.view(-1).float()

    TP = (pred * true).sum()
    FP = (pred * (1 - true)).sum()
    FN = ((1 - pred) * true).sum()
    TN = ((1 - pred) * (1 - true)).sum()

    accuracy  = (TP + TN) / (TP + FP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    dice      = 2 * precision * recall / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)

    return {
        'accuracy':  accuracy.item(),
        'precision': precision.item(),
        'recall':    recall.item(),
        'dice':      dice.item(),
        'iou':       iou.item(),
    }


def evaluate_model(model, loader, device, threshold=0.5):
    """
    Devuelve (metrics_dict, fpr_array, tpr_array).
    metrics_dict incluye accuracy, precision, recall, dice, iou, auc.
    """
    model.eval()
    acc_m = {k: [] for k in ('accuracy', 'precision', 'recall', 'dice', 'iou')}
    all_probs, all_gt = [], []

    with torch.no_grad():
        for images, masks in loader:
            logits     = model(images.to(device))
            pred_masks = get_segmentation_masks(logits, threshold)
            gt_binary  = (masks.to(device) > 0.5).float()

            for k, v in compute_metrics(pred_masks, gt_binary).items():
                acc_m[k].append(v)

            all_probs.append(torch.sigmoid(logits).cpu())
            all_gt.append(gt_binary.cpu())

    probs_flat = torch.cat(all_probs).numpy().flatten()
    gt_flat    = torch.cat(all_gt).numpy().flatten()

    metrics = {k: float(np.mean(v)) for k, v in acc_m.items()}
    metrics['auc'] = float(roc_auc_score(gt_flat, probs_flat))

    fpr, tpr, _ = roc_curve(gt_flat, probs_flat)
    return metrics, fpr, tpr


def threshold_sweep(model, loader, device, thresholds=None):
    """Devuelve lista de dicts {threshold, accuracy, precision, recall, dice, iou}."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    model.eval()
    probs_all, gt_all = [], []

    with torch.no_grad():
        for images, masks in loader:
            probs_all.append(torch.sigmoid(model(images.to(device))).cpu())
            gt_all.append(masks)

    probs_all = torch.cat(probs_all)
    gt        = (torch.cat(gt_all) > 0.5).float()

    records = []
    for thr in thresholds:
        m = compute_metrics((probs_all > thr).float(), gt)
        records.append({'threshold': float(thr), **m})
    return records
