import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def compute_metrics(pred_flat, true_flat, eps=1e-7):
    pred = pred_flat.float()
    true = true_flat.float()
    TP = (pred * true).sum()
    FP = (pred * (1 - true)).sum()
    FN = ((1 - pred) * true).sum()
    TN = ((1 - pred) * (1 - true)).sum()
    accuracy = (TP + TN) / (TP + FP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    return {
        'accuracy':  accuracy.item(),
        'precision': precision.item(),
        'recall':    recall.item(),
        'dice':      dice.item(),
        'iou':       iou.item(),
    }


def _collect(model, loader, device):
    """Single forward pass over the loader. Returns (images, probs, gt) as CPU tensors."""
    model.eval()
    imgs_l, probs_l, gt_l = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            probs_l.append(torch.sigmoid(logits).cpu())
            gt_l.append((masks > 0.5).float())
            imgs_l.append(images.cpu())
    return torch.cat(imgs_l), torch.cat(probs_l), torch.cat(gt_l)


def find_optimal_threshold(probs_flat, gt_flat, metric='dice', n=37):
    """Grid search over n thresholds; returns the one maximising `metric`."""
    best_thr, best_val = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, n):
        m = compute_metrics((probs_flat > thr).float(), gt_flat)
        if m[metric] > best_val:
            best_val, best_thr = m[metric], float(thr)
    return best_thr


def run_evaluation(model, loader, device, threshold=0.5, opt_metric='dice'):
    images, probs, gt = _collect(model, loader, device)
    probs_flat = probs.flatten()
    gt_flat = gt.flatten()

    metrics_05 = compute_metrics((probs_flat > threshold).float(), gt_flat)
    auc = float(roc_auc_score(gt_flat.numpy(), probs_flat.numpy()))
    metrics_05['auc'] = auc
    fpr, tpr, _ = roc_curve(gt_flat.numpy(), probs_flat.numpy())

    opt_thr = find_optimal_threshold(probs_flat, gt_flat, metric=opt_metric)
    metrics_opt = compute_metrics((probs_flat > opt_thr).float(), gt_flat)
    metrics_opt['auc'] = auc

    return {
        'images':        images,
        'probs':         probs,
        'gt':            gt,
        'metrics_05':    metrics_05,
        'metrics_opt':   metrics_opt,
        'opt_threshold': opt_thr,
        'fpr':           fpr,
        'tpr':           tpr,
        'auc':           auc,
    }
