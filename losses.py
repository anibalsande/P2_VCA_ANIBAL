import torch
import torch.nn as nn


def get_weighted_bce(pos_weight, device):
    """BCE con pos_weight para compensar el desequilibrio de clases (E1, E2)."""
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))


def dice_loss(logits, target, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum()
    return 1.0 - (2.0 * inter + eps) / (probs.sum() + target.sum() + eps)


class BCEDiceLoss:
    """
    BCE plano (sin pos_weight) + Dice en proporciones iguales (E3, E4, E5).
    Dice ya maneja el desequilibrio de clases por normalización interna.
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._bce  = nn.BCEWithLogitsLoss()

    def __call__(self, logits, target):
        return (self.alpha * self._bce(logits, target)
                + (1.0 - self.alpha) * dice_loss(logits, target))
