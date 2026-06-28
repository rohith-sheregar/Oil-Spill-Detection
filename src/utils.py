import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(preds, targets, gamma=2.0, alpha=0.25, eps=1e-6):
    """
    Focal Loss — forces model to focus on hard examples (thin streak edges).
    gamma=2 is standard; higher values = more focus on hard cases.
    """
    preds = preds.clamp(eps, 1 - eps)
    bce = F.binary_cross_entropy(preds, targets, reduction='none')
    p_t = preds * targets + (1 - preds) * (1 - targets)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def tversky_loss(preds, targets, alpha=0.3, beta=0.7, eps=1e-6):
    """
    Tversky Loss — penalises False Negatives more than False Positives.
    alpha=0.3, beta=0.7 means missing oil pixels (FN) is penalised
    2.3x more than false alarms (FP). Critical for sparse oil spill pixels.
    """
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky).mean()


def combined_loss(preds, targets):
    """
    Final loss = 0.4 * Focal + 0.6 * Tversky
    Replaces old BCE + Dice.
    Focal handles class imbalance (95% background).
    Tversky ensures we don't miss thin oil streaks.
    """
    return 0.4 * focal_loss(preds, targets) + 0.6 * tversky_loss(preds, targets)


def bce_dice_loss(preds, targets):
    """Keep this name so train.py doesn't break — now calls combined_loss."""
    return combined_loss(preds, targets)


def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    return ((2 * intersection + eps) /
            (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)).mean()


def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    return ((intersection + eps) / (union + eps)).mean()


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    # Handle the fact that older checkpoints used "model_state" instead of "model_state_dict"
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        
    if optimizer:
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
    return checkpoint.get("epoch", 0)