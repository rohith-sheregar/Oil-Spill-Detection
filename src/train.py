import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import config
from src.dataset import get_loaders
from src.model import get_model
from src.utils import bce_dice_loss, dice_score, iou_score, save_checkpoint

# Gradient scaler for mixed precision
scaler = GradScaler()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            preds = model(images)
            if preds.shape != masks.shape:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            loss = bce_dice_loss(preds, masks)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_dice, total_iou = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            with autocast():
                preds = model(images)
                if preds.shape != masks.shape:
                    preds = torch.nn.functional.interpolate(
                        preds, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
            total_dice += dice_score(preds, masks).item()
            total_iou  += iou_score(preds, masks).item()
    n = len(loader)
    if n == 0:
        return 0.0, 0.0
    return total_dice / n, total_iou / n


def run_training():
    device = config.DEVICE
    print(f"Using device: {device}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader, _ = get_loaders()
    model = get_model(device)

    # Warm up backbone first — freeze encoder for first 5 epochs
    for param in model.base.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_iou = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):

        # Unfreeze encoder after epoch 5
        if epoch == 6:
            print("  → Unfreezing encoder backbone")
            for param in model.base.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE / 5,  # lower LR for fine-tuning encoder
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice, val_iou = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            )
            print(f"  → Saved best model (IoU: {best_iou:.4f})")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")


def run_training_with_loaders(train_loader, val_loader, test_loader=None):
    device = config.DEVICE
    print(f"Using device: {device}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    model = get_model(device)

    for param in model.base.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_iou = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        if epoch == 6:
            print("  → Unfreezing encoder backbone")
            for param in model.base.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE / 5, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice, val_iou = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            )
            print(f"  → Saved best model (IoU: {best_iou:.4f})")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

if __name__ == "__main__":
    run_training()