import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import config
from src.dataset import get_loaders
from src.model import get_model
from src.utils import bce_dice_loss, dice_score, iou_score, save_checkpoint, load_checkpoint

# Gradient scaler for mixed precision
scaler = GradScaler('cuda')

def plot_and_save(train_vals, val_vals, ylabel, title, filename):
    plt.figure(figsize=(8,5))
    plt.plot(train_vals, label=f"Train {ylabel}")
    plt.plot(val_vals, label=f"Validation {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # Save to /kaggle/working if it exists, else to outputs directory
    save_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else config.CHECKPOINT_DIR
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def freeze_batchnorm(model):
    """
    Freeze BatchNorm2d layers — use pretrained running stats, don't 
    re-estimate them from small batches. Critical when batch_size < ~8.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    freeze_batchnorm(model)
    total_loss = 0
    total_dice = 0
    total_iou = 0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast('cuda'):
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
        with torch.no_grad():
            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

    n = len(loader)
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_dice / n, total_iou / n

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            with autocast('cuda'):
                preds = model(images)
                if preds.shape != masks.shape:
                    preds = torch.nn.functional.interpolate(
                        preds, size=masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                loss = bce_dice_loss(preds, masks)
            total_loss += loss.item()
            total_dice += dice_score(preds, masks).item()
            total_iou  += iou_score(preds, masks).item()
    n = len(loader)
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_dice / n, total_iou / n

def get_unwrapped_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

def run_training_with_loaders(train_loader, val_loader, test_loader=None):
    device = config.DEVICE
    print(f"Using device: {device}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else config.CHECKPOINT_DIR

    model = get_model(device)

    # Warm up backbone first — freeze encoder for first 5 epochs
    for param in model.base.encoder.parameters():
        param.requires_grad = False
        
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - 5, eta_min=1e-6
    )

    best_iou = 0
    
    start_epoch = 1
    resume_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(resume_path):
        print(f"Found existing checkpoint at {resume_path} — resuming training state")
        last_epoch = load_checkpoint(resume_path, get_unwrapped_model(model), optimizer, device)
        start_epoch = last_epoch + 1
        print(f"  → Resuming from epoch {start_epoch}")
        
        if start_epoch > 5:
            print("  → Resuming past epoch 5: ensuring encoder is unfrozen")
            unwrapped_model = get_unwrapped_model(model)
            for param in unwrapped_model.base.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE / 5, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS - 5, eta_min=1e-6
            )
            for _ in range(last_epoch - 5):
                scheduler.step()
                
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        # Unfreeze encoder after epoch 5
        if epoch == 6:
            print("  → Unfreezing encoder backbone")
            unwrapped_model = get_unwrapped_model(model)
            for param in unwrapped_model.base.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE / 5, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS - 5, eta_min=1e-6
            )
            torch.cuda.empty_cache()

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Dice: {train_dice:.4f}/{val_dice:.4f} | "
            f"IoU: {train_iou:.4f}/{val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            unwrapped_model = get_unwrapped_model(model)
            save_checkpoint(
                unwrapped_model, optimizer, epoch,
                os.path.join(save_dir, "best_model.pth")
            )
            # also save to outputs/checkpoints for local sync
            if save_dir != config.CHECKPOINT_DIR:
                save_checkpoint(
                    unwrapped_model, optimizer, epoch,
                    os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                )
            print(f"  → Saved best model (IoU: {best_iou:.4f})")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    
    # Save final model
    unwrapped_model = get_unwrapped_model(model)
    torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    
    # Save training history
    history = pd.DataFrame({
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train IoU": train_ious,
        "Validation IoU": val_ious,
        "Train Dice": train_dices,
        "Validation Dice": val_dices,
    })
    history.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    
    # Plot and save graphs
    plot_and_save(train_losses, val_losses, "Loss", "Training vs Validation Loss", "loss_curve.png")
    plot_and_save(train_ious, val_ious, "IoU", "IoU Curve", "iou_curve.png")
    plot_and_save(train_dices, val_dices, "Dice Score", "Dice Score Curve", "dice_curve.png")
    
    return best_iou

def run_training():
    train_loader, val_loader, test_loader = get_loaders()
    return run_training_with_loaders(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    run_training()