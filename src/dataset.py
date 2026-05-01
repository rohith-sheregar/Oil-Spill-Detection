import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.5, 0.5), std=(0.5, 0.5)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5), std=(0.5, 0.5)),
        ToTensorV2(),
    ])

class SOSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(('.png', '.jpg', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"SOSDataset(samples={len(self.images)})"

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)

def get_loaders(batch_size=config.BATCH_SIZE):
    full_dataset = SOSDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        transform=get_transforms(train=True)
    )
    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = get_transforms(train=False)

    test_ds = SOSDataset(
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    return train_loader, val_loader, test_loader