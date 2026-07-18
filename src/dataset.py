import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        # CLAHE removed from eval — keeps eval distribution consistent with the
        # ~70% of training batches that had no CLAHE applied
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


# ── SOS Dataset (binary: oil vs background) ──────────────────────────────────

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


# ── MKLab Dataset (5-class labels, converted to binary for Module 1) ─────────

class MKLabDataset(Dataset):
    """
    MKLab Oil Spill Detection Dataset.

    Labels (1D):
        0 = Sea Surface
        1 = Oil Spill
        2 = Look-alike
        3 = Ship
        4 = Land

    For binary segmentation (Module 1), only class 1 (Oil Spill) is treated as
    the positive class; all other classes become background (0).
    """

    def __init__(self, img_dir, label_dir, transform=None, target_size=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size or config.IMAGE_SIZE
        self.is_train = transform is not None

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))
        ])

        # Build matching label filenames (.jpg images -> .png labels)
        self.labels = []
        for img_name in self.images:
            base = os.path.splitext(img_name)[0]
            label_name = base + ".png"
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(label_path):
                self.labels.append(label_name)
            else:
                raise FileNotFoundError(
                    f"Label file not found for image '{img_name}': {label_path}"
                )

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"MKLabDataset(samples={len(self.images)})"

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Load full resolution — NO resize here anymore
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(
            Image.open(label_path).convert("L").resize(
                (image.shape[1], image.shape[0]), Image.NEAREST
            )
        )

        # Convert to binary mask first
        mask = (label == 1).astype(np.float32)

        # Random or Center crop 256x256 from full resolution image
        h, w = image.shape[:2]
        if h > self.target_size and w > self.target_size:
            if self.is_train:
                oil_pixels = np.argwhere(mask == 1)
                if len(oil_pixels) > 0 and np.random.rand() < 0.7:
                    anchor_y, anchor_x = oil_pixels[np.random.randint(len(oil_pixels))]
                    jitter = self.target_size // 3
                    top  = np.clip(anchor_y - self.target_size // 2 + 
                                    np.random.randint(-jitter, jitter), 
                                    0, h - self.target_size)
                    left = np.clip(anchor_x - self.target_size // 2 + 
                                    np.random.randint(-jitter, jitter), 
                                    0, w - self.target_size)
                else:
                    top  = np.random.randint(0, h - self.target_size)
                    left = np.random.randint(0, w - self.target_size)
            else:
                # Centre crop for val/test — reproducible
                top  = (h - self.target_size) // 2
                left = (w - self.target_size) // 2
            image = image[top:top+self.target_size, left:left+self.target_size]
            mask  = mask[top:top+self.target_size,  left:left+self.target_size]
        else:
            # Fallback: if image smaller than 256, resize as before
            image = np.array(
                Image.fromarray(image).resize(
                    (self.target_size, self.target_size), Image.BILINEAR
                )
            )
            mask = np.array(
                Image.fromarray(mask).resize(
                    (self.target_size, self.target_size), Image.NEAREST
                )
            )

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
            return image, mask.unsqueeze(0)

        image = torch.from_numpy(
            image.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


# ── Loader factory ────────────────────────────────────────────────────────────

def get_loaders(batch_size=config.BATCH_SIZE, dataset=None):
    """
    Create train/val/test DataLoaders.

    Args:
        batch_size: Batch size for loaders.
        dataset:    Override config.DATASET. Options: "sos", "mklab", "combined".

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    ds_choice = dataset or config.DATASET

    if ds_choice == "sos":
        full_dataset = SOSDataset(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            transform=get_transforms(train=True)
        )
        test_ds = SOSDataset(
            config.TEST_IMG_DIR,
            config.TEST_MASK_DIR,
            transform=get_transforms(train=False)
        )

    elif ds_choice == "mklab":
        full_dataset = MKLabDataset(
            config.MKLAB_TRAIN_IMG_DIR,
            config.MKLAB_TRAIN_LABEL_DIR,
            transform=get_transforms(train=True),
        )
        test_ds = MKLabDataset(
            config.MKLAB_TEST_IMG_DIR,
            config.MKLAB_TEST_LABEL_DIR,
            transform=get_transforms(train=False),
        )

    elif ds_choice == "combined":
        sentinel_train = SOSDataset(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            transform=get_transforms(train=True)
        )
        palsar_train = SOSDataset(
            config.PALSAR_TRAIN_IMG_DIR,
            config.PALSAR_TRAIN_MASK_DIR,
            transform=get_transforms(train=True)
        )
        mklab_train = MKLabDataset(
            config.MKLAB_TRAIN_IMG_DIR,
            config.MKLAB_TRAIN_LABEL_DIR,
            transform=get_transforms(train=True),
        )
        full_dataset = ConcatDataset([sentinel_train, palsar_train, mklab_train])

        sentinel_test = SOSDataset(
            config.TEST_IMG_DIR,
            config.TEST_MASK_DIR,
            transform=get_transforms(train=False)
        )
        palsar_test = SOSDataset(
            config.PALSAR_TEST_IMG_DIR,
            config.PALSAR_TEST_MASK_DIR,
            transform=get_transforms(train=False)
        )
        mklab_test = MKLabDataset(
            config.MKLAB_TEST_IMG_DIR,
            config.MKLAB_TEST_LABEL_DIR,
            transform=get_transforms(train=False),
        )
        test_ds = ConcatDataset([sentinel_test, palsar_test, mklab_test])
    else:
        raise ValueError(f"Unknown dataset: '{ds_choice}'. Use 'sos', 'mklab', or 'combined'.")

    # Train/val split
    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # For val set, disable training augmentations if it's a single dataset
    if hasattr(full_dataset, 'transform'):
        val_ds.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )

    print(f"Dataset: {ds_choice.upper()}")
    print(f"  Train: {train_size} | Val: {val_size} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader