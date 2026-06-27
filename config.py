import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset selector ──────────────────────────────────────────────────────────
# Options: "sos", "mklab", "combined"
DATASET = "mklab"

# ── SOS Dataset paths ─────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data", "sos")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "sentinel", "image")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "sentinel", "label")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "sentinel", "image")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "sentinel", "label")

# ── MKLab Dataset paths ──────────────────────────────────────────────────────
MKLAB_DIR = os.path.join(BASE_DIR, "data", "mklabs")
MKLAB_TRAIN_IMG_DIR = os.path.join(MKLAB_DIR, "train", "images")
MKLAB_TRAIN_LABEL_DIR = os.path.join(MKLAB_DIR, "train", "labels_1D")
MKLAB_TEST_IMG_DIR = os.path.join(MKLAB_DIR, "test", "images")
MKLAB_TEST_LABEL_DIR = os.path.join(MKLAB_DIR, "test", "labels_1D")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predictions")

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 2e-4
NUM_WORKERS = 2
PIN_MEMORY = True

TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

NUM_CLASSES = 1