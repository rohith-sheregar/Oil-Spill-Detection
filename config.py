import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "sos")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "sentinel", "image")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "sentinel", "label")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "sentinel", "image")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "sentinel", "label")

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