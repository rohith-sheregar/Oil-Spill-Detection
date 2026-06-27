"""
Script to train Module 1 (DeepLabV3+ semantic segmentation model) for oil spill detection.
Usage:
  python train_module1.py                  # Uses config.DATASET (default: mklab)
  python train_module1.py --dataset sos    # Train on SOS dataset
  python train_module1.py --dataset mklab  # Train on MKLab dataset
  python train_module1.py --dataset combined  # Train on both datasets
"""
import argparse
import config
from src.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Module 1 segmentation model")
    parser.add_argument(
        "--dataset", type=str, default=config.DATASET,
        choices=["sos", "mklab", "combined"],
        help="Dataset to train on (default: from config.py)"
    )
    args = parser.parse_args()

    # Override the global config with CLI argument
    config.DATASET = args.dataset

    print("=" * 60)
    print("  MODULE 1 -- DeepLabV3+ Segmentation Training")
    print(f"  Dataset: {args.dataset.upper()}")
    print("=" * 60)
    print()
    run_training()
