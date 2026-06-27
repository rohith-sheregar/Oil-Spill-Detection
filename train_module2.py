"""
Script to train Module 2 (Random Forest look-alike classifier).

Uses the MKLab dataset by default, which has real oil_spill and look_alike labels.
Pass --no-mklab to fall back to synthetic negatives from the SOS dataset.

Usage:
  python train_module2.py                # MKLab real labels (recommended)
  python train_module2.py --no-mklab     # SOS synthetic negatives (legacy)
"""
import argparse
import config
from src.train_module2 import train_classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Module 2 look-alike classifier")
    parser.add_argument(
        "--no-mklab", action="store_true",
        help="Disable MKLab data and use SOS synthetic negatives instead"
    )
    args = parser.parse_args()

    use_mklab = not args.no_mklab

    print("=" * 60)
    print("  MODULE 2 -- Look-alike Classifier Training")
    print(f"  Data source: {'MKLab (real labels)' if use_mklab else 'SOS (synthetic)'}")
    print("=" * 60)
    print()

    trained_clf = train_classifier(
        oil_img_dir=config.TRAIN_IMG_DIR,
        oil_mask_dir=config.TRAIN_MASK_DIR,
        use_mklab=use_mklab,
    )

    print("\nModule 2 classifier training complete")
    print()
    print(trained_clf.get_feature_importance_report())
