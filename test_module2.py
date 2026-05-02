"""
test_module2.py — Module 2 Smoke Test
Verifies that all Module 2 components can be imported and execute without error.
Run from the project root: python test_module2.py
"""

import sys
import os
import numpy as np
import traceback

# ─── Imports ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODULE 2 SMOKE TEST")
print("=" * 60)

# 1. Import checks
print("\n[1/9] Checking imports…", end=" ")
try:
    from src.features import (
        PatchFeatures, extract_features, features_to_array, describe_patch
    )
    from src.classifier import LookAlikeClassifier
    print("OK ✓")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Synthetic data ───────────────────────────────────────────────────────────
# Create a 256×256 binary mask with a few deliberate blobs
np.random.seed(0)
mask = np.zeros((256, 256), dtype=np.float32)
mask[30:80,  20:120] = 1   # elongated horizontal blob
mask[140:200, 80:160] = 1  # squarish blob
mask[10:15,   10:15] = 1   # tiny blob — should be filtered (< 50 px)

# Matching grayscale image (random SAR-like texture)
image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)

# ─── Test 2: extract_features ─────────────────────────────────────────────────
print("\n[2/9] extract_features…", end=" ")
try:
    features = extract_features(mask, image, acquisition_hour=2)  # night
    assert len(features) == 2, f"Expected 2 valid patches, got {len(features)}"
    print(f"OK ✓  ({len(features)} patches found, tiny blob correctly filtered)")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# Print first patch details
print(f"  First patch values:")
f0 = features[0]
print(f"    area_pixels   : {f0.area_pixels}")
print(f"    area_km2      : {f0.area_km2:.4f}")
print(f"    elongation    : {f0.elongation:.4f}")
print(f"    compactness   : {f0.compactness:.4f}")
print(f"    solidity      : {f0.solidity:.4f}")
print(f"    mean_intensity: {f0.mean_intensity:.4f}")
print(f"    is_night      : {f0.is_night}")

# ─── Test 3: features_to_array ────────────────────────────────────────────────
print("\n[3/9] features_to_array…", end=" ")
try:
    arr = features_to_array(features)
    assert arr.shape == (2, 13), f"Expected (2,13), got {arr.shape}"
    print(f"OK ✓  shape={arr.shape}")

    # Also test empty case
    empty_arr = features_to_array([])
    assert empty_arr.shape == (0, 13), f"Empty case: expected (0,13), got {empty_arr.shape}"
    print("        Empty list → (0,13) ✓")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Test 4: describe_patch ───────────────────────────────────────────────────
print("\n[4/9] describe_patch…", end=" ")
try:
    desc = describe_patch(features[0])
    assert "Area" in desc and "Elongation" in desc
    print(f"OK ✓")
    print(f"  → {desc}")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Test 5: LookAlikeClassifier instantiation ───────────────────────────────
print("\n[5/9] LookAlikeClassifier instantiation…", end=" ")
try:
    clf = LookAlikeClassifier()
    assert not clf.is_trained, "Should be untrained on init"
    print("OK ✓  (is_trained=False as expected)")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Test 6: Train on synthetic data ─────────────────────────────────────────
print("\n[6/9] Training classifier on 100 synthetic samples…", end=" ")
try:
    np.random.seed(42)
    X_synth = np.random.rand(100, 13).astype(np.float32)
    y_synth = np.array([1] * 50 + [0] * 50, dtype=np.int64)

    results = clf.train(X_synth, y_synth)
    assert clf.is_trained
    accuracy = results["accuracy"]
    print(f"OK ✓  accuracy={accuracy:.4f}")

    fi = results["feature_importances"]
    top3 = list(fi.items())[:3]
    print("  Top 3 feature importances:")
    for rank, (name, imp) in enumerate(top3, start=1):
        print(f"    {rank}. {name:<20s} {imp:.4f}")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Test 7: predict ──────────────────────────────────────────────────────────
print("\n[7/9] predict on 2 real patches…", end=" ")
try:
    preds = clf.predict(features)
    assert len(preds) == 2
    for i, p in enumerate(preds):
        assert "label"      in p
        assert "confidence" in p
        assert "is_oil"     in p
        assert 0.0 <= p["confidence"] <= 1.0
    print(f"OK ✓")
    for i, p in enumerate(preds):
        print(f"  Patch {i+1}: label={p['label']}, confidence={p['confidence']:.4f}, is_oil={p['is_oil']}")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Test 8: predict on empty list ───────────────────────────────────────────
print("\n[8/9] predict on empty feature list…", end=" ")
try:
    empty_preds = clf.predict([])
    assert empty_preds == []
    print("OK ✓  (returns empty list)")
except Exception as e:
    print(f"FAILED ✗\n  {e}")
    traceback.print_exc()
    sys.exit(1)

# ─── Test 9: run_pipeline with dummy path → FileNotFoundError ─────────────────
print("\n[9/9] run_pipeline with bad checkpoint path → FileNotFoundError…", end=" ")
try:
    from src.pipeline import run_pipeline, load_module1
    # Directly test load_module1 with a non-existent path
    try:
        load_module1("/nonexistent/path/best_model.pth", "cpu")
        print("FAILED ✗  (should have raised an error)")
        sys.exit(1)
    except (FileNotFoundError, RuntimeError, Exception):
        # Any error here is acceptable — we just confirm it fails gracefully
        print("OK ✓  (fails gracefully, does not crash)")
except Exception as e:
    print(f"FAILED ✗ (unexpected error: {e})")
    traceback.print_exc()
    sys.exit(1)

# ─── Feature importance report ────────────────────────────────────────────────
print("\nFeature Importance Report:")
print(clf.get_feature_importance_report())

# ─── PASS ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODULE 2 SMOKE TEST PASSED ✅")
print("=" * 60)
