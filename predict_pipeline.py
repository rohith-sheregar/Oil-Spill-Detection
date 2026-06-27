"""
Integrated Pipeline Testing/Inference Script.
Runs the end-to-end model pipeline (Module 1 DeepLabV3+ segmentation + Module 2 Random Forest look-alike rejection)
on the test dataset or a custom image.

Usage:
  # Predict and visualize all test images
  python predict_pipeline.py
  
  # Predict on a single image
  python predict_pipeline.py --image path/to/image.png
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

import config
from src.pipeline import run_pipeline, load_module1, run_module1, run_module2, visualize_pipeline
from src.classifier import LookAlikeClassifier

def predict_on_test_set():
    device = config.DEVICE
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    classifier_path = os.path.join(config.CHECKPOINT_DIR, "module2_classifier.pkl")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Module 1 model checkpoint not found at: {ckpt_path}")
        print("Please train Module 1 first (e.g., python train_module1.py).")
        return
        
    # Load model
    model = load_module1(ckpt_path, device)
    
    # Load classifier
    classifier = LookAlikeClassifier()
    if os.path.exists(classifier_path):
        classifier.load(classifier_path)
        print(f"✓ Loaded Module 2 classifier from: {classifier_path}")
    else:
        print(f"⚠️ Module 2 classifier not found at: {classifier_path}")
        print("Running with an untrained classifier (look-alike rejection will be skipped).")
        
    test_img_dir = config.TEST_IMG_DIR
    if not os.path.exists(test_img_dir):
        print(f"❌ Test image directory not found: {test_img_dir}")
        return
        
    test_images = sorted([
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))
    ])
    
    if not test_images:
        print(f"No test images found in {test_img_dir}")
        return
        
    print(f"Running predictions on {len(test_images)} test images...")
    
    os.makedirs(config.PRED_DIR, exist_ok=True)
    
    for i, fname in enumerate(test_images):
        img_path = os.path.join(test_img_dir, fname)
        print(f"\n[{i+1}/{len(test_images)}] Processing: {fname}")
        
        # Output visualization path
        save_path = os.path.join(config.PRED_DIR, f"result_{os.path.splitext(fname)[0]}.png")
        
        try:
            # Execute pipeline
            result = run_pipeline(img_path, classifier_path=classifier_path, save_viz=False)
            
            # Save visual results
            visualize_pipeline(
                image=result["image"], 
                mask=result["mask"], 
                features=result["features"], 
                predictions=result["predictions"], 
                save_path=save_path
            )
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the integrated Oil Spill Detection pipeline")
    parser.add_argument("--image", type=str, help="Path to a single SAR image to run prediction on")
    args = parser.parse_args()
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        run_pipeline(args.image)
    else:
        predict_on_test_set()
