"""
CrowdSense — Real Test Set Evaluation & Comparison Pipeline
===========================================================
1. Runs real ShanghaiTech Part A test images through trained MCNN & CV Fallback.
2. Generates model_comparison.csv with real measured predictions and errors.
3. Runs the 2,000-person synthetic sanity test image.
"""

import os
import sys
import csv
import time
import glob
import json
import scipy.io as sio
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import config
from model.crowd_model import CrowdDensityPredictor
from app import generate_cv_fallback_density

def run_comparison():
    print("=" * 65)
    print("🔬 REAL TEST IMAGES EVALUATION & COMPARISON")
    print("=" * 65)

    predictor = CrowdDensityPredictor(weights_path=config.MODEL_WEIGHTS_PATH)
    print(f"Model mode: {predictor.mode} (weights_loaded={predictor.weights_loaded})")

    # Select representative real ShanghaiTech Part A test samples
    test_img_dir = "data/ShanghaiTech/part_A/test_data/images"
    test_gt_dir = "data/ShanghaiTech/part_A/test_data/ground-truth"

    sample_names = ["IMG_1", "IMG_10", "IMG_25", "IMG_50", "IMG_100"]
    comparison_rows = []

    print("\nEvaluating real ShanghaiTech Part A test samples:")
    print("-" * 65)

    for name in sample_names:
        img_path = os.path.join(test_img_dir, f"{name}.jpg")
        gt_path = os.path.join(test_gt_dir, f"GT_{name}.mat")
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        # Load ground truth count
        mat = sio.loadmat(gt_path)
        gt_pts = mat['image_info'][0][0][0][0][0]
        gt_count = float(len(gt_pts))

        # MCNN prediction
        t0 = time.perf_counter()
        mcnn_res = predictor.predict(img_path)
        mcnn_time_ms = (time.perf_counter() - t0) * 1000.0
        mcnn_pred = mcnn_res['count']
        mcnn_raw = mcnn_res['raw_count']
        mcnn_err = abs(mcnn_pred - gt_count)

        # CV fallback prediction
        t_cv = time.perf_counter()
        cv_dmap, cv_raw, cv_pred = generate_cv_fallback_density(img_path)
        cv_time_ms = (time.perf_counter() - t_cv) * 1000.0
        cv_err = abs(cv_pred - gt_count)

        print(f"[{name}.jpg] GT: {gt_count:5.0f} | MCNN: {mcnn_pred:5d} (err: {mcnn_err:5.1f}, {mcnn_time_ms:5.1f}ms) | CV: {cv_pred:5d} (err: {cv_err:5.1f}, {cv_time_ms:5.1f}ms)")

        comparison_rows.append({
            "image": f"{name}.jpg",
            "ground_truth": round(gt_count, 1),
            "MCNN_prediction": mcnn_pred,
            "MCNN_absolute_error": round(mcnn_err, 2),
            "CV_prediction": cv_pred,
            "CV_absolute_error": round(cv_err, 2),
        })

    # Save model_comparison.csv
    csv_path = "model_comparison.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "ground_truth", "MCNN_prediction", "MCNN_absolute_error",
            "CV_prediction", "CV_absolute_error"
        ])
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"\nSaved model comparison table to: {csv_path}")

    # ── Synthetic 2,000-Person Sanity Test ──
    print("\n" + "=" * 65)
    print("🧪 SYNTHETIC 2,000-PERSON SANITY TEST")
    print("=" * 65)
    synth_path = "C:/Users/abhignya/Downloads/crowdsense_test_2000_people.png"
    if os.path.exists(synth_path):
        t0 = time.perf_counter()
        synth_res = predictor.predict(synth_path)
        synth_time = (time.perf_counter() - t0) * 1000.0
        synth_pred = synth_res['count']
        synth_gt = 2000.0
        synth_err = abs(synth_pred - synth_gt)
        print(f"Known synthetic ground truth: {synth_gt:.0f}")
        print(f"MCNN prediction:              {synth_pred}")
        print(f"Absolute difference:          {synth_err:.2f} ({synth_err/synth_gt*100:.1f}%)")
        print(f"Inference processing time:    {synth_time:.2f} ms")
        print(f"Integral density map sum:     {float(synth_res['density_map'].sum()):.2f}")
    else:
        print(f"Synthetic image not found at {synth_path}")

if __name__ == '__main__':
    run_comparison()

