"""
CrowdSense — Individual Image Inference & Inspection Utility
=============================================================
Runs MCNN prediction on a single image and displays the model mode,
estimated count, processing time, and optional ground-truth comparison.

Usage:
------
python predict.py --image path/to/image.jpg
python predict.py --image path/to/image.jpg --ground-truth 243
python predict.py --image path/to/image.jpg --compare-cv
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import cv2
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import config
from model.crowd_model import CrowdDensityPredictor, validate_density_map
from app import generate_cv_fallback_density

def run_predict(image_path: str, ground_truth: float = None,
                compare_cv: bool = False, weights_path: str = None,
                output_dir: str = None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    weights_to_use = weights_path or config.MODEL_WEIGHTS_PATH
    predictor = CrowdDensityPredictor(weights_path=weights_to_use)

    # ── MCNN Inference ──
    t0 = time.perf_counter()
    if predictor.trained_model:
        res = predictor.predict(image_path)
        mcnn_raw = res['raw_count']
        mcnn_count = res['count']
        density_map = res['density_map']
        mode_str = "trained"
    else:
        # If weights are missing, predictor runs in demo mode
        dmap, raw_cnt, est_cnt = generate_cv_fallback_density(image_path)
        mcnn_raw = raw_cnt
        mcnn_count = est_cnt
        density_map = dmap
        mode_str = "demo (Computer Vision Fallback — trained weights unavailable)"
    
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print("=" * 60)
    print("CROWDSENSE — INFERENCE RESULT")
    print("=" * 60)
    print(f"Model:           {config.MODEL_NAME}")
    print(f"Mode:            {mode_str}")
    print(f"Estimated Count: {mcnn_count:,} (raw integral: {mcnn_raw:.2f})")
    print(f"Processing Time: {elapsed_ms:.2f} ms")

    if ground_truth is not None:
        abs_err = abs(mcnn_count - ground_truth)
        pct_err = (abs_err / max(ground_truth, 1.0)) * 100.0
        print(f"Ground Truth:    {ground_truth:g}")
        print(f"Prediction:      {mcnn_count}")
        print(f"Absolute Error:  {abs_err:.2f} ({pct_err:.1f}%)")

    # ── Comparison with CV Fallback ──
    if compare_cv:
        t_cv0 = time.perf_counter()
        cv_dmap, cv_raw, cv_count = generate_cv_fallback_density(image_path)
        cv_elapsed_ms = (time.perf_counter() - t_cv0) * 1000.0

        print("\n" + "-" * 60)
        print("MODEL COMPARISON: MCNN vs CV FALLBACK")
        print("-" * 60)
        print(f"MCNN Count:         {mcnn_count:,} (time: {elapsed_ms:.2f} ms)")
        print(f"CV Fallback Count:  {cv_count:,} (time: {cv_elapsed_ms:.2f} ms)")
        if ground_truth is not None:
            mcnn_diff = abs(mcnn_count - ground_truth)
            cv_diff = abs(cv_count - ground_truth)
            print(f"MCNN Absolute Error: {mcnn_diff:.2f}")
            print(f"CV Absolute Error:   {cv_diff:.2f}")

    # ── Optional Visual Output Saving ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save density map numpy array
        np.save(os.path.join(output_dir, f"{base}_density.npy"), density_map)
        
        # Save colorized heatmap
        dm_min, dm_max = float(density_map.min()), float(density_map.max())
        if dm_max - dm_min > 1e-7:
            norm_map = ((density_map - dm_min) / (dm_max - dm_min) * 255.0).astype(np.uint8)
        else:
            norm_map = np.zeros_like(density_map, dtype=np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, f"{base}_heatmap.png"), heatmap)
        print(f"\nSaved density map and heatmap to: {output_dir}")

    print("=" * 60 + "\n")
    return mcnn_count, mode_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrowdSense Single-Image Predictor')
    parser.add_argument('--image', required=True, help='Path to crowd image')
    parser.add_argument('--ground-truth', type=float, default=None, help='Known ground truth count for verification')
    parser.add_argument('--compare-cv', action='store_true', help='Compare MCNN prediction with CV fallback')
    parser.add_argument('--weights', default=None, help='Custom model weights path')
    parser.add_argument('--output-dir', default=None, help='Directory to save heatmap and density map')
    args = parser.parse_args()

    run_predict(
        image_path=args.image,
        ground_truth=args.ground_truth,
        compare_cv=args.compare_cv,
        weights_path=args.weights,
        output_dir=args.output_dir
    )

