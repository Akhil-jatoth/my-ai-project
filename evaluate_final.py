"""
CrowdSense — Final Accuracy & Benchmark Evaluation Script
Evaluates the final best validation checkpoint on the untouched ShanghaiTech Part A test set (182 images).
Calculates exact MAE, RMSE, Pearson correlation, baseline comparisons, best/worst predictions,
and generates final_accuracy_report.json and final_accuracy_report.csv.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
import csv
import json
import time
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from PIL import Image

import config
from model.crowd_model import MCNN, validate_density_map
from train import ShanghaiTechDataset

def run_final_evaluation():
    print("=" * 70)
    print("CROWDSENSE - FINAL BENCHMARK EVALUATION (SHANGHAITECH PART A)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluation Device: {device}")

    weights_path = config.MODEL_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        print(f"ERROR: Checkpoint file not found at {weights_path}")
        sys.exit(1)

    # 1. Load Model
    model = MCNN().float().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    # 2. Load untouched 182 official test images
    test_ds = ShanghaiTechDataset(root="data/ShanghaiTech", part="A", split="test", image_size=(640, 480))
    print(f"Official test dataset size: {len(test_ds)} images")

    # 3. Calculate Training Constant-Mean Baseline
    train_gt_dir = "data/ShanghaiTech/part_A/train_data/ground-truth"
    train_counts = []
    for mat_file in os.listdir(train_gt_dir):
        if mat_file.endswith(".mat"):
            mat = sio.loadmat(os.path.join(train_gt_dir, mat_file))
            pts = mat['image_info'][0][0][0][0][0]
            train_counts.append(len(pts))
    mean_train_baseline = float(np.mean(train_counts))
    print(f"Training set constant-mean baseline count: {mean_train_baseline:.2f} (from {len(train_counts)} train images)")

    # 4. Evaluate Test Set
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    rows = []
    abs_errors = []
    sq_errors = []
    gt_counts = []
    pred_counts = []
    baseline_abs_errors = []
    baseline_sq_errors = []

    print("\nRunning inference on 182 test images...")
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, densities) in enumerate(test_loader):
            imgs = imgs.to(device)
            densities = densities.to(device)

            preds = model(imgs)
            clamped = torch.clamp(preds, min=0.0)

            pred_val = float(clamped.sum().item())
            gt_val   = float(densities.sum().item())

            abs_err = abs(pred_val - gt_val)
            sq_err  = (pred_val - gt_val) ** 2

            abs_errors.append(abs_err)
            sq_errors.append(sq_err)
            gt_counts.append(gt_val)
            pred_counts.append(pred_val)

            # Baseline error
            b_err = abs(mean_train_baseline - gt_val)
            baseline_abs_errors.append(b_err)
            baseline_sq_errors.append(b_err ** 2)

            img_path = test_ds.samples[i][0]
            filename = os.path.basename(img_path)

            rows.append({
                "image": filename,
                "ground_truth": round(gt_val, 2),
                "prediction": round(pred_val, 2),
                "absolute_error": round(abs_err, 2),
                "squared_error": round(sq_err, 2),
                "percentage_error": round((abs_err / max(gt_val, 1.0)) * 100.0, 2)
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(test_ds):
                print(f"  Processed {i + 1}/{len(test_ds)} images...")

    total_eval_time = time.time() - t0
    n = len(rows)

    # Metrics calculation
    mae = float(np.mean(abs_errors))
    rmse = float(math.sqrt(np.mean(sq_errors)))
    median_ae = float(np.median(abs_errors))
    mean_gt = float(np.mean(gt_counts))
    mean_pred = float(np.mean(pred_counts))
    min_pred = float(np.min(pred_counts))
    max_pred = float(np.max(pred_counts))
    min_err = float(np.min(abs_errors))
    max_err = float(np.max(abs_errors))
    zero_preds = int(sum(1 for p in pred_counts if p < 1.0))
    valid_preds = n - zero_preds

    # Pearson correlation
    r_corr = float(np.corrcoef(gt_counts, pred_counts)[0, 1]) if np.std(pred_counts) > 1e-6 else 0.0

    # Baseline metrics
    base_mae = float(np.mean(baseline_abs_errors))
    base_rmse = float(math.sqrt(np.mean(baseline_sq_errors)))

    # Sort for best and worst
    sorted_rows = sorted(rows, key=lambda x: x["absolute_error"])
    best_5 = sorted_rows[:5]
    worst_5 = sorted_rows[-5:]

    # Read training metadata for best validation epoch
    val_mae, val_rmse, best_epoch = None, None, None
    meta_path = config.TRAINING_META_PATH
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as mf:
            m_data = json.load(mf)
            val_mae = m_data.get("best_val_mae")
            val_rmse = m_data.get("best_val_rmse")
            best_epoch = m_data.get("best_epoch")

    # Build final report dictionary
    final_report = {
        "dataset": "ShanghaiTech Part A",
        "training_images": 270,
        "validation_images": 30,
        "test_images": n,
        "best_checkpoint": weights_path,
        "best_epoch": best_epoch,
        "best_validation_mae": val_mae,
        "best_validation_rmse": val_rmse,
        "test_mae": round(mae, 2),
        "test_rmse": round(rmse, 2),
        "median_absolute_error": round(median_ae, 2),
        "mean_ground_truth": round(mean_gt, 2),
        "mean_prediction": round(mean_pred, 2),
        "min_prediction": round(min_pred, 2),
        "max_prediction": round(max_pred, 2),
        "min_error": round(min_err, 2),
        "max_error": round(max_err, 2),
        "zero_near_zero_predictions": zero_preds,
        "valid_predictions": valid_preds,
        "pearson_correlation": round(r_corr, 4),
        "baseline_mae": round(base_mae, 2),
        "baseline_rmse": round(base_rmse, 2),
        "total_evaluation_time_sec": round(total_eval_time, 2),
        "best_5_predictions": best_5,
        "worst_5_predictions": worst_5,
        "evaluated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    # Save final_accuracy_report.json
    json_path = "final_accuracy_report.json"
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(final_report, jf, indent=2)
    print(f"\nSaved final accuracy report to {json_path}")

    # Also update model/evaluation_metrics.json for dashboard
    dashboard_metrics = {
        "evaluated": True,
        "model": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "total_test_images": n,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "min_error": round(min_err, 2),
        "max_error": round(max_err, 2),
        "mean_ground_truth": round(mean_gt, 2),
        "mean_prediction": round(mean_pred, 2),
        "baseline_mae": round(base_mae, 2),
        "baseline_rmse": round(base_rmse, 2),
        "zero_predictions": zero_preds,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device
    }
    with open(config.EVALUATION_METRICS_PATH, 'w', encoding='utf-8') as df:
        json.dump(dashboard_metrics, df, indent=2)

    # Save final_accuracy_report.csv
    csv_path = "final_accuracy_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=["image", "ground_truth", "prediction", "absolute_error", "squared_error", "percentage_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved image-by-image benchmark report to {csv_path}")

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test MAE:                 {mae:.2f}")
    print(f"Test RMSE:                {rmse:.2f}")
    print(f"Baseline MAE:             {base_mae:.2f}")
    print(f"Baseline RMSE:            {base_rmse:.2f}")
    print(f"Mean Predicted Count:     {mean_pred:.2f}")
    print(f"Mean Ground-Truth Count:  {mean_gt:.2f}")
    print(f"Zero/Near-Zero Count:     {zero_preds} / {n}")
    print(f"Pearson Correlation (r):  {r_corr:.4f}")
    print("=" * 70)

    return final_report

if __name__ == '__main__':
    run_final_evaluation()

