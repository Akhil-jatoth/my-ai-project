"""
CrowdSense — Model Evaluation & Benchmark Script
=================================================
Evaluates the trained MCNN model against benchmark test datasets with
annotated ground-truth counts.

Metrics Computed
----------------
1. Mean Absolute Error (MAE):
       MAE = (1 / N) * sum_{i=1}^N |pred_i - gt_i|
2. Root Mean Squared Error (RMSE):
       RMSE = sqrt((1 / N) * sum_{i=1}^N (pred_i - gt_i)^2)
3. Error range: Min and Max absolute errors
4. Sample mean comparisons: Mean predicted vs. Mean ground truth count

Outputs
-------
- CSV file (default: evaluation_results.csv):
    image,ground_truth,prediction,absolute_error,squared_error
- JSON metrics file (model/evaluation_metrics.json):
    Saved for display in the dashboard performance panel.

Usage
-----
# Evaluate on ShanghaiTech Part A test set
python evaluate.py --dataset shanghaitech --part A --data-root data/ShanghaiTech

# Evaluate on UCF_CC_50
python evaluate.py --dataset ucf --data-root data/UCF_CC_50

# Evaluate using custom image folder and ground-truth CSV (columns: filename,count)
python evaluate.py --image-dir path/to/imgs --gt-csv path/to/gt.csv

Scientific Integrity Notice
---------------------------
Never invent evaluation results. Metrics are computed strictly from genuine
evaluated images and annotations.
"""

import argparse
import os
import math
import csv
import json
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from model.crowd_model import MCNN, CrowdDensityPredictor
from train import ShanghaiTechDataset, UCFDataset, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('evaluate')


def evaluate_dataset(model: torch.nn.Module, dataset, device: str, output_csv: str) -> dict:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    results = []
    abs_errors = []
    sq_errors = []
    gt_counts = []
    pred_counts = []

    log.info(f"Evaluating {len(dataset)} test samples on device: {device}...")

    with torch.no_grad():
        for i, (imgs, densities) in enumerate(loader):
            imgs = imgs.to(device)
            densities = densities.to(device)

            preds = model(imgs)

            pred_val = float(torch.clamp(preds, min=0.0).sum().item())
            gt_val   = float(densities.sum().item())
            abs_err  = abs(pred_val - gt_val)
            sq_err   = (pred_val - gt_val) ** 2

            abs_errors.append(abs_err)
            sq_errors.append(sq_err)
            gt_counts.append(gt_val)
            pred_counts.append(pred_val)

            # Get sample image path
            img_path = getattr(dataset, 'samples', [('', '')])[i][0]
            filename = os.path.basename(img_path) if img_path else f"sample_{i+1:04d}"

            results.append({
                "image": filename,
                "ground_truth": round(gt_val, 2),
                "prediction": round(pred_val, 2),
                "absolute_error": round(abs_err, 2),
                "squared_error": round(sq_err, 2),
            })

            if (i + 1) % 25 == 0 or (i + 1) == len(dataset):
                log.info(f"Processed {i + 1}/{len(dataset)} images...")

    n = max(len(results), 1)
    mae  = float(np.mean(abs_errors))
    rmse = float(math.sqrt(np.mean(sq_errors)))
    min_err = float(np.min(abs_errors))
    max_err = float(np.max(abs_errors))
    mean_gt = float(np.mean(gt_counts))
    mean_pred = float(np.mean(pred_counts))

    # Write CSV
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "absolute_error", "squared_error"])
            writer.writeheader()
            writer.writerows(results)
        log.info(f"Saved detailed image-by-image evaluation to: {output_csv}")

    metrics = {
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
        "evaluation_timestamp": datetime.now().isoformat(),
        "device": device,
    }

    # Save to metrics JSON for dashboard display
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.EVALUATION_METRICS_PATH, 'w', encoding='utf-8') as jf:
        json.dump(metrics, jf, indent=2)
    log.info(f"Saved model performance metrics to: {config.EVALUATION_METRICS_PATH}")

    return metrics


def evaluate_custom_folder(predictor: CrowdDensityPredictor, img_dir: str, gt_csv: str, output_csv: str) -> dict:
    """Evaluate against a custom directory of images with a CSV of ground truth counts."""
    gt_map = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#') or row[0].lower() in ('filename', 'image'):
                continue
            gt_map[row[0].strip()] = float(row[1].strip())

    results = []
    abs_errors = []
    sq_errors = []
    gt_counts = []
    pred_counts = []

    for fname, gt_val in gt_map.items():
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            log.warning(f"Image not found: {img_path}")
            continue

        res = predictor.predict(img_path)
        pred_val = float(res['raw_count'])
        abs_err = abs(pred_val - gt_val)
        sq_err = (pred_val - gt_val) ** 2

        abs_errors.append(abs_err)
        sq_errors.append(sq_err)
        gt_counts.append(gt_val)
        pred_counts.append(pred_val)

        results.append({
            "image": fname,
            "ground_truth": round(gt_val, 2),
            "prediction": round(pred_val, 2),
            "absolute_error": round(abs_err, 2),
            "squared_error": round(sq_err, 2),
        })

    n = max(len(results), 1)
    mae  = float(np.mean(abs_errors)) if abs_errors else 0.0
    rmse = float(math.sqrt(np.mean(sq_errors))) if sq_errors else 0.0

    if output_csv and results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "absolute_error", "squared_error"])
            writer.writeheader()
            writer.writerows(results)

    metrics = {
        "evaluated": True,
        "model": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "total_test_images": len(results),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "min_error": round(float(np.min(abs_errors)), 2) if abs_errors else 0.0,
        "max_error": round(float(np.max(abs_errors)), 2) if abs_errors else 0.0,
        "mean_ground_truth": round(float(np.mean(gt_counts)), 2) if gt_counts else 0.0,
        "mean_prediction": round(float(np.mean(pred_counts)), 2) if pred_counts else 0.0,
        "evaluation_timestamp": datetime.now().isoformat(),
        "device": predictor.device,
    }

    with open(config.EVALUATION_METRICS_PATH, 'w', encoding='utf-8') as jf:
        json.dump(metrics, jf, indent=2)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CrowdSense MCNN model')
    parser.add_argument('--dataset', choices=['shanghaitech', 'ucf'], default=None)
    parser.add_argument('--part', default='A', choices=['A', 'B'])
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--gt-csv', default=None)
    parser.add_argument('--weights', default=config.MODEL_WEIGHTS_PATH)
    parser.add_argument('--output-csv', default='evaluation_results.csv')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.weights):
        log.error(f"Cannot evaluate: weights file '{args.weights}' does not exist.")
        log.error("Train the model first using: python train.py ...")
        exit(1)

    log.info(f"Loading weights from: {args.weights}")
    model = MCNN().float().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    if args.dataset == 'shanghaitech':
        if not args.data_root:
            raise ValueError("--data-root must be provided for shanghaitech")
        test_ds = ShanghaiTechDataset(args.data_root, part=args.part, split='test')
        metrics = evaluate_dataset(model, test_ds, device, args.output_csv)
    elif args.dataset == 'ucf':
        if not args.data_root:
            raise ValueError("--data-root must be provided for ucf")
        test_ds = UCFDataset(args.data_root, split='val', seed=args.seed)
        metrics = evaluate_dataset(model, test_ds, device, args.output_csv)
    elif args.image_dir and args.gt_csv:
        predictor = CrowdDensityPredictor(weights_path=args.weights, device=device)
        metrics = evaluate_custom_folder(predictor, args.image_dir, args.gt_csv, args.output_csv)
    else:
        log.error("Specify either --dataset (shanghaitech/ucf) with --data-root, OR --image-dir with --gt-csv")
        exit(1)

    # Generate research experiment summary (Section 18)
    exp_summary = {
        "model": metrics.get("model", "MCNN"),
        "dataset": args.dataset or "Custom",
        "total_test_images": metrics.get("total_test_images"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "min_error": metrics.get("min_error"),
        "max_error": metrics.get("max_error"),
        "mean_ground_truth": metrics.get("mean_ground_truth"),
        "mean_prediction": metrics.get("mean_prediction"),
        "evaluation_timestamp": metrics.get("evaluation_timestamp"),
        "weights_path": args.weights,
    }
    if os.path.exists(config.TRAINING_META_PATH):
        try:
            with open(config.TRAINING_META_PATH, 'r', encoding='utf-8') as mf:
                exp_summary["training_provenance"] = json.load(mf)
        except Exception:
            pass

    exp_summary_path = os.path.join(config.BASE_DIR, 'experiment_summary.json')
    with open(exp_summary_path, 'w', encoding='utf-8') as ef:
        json.dump(exp_summary, ef, indent=2)
    log.info(f"Saved research experiment summary to: {exp_summary_path}")

    print("\n" + "=" * 60)
    print("[*] CROWDSENSE - MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model:                {metrics['model']} (v{metrics['model_version']})")
    print(f"  Test Samples:         {metrics['total_test_images']}")
    print(f"  Mean Absolute Error:  {metrics['mae']} (MAE)")
    print(f"  Root Mean Sq. Error:  {metrics['rmse']} (RMSE)")
    print(f"  Min Absolute Error:   {metrics['min_error']}")
    print(f"  Max Absolute Error:   {metrics['max_error']}")
    print(f"  Mean Ground Truth:    {metrics['mean_ground_truth']}")
    print(f"  Mean Predicted:       {metrics['mean_prediction']}")
    print("=" * 60)
    print(f"Results exported to: {args.output_csv}")
    print(f"Metrics saved to:    {config.EVALUATION_METRICS_PATH}")
    print(f"Experiment summary:  {exp_summary_path}\n")

