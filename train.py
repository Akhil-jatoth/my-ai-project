"""
CrowdSense — MCNN Training Script
==================================
Trains the MCNN model on ShanghaiTech or UCF_CC_50 crowd-counting datasets
and saves the best weights to model/weights.pth along with training metadata.

Usage
-----
# ShanghaiTech Part A
python train.py --dataset shanghaitech --part A --data-root data/ShanghaiTech

# ShanghaiTech Part B
python train.py --dataset shanghaitech --part B --data-root data/ShanghaiTech

# UCF_CC_50
python train.py --dataset ucf --data-root data/UCF_CC_50

Requirements
------------
  pip install torch torchvision scipy numpy opencv-python Pillow

Scientific Guarantees
---------------------
- Seeds set for Python, NumPy, and PyTorch for full experiment reproducibility.
- Density-map Euclidean Loss (MSE) between ground-truth and predicted density maps.
- Checkpointing: weights.pth is saved ONLY when validation MAE strictly improves.
- Logs epoch-by-epoch loss, validation MAE, and RMSE.
- Generates model/training_metadata.json with full training provenance.
"""

import argparse
import os
import sys
import math
import time
import glob
import json
import random
import logging
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config
from model.crowd_model import MCNN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Density Map Generation
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_kernel(kernel_size: int = 15, sigma: float = 4.0) -> np.ndarray:
    """Return a normalized 2-D Gaussian kernel."""
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / max(float(kernel.sum()), 1e-7)


def generate_density_map(image_shape: tuple, points: np.ndarray,
                          kernel_size: int = 15, sigma: float = 4.0) -> np.ndarray:
    """
    Generate ground-truth density map from head point annotations.
    sum(density) ≈ number of annotated heads.
    """
    H, W = image_shape
    density = np.zeros((H, W), dtype=np.float32)
    kernel = gaussian_kernel(kernel_size, sigma)
    half = kernel_size // 2

    for pt in points:
        r, c = int(round(pt[0])), int(round(pt[1]))
        if r < 0 or r >= H or c < 0 or c >= W:
            continue
        r0, r1 = max(0, r - half), min(H, r + half + 1)
        c0, c1 = max(0, c - half), min(W, c + half + 1)
        kr0 = r0 - (r - half)
        kc0 = c0 - (c - half)
        kr1 = kr0 + (r1 - r0)
        kc1 = kc0 + (c1 - c0)
        density[r0:r1, c0:c1] += kernel[kr0:kr1, kc0:kc1]

    return density


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_mat_points(mat_path: str) -> np.ndarray:
    """Load head point coordinates from MATLAB annotation files."""
    try:
        import scipy.io as sio
        mat = sio.loadmat(mat_path)
        if 'image_info' in mat:
            pts = mat['image_info'][0][0][0][0][0].astype(np.float32)
        elif 'annPoints' in mat:
            pts = mat['annPoints'].astype(np.float32)
        else:
            pts = np.zeros((0, 2), dtype=np.float32)
        return pts
    except NotImplementedError:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            if 'image_info' in f:
                pts = np.array(f['image_info']['location']).T.astype(np.float32)
            elif 'annPoints' in f:
                pts = np.array(f['annPoints']).T.astype(np.float32)
            else:
                pts = np.zeros((0, 2), dtype=np.float32)
        return pts


class ShanghaiTechDataset(Dataset):
    def __init__(self, root: str, part: str = 'A', split: str = 'train',
                 image_size: tuple = (640, 480), crop_size: int = None, seed: int = 42,
                 val_fraction: float = 0.1):
        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.rng = np.random.RandomState(seed)

        # Protocol enforcement (Phase 2):
        # train and val splits come strictly from official train_data (300 images)
        # test split comes strictly from official test_data (182 images)
        split_dir = 'test_data' if split == 'test' else 'train_data'
        if os.path.exists(os.path.join(root, split_dir, 'images')):
            img_dir = os.path.join(root, split_dir, 'images')
            gt_dir  = os.path.join(root, split_dir, 'ground-truth')
        else:
            img_dir = os.path.join(root, f'part_{part}', split_dir, 'images')
            gt_dir  = os.path.join(root, f'part_{part}', split_dir, 'ground-truth')

        all_samples = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
            name      = os.path.splitext(os.path.basename(img_path))[0]
            mat_path  = os.path.join(gt_dir, f'GT_{name}.mat')
            if os.path.exists(mat_path):
                all_samples.append((img_path, mat_path))

        if not all_samples:
            raise FileNotFoundError(
                f"No image/annotation pairs found in '{img_dir}'. "
                "Check data/README.md for the correct directory layout."
            )

        if split == 'test':
            self.samples = all_samples
            log.info(f"ShanghaiTech Part-{part} test (untouched official test set): {len(self.samples)} samples")
        else:
            # Deterministic 90/10 split of the 300 official training images (seed=42)
            perm = np.random.RandomState(seed).permutation(len(all_samples))
            val_count = max(1, int(len(all_samples) * val_fraction))  # 30 validation samples
            if split == 'val':
                self.samples = [all_samples[i] for i in perm[:val_count]]
                log.info(f"ShanghaiTech Part-{part} validation split: {len(self.samples)} samples (from train_data)")
                log.info(f"Precomputing {len(self.samples)} validation density maps for high-speed evaluation...")
                self.cached_val = []
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                W, H = self.image_size
                for img_path, mat_path in self.samples:
                    img = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = img.size
                    pts = load_mat_points(mat_path)
                    img = img.resize((W, H), Image.BILINEAR)
                    if pts.shape[0] > 0:
                        scaled_pts = np.zeros_like(pts)
                        scaled_pts[:, 0] = pts[:, 1] * (H / max(float(orig_h), 1.0))
                        scaled_pts[:, 1] = pts[:, 0] * (W / max(float(orig_w), 1.0))
                    else:
                        scaled_pts = np.zeros((0, 2), dtype=np.float32)
                    density = generate_density_map((H, W), scaled_pts)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    arr = (arr - mean) / std
                    img_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
                    density_tensor = torch.from_numpy(density).unsqueeze(0).float()
                    self.cached_val.append((img_tensor, density_tensor))
            else:  # train
                self.samples = [all_samples[i] for i in perm[val_count:]]
                log.info(f"ShanghaiTech Part-{part} training split: {len(self.samples)} samples (from train_data)")
                log.info(f"Preloading {len(self.samples)} training images and annotations into RAM...")
                self.cached_train = []
                for img_path, mat_path in self.samples:
                    im = Image.open(img_path).convert('RGB')
                    pts = load_mat_points(mat_path)
                    self.cached_train.append((im, pts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == 'val' and hasattr(self, 'cached_val'):
            return self.cached_val[idx]

        if self.split == 'train' and hasattr(self, 'cached_train'):
            orig_img, raw_pts = self.cached_train[idx]
            img = orig_img.copy()
            pts = raw_pts.copy()
        else:
            img_path, mat_path = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            pts = load_mat_points(mat_path)

        orig_w, orig_h = img.size

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.split == 'train' and self.crop_size:
            cs = self.crop_size
            if orig_w < cs or orig_h < cs:
                scale = max(cs / max(float(orig_w), 1.0), cs / max(float(orig_h), 1.0))
                nw, nh = int(math.ceil(orig_w * scale)), int(math.ceil(orig_h * scale))
                img = img.resize((nw, nh), Image.BILINEAR)
                pts = pts * scale
                orig_w, orig_h = nw, nh

            # Phase 8: Balanced crowd-centric sampling (80% head-centered with jitter, 20% random context)
            if len(pts) > 0 and self.rng.rand() < 0.8:
                target_pt = pts[self.rng.randint(0, len(pts))]
                cx, cy = target_pt[0], target_pt[1]
                x0 = int(np.clip(cx - self.rng.randint(32, cs - 32), 0, orig_w - cs))
                y0 = int(np.clip(cy - self.rng.randint(32, cs - 32), 0, orig_h - cs))
            else:
                x0 = self.rng.randint(0, orig_w - cs + 1)
                y0 = self.rng.randint(0, orig_h - cs + 1)

            img = img.crop((x0, y0, x0 + cs, y0 + cs))

            if len(pts) > 0:
                mask = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + cs) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + cs)
                c_pts = pts[mask].copy()
                c_pts[:, 0] -= x0
                c_pts[:, 1] -= y0
            else:
                c_pts = np.zeros((0, 2), dtype=np.float32)

            if self.rng.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(c_pts) > 0:
                    c_pts[:, 0] = cs - 1 - c_pts[:, 0]

            scaled_pts = np.zeros_like(c_pts)
            if len(c_pts) > 0:
                scaled_pts[:, 0] = c_pts[:, 1]  # row (y)
                scaled_pts[:, 1] = c_pts[:, 0]  # col (x)

            density = generate_density_map((cs, cs), scaled_pts)
        else:
            W, H = self.image_size
            img = img.resize((W, H), Image.BILINEAR)
            if pts.shape[0] > 0:
                scaled_pts = np.zeros_like(pts)
                scaled_pts[:, 0] = pts[:, 1] * (H / max(float(orig_h), 1.0))  # row (y)
                scaled_pts[:, 1] = pts[:, 0] * (W / max(float(orig_w), 1.0))  # col (x)
            else:
                scaled_pts = np.zeros((0, 2), dtype=np.float32)
            density = generate_density_map((H, W), scaled_pts)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        img_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        return img_tensor, density_tensor


class UCFDataset(Dataset):
    def __init__(self, root: str, split: str = 'train',
                 val_fraction: float = 0.2, image_size: tuple = (640, 480),
                 seed: int = 42):
        self.image_size = image_size
        rng = np.random.RandomState(seed)

        pairs = []
        for img_path in sorted(glob.glob(os.path.join(root, '*.jpg'))):
            base     = os.path.splitext(img_path)[0]
            mat_path = base + '_ann.mat'
            if os.path.exists(mat_path):
                pairs.append((img_path, mat_path))

        if not pairs:
            raise FileNotFoundError(
                f"No image/annotation pairs found in '{root}'. "
                "Check data/README.md for the correct directory layout."
            )

        idx    = rng.permutation(len(pairs))
        n_val  = max(1, int(len(pairs) * val_fraction))
        if split == 'train':
            self.samples = [pairs[i] for i in idx[n_val:]]
        else:
            self.samples = [pairs[i] for i in idx[:n_val]]

        log.info(f"UCF_CC_50 {split}: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mat_path = self.samples[idx]
        W, H = self.image_size

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((W, H), Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr  = (arr - mean) / std
        img_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()

        pts = load_mat_points(mat_path)
        if pts.shape[0] > 0:
            scaled_pts = np.zeros_like(pts)
            scaled_pts[:, 0] = pts[:, 1] * (H / max(float(orig_h), 1.0))  # row (y)
            scaled_pts[:, 1] = pts[:, 0] * (W / max(float(orig_w), 1.0))  # col (x)
        else:
            scaled_pts = np.zeros((0, 2), dtype=np.float32)
        density = generate_density_map((H, W), scaled_pts)
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        return img_tensor, density_tensor


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Subroutine
# ══════════════════════════════════════════════════════════════════════════════

class EuclideanLoss(nn.Module):
    """
    Euclidean loss for crowd density map regression (Zhang et al. CVPR 2016 Eq. 1):
        L = (1 / 2N) * sum_{i=1}^N || F(X_i; Theta) - D_i ||_2^2
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum((pred - target) ** 2) / (2.0 * pred.size(0))


def evaluate_loader(model: nn.Module, loader: DataLoader, device: str) -> dict:
    """Evaluate MAE, RMSE, and validation loss on DataLoader with detailed prediction metrics."""
    model.eval()
    mae_total, mse_total, val_loss_total, n = 0.0, 0.0, 0.0, 0
    all_preds_list = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for imgs, densities in loader:
            imgs      = imgs.to(device)
            densities = densities.to(device)
            preds     = model(imgs)

            loss = criterion(preds, densities)
            val_loss_total += loss.item() * imgs.size(0)

            # Enforce non-negativity at evaluation time
            clamped_preds = torch.clamp(preds, min=0.0)
            pred_counts = clamped_preds.sum(dim=(1, 2, 3)).cpu().numpy()
            gt_counts   = densities.sum(dim=(1, 2, 3)).cpu().numpy()
            diff        = pred_counts - gt_counts

            mae_total += np.abs(diff).sum()
            mse_total += (diff ** 2).sum()
            n         += len(diff)
            all_preds_list.extend(pred_counts.tolist())

    model.train()
    sample_count = max(n, 1)
    preds_arr = np.array(all_preds_list) if all_preds_list else np.zeros(1)
    return {
        'val_loss':    val_loss_total / sample_count,
        'mae':         mae_total / sample_count,
        'rmse':        math.sqrt(mse_total / sample_count),
        'mean_pred':   float(np.mean(preds_arr)),
        'median_pred': float(np.median(preds_arr)),
        'min_pred':    float(np.min(preds_arr)),
        'max_pred':    float(np.max(preds_arr)),
        'count':       n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def verify_ground_truth_density_integral(points: np.ndarray, density_map: np.ndarray, tolerance: float = 0.05) -> tuple[bool, float, float]:
    """
    Section 5: Verify that the integral of the generated ground-truth density map
    approximately equals the actual number of people in the image.
    Boundary heads whose Gaussian supports are clipped by image edges are accounted for.
    """
    gt_count = float(len(points))
    density_sum = float(density_map.sum())
    diff = abs(density_sum - gt_count)
    allowed_diff = max(1.0, gt_count * tolerance)
    is_valid = diff <= allowed_diff
    return is_valid, gt_count, density_sum


def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Training device: {device} | Random seed: {args.seed}")

    # Validate dataset existence (Section 4)
    data_root = args.data_root or getattr(config, 'DATASET_PATH', 'data/ShanghaiTech')
    if not os.path.exists(data_root) or not os.path.isdir(data_root):
        print("\n" + "=" * 60)
        print("Crowd dataset not found.")
        print("")
        print("Please download ShanghaiTech or UCF_CC_50")
        print("and configure DATASET_PATH in config.py or pass --data-root.")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Load dataset (Phase 2 Protocol: 270 train / 30 val from 300 training samples)
    crop_size = args.crop_size if getattr(args, 'crop_size', 0) > 0 else None
    if args.dataset == 'shanghaitech':
        train_ds = ShanghaiTechDataset(args.data_root, part=args.part, split='train',
                                       crop_size=crop_size, seed=args.seed)
        val_ds   = ShanghaiTechDataset(args.data_root, part=args.part, split='val',
                                       image_size=(640, 480), seed=args.seed)
    elif args.dataset == 'ucf':
        train_ds = UCFDataset(args.data_root, split='train', seed=args.seed)
        val_ds   = UCFDataset(args.data_root, split='val', seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # Initialize MCNN
    model = MCNN().float().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"MCNN Parameter count: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    loss_fn_name = getattr(args, 'loss', 'euclidean')
    if loss_fn_name == 'euclidean':
        criterion = EuclideanLoss()
        log.info("Loss function: Euclidean Density Loss (Zhang et al. CVPR 2016 Eq. 1)")
    else:
        criterion = nn.MSELoss()
        log.info("Loss function: Mean Squared Error (MSE)")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    log_path = os.path.join(config.MODEL_DIR, 'train_log.txt')
    history_csv_path = os.path.join(config.BASE_DIR, 'training_history.csv')

    best_mae   = float('inf')
    best_rmse  = float('inf')
    best_epoch = -1

    with open(log_path, 'w', encoding='utf-8') as log_file, \
         open(history_csv_path, 'w', encoding='utf-8') as csv_file:
        header = 'epoch,train_loss,val_loss,val_mae,val_rmse,mean_pred,median_pred,min_pred,max_pred,grad_norm,time_sec\n'
        log_file.write(header)
        csv_file.write(header)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_grads = []
            t0 = time.time()

            for imgs, densities in train_loader:
                imgs      = imgs.to(device)
                densities = densities.to(device)

                optimizer.zero_grad()
                preds = model(imgs)
                loss  = criterion(preds, densities)
                loss.backward()
                
                # Track gradient norm
                total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()
                epoch_grads.append(total_norm)
                optimizer.step()

                epoch_loss += loss.item() * imgs.size(0)

            scheduler.step()
            train_loss = epoch_loss / max(len(train_ds), 1)
            mean_grad_norm = float(np.mean(epoch_grads)) if epoch_grads else 0.0
            elapsed    = time.time() - t0

            val_metrics = evaluate_loader(model, val_loader, device)
            val_loss = val_metrics['val_loss']
            val_mae  = val_metrics['mae']
            val_rmse = val_metrics['rmse']

            log.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val MAE: {val_mae:.2f} | "
                f"Val RMSE: {val_rmse:.2f} | "
                f"Mean Pred: {val_metrics['mean_pred']:.2f} | "
                f"Grad Norm: {mean_grad_norm:.2f} | "
                f"Time: {elapsed:.1f}s"
            )
            line = (
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_mae:.4f},{val_rmse:.4f},"
                f"{val_metrics['mean_pred']:.2f},{val_metrics['median_pred']:.2f},"
                f"{val_metrics['min_pred']:.2f},{val_metrics['max_pred']:.2f},"
                f"{mean_grad_norm:.4f},{elapsed:.1f}\n"
            )
            log_file.write(line)
            log_file.flush()
            csv_file.write(line)
            csv_file.flush()

            # Save strictly when validation MAE improves on the 30-sample validation subset
            if val_mae < best_mae:
                best_mae   = val_mae
                best_rmse  = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), config.MODEL_WEIGHTS_PATH)
                log.info(f"  ✔ Best model checkpoint saved to {config.MODEL_WEIGHTS_PATH} (Val MAE={best_mae:.2f})")

    # Save training metadata JSON for complete provenance (Section 8)
    metadata = {
        "model_name": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "dataset": args.dataset,
        "part": args.part if args.dataset == 'shanghaitech' else None,
        "total_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_mae": round(float(best_mae), 3),
        "best_val_rmse": round(float(best_rmse), 3),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "random_seed": args.seed,
        "crop_size": crop_size,
        "loss_function": loss_fn_name,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "official_test_samples": 182,
        "trained_timestamp": datetime.now().isoformat(),
        "weights_path": config.MODEL_WEIGHTS_PATH,
    }
    with open(config.TRAINING_META_PATH, 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, indent=2)
    # Also save model_metadata.json for naming compatibility
    model_meta_alt = os.path.join(config.MODEL_DIR, 'model_metadata.json')
    with open(model_meta_alt, 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, indent=2)
    log.info(f"Saved training metadata to {config.TRAINING_META_PATH} and {model_meta_alt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CrowdSense MCNN')
    parser.add_argument('--dataset', choices=['shanghaitech', 'ucf'], required=True)
    parser.add_argument('--part', default='A', choices=['A', 'B'])
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch-size',   type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-step',      type=int,   default=30)
    parser.add_argument('--lr-gamma',     type=float, default=0.5)
    parser.add_argument('--workers',      type=int,   default=0)
    parser.add_argument('--crop-size',   type=int,   default=256, help='Crop size for training patches (0 for full images)')
    parser.add_argument('--loss',        choices=['euclidean', 'mse'], default='euclidean', help='Loss function')
    parser.add_argument('--seed',         type=int,   default=config.RANDOM_SEED)
    args = parser.parse_args()
    train(args)
