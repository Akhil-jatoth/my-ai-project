import os
import glob
import math
import numpy as np
from PIL import Image
from train import load_mat_points

def analyze_patches():
    img_dir = "data/ShanghaiTech/part_A/train_data/images"
    gt_dir = "data/ShanghaiTech/part_A/train_data/ground-truth"
    
    samples = []
    for p in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        name = os.path.splitext(os.path.basename(p))[0]
        mat = os.path.join(gt_dir, f"GT_{name}.mat")
        if os.path.exists(mat):
            samples.append((p, mat))
            
    print("=" * 80)
    print(f"PHASE 8: PATCH SAMPLING ANALYSIS (on all {len(samples)} training samples)")
    print("=" * 80)
    
    rng = np.random.RandomState(42)
    cs = 256
    
    # 1. Pure Random Cropping (1000 trials across the dataset)
    zero_count = 0
    non_zero_heads = []
    all_heads = []
    
    for _ in range(1000):
        idx = rng.randint(0, len(samples))
        img_path, mat_path = samples[idx]
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        pts = load_mat_points(mat_path)
        
        if orig_w < cs or orig_h < cs:
            scale = max(cs / orig_w, cs / orig_h)
            pts = pts * scale
            orig_w, orig_h = int(math.ceil(orig_w * scale)), int(math.ceil(orig_h * scale))
            
        x0 = rng.randint(0, orig_w - cs + 1)
        y0 = rng.randint(0, orig_h - cs + 1)
        
        if len(pts) > 0:
            mask = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + cs) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + cs)
            h_count = int(mask.sum())
        else:
            h_count = 0
            
        all_heads.append(h_count)
        if h_count == 0:
            zero_count += 1
        else:
            non_zero_heads.append(h_count)
            
    pct_zero = (zero_count / 1000.0) * 100.0
    pct_non_zero = 100.0 - pct_zero
    avg_heads = np.mean(all_heads)
    min_heads = np.min(all_heads)
    max_heads = np.max(all_heads)
    
    print("--- Pure Random Cropping Statistics (1,000 crops) ---")
    print(f"  Percentage containing ZERO heads:     {pct_zero:.1f}%")
    print(f"  Percentage containing AT LEAST 1 head: {pct_non_zero:.1f}%")
    print(f"  Average heads/crop:                   {avg_heads:.1f}")
    print(f"  Minimum heads/crop:                   {min_heads}")
    print(f"  Maximum heads/crop:                   {max_heads}")
    
    # 2. Balanced Crop Strategy:
    # 80% of crops sampled centered around a real head point (crowd crop), 20% random (background/context crop)
    balanced_all_heads = []
    balanced_zero = 0
    for _ in range(1000):
        idx = rng.randint(0, len(samples))
        img_path, mat_path = samples[idx]
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        pts = load_mat_points(mat_path)
        
        if orig_w < cs or orig_h < cs:
            scale = max(cs / orig_w, cs / orig_h)
            pts = pts * scale
            orig_w, orig_h = int(math.ceil(orig_w * scale)), int(math.ceil(orig_h * scale))
            
        if len(pts) > 0 and rng.rand() < 0.8:
            # Sample around a real head
            target_pt = pts[rng.randint(0, len(pts))]
            # center crop on target_pt with random jitter
            cx, cy = target_pt[0], target_pt[1]
            x0 = int(np.clip(cx - rng.randint(32, cs - 32), 0, orig_w - cs))
            y0 = int(np.clip(cy - rng.randint(32, cs - 32), 0, orig_h - cs))
        else:
            x0 = rng.randint(0, orig_w - cs + 1)
            y0 = rng.randint(0, orig_h - cs + 1)
            
        if len(pts) > 0:
            mask = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + cs) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + cs)
            h_count = int(mask.sum())
        else:
            h_count = 0
            
        balanced_all_heads.append(h_count)
        if h_count == 0:
            balanced_zero += 1
            
    b_pct_zero = (balanced_zero / 1000.0) * 100.0
    b_pct_non_zero = 100.0 - b_pct_zero
    b_avg_heads = np.mean(balanced_all_heads)
    b_min_heads = np.min(balanced_all_heads)
    b_max_heads = np.max(balanced_all_heads)
    
    print("\n--- Balanced Crowd-Centric Sampling Statistics (1,000 crops) ---")
    print(f"  Percentage containing ZERO heads:     {b_pct_zero:.1f}%")
    print(f"  Percentage containing AT LEAST 1 head: {b_pct_non_zero:.1f}%")
    print(f"  Average heads/crop:                   {b_avg_heads:.1f}")
    print(f"  Minimum heads/crop:                   {b_min_heads}")
    print(f"  Maximum heads/crop:                   {b_max_heads}")
    print("=" * 80)

if __name__ == '__main__':
    analyze_patches()

