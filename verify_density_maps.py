import os
import glob
import math
import numpy as np
from PIL import Image
from train import load_mat_points, generate_density_map

def verify_density():
    img_dir = "data/ShanghaiTech/part_A/train_data/images"
    gt_dir = "data/ShanghaiTech/part_A/train_data/ground-truth"
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:20]
    print("=" * 80)
    print("PHASE 3: FULL IMAGE DENSITY MAP VERIFICATION (20 Real Samples)")
    print("=" * 80)
    print(f"{'Image':<12} {'Ann Count':<10} {'DMap Sum':<12} {'Ratio':<8} {'Min':<10} {'Max':<10} {'Mean':<12}")
    print("-" * 80)
    
    full_ratios = []
    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        mat_path = os.path.join(gt_dir, f"GT_{name}.mat")
        pts = load_mat_points(mat_path)
        img = Image.open(p)
        orig_w, orig_h = img.size
        
        W, H = 640, 480
        scaled_pts = np.zeros_like(pts)
        if pts.shape[0] > 0:
            scaled_pts[:, 0] = pts[:, 1] * (H / max(float(orig_h), 1.0)) # row (y)
            scaled_pts[:, 1] = pts[:, 0] * (W / max(float(orig_w), 1.0)) # col (x)
        
        dmap = generate_density_map((H, W), scaled_pts)
        ann_count = float(len(pts))
        dmap_sum = float(dmap.sum())
        ratio = dmap_sum / max(ann_count, 1e-5)
        full_ratios.append(ratio)
        
        print(f"{name:<12} {ann_count:<10.0f} {dmap_sum:<12.2f} {ratio:<8.4f} {dmap.min():<10.6f} {dmap.max():<10.6f} {dmap.mean():<12.6e}")
        
    print("-" * 80)
    print(f"Mean Ratio across 20 full images: {np.mean(full_ratios):.4f} (std: {np.std(full_ratios):.4f})")
    
    print("\n" + "=" * 80)
    print("PHASE 3: 256x256 CROP DENSITY MAP VERIFICATION (20 Real Samples)")
    print("=" * 80)
    print(f"{'Image':<12} {'Crop Box':<18} {'Heads In':<10} {'DMap Sum':<12} {'Ratio':<8} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    crop_ratios = []
    rng = np.random.RandomState(42)
    cs = 256
    
    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        mat_path = os.path.join(gt_dir, f"GT_{name}.mat")
        pts = load_mat_points(mat_path)
        img = Image.open(p)
        orig_w, orig_h = img.size
        
        # Scale if smaller than crop_size
        if orig_w < cs or orig_h < cs:
            scale = max(cs / orig_w, cs / orig_h)
            nw, nh = int(math.ceil(orig_w * scale)), int(math.ceil(orig_h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
            pts = pts * scale
            orig_w, orig_h = nw, nh
            
        x0 = rng.randint(0, orig_w - cs + 1)
        y0 = rng.randint(0, orig_h - cs + 1)
        
        if len(pts) > 0:
            mask = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + cs) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + cs)
            c_pts = pts[mask].copy()
            c_pts[:, 0] -= x0 # col
            c_pts[:, 1] -= y0 # row
        else:
            c_pts = np.zeros((0, 2), dtype=np.float32)
            
        scaled_c_pts = np.zeros_like(c_pts)
        if len(c_pts) > 0:
            scaled_c_pts[:, 0] = c_pts[:, 1] # row
            scaled_c_pts[:, 1] = c_pts[:, 0] # col
            
        crop_dmap = generate_density_map((cs, cs), scaled_c_pts)
        heads_in = float(len(c_pts))
        crop_sum = float(crop_dmap.sum())
        ratio = crop_sum / max(heads_in, 1e-5) if heads_in > 0 else 1.0
        if heads_in > 0:
            crop_ratios.append(ratio)
            
        box_str = f"[{x0},{y0},{x0+cs},{y0+cs}]"
        print(f"{name:<12} {box_str:<18} {heads_in:<10.0f} {crop_sum:<12.2f} {ratio:<8.4f} {crop_dmap.min():<10.6f} {crop_dmap.max():<10.6f}")

    print("-" * 80)
    if crop_ratios:
        print(f"Mean Crop Ratio (where heads > 0): {np.mean(crop_ratios):.4f} (std: {np.std(crop_ratios):.4f})")
    print("=" * 80)

if __name__ == '__main__':
    verify_density()

