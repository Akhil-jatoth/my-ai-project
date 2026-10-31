import os
import sys
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model.crowd_model import MCNN
from train import ShanghaiTechDataset

class EuclideanLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sum((pred - target) ** 2) / (2.0 * pred.size(0))

def run_tiny_test():
    os.makedirs("diagnostics", exist_ok=True)
    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 3 real training images
    ds = ShanghaiTechDataset('data/ShanghaiTech', part='A', split='train')
    indices = [0, 9, 24] # IMG_1, IMG_10, IMG_25
    batch_imgs, batch_gts, sample_names = [], [], []
    
    for idx in indices:
        img_t, gt_t = ds[idx]
        batch_imgs.append(img_t)
        batch_gts.append(gt_t)
        sample_names.append(os.path.basename(ds.samples[idx][0]))
        
    imgs = torch.stack(batch_imgs).to(device)
    gts = torch.stack(batch_gts).to(device)
    gt_counts = [float(gts[i].sum().item()) for i in range(len(indices))]
    
    print("=" * 80, flush=True)
    print("PHASE 6 & 7: TINY DATASET MEMORIZATION & OUTPUT LAYER COMPARISON", flush=True)
    print(f"Target Samples: {sample_names}", flush=True)
    print(f"Ground Truth Counts: {[round(c, 1) for c in gt_counts]}", flush=True)
    print("=" * 80, flush=True)
    
    configs = [
        ("Linear Final Conv", False),
        ("LeakyReLU(0.01) Final Conv", True)
    ]
    
    results = {}
    
    for label, use_leaky in configs:
        print(f"\n--- Testing Output Layer: {label} ---", flush=True)
        torch.manual_seed(42)
        model = MCNN().to(device)
        if use_leaky:
            model.density_layer = nn.Sequential(
                nn.Conv2d(30, 1, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
        else:
            model.density_layer = nn.Sequential(
                nn.Conv2d(30, 1, kernel_size=1)
            )
            
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = EuclideanLoss()
        
        # Initial evaluation
        model.eval()
        with torch.no_grad():
            init_pred = model(imgs)
            init_loss = criterion(init_pred, gts).item()
            init_counts = [float(torch.clamp(init_pred[i], min=0.0).sum().item()) for i in range(len(indices))]
            
        print(f"Initial Loss: {init_loss:.4f} | Init Pred Counts: {[round(c, 1) for c in init_counts]}", flush=True)
        
        # Train for 20 iterations
        model.train()
        grad_norms = []
        for step in range(1, 21):
            opt.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, gts)
            loss.backward()
            
            total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()
            grad_norms.append(total_norm)
            opt.step()
            
            step_counts = [float(torch.clamp(pred[i], min=0.0).sum().item()) for i in range(len(indices))]
            print(f"Step {step:2d}/20 | Loss: {loss.item():8.4f} | Grad Norm: {total_norm:6.2f} | Pred Counts: {[round(c, 1) for c in step_counts]}", flush=True)
                
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_pred = model(imgs)
            final_loss = criterion(final_pred, gts).item()
            final_counts = [float(torch.clamp(final_pred[i], min=0.0).sum().item()) for i in range(len(indices))]
            
        results[label] = {
            "init_loss": init_loss,
            "final_loss": final_loss,
            "init_counts": init_counts,
            "final_counts": final_counts,
            "final_pred": final_pred.cpu().numpy(),
            "grad_norm_final": grad_norms[-1],
            "model": model
        }
        
    print("\n" + "=" * 80, flush=True)
    print("COMPARATIVE SUMMARY (20-Step Controlled Memorization)", flush=True)
    print("=" * 80, flush=True)
    for label in configs:
        res = results[label[0]]
        print(f"Config: {label[0]}", flush=True)
        print(f"  Loss: {res['init_loss']:.4f} -> {res['final_loss']:.4f} ({((res['final_loss']-res['init_loss'])/res['init_loss']*100):.1f}%)", flush=True)
        print(f"  Pred Counts: {[round(c, 1) for c in res['final_counts']]} (GT: {[round(c, 1) for c in gt_counts]})", flush=True)
        print(f"  Final Gradient Norm: {res['grad_norm_final']:.4f}", flush=True)

    # Plot & Save Visualizations for Linear Final Conv
    best_config_name = "Linear Final Conv"
    best_preds = results[best_config_name]["final_pred"]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(3):
        # 1. Original Image (denormalized)
        img_np = batch_imgs[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0.0, 1.0)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{sample_names[i]} (Original)")
        axes[i, 0].axis('off')
        
        # 2. Ground Truth Density Map
        gt_np = batch_gts[i].squeeze().cpu().numpy()
        im1 = axes[i, 1].imshow(gt_np, cmap='jet')
        axes[i, 1].set_title(f"Ground Truth DMap (Count: {gt_counts[i]:.0f})")
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # 3. Predicted Density Map
        pred_np = np.maximum(0.0, best_preds[i].squeeze())
        im2 = axes[i, 2].imshow(pred_np, cmap='jet')
        axes[i, 2].set_title(f"Predicted DMap (Count: {pred_np.sum():.1f})")
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    viz_path = "diagnostics/tiny_overfit_diagnostic.png"
    plt.savefig(viz_path, dpi=150)
    plt.close()
    print(f"\nVisualizations saved to: {viz_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_tiny_test()
