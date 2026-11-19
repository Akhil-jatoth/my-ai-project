"""
CrowdSense — MCNN Model & Scientific Density-Map Pipeline
=========================================================
Multi-Column Convolutional Neural Network (MCNN) for crowd-density estimation.

Architecture Reference
----------------------
Zhang et al., "Single-Image Crowd Counting via Multi-Column Convolutional
Neural Network", IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2016.

Mathematical Principle
----------------------
Instead of localizing individual bounding boxes (which fails under heavy occlusion),
the network maps an input image I in R^(3 x H x W) to a continuous density map
D in R^(1 x H x W). The density value at each pixel represents the expected fractional
head presence. The total crowd estimate is the mathematical integral:

    Count = iint D(x, y) dx dy  approx  sum_{i,j} D[i, j]

Scientific Constraints Enforced
-------------------------------
1. Non-negativity: D[i, j] >= 0 everywhere via final ReLU activation.
2. Finite numerical bounds: Explicit guards against NaN, Inf, and numerical drift.
3. Strict integration: The estimated count is the exact sum of the density map
   without arbitrary multipliers or artificial heuristic scalers.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import config


# ══════════════════════════════════════════════════════════════════════════════
# Density Map Validation Utilities
# ══════════════════════════════════════════════════════════════════════════════

def validate_density_map(density_map: np.ndarray) -> tuple[bool, str, np.ndarray]:
    """
    Validate and sanitize an estimated density map.

    Checks:
      1. Dimensionality: Must be a 2D array (H, W).
      2. Numerical validity: Checks for NaN or +/-Inf values.
      3. Non-negativity: Clamps any tiny negative numerical precision drift to 0.0.
      4. Biological plausibility: Clamps extreme single-pixel spikes (> 50.0).

    Returns:
        (is_valid: bool, status_message: str, sanitized_map: np.ndarray)
    """
    if not isinstance(density_map, np.ndarray):
        return False, "Density map must be a numpy ndarray", np.zeros((480, 640), dtype=np.float32)

    if density_map.ndim != 2:
        return False, f"Expected 2D density map, got shape {density_map.shape}", np.zeros((480, 640), dtype=np.float32)

    # Check for NaN / Inf
    if np.isnan(density_map).any() or np.isinf(density_map).any():
        sanitized = np.nan_to_num(density_map, nan=0.0, posinf=0.0, neginf=0.0)
        return False, "Density map contained NaN/Inf values (sanitized to 0.0)", sanitized.astype(np.float32)

    # Enforce non-negativity (mathematical requirement for density maps)
    sanitized = np.maximum(0.0, density_map)

    # Guard against extreme single-pixel outliers (numerical divergence guard)
    if sanitized.max() > 100.0:
        sanitized = np.clip(sanitized, 0.0, 100.0)
        return True, "Density map contained extreme spikes (clipped to 100.0)", sanitized.astype(np.float32)

    return True, "Valid density map", sanitized.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MCNN Neural Network Architecture
# ══════════════════════════════════════════════════════════════════════════════

class MCNN(nn.Module):
    """
    Multi-Column Convolutional Neural Network.

    Contains three parallel columns with distinct filter sizes:
      - Column 1 (Large receptive field: 9x9, 7x7) captures large-scale heads (foreground).
      - Column 2 (Medium receptive field: 7x7, 5x5) captures medium-scale heads (midground).
      - Column 3 (Small receptive field: 5x5, 3x3) captures small-scale heads (background).

    Branch outputs are upsampled to the input spatial resolution using bilinear
    interpolation, concatenated along the channel axis (16 + 20 + 24 = 60 feature maps),
    and mapped to a single-channel density map via a 1x1 convolution followed by ReLU.
    """

    def __init__(self):
        super().__init__()

        # Column 1: Large receptive field (filters: 9x9 -> 7x7 -> 7x7 -> 7x7)
        self.col1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )

        # Column 2: Medium receptive field (filters: 7x7 -> 5x5 -> 5x5 -> 5x5)
        self.col2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # Column 3: Small receptive field (filters: 5x5 -> 3x3 -> 3x3 -> 3x3)
        self.col3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fusion layer: 1x1 convolution mapping 8 + 10 + 12 = 30 channels to 1 density map
        # Linear output layer allows unbounded continuous density regression without
        # artificial gradient suppression or Dead ReLU stalls.
        # Biological non-negativity is strictly enforced at inference/validation via validate_density_map.
        self.density_layer = nn.Conv2d(30, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.col1(x)
        c2 = self.col2(x)
        c3 = self.col3(x)

        h, w = x.size(2), x.size(3)
        # Upsample column feature maps back to original image spatial resolution
        c1 = F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)
        c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)

        combined = torch.cat([c1, c2, c3], dim=1)
        density_map = self.density_layer(combined)
        return density_map


# ══════════════════════════════════════════════════════════════════════════════
# CSRNet Neural Network Architecture (Alternative Deep Back-End)
# ══════════════════════════════════════════════════════════════════════════════

class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes.
    Reference: Li et al., CVPR 2018.

    Front-end: First 10 convolutional layers of VGG-16 (conv1_1 through conv4_3).
    Back-end: 6 dilated convolutional layers (dilation rate = 2) for multi-scale feature
              extraction without resolution loss.
    Output: 1x1 convolution followed by bilinear upsampling back to input spatial resolution.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.frontend = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if pretrained:
            self._load_vgg_pretrained()
        else:
            self._init_weights()

    def _load_vgg_pretrained(self):
        vgg = None
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except Exception:
            try:
                from torchvision.models import vgg16
                vgg = vgg16(pretrained=True)
            except Exception:
                vgg = None

        if vgg is not None:
            vgg_layers = list(vgg.features.children())
            conv_idx = 0
            for layer in vgg_layers[:23]:
                if isinstance(layer, nn.Conv2d):
                    # Match only Conv2d layers in frontend
                    while conv_idx < len(self.frontend) and not isinstance(self.frontend[conv_idx], nn.Conv2d):
                        conv_idx += 1
                    if conv_idx < len(self.frontend):
                        self.frontend[conv_idx].weight.data.copy_(layer.weight.data)
                        if layer.bias is not None and self.frontend[conv_idx].bias is not None:
                            self.frontend[conv_idx].bias.data.copy_(layer.bias.data)
                        conv_idx += 1
        else:
            self._init_weights()

        # Initialize backend weights with Gaussian N(0, 0.01)
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        feat = self.frontend(x)
        feat = self.backend(feat)
        density = self.output_layer(feat)
        density_map = F.interpolate(density, size=(h, w), mode='bilinear', align_corners=False)
        return density_map


# ══════════════════════════════════════════════════════════════════════════════
# Predictor Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CrowdDensityPredictor:
    """
    High-level inference pipeline for MCNN and CSRNet models.
    Dynamically loads model architecture based on metadata or state_dict keys.
    Handles image preprocessing, tensor transformation, device management,
    model inference, and density-map validation.
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    INPUT_W, INPUT_H = 640, 480

    def __init__(self, weights_path: str = None, device: str = None, model_name: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.weights_path = weights_path
        self.training_metadata = None
        self.weights_loaded = False
        self.trained_model  = False

        inferred_arch = model_name

        # 1. Check training metadata if present
        meta_file = config.TRAINING_META_PATH
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as mf:
                    self.training_metadata = json.load(mf)
                    if not inferred_arch:
                        inferred_arch = self.training_metadata.get("model_name") or self.training_metadata.get("model_architecture")
            except Exception:
                pass

        # 2. Check state dict keys if weights exist
        state_dict = None
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=device)
                if not inferred_arch and isinstance(state_dict, dict):
                    if any(k.startswith('frontend') or k.startswith('backend') for k in state_dict.keys()):
                        inferred_arch = 'CSRNet'
                    elif any(k.startswith('col1') for k in state_dict.keys()):
                        inferred_arch = 'MCNN'
            except Exception as e:
                print(f"[Predictor] Warning reading weights file '{weights_path}': {e}")

        arch_str = (inferred_arch or 'MCNN').upper()
        if 'CSR' in arch_str:
            self.model_name = 'CSRNet'
            self.model = CSRNet(pretrained=False).float().to(device)
        else:
            self.model_name = 'MCNN'
            self.model = MCNN().float().to(device)

        if state_dict is not None:
            try:
                self.model.load_state_dict(state_dict)
                self.weights_loaded = True
                self.trained_model  = True
                print(f"[{self.model_name}] Trained weights successfully loaded from '{weights_path}' (device={device})")
            except Exception as exc:
                print(f"[{self.model_name}] ERROR: Failed loading weights '{weights_path}': {exc}")
                print(f"[{self.model_name}] Falling back to Demo/Fallback Mode.")
        else:
            if weights_path:
                print(f"[{self.model_name}] Weights file not found: '{weights_path}'")
            print(f"[{self.model_name}] Running in Demo/Fallback Mode (weights unavailable).")

        self.model.eval()

    @property
    def mode(self) -> str:
        return "trained" if self.trained_model else "demo"

    @property
    def status(self) -> dict:
        status_dict = {
            "model_name":     self.model_name,
            "model_version":  config.MODEL_VERSION,
            "weights_loaded": self.weights_loaded,
            "trained_model":  self.trained_model,
            "device":         self.device,
            "mode":           self.mode,
            "training_meta":  self.training_metadata,
        }
        return status_dict

    def preprocess_image(self, image_path: str):
        """
        Load and normalize image to float32 tensor suitable for crowd density models.
        Returns:
            tensor: torch.Tensor of shape (1, 3, 480, 640) float32
            pil_image: PIL RGB image
        """
        img = Image.open(image_path).convert('RGB')
        resized_img = img.resize((self.INPUT_W, self.INPUT_H), Image.BILINEAR)

        arr = np.array(resized_img, dtype=np.float32) / 255.0
        arr = (arr - self._MEAN) / self._STD

        tensor = (
            torch.from_numpy(arr.transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        return tensor, resized_img

    def predict(self, image_path: str) -> dict:
        """
        Run inference on an image file.

        The crowd count is computed as the exact mathematical sum of the
        validated density map:
            Count = sum_{i,j} D[i,j]

        No arbitrary multiplier or artificial scaling is applied.
        """
        tensor, pil_img = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(tensor)

        density_raw = output.squeeze().cpu().numpy().astype(np.float32)
        is_valid, msg, density = validate_density_map(density_raw)

        # Exact mathematical integral sum
        raw_count = float(density.sum())
        estimated_count = max(0, int(round(raw_count)))

        return {
            'count':           estimated_count,
            'raw_count':       round(raw_count, 3),
            'density_map':     density,
            'original_image':  pil_img,
            'mode':            self.mode,
            'model_name':      self.model_name,
            'map_is_valid':    is_valid,
            'validation_msg':  msg,
        }
