"""
CrowdSense — Centralised Configuration & Scientific Parameters
==============================================================
All application paths, upload limits, density classification thresholds,
quality scoring parameters, and reproducibility seeds are defined here.

NOTE ON DENSITY & CAPACITY THRESHOLDS
-------------------------------------
Crowd density categories (Low, Medium, High, Very High) are heuristic
thresholds configured for general monitoring purposes. They are NOT universal
laws of physics and should be adjusted to match the spatial geometry and safety
standards of the specific monitored venue.
"""

import os

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Directory Layout ───────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER       = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER       = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_DIR           = os.path.join(BASE_DIR, 'model')
MODEL_WEIGHTS_PATH  = os.path.join(MODEL_DIR, 'weights.pth')
TRAINING_META_PATH  = os.path.join(MODEL_DIR, 'training_metadata.json')
EVALUATION_METRICS_PATH = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
DATABASE_PATH       = os.path.join(BASE_DIR, 'crowdsense.db')
DATASET_PATH        = os.path.join(BASE_DIR, 'data', 'ShanghaiTech')

# ── Model Metadata ─────────────────────────────────────────────────────────
MODEL_NAME    = "MCNN"
MODEL_VERSION = "1.0.0"

# ── Upload & Security Settings ─────────────────────────────────────────────
MAX_FILE_SIZE       = 16 * 1024 * 1024          # 16 MB limit
ALLOWED_EXTENSIONS  = {'jpg', 'jpeg', 'png', 'webp'}

# Minimum and maximum allowable image dimensions for processing
MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 8192

# ── Configurable Crowd-Density Classification Thresholds ───────────────────
# Categorization based on estimated headcount.
DENSITY_THRESHOLDS = {
    "low": 50,       # count < 50        -> "Low"
    "medium": 150,   # 50 <= count < 150 -> "Medium"
    "high": 300,     # 150 <= count < 300-> "High"
                     # count >= 300      -> "Very High"
}

# Base overcrowding alert headcount (triggers advisory notification)
OVERCROWDING_ALERT_THRESHOLD = 300

# Default scene/venue capacity limit (None = capacity not configured)
DEFAULT_SCENE_CAPACITY = None

# ── Input Image Quality Analysis Thresholds ────────────────────────────────
QUALITY_THRESHOLDS = {
    "min_width": 320,
    "min_height": 240,
    "blur_laplacian_threshold": 60.0,    # below this indicates substantial blur
    "underexposed_mean": 35.0,           # below this indicates very dark image
    "overexposed_mean": 225.0,           # above this indicates washed-out image
    "low_contrast_std": 20.0,            # below this indicates low dynamic range
}

# ── Estimation Reliability Normalization Constants ─────────────────────────
# Factors used to compute a normalized 0-100% estimation reliability score.
# This represents signal quality and stability, NOT a statistical probability.
RELIABILITY_CAPS = {
    "sharpness_cap": 1200.0,    # Laplacian variance upper bound
    "variance_cap": 40.0,       # Density map variance upper bound
    "contrast_cap": 80.0,       # Standard deviation of luminance upper bound
}

# Weights for reliability composition (sum = 1.0)
RELIABILITY_WEIGHTS = {
    "sharpness": 0.35,
    "exposure_contrast": 0.25,
    "density_stability": 0.25,
    "resolution": 0.15,
}

# ── Server ─────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT  = int(os.getenv("PORT", "5000"))
