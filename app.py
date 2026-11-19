"""
CrowdSense — Flask Application & Scientific API
================================================
AI-Powered Crowd Density Estimation and Monitoring System.

Features & Technical Standards:
  - Dual Inference Engine: Trained MCNN (PyTorch) vs. Computer Vision Fallback.
  - Mathematically valid density-map integration without artificial multipliers.
  - Density map validation: non-negativity, NaN/Inf guards, spike filtering.
  - Multi-factor input image quality analysis (resolution, sharpness, brightness, contrast).
  - Normalized Estimation Reliability Score (30-95%) based on image and signal quality.
  - Uncertainty bound indicators (heuristic or MAE-referenced).
  - High-density zone detection with spatial bounding boxes and relative densities.
  - Configurable density thresholds and optional scene capacity input.
  - Persistent SQLite history tracking with detailed timings.
  - Model Performance endpoint (/api/model-performance) surfacing real evaluation metrics.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import uuid
import json
import sqlite3
import logging
import traceback
import numpy as np
from datetime import datetime
from PIL import Image
import base64
import time

from flask import Flask, render_template, request, jsonify

import config

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format='%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('crowdsense')

# ── Flask Application Initialisation ───────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Database (SQLite Persistence)
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialise analyses table with support for scientific tracking fields."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp               TEXT    NOT NULL,
                filename                TEXT    NOT NULL,
                crowd_count             INTEGER NOT NULL,
                raw_count               REAL    DEFAULT 0.0,
                uncertainty_range       INTEGER DEFAULT 0,
                density_level           TEXT    NOT NULL,
                risk_level              TEXT    NOT NULL,
                mode                    TEXT    NOT NULL,
                estimation_reliability  INTEGER NOT NULL,
                scene_capacity          INTEGER DEFAULT NULL,
                occupancy_percent       REAL    DEFAULT NULL,
                processing_time_ms      REAL    NOT NULL
            )
        """)
        # Schema migration check: add columns if migrating from older versions
        cursor = conn.execute("PRAGMA table_info(analyses)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'raw_count' not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN raw_count REAL DEFAULT 0.0")
        if 'uncertainty_range' not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN uncertainty_range INTEGER DEFAULT 0")
        if 'scene_capacity' not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN scene_capacity INTEGER DEFAULT NULL")
        if 'occupancy_percent' not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN occupancy_percent REAL DEFAULT NULL")
        if 'processing_time_ms' not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN processing_time_ms REAL DEFAULT 0.0")
        conn.commit()
    log.info(f"SQLite database ready: {config.DATABASE_PATH}")


def save_analysis(filename: str, crowd_count: int, raw_count: float,
                  uncertainty_range: int, density_level: str, risk_level: str,
                  mode: str, estimation_reliability: int, scene_capacity: int,
                  occupancy_percent: float, processing_time_ms: float):
    try:
        cols = ["timestamp", "filename", "crowd_count", "raw_count", "uncertainty_range",
                "density_level", "risk_level", "mode", "estimation_reliability",
                "scene_capacity", "occupancy_percent"]
        vals = [datetime.now().isoformat(), filename, crowd_count, round(raw_count, 2),
                uncertainty_range, density_level, risk_level, mode, estimation_reliability,
                scene_capacity, occupancy_percent]

        with get_db() as conn:
            cursor = conn.execute("PRAGMA table_info(analyses)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if "processing_time" in existing_cols:
                cols.append("processing_time")
                vals.append(round(processing_time_ms / 1000.0, 3))
            if "processing_time_ms" in existing_cols:
                cols.append("processing_time_ms")
                vals.append(round(processing_time_ms, 2))

            placeholders = ",".join(["?"] * len(vals))
            col_names = ",".join(cols)
            conn.execute(f"INSERT INTO analyses ({col_names}) VALUES ({placeholders})", vals)
            conn.commit()
    except Exception as exc:
        log.error(f"Database write failed: {exc}")


def load_history(limit: int = 10):
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        return [dict(r) for r in rows], total
    except Exception as exc:
        log.error(f"Database read failed: {exc}")
        return [], 0


# ══════════════════════════════════════════════════════════════════════════════
# Model Setup
# ══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    from model.crowd_model import CrowdDensityPredictor, validate_density_map

    predictor = CrowdDensityPredictor(
        weights_path=config.MODEL_WEIGHTS_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    PYTORCH_AVAILABLE = True
    log.info(f"MCNN Predictor: mode={predictor.mode} | device={predictor.device} | weights_loaded={predictor.weights_loaded}")

except ImportError:
    predictor = None
    PYTORCH_AVAILABLE = False
    log.warning("PyTorch not installed. System will operate exclusively in Computer Vision Fallback Mode.")


# ══════════════════════════════════════════════════════════════════════════════
# Scientific & Computer Vision Pipelines
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in config.ALLOWED_EXTENSIONS


def validate_image_file(path: str) -> tuple[bool, str]:
    """Validate image presence, integrity, and basic resolution constraints."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w < config.MIN_IMAGE_DIM or h < config.MIN_IMAGE_DIM:
                return False, f"Image dimensions ({w}x{h}) are too small (minimum {config.MIN_IMAGE_DIM}x{config.MIN_IMAGE_DIM})."
            if w > config.MAX_IMAGE_DIM or h > config.MAX_IMAGE_DIM:
                return False, f"Image dimensions ({w}x{h}) exceed maximum allowable size ({config.MAX_IMAGE_DIM}x{config.MAX_IMAGE_DIM})."
        return True, "Valid image"
    except Exception as exc:
        return False, f"Corrupted or invalid image file: {str(exc)}"


def unique_filename(original: str) -> str:
    ext = original.rsplit('.', 1)[-1].lower() if '.' in original else 'png'
    return f"{uuid.uuid4().hex}.{ext}"


def image_to_base64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def to_json_safe(value):
    """Recursively convert NumPy scalars/arrays to native JSON-serializable types."""
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ── Phase 13: Input Quality Checking ───────────────────────────────────────

def analyze_image_quality(image_path: str) -> dict:
    """
    Evaluate multi-factor quality indicators of the input photograph.
    Computes dimensions, aspect ratio, sharpness (Laplacian variance),
    mean brightness, and contrast (grayscale std).
    """
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(image_path)
    if img is None:
        return {
            "dimensions": [0, 0],
            "aspect_ratio": 0.0,
            "sharpness": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "rating": "Invalid",
            "advisories": ["Failed to decode image pixels."]
        }

    h, w = img.shape[:2]
    aspect = round(float(w) / max(float(h), 1.0), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    advisories = []
    if w < config.QUALITY_THRESHOLDS["min_width"] or h < config.QUALITY_THRESHOLDS["min_height"]:
        advisories.append("Low resolution: small or distant heads may lack sufficient detail for counting.")
    if sharpness < config.QUALITY_THRESHOLDS["blur_laplacian_threshold"]:
        advisories.append("Motion or lens blur detected: edges are degraded.")
    if brightness < config.QUALITY_THRESHOLDS["underexposed_mean"]:
        advisories.append("Severe underexposure: dark scene reduces visible contrast.")
    elif brightness > config.QUALITY_THRESHOLDS["overexposed_mean"]:
        advisories.append("Severe overexposure: washed-out highlights may cause missed detections.")
    if contrast < config.QUALITY_THRESHOLDS["low_contrast_std"]:
        advisories.append("Low dynamic contrast: scene has flat luminance.")

    if len(advisories) == 0:
        rating = "Good"
    elif len(advisories) == 1:
        rating = "Acceptable"
    else:
        rating = "Low Quality"

    return {
        "dimensions": [w, h],
        "aspect_ratio": aspect,
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "rating": rating,
        "advisories": advisories,
    }


# ── Phase 6: Computer Vision Fallback Estimator ────────────────────────────

def generate_cv_fallback_density(image_path: str) -> tuple[np.ndarray, float, int]:
    """
    Computer Vision Fallback Estimator (Heuristic Mode).

    Active ONLY when trained MCNN weights are not available.
    Processes the image using:
      1. Multi-scale Difference of Gaussians (DoG) for head/blob candidate detection.
      2. Non-maximum suppression (NMS) over spatial candidate peaks.
      3. High-frequency edge gradient salience for crowded cluster regions.
      4. 2D Gaussian density distribution mapping.

    Returns:
        density_map : (480, 640) float32 array
        raw_count   : float exact sum
        est_count   : int rounded count
    """
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(image_path)
    if img is None:
        return np.zeros((480, 640), dtype=np.float32), 0.0, 0

    orig_h, orig_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Process at scale normalized for head detection
    scale = min(1.0, 1000.0 / max(orig_h, orig_w))
    proc_w = max(64, int(orig_w * scale))
    proc_h = max(48, int(orig_h * scale))
    small_gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    # Difference of Gaussians for blob candidate extraction
    b1 = cv2.GaussianBlur(small_gray, (5, 5), 1.0)
    b2 = cv2.GaussianBlur(small_gray, (11, 11), 2.5)
    dog = cv2.absdiff(b1, b2).astype(np.float32)

    # Non-maximum suppression over local candidate peaks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(dog, kernel)
    thresh = max(6.0, float(np.percentile(dog, 85)))
    peaks = (dog == dilated) & (dog >= thresh)

    # Connected components over candidate foreground
    fg = (dog > thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)

    valid_mass = 0.0
    density_accum = np.zeros((proc_h, proc_w), dtype=np.float32)

    for i in range(1, num_labels):
        cy = centroids[i, 1]
        if cy < 0.03 * proc_h or cy > 0.97 * proc_h:
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-3)
        if aspect > 7.0:
            continue

        n_pts = max(1.0, area / 35.0)
        valid_mass += n_pts
        cx = int(centroids[i, 0])
        cy = int(centroids[i, 1])
        if 0 <= cy < proc_h and 0 <= cx < proc_w:
            density_accum[cy, cx] += n_pts

    # Calibrated headcount integration
    raw_estimate = (valid_mass * 0.75 + float(np.sum(peaks)) * 0.25) / 6.2
    raw_count = max(0.0, float(raw_estimate))
    calibrated_count = int(round(raw_count))

    # Standardize density map to (480, 640)
    density_accum_resized = cv2.resize(density_accum, (640, 480), interpolation=cv2.INTER_LINEAR)
    density_map = cv2.GaussianBlur(density_accum_resized, (31, 31), 9.0)

    if density_map.sum() > 0:
        density_map = (density_map / density_map.sum()) * float(calibrated_count)
    else:
        density_map = np.full((480, 640), float(calibrated_count) / (480.0 * 640.0), dtype=np.float32)

    return density_map, raw_count, calibrated_count


# ── Phase 7: Estimation Reliability Scoring ────────────────────────────────

def calculate_estimation_reliability(density_map: np.ndarray, quality_info: dict) -> int:
    """
    Calculate normalized 0-100% Estimation Reliability score.

    Mathematical Definition
    -----------------------
    R = w1 * f_sharp + w2 * f_contrast + w3 * f_stability + w4 * f_res

    Each component is strictly bounded in [0.0, 1.0].
    The final percentage is clipped to [25, 95] to prevent unrealistic claims
    of 100% confidence without true ground-truth validation.
    """
    # 1. Sharpness component
    sharpness = quality_info.get('sharpness', 0.0)
    f_sharp = min(1.0, max(0.0, sharpness / config.RELIABILITY_CAPS["sharpness_cap"]))

    # 2. Contrast & Exposure component
    contrast = quality_info.get('contrast', 0.0)
    f_contrast = min(1.0, max(0.0, contrast / config.RELIABILITY_CAPS["contrast_cap"]))
    brightness = quality_info.get('brightness', 128.0)
    f_exposure = 1.0 - (abs(brightness - 128.0) / 128.0)
    f_exposure_contrast = max(0.0, 0.6 * f_contrast + 0.4 * f_exposure)

    # 3. Density map signal stability
    dm_min, dm_max = float(density_map.min()), float(density_map.max())
    if dm_max - dm_min > 1e-6:
        dm_var = float(np.var(density_map))
        f_stability = min(1.0, max(0.0, dm_var / config.RELIABILITY_CAPS["variance_cap"]))
    else:
        f_stability = 0.4

    # 4. Image resolution factor
    w, h = quality_info.get('dimensions', [640, 480])
    f_res = min(1.0, max(0.0, float(w * h) / (640.0 * 480.0)))

    # Weighted combination
    w_s = config.RELIABILITY_WEIGHTS["sharpness"]
    w_c = config.RELIABILITY_WEIGHTS["exposure_contrast"]
    w_d = config.RELIABILITY_WEIGHTS["density_stability"]
    w_r = config.RELIABILITY_WEIGHTS["resolution"]

    score = (w_s * f_sharp) + (w_c * f_exposure_contrast) + (w_d * f_stability) + (w_r * f_res)
    reliability_pct = int(round(score * 100.0))
    return max(25, min(95, reliability_pct))


# ── Phase 8: Uncertainty Bounds ────────────────────────────────────────────

def calculate_uncertainty(count: int, reliability: int, is_trained: bool, eval_mae: float = None) -> tuple[int, tuple[int, int]]:
    """
    Compute uncertainty bounds for the estimated crowd count.
    Returns:
        delta       : int (uncertainty +/- value)
        bound_range : (lower_bound, upper_bound)
    """
    if is_trained and eval_mae is not None and eval_mae > 0:
        delta = max(2, int(round(eval_mae)))
    else:
        rel_frac = float(reliability) / 100.0
        # Heuristic error scale: lower reliability -> wider uncertainty bound
        scale_factor = 0.08 + 0.22 * (1.0 - rel_frac)
        delta = max(3, int(round(count * scale_factor)))

    lower = max(0, count - delta)
    upper = count + delta
    return delta, (lower, upper)


# ── Phase 9: Density Classification ────────────────────────────────────────

def classify_density(count: float) -> str:
    """Classify crowd density using centralised configuration thresholds."""
    if count < config.DENSITY_THRESHOLDS["low"]:
        return 'Low'
    if count < config.DENSITY_THRESHOLDS["medium"]:
        return 'Medium'
    if count < config.DENSITY_THRESHOLDS["high"]:
        return 'High'
    return 'Very High'


# ── Phase 11: High-Density Zone Detection ──────────────────────────────────

def detect_high_density_zones(density_map: np.ndarray) -> list[dict]:
    """
    Identify spatial clusters of concentrated density.

    Divides the continuous map into local spatial regions, thresholds upper
    density percentiles, merges adjacent cells, and calculates bounding boxes.
    Termed "High-Density Zone" rather than unvalidated "Dangerous Zone".
    """
    dm_min, dm_max = float(density_map.min()), float(density_map.max())
    if dm_max - dm_min < 1e-7:
        return []

    norm = (density_map - dm_min) / (dm_max - dm_min)
    thresh = float(norm.mean() + 1.2 * norm.std())
    thresh = max(0.40, min(0.85, thresh))
    mask = (norm >= thresh).astype(np.uint8)

    # Morphological closing to merge adjacent density cells
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    total_area = float(density_map.size)

    for i, c in enumerate(contours, 1):
        area = cv2.contourArea(c)
        if area < 50:  # Filter out single-pixel noise
            continue
        x, y, w, h = cv2.boundingRect(c)
        sub_density = float(norm[y:y+h, x:x+w].mean())
        severity = 'Critical' if sub_density > 0.8 else ('High' if sub_density > 0.6 else 'Moderate')

        zones.append({
            'zone_id': i,
            'label': f'High-Density Zone {i}',
            'severity': severity,
            'relative_density': round(sub_density, 2),
            'area_percentage': round(area / total_area * 100.0, 2),
            'bbox': [int(x), int(y), int(w), int(h)],
        })
    return zones


# ── Phase 12: Heatmap Overlay Generation ───────────────────────────────────

def generate_heatmap_overlay(original_path: str, density_map: np.ndarray, output_path: str) -> bool:
    """
    Render Jet colormap heatmap overlay onto original photograph.
    Guaranteed finite bounds and divide-by-zero protection.
    """
    try:
        img = cv2.imread(original_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        dm_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)

        dm_min, dm_max = float(dm_resized.min()), float(dm_resized.max())
        if dm_max - dm_min < 1e-7:
            dm_norm = np.zeros((h, w), dtype=np.uint8)
        else:
            dm_norm = np.uint8(255.0 * np.clip((dm_resized - dm_min) / (dm_max - dm_min), 0.0, 1.0))

        heatmap = cv2.applyColorMap(dm_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.60, heatmap, 0.40, 0)
        cv2.imwrite(output_path, overlay)
        return True
    except Exception as exc:
        log.error(f"Heatmap generation error: {exc}")
        return False


def get_density_distribution(density_map: np.ndarray) -> dict:
    dm_min, dm_max = float(density_map.min()), float(density_map.max())
    if dm_max - dm_min < 1e-7:
        return {'low': 100.0, 'medium': 0.0, 'high': 0.0}
    norm = (density_map - dm_min) / (dm_max - dm_min)
    total = float(norm.size)
    return {
        'low':    round(float(np.sum(norm < 0.33) / total * 100.0), 1),
        'medium': round(float(np.sum((norm >= 0.33) & (norm < 0.66)) / total * 100.0), 1),
        'high':   round(float(np.sum(norm >= 0.66) / total * 100.0), 1),
    }


def generate_recommendations(crowd_count: int, level: str, alert: bool,
                             high_density_zones: list, scene_capacity: int, occupancy_pct: float) -> list[dict]:
    recs = []
    if alert:
        recs.append({
            'type': 'critical',
            'title': 'High Crowd Density Alert',
            'text': 'Estimated crowd exceeds safe operating parameters. Deploy personnel to active high-density zones.'
        })
    elif level in ('High', 'Very High'):
        recs.append({
            'type': 'warning',
            'title': 'Elevated Density',
            'text': 'Crowd concentration is approaching capacity thresholds. Increase observation frequency.'
        })

    crit_zones = [z for z in high_density_zones if z['severity'] == 'Critical']
    if crit_zones:
        recs.append({
            'type': 'critical',
            'title': 'Hotspot Chokepoint Detected',
            'text': f"{len(crit_zones)} localized area(s) show high crowd concentration. Ensure exit channels remain unobstructed."
        })

    if occupancy_pct is not None and occupancy_pct >= 85.0:
        recs.append({
            'type': 'warning',
            'title': 'Venue Capacity Threshold Reached',
            'text': f"Venue occupancy is currently at {occupancy_pct}% of configured capacity limit."
        })

    if not recs:
        recs.append({
            'type': 'success',
            'title': 'Normal Density Distribution',
            'text': 'Observed crowd patterns are within standard operating ranges.'
        })
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# Application Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Multipart form-data:
      - 'file': Image file (JPG, PNG, WEBP)
      - 'scene_capacity': (Optional) Integer venue capacity limit
    """
    t_start = time.perf_counter()
    timings = {}
    request_id = uuid.uuid4().hex[:8]

    # 1. File verification
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided in request.'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f"Disallowed file extension. Supported formats: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        }), 400

    # 2. Save with secure UUID filename
    safe_name = unique_filename(file.filename)
    upload_path = os.path.join(config.UPLOAD_FOLDER, safe_name)
    try:
        file.save(upload_path)
    except Exception as exc:
        log.error(f"[{request_id}] Save failed: {exc}")
        return jsonify({'success': False, 'error': 'Failed to save uploaded file.'}), 500

    # 3. File integrity & quality check
    is_valid_img, val_msg = validate_image_file(upload_path)
    if not is_valid_img:
        os.remove(upload_path)
        return jsonify({'success': False, 'error': val_msg}), 400

    t_prep_start = time.perf_counter()
    quality_info = analyze_image_quality(upload_path)
    timings['preprocessing_ms'] = round((time.perf_counter() - t_prep_start) * 1000.0, 2)

    # 4. Optional Scene Capacity Parsing (Phase 10)
    scene_capacity = None
    capacity_raw = request.form.get('scene_capacity')
    if capacity_raw:
        try:
            c_val = int(capacity_raw.strip())
            if c_val > 0:
                scene_capacity = c_val
        except ValueError:
            pass

    # 5. Dual-Mode Inference
    t_infer_start = time.perf_counter()
    try:
        if PYTORCH_AVAILABLE and predictor is not None and predictor.trained_model:
            result = predictor.predict(upload_path)
            density_map = result['density_map']
            crowd_count = result['count']
            raw_count   = result['raw_count']
            mode        = 'trained'
            model_info  = {
                "name": config.MODEL_NAME,
                "version": config.MODEL_VERSION,
                "type": "Neural Network (MCNN)",
                "weights_loaded": True,
                "device": predictor.device
            }
        else:
            density_map, raw_count, crowd_count = generate_cv_fallback_density(upload_path)
            mode        = 'demo'
            model_info  = {
                "name": "Computer Vision Fallback",
                "version": "1.0.0",
                "type": "Heuristic Feature Extractor",
                "weights_loaded": False,
                "device": "cpu"
            }
    except Exception as exc:
        log.error(f"[{request_id}] Inference error: {traceback.format_exc()}")
        if os.path.exists(upload_path):
            os.remove(upload_path)
        return jsonify({'success': False, 'error': f'Inference execution failed: {str(exc)}'}), 500

    timings['inference_ms'] = round((time.perf_counter() - t_infer_start) * 1000.0, 2)

    # 6. Analytics computation
    t_analytics_start = time.perf_counter()
    level = classify_density(crowd_count)
    reliability = calculate_estimation_reliability(density_map, quality_info)

    eval_mae = None
    if os.path.exists(config.EVALUATION_METRICS_PATH):
        try:
            with open(config.EVALUATION_METRICS_PATH, 'r') as ef:
                eval_data = json.load(ef)
                eval_mae = eval_data.get('mae')
        except Exception:
            pass

    uncertainty_delta, uncertainty_bounds = calculate_uncertainty(
        crowd_count, reliability, is_trained=(mode == 'trained'), eval_mae=eval_mae
    )

    high_density_zones = detect_high_density_zones(density_map)
    distribution = get_density_distribution(density_map)

    # Phase 10: Capacity & Occupancy logic
    if scene_capacity is not None:
        occupancy_percent = round((crowd_count / float(scene_capacity)) * 100.0, 1)
        capacity_status = f"{scene_capacity} (Configured)"
        capacity_configured = True
        alert = (occupancy_percent >= 90.0) or (level == 'Very High')
    else:
        occupancy_percent = None
        capacity_status = "Not configured"
        capacity_configured = False
        alert = crowd_count >= config.OVERCROWDING_ALERT_THRESHOLD

    alert_msg = "⚠️ HIGH CROWD DENSITY DETECTED!" if alert else "✅ Density Within Standard Range"
    recommendations = generate_recommendations(
        crowd_count, level, alert, high_density_zones, scene_capacity, occupancy_percent
    )
    timings['analytics_ms'] = round((time.perf_counter() - t_analytics_start) * 1000.0, 2)

    # 7. Heatmap Visualization
    t_viz_start = time.perf_counter()
    heatmap_name = safe_name.rsplit('.', 1)[0] + '_heatmap.png'
    heatmap_path = os.path.join(config.OUTPUT_FOLDER, heatmap_name)
    heatmap_ok = generate_heatmap_overlay(upload_path, density_map, heatmap_path)
    if not heatmap_ok:
        import shutil
        shutil.copy2(upload_path, heatmap_path)

    try:
        original_b64 = image_to_base64(upload_path)
        heatmap_b64  = image_to_base64(heatmap_path)
    except Exception as exc:
        log.error(f"[{request_id}] Base64 encoding error: {exc}")
        return jsonify({'success': False, 'error': 'Failed to encode visualization.'}), 500
    timings['visualization_ms'] = round((time.perf_counter() - t_viz_start) * 1000.0, 2)

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    timings['total_ms'] = total_time_ms

    # 8. Database Persistence
    t_db_start = time.perf_counter()
    save_analysis(
        filename=safe_name,
        crowd_count=crowd_count,
        raw_count=raw_count,
        uncertainty_range=uncertainty_delta,
        density_level=level,
        risk_level=high_density_zones[0]['severity'] if high_density_zones else 'Moderate',
        mode=mode,
        estimation_reliability=reliability,
        scene_capacity=scene_capacity,
        occupancy_percent=occupancy_percent,
        processing_time_ms=total_time_ms,
    )
    timings['database_ms'] = round((time.perf_counter() - t_db_start) * 1000.0, 2)

    log.info(
        f"[{request_id}] Mode={mode} | Count={crowd_count} | Level={level} | "
        f"Reliability={reliability}% | Uncertainty=+/-{uncertainty_delta} | TotalTime={total_time_ms}ms"
    )

    # 9. Structured Response (Phase 20) with backward-compatible top-level keys
    response = {
        "success": True,
        "request_id": request_id,
        "mode": mode,
        "model": model_info,

        # Standardized prediction structure
        "prediction": {
            "estimated_count": crowd_count,
            "raw_count": raw_count,
            "uncertainty_delta": uncertainty_delta,
            "uncertainty_bounds": uncertainty_bounds,
            "density_level": level,
            "estimation_reliability": reliability,
        },

        # Scene capacity & occupancy structure (Phase 10)
        "scene_capacity": {
            "configured": capacity_configured,
            "capacity": scene_capacity,
            "occupancy_percent": occupancy_percent,
            "status": capacity_status,
        },

        # Spatial high-density zones (Phase 11)
        "high_density_zones": high_density_zones,
        "risk_zones": {
            "count": len(high_density_zones),
            "zones": high_density_zones,
            "risk_level": high_density_zones[0]['severity'] if high_density_zones else 'Moderate',
        },

        # Input image quality (Phase 13)
        "image_quality": quality_info,

        # Detailed breakdown of processing timings (Phase 18)
        "processing_time": timings,

        # Recommendations & distribution
        "recommendations": recommendations,
        "density_distribution": distribution,
        "alert": alert,
        "alert_message": alert_msg,

        # Visualizations
        "original_image": f"data:image/png;base64,{original_b64}",
        "heatmap_image":  f"data:image/png;base64,{heatmap_b64}",
        "timestamp": datetime.now().isoformat(),

        # Backward compatibility aliases for existing UI and scripts
        "count": crowd_count,
        "level": level,
        "confidence": reliability,
        "estimation_reliability": reliability,
        "uncertainty": f"±{uncertainty_delta}",
        "occupancy_percent": occupancy_percent if occupancy_percent is not None else 0.0,

        "demo_notice": (
            "Computer Vision Fallback estimate — not validated against a trained crowd-counting model."
            if mode == 'demo' else None
        )
    }

    return jsonify(to_json_safe(response))


@app.route('/api/health', methods=['GET'])
def health():
    """
    GET /api/health
    Returns operational state, model metadata, weights availability,
    and database connectivity status.
    """
    if PYTORCH_AVAILABLE and predictor is not None:
        model_status = predictor.status
    else:
        model_status = {
            'model_name': config.MODEL_NAME,
            'model_version': config.MODEL_VERSION,
            'weights_loaded': False,
            'trained_model': False,
            'device': 'cpu',
            'mode': 'demo',
            'training_meta': None,
        }

    try:
        with get_db() as conn:
            conn.execute("SELECT 1")
        db_status = 'connected'
    except Exception:
        db_status = 'error'

    weights_loaded = model_status['weights_loaded']
    eval_exists = os.path.exists(config.EVALUATION_METRICS_PATH)
    if weights_loaded:
        if eval_exists:
            validation_status = "MCNN — Trained & Evaluated"
        else:
            validation_status = "MCNN — Trained / Evaluation pending"
    else:
        validation_status = "MCNN — Weights unavailable"

    return jsonify({
        'status':            'healthy',
        'model':             model_status['model_name'],
        'model_version':     config.MODEL_VERSION,
        'mode':              model_status['mode'],
        'weights_loaded':    model_status['weights_loaded'],
        'validation_status': validation_status,
        'device':            model_status['device'],
        'database':          db_status,
        'training_meta':     model_status.get('training_meta'),
        'model_loaded':      model_status['weights_loaded'],
        'device_label':      'CUDA (GPU)' if model_status['device'] == 'cuda' else 'CPU',
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """GET /api/history — Retrieve up to 10 most recent persistent analyses."""
    rows, total = load_history(limit=10)
    history = []
    for r in rows:
        history.append({
            'id':                     r.get('id'),
            'timestamp':              r.get('timestamp'),
            'filename':               r.get('filename'),
            'count':                  r.get('crowd_count'),
            'raw_count':              r.get('raw_count', 0.0),
            'uncertainty_range':      r.get('uncertainty_range', 0),
            'level':                  r.get('density_level'),
            'confidence':             r.get('estimation_reliability'),
            'estimation_reliability': r.get('estimation_reliability'),
            'scene_capacity':         r.get('scene_capacity'),
            'occupancy_percent':      r.get('occupancy_percent'),
            'mode':                   r.get('mode'),
            'processing_time':        round(r.get('processing_time_ms', 0.0) / 1000.0, 2),
            'processing_time_ms':     r.get('processing_time_ms', 0.0),
        })
    return jsonify({'history': history, 'total_analyses': total})


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear all analysis history records from the database."""
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM analyses")
            conn.commit()
            conn.execute("VACUUM")
        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/model-performance', methods=['GET'])
def model_performance():
    """
    GET /api/model-performance (Phase 5)
    Returns real benchmark evaluation metrics if evaluate.py has been executed,
    or clearly indicates that evaluation has not been conducted.
    Never fabricates metrics.
    """
    if os.path.exists(config.EVALUATION_METRICS_PATH):
        try:
            with open(config.EVALUATION_METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        except Exception as exc:
            return jsonify({
                "evaluated": False,
                "error": f"Failed to read evaluation metrics file: {str(exc)}"
            }), 500

    return jsonify({
        "evaluated": False,
        "model": config.MODEL_NAME,
        "message": "Model evaluation has not been performed yet. Run python evaluate.py on a test dataset to generate verified MAE/RMSE metrics."
    })


# ── Error Handlers ────────────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'success': False,
        'error': f'Payload exceeds maximum allowable file size of {config.MAX_FILE_SIZE // (1024*1024)} MB.'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Requested API endpoint not found.'}), 404


@app.errorhandler(500)
def internal_error(e):
    log.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server processing error.'}), 500


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    app.run(
        debug=config.DEBUG,
        host='0.0.0.0',
        port=config.PORT,
    )
else:
    init_db()
