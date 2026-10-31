# 👥 CrowdSense — AI-Powered Crowd Density Estimation and Monitoring System

A full-stack web application that estimates crowd size in images using deep learning density-map estimation. Upload or capture a crowd photograph and receive an interactive dashboard with head-count estimates, heatmap overlays, risk scoring, and safety recommendations.

---

## 📌 Project Overview

Traditional object detection (YOLO, Faster R-CNN) struggles in highly congested scenes where people overlap or are partially visible. **Density-map estimation** bypasses per-person detection entirely: a convolutional neural network produces a continuous density map where each pixel value represents a fractional person count, and the total crowd estimate is simply the pixel sum.

CrowdSense uses the **MCNN** (Multi-Column CNN) architecture with three parallel branches at different receptive-field scales, allowing it to handle crowds at varying distances from the camera.

### Key Features

| Feature | Description |
|---------|-------------|
| **Crowd Count** | Estimated number of people via MCNN density map |
| **Density Heatmap** | Interactive overlay with compare-slider and opacity control |
| **Density Level** | Low / Medium / High / Very High classification |
| **Overcrowding Alerts** | Automatic warning when count exceeds configured threshold |
| **Estimation Reliability** | Heuristic 0–100 score based on density-map variance and image sharpness |
| **Risk Zone Detection** | Identifies hotspot areas exceeding the 80th-percentile density |
| **Safety Recommendations** | Rule-based advisory cards |
| **Webcam Capture** | Analyse a live frame from device camera |
| **Session History** | Persistent SQLite storage — survives server restarts |
| **Export** | Download JSON, save heatmap PNG, copy text, print report |

---

## 🏗️ Architecture

```
Browser (HTML / CSS / JS / Chart.js)
        ↕ Fetch API
Flask REST API (app.py)
        ↕
PyTorch MCNN Model (model/crowd_model.py)
  + OpenCV (heatmap)
  + Analytics functions
        ↕
SQLite (crowdsense.db) — analysis history
File system — uploaded images + heatmap outputs
```

---

## 📁 Project Structure

```
CrowdSense/
├── app.py                  Flask application (routes + analytics)
├── config.py               Centralised settings and thresholds
├── train.py                MCNN training script (ShanghaiTech / UCF_CC_50)
├── demo.py                 Smoke-test script (requires server running)
├── requirements.txt        Python dependencies
├── render.yaml             Render.com deployment config
├── README.md               This file
│
├── model/
│   ├── __init__.py
│   ├── crowd_model.py      MCNN architecture + CrowdDensityPredictor
│   └── weights.pth         Trained weights (only if training has been run)
│
├── data/
│   └── README.md           Dataset download & layout instructions
│
├── templates/
│   └── index.html          Frontend UI (single-page application)
│
├── static/
│   ├── css/style.css       Styling
│   ├── js/script.js        Frontend logic
│   ├── uploads/            Uploaded images (generated at runtime)
│   └── outputs/            Heatmap outputs (generated at runtime)
│
└── crowdsense.db           SQLite database (generated at runtime)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install PyTorch (optional but recommended)

Without PyTorch the app runs in **Demo mode** — predictions are illustrative only.

```bash
# CPU-only (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA GPU support
pip install torch torchvision
```

---

## ▶️ Running the Application

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Set `true` to enable Flask debug mode |
| `PORT` | `5000` | Server port |

---

## 🧠 Model Training

The MCNN needs to be **trained** before it can produce meaningful predictions. Without `model/weights.pth`, the app runs in Demo mode.

### Supported datasets

| Dataset | Size | Reference |
|---------|------|-----------|
| **ShanghaiTech** (Part A/B) | ~3 000 images, 1.3 M annotations | Zhang et al., CVPR 2016 |
| **UCF_CC_50** | 50 images, 63–4 543 people each | Idrees et al., CVPR 2013 |

### Steps

1. Download the dataset and place it under `data/` (see `data/README.md` for layout).

2. Run training:

```bash
# ShanghaiTech Part A
python train.py --dataset shanghaitech --part A --data-root data/ShanghaiTech

# ShanghaiTech Part B
python train.py --dataset shanghaitech --part B --data-root data/ShanghaiTech

# UCF_CC_50
python train.py --dataset ucf --data-root data/UCF_CC_50
```

3. The best model is saved to `model/weights.pth`. A training log is written to `model/train_log.txt`.

4. Restart the app — it will load the weights automatically and switch from Demo to Trained mode.

> **Note:** MAE / RMSE reported during training are measured on the dataset's own validation split and should not be quoted as general benchmark results without proper evaluation on a held-out test set.

---

## 🔄 Trained vs Demo Mode

| Aspect | Trained Mode | Demo Mode |
|--------|-------------|-----------|
| **When** | `model/weights.pth` exists and loads successfully | Weights missing, corrupt, or PyTorch not installed |
| **Inference** | Real MCNN forward pass | Deterministic synthetic density map (fixed seed) |
| **Output quality** | Depends on training data and epochs | Illustrative only — **not** a real crowd estimate |
| **`/api/health` mode** | `"trained"` | `"demo"` |
| **Dashboard indicator** | **MCNN • Trained Model** | **MCNN • Demo Mode** + yellow banner |

> **Important:** Demo-mode results are clearly labelled in the UI, the API response (`"mode": "demo"`), and the database (`mode` column). They must not be presented as genuine ML predictions.

---

## 📡 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Serve the dashboard HTML |
| `POST` | `/api/predict` | Upload image → crowd analysis → JSON response |
| `GET` | `/api/health` | Server + model + database status |
| `GET` | `/api/history` | Last 10 persisted analyses |

### `POST /api/predict` — Response

```json
{
  "success": true,
  "count": 142,
  "level": "Medium",
  "alert": false,
  "alert_message": "✅ Crowd Level Normal",
  "mode": "trained",
  "estimation_reliability": 72,
  "occupancy_percent": 47.3,
  "density_distribution": { "low": 65.2, "medium": 28.1, "high": 6.7 },
  "risk_zones": { "risk_level": "Moderate", "risk_area_percentage": 8.4 },
  "metrics": { "density_per_sqm": 0.46, "crowd_spread_score": 34.2 },
  "recommendations": [{ "type": "success", "title": "Situation Under Control", "text": "..." }],
  "original_image": "data:image/png;base64,...",
  "heatmap_image": "data:image/png;base64,...",
  "processing_time": 0.54,
  "timestamp": "2026-01-15T10:30:00",
  "demo_notice": null
}
```

### `GET /api/health` — Response

```json
{
  "status": "healthy",
  "model": "MCNN",
  "mode": "trained",
  "weights_loaded": true,
  "device": "cpu",
  "database": "connected"
}
```

---

## 🗄️ Database

CrowdSense uses **SQLite** (`crowdsense.db`) for persistent analysis history.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TEXT | ISO 8601 analysis timestamp |
| `filename` | TEXT | UUID-based stored filename |
| `crowd_count` | INTEGER | Estimated count |
| `density_level` | TEXT | Low / Medium / High / Very High |
| `risk_level` | TEXT | Moderate / High / Critical |
| `mode` | TEXT | `trained` or `demo` |
| `estimation_reliability` | INTEGER | 0–100 heuristic score |
| `processing_time` | REAL | Seconds |

History is accessible via `GET /api/history` and displayed in the sidebar chart.

---

## 🔒 Security

| Measure | Implementation |
|---------|---------------|
| File extension validation | Only jpg, jpeg, png, webp accepted |
| Image integrity check | Pillow `.verify()` on uploaded bytes |
| UUID filenames | Original filenames are never used on disk |
| Upload size limit | 16 MB (configurable in `config.py`) |
| Debug mode off by default | `DEBUG=false`; enable via environment variable |
| JSON error responses | No Python stack traces exposed to users |

---

## ⚠️ Limitations

These are genuine, honest limitations:

1. **Model accuracy depends entirely on training data.** Without `weights.pth`, all predictions are in Demo mode and meaningless.
2. **The estimation-reliability score is heuristic** — it measures density-map variance and image sharpness, not a calibrated ML probability.
3. **Density thresholds are configurable assumptions**, not experimentally validated crowd-safety benchmarks. Tune them for your venue.
4. **No authentication.** Any user with the URL can access all endpoints.
5. **Static images only.** Real-time CCTV/video stream processing is not implemented.
6. **SQLite is local.** History is per-server-instance, not shared across deployments.
7. **No multi-user isolation.** All uploads and history are shared.

---

## 🔮 Future Enhancements

- Train on larger datasets (ShanghaiTech Part A + B combined)
- Upgrade to CSRNet or CANNet architectures
- Real-time video / CCTV stream processing
- User authentication and per-user history
- PostgreSQL or cloud database for shared deployments
- Email/SMS overcrowding alerts
- Mobile-responsive PWA

---

## 📄 License

This project is open source and available under the MIT License.

---

**Built with Python, Flask, PyTorch, OpenCV, and Chart.js**
