# 👥 CrowdSense — AI-Powered Crowd Density Estimation & Spatial Analytics

A production-ready deep learning system and web application for automated crowd counting, spatial density mapping, temporal surveillance video analysis, and safety risk evaluation.

CrowdSense reformulates discrete object detection into continuous **spatial density-map regression**. Bypassing individual bounding-box detection—which suffers catastrophic failure in congested, highly occluded scenes—CrowdSense maps input images to continuous 2D fractional head distributions. The total crowd estimate is derived through continuous spatial integration over the image domain.

---

## 📌 Benchmark Results (ShanghaiTech Part A)

All metrics below are **empirically measured** on the official untouched ShanghaiTech Part A test set (182 images, crowd counts ranging from 33 to 2,222 people).

### Global Model Performance

| Metric | Trained MCNN | Baseline (Constant Mean) | Heuristic CV Fallback | Relative vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **205.83** | 303.50 | 649.20 | **-32.18%** |
| **Root Mean Squared Error (RMSE)** | **302.98** | 368.69 | 812.45 | **-17.82%** |
| **Mean Absolute Percentage Error (MAPE)** | **58.28%** | 89.14% | 154.20% | **-34.62%** |
| **Median Absolute Error** | **139.92** | 225.38 | 580.00 | **-37.92%** |
| **Pearson Correlation ($r$)** | **0.5515** | 0.0000 | 0.1840 | **Statistically Significant ($p < 0.001$)** |
| **Mean Predicted Count** | **369.19** | 541.38 | 1109.80 | Ground Truth Mean: 431.64 |

### Performance Breakdown by Density Tier

| Ground-Truth Crowd Bucket | Images ($N$) | Mean GT Count | Mean Predicted Count | MAE | RMSE | MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 – 200 (Sparse)** | 43 | 143.9 | 282.6 | 145.58 | 180.28 | 112.6% |
| **200 – 500 (Moderate)** | 94 | 323.9 | 347.8 | **121.60** | **152.81** | **41.0%** |
| **500 – 1000 (Dense)** | 27 | 692.7 | 447.5 | 252.35 | 300.41 | 34.1% |
| **1000+ (Extreme Congestion)** | 18 | 1290.0 | 570.2 | 719.83 | 770.21 | 55.1% |

*Key finding:* The model demonstrates highest relative precision on moderate crowd levels (200–500, MAE 121.60), but exhibits systemic underestimation in extreme congestions ($1000+$ heads), reaching a maximum prediction ceiling of $898.7$ on a scene with $2,221.8$ ground-truth individuals.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client["Frontend Interface (Web / Mobile PWA)"]
        UI["SPA Interface (HTML5 / Vanilla CSS / JavaScript)"]
        Cam["Webcam Live Stream"]
        Vid["Surveillance MP4 / AVI Video Uploader"]
        ZoneTool["Interactive Canvas ROI Box Drawer"]
        ChartUI["Chart.js Real-time Crowd Dynamics"]
    end

    subgraph API["Backend Web Application (Flask REST API)"]
        App["Flask Controller (app.py)"]
        Health["/api/health (Mode, Device, DB)"]
        Predict["/api/predict (TTA Uncertainty, Heatmap, Risk)"]
        PredictVid["/api/predict-video (Frame Sampling, Time Series)"]
        ZoneCount["/api/zone-count (Exact Sub-Region Integration)"]
        DB[(SQLite Persistent Storage: crowdsense.db)]
    end

    subgraph CoreML["Deep Learning Inference Engine (PyTorch)"]
        Predictor["CrowdDensityPredictor (model/crowd_model.py)"]
        MCNN["MCNN (3 Parallel Multi-Scale Columns)"]
        CSRNet["CSRNet (VGG-16 Dilated Conv Back-End)"]
        Fallback["CV Heuristic Fallback Engine (Demo Mode)"]
        TTA["Test-Time Augmentation Engine (Flip, 0.9x, 1.1x)"]
    end

    UI -->|Image Upload| Predict
    Cam -->|Snapshot| Predict
    Vid -->|Video Upload| PredictVid
    ZoneTool -->|ROI Normalized Box [x, y, w, h]| ZoneCount
    
    Predict --> App
    PredictVid --> App
    ZoneCount --> App
    
    App --> Predictor
    Predictor -->|Auto-Detect Arch| MCNN
    Predictor -->|Auto-Detect Arch| CSRNet
    Predictor -.->|Fallback if weights missing| Fallback
    Predictor --> TTA
    
    App --> DB
    App --> UI
    PredictVid --> ChartUI
```

### Supported Neural Architectures

1. **MCNN (Multi-Column Convolutional Neural Network)**
   - Three independent parallel convolutional branches with varying filter kernel sizes ($9\times9$, $7\times7$, $5\times5$) tailored to capture large, medium, and small head scales resulting from perspective geometry.
   - Merged via $1\times1$ convolutional fusion into a continuous density map.
   - Output density layer is linear with clamped positive integration to prevent dead ReLU zero-collapse.
   - Parameter Count: **127,489 parameters** (~510 KB).

2. **CSRNet (Congested Scene Recognition Network)**
   - Front-end: First 10 convolutional layers of ImageNet-pretrained VGG-16 ($3\times3$ filters) for deep feature extraction.
   - Back-end: 6 dilated convolutional layers ($d=2$) expanding receptive fields without resolution loss, resolving dense crowd underestimation.
   - Parameter Count: **16,260,097 parameters** (~65 MB).

---

## 📁 Repository Directory Structure

```
CrowdSense/
├── app.py                      # Production Flask application & REST endpoints
├── config.py                   # Central configuration & hyperparameters
├── train.py                    # Advanced training pipeline (CV, Augment, Schedulers)
├── evaluate_final.py           # Single source of truth benchmark evaluation script
├── crowdsense.db               # SQLite persistent analysis database
├── requirements.txt            # Production dependencies (Torch, OpenCV, Flask)
├── requirements-dev.txt        # Development, testing, and linting dependencies
├── Dockerfile                  # Containerized production deployment
├── .dockerignore               # Docker build exclusions
├── render.yaml                 # Cloud deployment manifest
├── README.md                   # Comprehensive project documentation
├── SCIENTIFIC_REPORT.md        # Formal scientific technical report
├── PRESENTATION_NOTES.md       # 7-Slide presentation outline with figure mappings
│
├── model/
│   ├── __init__.py
│   ├── crowd_model.py          # MCNN, CSRNet & CrowdDensityPredictor classes
│   ├── weights.pth             # Trained weights checkpoint
│   ├── model_metadata.json     # Architecture provenance & parameters
│   └── training_metadata.json  # Loss history and hyperparameter provenance
│
├── results/                    # Ground-truth evaluation outputs & single source of truth
│   ├── metrics.json            # Verified official benchmark evaluation metrics
│   ├── comparison_table.md     # Markdown comparison table (MCNN vs Baselines)
│   ├── scatter_prediction_vs_gt.png # Predicted vs Ground Truth scatter plot (y=x)
│   ├── error_histogram.png     # Absolute error distribution histogram
│   ├── error_by_bucket.png     # Error across crowd density tiers bar chart
│   └── best_worst_predictions.png # Qualitative comparisons (Best 5 vs Worst 5)
│
├── scripts/                    # Utilities, diagnostics & analysis scripts
│   ├── analyze_patch_sampling.py # Patch sampling head-density distribution analysis
│   ├── compare_pipeline.py     # Comparison pipeline against CV baseline
│   ├── demo.py                 # Smoke-test client sending test images to API
│   ├── evaluate.py             # Legacy evaluation script
│   ├── overfit_tiny_test.py    # Micro-batch overfit diagnostic
│   ├── predict.py              # Single-image command-line inference utility
│   ├── verify_architecture.py  # Layer-by-layer tensor dimension validator
│   ├── verify_density_maps.py  # Gaussian kernel integral verification
│   └── view_db.py              # SQLite database CLI viewer
│
├── tests/                      # Automated test suite (21 unit & integration tests)
│   ├── __init__.py
│   ├── test_crowdsense.py      # Core unit tests (math, API, security, TTA, zones)
│   └── smoke_test.py           # 11-step complete pipeline smoke test
│
├── templates/
│   └── index.html              # Frontend user interface (SPA)
│
└── static/
    ├── css/style.css           # UI styling and responsive layouts
    ├── js/script.js            # Video analytics, ROI zone counting, and charts
    └── heatmaps/               # Generated Jet colormap overlay artifacts
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/crowdsense.git
cd crowdsense

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies (CPU or CUDA)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Run the Application

```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

If `model/weights.pth` is present, the app automatically runs in **TRAINED mode**. If weights are absent, it operates in **DEMO mode** (computer vision fallback) with a visible advisory banner.

---

## 🐳 Docker Deployment

The application includes a production-grade Docker configuration:

```bash
# Build the Docker image
docker build -t crowdsense:latest .

# Run container on port 5000
docker run -d -p 5000:5000 --name crowdsense crowdsense:latest

# Verify health
curl http://localhost:5000/api/health
```

---

## 🧠 Model Training

The training script `train.py` supports advanced data augmentation, 5-fold cross-validation, multiple architectures, and count-aware loss functions.

### Command-Line Arguments

| Flag | Options / Default | Description |
| :--- | :--- | :--- |
| `--model` | `mcnn` (default), `csrnet` | Neural network architecture |
| `--loss` | `euclidean` (default), `euclidean+count`, `mse` | Loss function objective |
| `--count-loss-weight`| `0.1` (default) | Weight for auxiliary count error loss |
| `--dataset` | `shanghaitech` (default), `ucf` | Target dataset |
| `--part` | `A` (default), `B` | Dataset partition |
| `--data-root` | `data/ShanghaiTech` | Path to dataset directory |
| `--augment` | Flag (toggle) | Enables horizontal flip, scaling (0.8–1.2), and jitter |
| `--kfold` | Integer (e.g. `5`) | Enables K-fold cross validation split |
| `--fold` | Integer (0 to K-1) | Active validation fold index |
| `--full-train` | Flag (toggle) | Trains on all 300 images without val split |
| `--scheduler` | `cosine` (default), `step` | Learning rate schedule policy |
| `--warmup-epochs` | `5` (default) | Linear warmup epochs before cosine decay |
| `--grad-clip` | `1.0` (default) | Maximum gradient norm clipping |
| `--early-stopping`| `15` (default) | Early stopping patience epochs |

### Example Training Recipes

```bash
# 1. Train MCNN with data augmentation, cosine annealing, and gradient clipping
python train.py --model mcnn --augment --scheduler cosine --grad-clip 1.0 --epochs 50

# 2. Train CSRNet with auxiliary count loss to combat dense-crowd underestimation
python train.py --model csrnet --loss euclidean+count --count-loss-weight 0.1 --augment --epochs 60

# 3. Run 5-fold cross validation (Fold 0)
python train.py --kfold 5 --fold 0 --augment --scheduler cosine
```

---

## 📊 Evaluation & Verification

To run official benchmarks and update `results/metrics.json` and diagnostic plots:

```bash
# Evaluate on ShanghaiTech Part A
python evaluate_final.py --dataset-part A

# Evaluate on ShanghaiTech Part B
python evaluate_final.py --dataset-part B
```

### Running Automated Test Suite

```bash
# Run all 21 unit and integration tests
pytest tests/ -v

# Run 11-step end-to-end pipeline smoke test
python tests/smoke_test.py
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System status, active model, compute device, and operational mode (`trained` or `demo`) |
| `POST` | `/api/predict` | Single image crowd inference, TTA uncertainty bounds, and risk zones |
| `POST` | `/api/predict-video` | Surveillance video processing; samples frames and returns temporal crowd dynamics |
| `POST` | `/api/zone-count` | Exact spatial density integration over a user-drawn ROI box `[x, y, w, h]` |
| `GET` | `/api/history` | Retrieves stored historical analyses from SQLite database |
| `POST` | `/api/history/clear` | Flushes session analysis history |
| `GET` | `/api/model-performance`| Serves single-source-of-truth metrics from `results/metrics.json` |

---

## ⚠️ Limitations & Future Work

### Known Limitations
1. **Dense Crowd Saturation ($1000+$ individuals):**
   MCNN's receptive fields ($9\times9$ max) limit its capacity to distinguish heads in ultra-dense scenes. On images with $>1,000$ people, the model exhibits a mean absolute error of **719.83**, predicting a maximum of $898.7$ on a scene containing $2,221.8$ individuals.
2. **Perspective & Out-of-Distribution Shift:**
   Extreme ground-level perspective distortion (where foreground heads are $100\times$ larger than background heads) causes localized density dispersion.
3. **Video Processing Latency:**
   Video analytics processes 1 frame per second sequentially on CPU; high-frame-rate real-time video surveillance requires dedicated GPU deployment.

### Future Work & Planned Upgrades
- **Full CSRNet GPU Retraining:** Retrain the newly integrated CSRNet architecture on ShanghaiTech Part A using the auxiliary count loss (`--loss euclidean+count`) on a GPU cluster to overcome the 1000+ underestimation ceiling.
- **Context-Aware Regressors:** Integrate CANNet (Context-Aware Network) or Bayesian loss formulation to penalize spatial annotation ambiguities.
- **Temporal Video Smoothing:** Implement Kalman filtering or optical flow tracking across consecutive video frames to prevent count jitter in surveillance streams.
