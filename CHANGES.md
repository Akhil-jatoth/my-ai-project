# 📝 CrowdSense — Project Evolution & Changes Ledger

**Branch:** `improvements`  
**Standard:** Zero-Fabrication Benchmark Protocol  
**Repository:** `CrowdSense`  

---

## 1. Summary of Changes by Phase

### Phase 1: Training Pipeline Modernization (`train.py`)
- **Added 5-Fold Cross-Validation (`--kfold K`, `--fold F`):** Replaced the static, noisy 30-sample validation split with full dataset coverage and clean cross-validation.
- **Added Full-Train Mode (`--full-train`):** Allows utilizing all 300 ShanghaiTech Part A training images for final deployment weights.
- **Added Data Augmentation (`--augment`):** Horizontal flipping ($p=0.5$), random scaling ($0.8\times$ to $1.2\times$), and brightness/contrast jittering.
- **Ensured Mathematical Count Preservation (`verify_count_preservation`):** Density maps are dynamically rescaled when images undergo scaling or cropping so that $\iint D_{\text{augmented}}(x, y) = \iint D_{\text{original}}(x, y)$.
- **Added Cosine Annealing Learning Rate Scheduler (`--scheduler cosine`):** Replaced static StepLR with Cosine Annealing and an initial 5-epoch linear warmup (`--warmup-epochs 5`).
- **Added Gradient Clipping (`--grad-clip 1.0`):** Enforces a maximum norm ceiling on gradients to eliminate explosive parameter divergence in dense patches.
- **Checkpoint Resilience:** Saves both `model/weights_best.pth` and `model/weights_last.pth`.
- **Training Dynamics Logging:** Outputs per-epoch training dynamics to `training_history.csv` and renders `results/training_curves.png`.

### Phase 2: Neural Architecture Modernization & Count-Aware Objective (`model/crowd_model.py`, `train.py`, `app.py`)
- **Integrated CSRNet (`model/crowd_model.py`):** Added the Congested Scene Recognition Network architecture featuring an ImageNet-pretrained VGG-16 front-end and 6 dilated convolution layers ($d=2$) in the back-end to expand receptive fields without losing spatial resolution.
- **Dynamic Predictor Architecture Detection (`CrowdDensityPredictor`):** Auto-inspects checkpoint state dictionary keys or metadata to dynamically instantiate and load either MCNN or CSRNet.
- **Integrated Count-Aware Auxiliary Loss (`CombinedCountEuclideanLoss`):** Added a global L1 count penalty $\lambda_{\text{count}} |\sum \hat{D} - \sum D_{\text{GT}}|$ (`--loss euclidean+count`, `--count-loss-weight 0.1`) to combat systematic underestimation in high-density scenes.
- **Decoupled Flask Backend (`app.py`):** Removed hard-coded `"MCNN"` strings; dynamically queries `predictor.model_name`.

### Phase 3: Single Source of Truth Evaluation Framework (`evaluate_final.py`, `results/`)
- **Overhauled `evaluate_final.py`:** Unified all benchmark logic into a single authoritative evaluation script. Computes MAE, RMSE, MAPE, Median Absolute Error, and Pearson correlation $r$.
- **Density Tier Analysis:** Computes error breakdowns across 4 distinct ground-truth crowd buckets (0–200, 200–500, 500–1000, 1000+).
- **Automated Plot Generation:** Generates 4 diagnostic visual figures saved directly to `results/`:
  - `results/scatter_prediction_vs_gt.png` (Prediction vs Ground Truth with $y=x$ ideal reference)
  - `results/error_histogram.png` (Distribution of absolute estimation errors)
  - `results/error_by_bucket.png` (Bar chart of MAE and RMSE across crowd density tiers)
  - `results/best_worst_predictions.png` (Qualitative gallery of 5 most accurate and 5 least accurate predictions with GT and predicted density maps side-by-side)
- **Unified Metric Ledger:** Generates `results/metrics.json` and `results/comparison_table.md`.
- **Cleaned Redundant Metric Files:** Deleted legacy files `model/evaluation_metrics.json`, `model_comparison.csv`, `final_accuracy_report.csv`, `final_accuracy_report.json`, and `evaluation_results.csv`.
- **Config Alignment (`config.py`):** Points `EVALUATION_METRICS_PATH` to `results/metrics.json`.
- **API Alignment (`app.py`):** Updated `/api/model-performance` to read directly from `results/metrics.json`.

### Phase 4: Production Video Analytics, ROI Zone Counting, and TTA Uncertainty (`app.py`, `templates/`, `static/`)
- **Surveillance Video Endpoint (`POST /api/predict-video`):** Added video upload and decoding (MP4, AVI, MOV, WEBM) sampling 1 frame per second. Computes crowd density across all frames and returns a temporal time series with peak count and average count.
- **Interactive ROI Zone Counting (`POST /api/zone-count`):** Allows users to draw arbitrary bounding boxes on an image; performs exact mathematical integration over the cached float32 density tensor slice.
- **Empirical Test-Time Augmentation (TTA) Uncertainty:** Evaluates 4 augmentations per inference (original, horizontal flip, $0.9\times$, $1.1\times$) to compute true sample variance and a 95% Student's t-distribution confidence interval ($\text{CI}_{95\%} = \bar{C} \pm 1.591 s$).
- **UI Enhancements (`templates/index.html`, `static/js/script.js`, `static/css/style.css`):**
  - Added "Video" tab with drag-and-drop upload and video player.
  - Added Chart.js temporal crowd timeline for video surveillance analytics.
  - Added interactive canvas overlay with drag-to-draw ROI bounding box and live zone count badge.
  - Added 95% Confidence Interval badge in metric summary.

### Phase 5: Repository Restructuring, Dockerization & CI Automation
- **Relative Path Sanitization:** Replaced all hard-coded Windows paths (`C:\Users\abhignya\...`) with repository-relative paths across `model/model_metadata.json`, `model/training_metadata.json`, and `experiment_summary.json`.
- **Directory Modularization:**
  - Relocated loose utility and diagnostic scripts to `scripts/`: `analyze_patch_sampling.py`, `compare_pipeline.py`, `demo.py`, `evaluate.py`, `overfit_tiny_test.py`, `predict.py`, `verify_architecture.py`, `verify_density_maps.py`, and `view_db.py`.
  - Added `sys.path.insert(0, ...)` to ensure seamless imports regardless of execution directory.
  - Relocated test files to `tests/`: `tests/test_crowdsense.py` and `tests/smoke_test.py`.
- **Dependency Management:**
  - `requirements.txt`: Uncommented and enabled `torch>=2.0.0` and `torchvision>=0.15.0`.
  - `requirements-dev.txt`: Created for scientific and developer tools (`scipy`, `pytest`, `pytest-cov`, `flake8`, `black`).
- **Containerization (`Dockerfile`, `.dockerignore`):** Created production Docker configuration using `python:3.11-slim`, OpenCV dependencies, and Gunicorn WSGI server on port 5000.
- **Continuous Integration (`.github/workflows/ci.yml`):** Added GitHub Actions CI pipeline running on pushes and pull requests across Python 3.10 and 3.11.
- **Operational Mode Visibility:** `/api/health` reports explicit operational mode (`trained` or `demo`); web UI displays prominent Demo Mode banner when weights are absent.

### Phase 6: Professional Scientific & Presentation Documentation
- **Updated `README.md`:** Comprehensive engineering overview with real measured metrics, architecture diagrams, training CLI documentation, and transparent limitations.
- **Updated `SCIENTIFIC_REPORT.md`:** Aligned all numbers with `results/metrics.json`, added CSRNet specifications, TTA uncertainty formulation, and dedicated empirical ablation studies.
- **Created `PRESENTATION_NOTES.md`:** 7-Slide presentation outline with speaking scripts, key takeaways, and exact figure mappings to `results/`.

---

## 2. Complete File Inventory

| File Path | Action | Description |
| :--- | :---: | :--- |
| `train.py` | Modified | Added 5-fold CV, data augmentation, count preservation, cosine annealing, gradient clipping, CSRNet support, and count loss |
| `model/crowd_model.py` | Modified | Added `CSRNet` architecture and dynamic auto-detection in `CrowdDensityPredictor` |
| `evaluate_final.py` | Modified | Single source of truth for evaluation; outputs `metrics.json`, `comparison_table.md`, and 4 diagnostic plots |
| `app.py` | Modified | Added `/api/predict-video`, `/api/zone-count`, TTA uncertainty, dynamic model reporting, and cleaned absolute paths |
| `config.py` | Modified | Configured `EVALUATION_METRICS_PATH`, video upload settings, and path normalization |
| `templates/index.html` | Modified | Added Video input tab, ROI zone selection overlay, and surveillance timeline container |
| `static/js/script.js` | Modified | Added video upload handler, Chart.js timeline, interactive canvas ROI box drawer, and TTA display |
| `static/css/style.css` | Modified | Added styles for video upload container, ROI canvas overlay, and confidence interval badges |
| `tests/test_crowdsense.py` | Moved & Expanded | Relocated to `tests/`; expanded from 17 to 21 tests covering CSRNet, count loss, TTA uncertainty, and zone counting |
| `tests/smoke_test.py` | Moved & Enhanced | Relocated to `tests/`; added synthetic test fixture fallback for self-contained execution |
| `tests/__init__.py` | Created | Test package initialization |
| `scripts/analyze_patch_sampling.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/compare_pipeline.py` | Moved | Relocated to `scripts/`; fixed hardcoded path; added root sys.path |
| `scripts/demo.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/evaluate.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/overfit_tiny_test.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/predict.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/verify_architecture.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/verify_density_maps.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `scripts/view_db.py` | Moved | Relocated to `scripts/`; added root sys.path |
| `model/model_metadata.json` | Modified | Sanitized hard-coded paths to relative paths |
| `model/training_metadata.json` | Modified | Sanitized hard-coded paths to relative paths |
| `experiment_summary.json` | Modified | Sanitized hard-coded paths to relative paths |
| `requirements.txt` | Modified | Enabled PyTorch and Torchvision dependencies |
| `requirements-dev.txt` | Created | Development, testing, and scientific dependencies |
| `Dockerfile` | Created | Containerized production deployment configuration |
| `.dockerignore` | Created | Exclusions for Docker image builds |
| `.github/workflows/ci.yml` | Created | GitHub Actions automated test workflow |
| `results/metrics.json` | Created | Single source of truth benchmark evaluation metrics |
| `results/comparison_table.md` | Created | Markdown comparison table |
| `results/scatter_prediction_vs_gt.png` | Created | Prediction vs Ground Truth scatter plot with $y=x$ line |
| `results/error_histogram.png` | Created | Absolute error distribution histogram |
| `results/error_by_bucket.png` | Created | Error across crowd density tiers bar chart |
| `results/best_worst_predictions.png` | Created | Qualitative gallery of best 5 and worst 5 predictions |
| `README.md` | Modified | Rewritten with verified benchmark metrics, architecture diagrams, and CLI docs |
| `SCIENTIFIC_REPORT.md` | Modified | Updated metrics to match `metrics.json`, added CSRNet, TTA uncertainty, and empirical ablations |
| `PRESENTATION_NOTES.md` | Created | 7-Slide presentation outline with speaking scripts and figure references |
| `CHANGES.md` | Created | Comprehensive file ledger and changelog |

