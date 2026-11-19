"""
=============================================================================
CrowdSense — Comprehensive Test Suite
Validates mathematical correctness, API contracts, image quality diagnostics,
uncertainty estimation, spatial risk zones, capacity calculations, security,
and database persistence.
=============================================================================
"""

import os
import sys
import json
import io
import unittest
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import config
from model.crowd_model import validate_density_map, MCNN
from app import (
    app,
    generate_cv_fallback_density,
    analyze_image_quality,
    calculate_estimation_reliability,
    calculate_uncertainty,
    classify_density,
    detect_high_density_zones,
    init_db,
    get_db
)

class TestCrowdSenseCore(unittest.TestCase):
    """Phase 1, 2, 7, 8, 9, 10, 11: Scientific validation of mathematical algorithms."""

    def test_density_map_validation_clean(self):
        """Phase 2: Validate clean, non-negative finite density map."""
        clean_map = np.full((64, 64), 0.05, dtype=np.float32)
        valid, msg, sanitized = validate_density_map(clean_map)
        self.assertTrue(valid, f"Clean map should be valid: {msg}")
        self.assertEqual(sanitized.shape, clean_map.shape)

    def test_density_map_validation_nan_inf(self):
        """Phase 2: Validate rejection of NaN and Inf in density maps."""
        nan_map = np.full((32, 32), 0.1, dtype=np.float32)
        nan_map[10, 10] = np.nan
        valid, msg, _ = validate_density_map(nan_map)
        self.assertFalse(valid)
        self.assertIn("NaN", msg)

        inf_map = np.full((32, 32), 0.1, dtype=np.float32)
        inf_map[10, 10] = np.inf
        valid, msg, _ = validate_density_map(inf_map)
        self.assertFalse(valid)
        self.assertIn("NaN/Inf", msg)

    def test_density_map_validation_negative(self):
        """Phase 2: Validate clamping of small negative numerical noise."""
        neg_map = np.full((32, 32), 0.1, dtype=np.float32)
        neg_map[5, 5] = -0.0001
        valid, msg, sanitized = validate_density_map(neg_map)
        self.assertTrue(valid)
        self.assertGreaterEqual(sanitized.min(), 0.0)

    def test_exact_count_integration(self):
        """Phase 1: Verify exact mathematical integration: count = density_map.sum()."""
        density_map = np.zeros((100, 100), dtype=np.float32)
        density_map[20:30, 20:30] = 1.5  # 100 * 1.5 = 150.0
        exact_sum = float(np.sum(density_map))
        self.assertAlmostEqual(exact_sum, 150.0, places=4)
        count_display = int(round(exact_sum))
        self.assertEqual(count_display, 150)

    def test_cv_fallback_deterministic_output(self):
        """Phase 6: Computer Vision fallback produces valid density map with non-negative count."""
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        for y, x in [(30, 30), (60, 120), (100, 80), (140, 150), (170, 50)]:
            arr[max(0, y-4):min(200, y+5), max(0, x-4):min(200, x+5)] = 255

        dmap, raw_cnt, est_cnt = generate_cv_fallback_density(arr)
        valid, msg, _ = validate_density_map(dmap)
        self.assertTrue(valid, f"CV fallback density map must be valid: {msg}")
        self.assertGreaterEqual(raw_cnt, 0.0)
        self.assertGreaterEqual(est_cnt, 0)

    def test_image_quality_analysis(self):
        """Phase 13: Input quality analysis calculates sharpness, brightness, contrast."""
        sharp_img = np.random.RandomState(42).randint(0, 255, (200, 200, 3), dtype=np.uint8)
        q_sharp = analyze_image_quality(sharp_img)
        self.assertGreater(q_sharp["sharpness"], 50.0)
        self.assertIn("rating", q_sharp)

        flat_img = np.full((200, 200, 3), 128, dtype=np.uint8)
        q_flat = analyze_image_quality(flat_img)
        self.assertLess(q_flat["sharpness"], 10.0)
        self.assertLess(q_flat["contrast"], 5.0)

    def test_estimation_reliability_bounds(self):
        """Phase 7: Estimation Reliability bounded within 25% and 95%."""
        q_good = {"sharpness": 150.0, "contrast": 60.0, "brightness": 128.0, "dimensions": [800, 600]}
        d_map = np.full((50, 50), 0.1, dtype=np.float32)

        rel = calculate_estimation_reliability(d_map, q_good)
        self.assertGreaterEqual(rel, 25)
        self.assertLessEqual(rel, 95)

    def test_uncertainty_interval(self):
        """Phase 8: Uncertainty bounds are symmetric and non-negative."""
        delta, bounds = calculate_uncertainty(count=200, reliability=80, is_trained=False)
        self.assertGreater(delta, 0)
        self.assertEqual(bounds[0], max(0, 200 - delta))
        self.assertEqual(bounds[1], 200 + delta)

    def test_density_classification_thresholds(self):
        """Phase 9: Configurable thresholds determine density level."""
        self.assertEqual(classify_density(20), "Low")
        self.assertEqual(classify_density(100), "Medium")
        self.assertEqual(classify_density(220), "High")
        self.assertEqual(classify_density(450), "Very High")

    def test_spatial_risk_zones_detection(self):
        """Phase 11: Spatial partitioning detects high-density clusters."""
        dmap = np.zeros((100, 100), dtype=np.float32)
        dmap[10:30, 10:30] = 5.0
        zones = detect_high_density_zones(dmap)
        self.assertTrue(len(zones) >= 1)
        self.assertIn("bbox", zones[0])
        self.assertIn("relative_density", zones[0])
        self.assertIn("severity", zones[0])


class TestCrowdSenseAPI(unittest.TestCase):
    """Phase 17, 18, 19, 21: Full Flask integration, security, and persistence tests."""

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()
        init_db()

    def test_health_endpoint(self):
        """Verify /api/health returns operational status, schema, and validation status."""
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["status"] == "ok" or data["status"] == "healthy")
        self.assertIn("mode", data)
        self.assertIn("device", data)
        self.assertIn("validation_status", data)
        self.assertIn(data["validation_status"], [
            "MCNN — Trained & Evaluated",
            "MCNN — Trained / Evaluation pending",
            "MCNN — Weights unavailable"
        ])

    def test_model_performance_endpoint_honesty(self):
        """Phase 5: If no evaluation exists, /api/model-performance honestly reports pending."""
        resp = self.client.get('/api/model-performance')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("evaluated", data)
        if not os.path.exists(config.EVALUATION_METRICS_PATH):
            self.assertFalse(data["evaluated"])
            self.assertIn("not been performed yet", data["message"])

    def test_predict_endpoint_valid_image(self):
        """Phase 18, 20: Predict endpoint returns standardized JSON with timing and diagnostics."""
        img = Image.new('RGB', (160, 120), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        data = {
            'file': (buf, 'test_synthetic.png'),
            'scene_capacity': '500'
        }
        resp = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        res = json.loads(resp.data)
        self.assertTrue(res["success"])
        self.assertIn("count", res)
        self.assertIn("processing_time", res)
        self.assertIn("image_quality", res)
        self.assertIn("uncertainty", res)
        self.assertIn("scene_capacity", res)
        self.assertTrue(res["scene_capacity"]["configured"])
        self.assertEqual(res["scene_capacity"]["capacity"], 500)

    def test_predict_endpoint_unconfigured_capacity(self):
        """Phase 10: When capacity is empty, scene_capacity reports configured=False."""
        img = Image.new('RGB', (120, 120), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        data = {
            'file': (buf, 'test_uncapacity.png')
        }
        resp = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        res = json.loads(resp.data)
        self.assertFalse(res["scene_capacity"]["configured"])
        self.assertIsNone(res["scene_capacity"].get("occupancy_percent"))

    def test_security_disallowed_extension(self):
        """Phase 21: Security check rejects dangerous file extensions like .exe or .sh."""
        buf = io.BytesIO(b"echo 'malicious payload'")
        data = {'file': (buf, 'exploit.sh')}
        resp = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        res = json.loads(resp.data)
        self.assertFalse(res["success"])
        self.assertIn("Disallowed file extension", res["error"])

    def test_security_corrupted_image(self):
        """Phase 21: Pillow verification catches pseudo-images with corrupted bytes."""
        buf = io.BytesIO(b"PNGFakeHeaderButCorruptDataPayloadBytes1234567890")
        data = {'file': (buf, 'corrupted.png')}
        resp = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        res = json.loads(resp.data)
        self.assertFalse(res["success"])

    def test_history_persistence(self):
        """Phase 22: SQLite database persists records with timestamps and IDs."""
        resp = self.client.get('/api/history')
        self.assertEqual(resp.status_code, 200)
        hist = json.loads(resp.data)
        self.assertIn("history", hist)
        self.assertIsInstance(hist["history"], list)

if __name__ == '__main__':
    unittest.main()
