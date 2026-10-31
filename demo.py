"""
CrowdSense — Demo / Smoke-test Script
======================================
Generates four synthetic test images, sends them to the running Flask
server at http://localhost:5000/api/predict, and prints the results.

Prerequisites:
  - The Flask server must already be running:  python app.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import requests
from PIL import Image, ImageDraw
import random

API_URL = 'http://localhost:5000'

# ── Ensure upload dir exists for temp images ──────────────────────────────
os.makedirs('static/uploads', exist_ok=True)


def run_prediction(image_path: str, label: str):
    """POST an image to /api/predict and print the result summary."""
    with open(image_path, 'rb') as f:
        response = requests.post(f'{API_URL}/api/predict',
                                 files={'file': f}, timeout=30)
    if not response.ok:
        print(f"  ❌  HTTP {response.status_code}: {response.text[:200]}")
        return None

    r = response.json()
    mode_label = r.get('mode', '?').upper()
    print(f"  Mode:        {mode_label}")
    print(f"  Crowd Count: {r.get('count')}")
    print(f"  Level:       {r.get('level')}")
    print(f"  Alert:       {'⚠️ YES' if r.get('alert') else '✅ NO'}")
    print(f"  Reliability: {r.get('estimation_reliability', r.get('confidence', '—'))}%")
    print(f"  Status:      {r.get('alert_message')}")
    if r.get('demo_notice'):
        print(f"  ⓘ {r['demo_notice']}")
    return r


def create_test_image(width, height, bg_color, n_dots, dot_color, label):
    """Create a simple synthetic crowd image with coloured dots."""
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    rng = random.Random(42)  # fixed seed for reproducibility
    for _ in range(n_dots):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        r = rng.randint(3, 8)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=dot_color, outline=dot_color)
    draw.text((20, height - 40), label, fill='white')
    return img


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🎯 CROWDSENSE — SMOKE TEST")
print("=" * 70)

# ── Health check ──────────────────────────────────────────────────────────
print("\n📡 Health check...")
try:
    hr = requests.get(f'{API_URL}/api/health', timeout=5)
    h = hr.json()
    print(f"  Status:  {h.get('status')}")
    print(f"  Model:   {h.get('model')}  (mode={h.get('mode')})")
    print(f"  Device:  {h.get('device')}")
    print(f"  DB:      {h.get('database')}")
except Exception as exc:
    print(f"  ❌  Could not reach server: {exc}")
    sys.exit(1)

# ── Test cases ────────────────────────────────────────────────────────────
cases = [
    ("Sparse Crowd",     'lightgreen',  30,  (100, 100, 100), 'sparse_crowd.jpg'),
    ("Moderate Crowd",   'lightyellow', 150, (120, 100, 80),  'moderate_crowd.jpg'),
    ("Dense Crowd",      'lightyellow', 250, (140, 100, 60),  'dense_crowd.jpg'),
    ("Overcrowded",      'lightcoral',  400, (160, 80, 80),   'overcrowded.jpg'),
]

results = []
for i, (title, bg, dots, dot_col, fname) in enumerate(cases, 1):
    print(f"\n📌 TEST CASE {i}: {title.upper()}")
    print("-" * 70)

    path = os.path.join('static', 'uploads', fname)
    img = create_test_image(640, 480, bg, dots, dot_col, f'{title} - ~{dots} people')
    img.save(path)

    r = run_prediction(path, title)
    if r:
        results.append((title, r))

# ── Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📊 SUMMARY TABLE")
print("-" * 70)
print(f"{'Test Case':<25} {'Count':<12} {'Level':<15} {'Alert':<10} {'Mode':<10}")
print("-" * 70)
for title, r in results:
    alert_str = '⚠️ YES' if r.get('alert') else '✅ NO'
    print(f"{title:<25} {r.get('count', '?'):<12} {r.get('level', '?'):<15} {alert_str:<10} {r.get('mode', '?'):<10}")
print("-" * 70)

# ── History check ─────────────────────────────────────────────────────────
print("\n📜 Checking history persistence...")
try:
    hr = requests.get(f'{API_URL}/api/history', timeout=5)
    hd = hr.json()
    print(f"  Total analyses in DB: {hd.get('total_analyses', 0)}")
    print(f"  Entries returned:     {len(hd.get('history', []))}")
except Exception as exc:
    print(f"  ❌  {exc}")

print("\n✨ Smoke test complete!")
