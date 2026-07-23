import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import requests
import numpy as np
from PIL import Image, ImageDraw
import random

print("\n" + "="*70)
print("🎯 CROWD DENSITY ESTIMATION SYSTEM - LIVE DEMONSTRATION")
print("="*70)

# Test Case 1: SPARSE CROWD
print("\n📌 TEST CASE 1: SPARSE CROWD")
print("-" * 70)

width, height = 640, 480
img = Image.new('RGB', (width, height), color='lightgreen')
draw = ImageDraw.Draw(img)

# Draw sparse crowd (30 people)
for i in range(30):
    x = random.randint(0, width)
    y = random.randint(0, height)
    radius = random.randint(3, 8)
    color = (100, 100, 100)
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color)

draw.text((20, height-40), 'Sparse Crowd - ~30 people', fill='white')
img.save('static/uploads/sparse_crowd.jpg')

with open('static/uploads/sparse_crowd.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', files={'file': f})
result = response.json()

print(f"  Input: Sparse crowd image")
print(f"  Crowd Count: {result['count']} people")
print(f"  Density Level: {result['level']}")
print(f"  Alert: {'⚠️ YES' if result['alert'] else '✅ NO'}")
print(f"  Status: {result['alert_message']}")

# Test Case 2: MODERATE CROWD
print("\n📌 TEST CASE 2: MODERATE CROWD")
print("-" * 70)

img = Image.new('RGB', (width, height), color='lightyellow')
draw = ImageDraw.Draw(img)

for i in range(150):
    x = random.randint(0, width)
    y = random.randint(0, height)
    radius = random.randint(3, 8)
    color = (120, 100, 80)
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color)

draw.text((20, height-40), 'Moderate Crowd - ~150 people', fill='black')
img.save('static/uploads/moderate_crowd.jpg')

with open('static/uploads/moderate_crowd.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', files={'file': f})
result = response.json()

print(f"  Input: Moderate crowd image")
print(f"  Crowd Count: {result['count']} people")
print(f"  Density Level: {result['level']}")
print(f"  Alert: {'⚠️ YES' if result['alert'] else '✅ NO'}")
print(f"  Status: {result['alert_message']}")

# Test Case 3: DENSE CROWD
print("\n📌 TEST CASE 3: DENSE CROWD")
print("-" * 70)

img = Image.new('RGB', (width, height), color='lightyellow')
draw = ImageDraw.Draw(img)

for i in range(250):
    x = random.randint(0, width)
    y = random.randint(0, height)
    radius = random.randint(3, 8)
    color = (140, 100, 60)
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color)

draw.text((20, height-40), 'Dense Crowd - ~250 people', fill='white')
img.save('static/uploads/dense_crowd.jpg')

with open('static/uploads/dense_crowd.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', files={'file': f})
result = response.json()

print(f"  Input: Dense crowd image")
print(f"  Crowd Count: {result['count']} people")
print(f"  Density Level: {result['level']}")
print(f"  Alert: {'⚠️ YES' if result['alert'] else '✅ NO'}")
print(f"  Status: {result['alert_message']}")

# Test Case 4: VERY DENSE CROWD (OVERCROWDED)
print("\n📌 TEST CASE 4: VERY DENSE CROWD (OVERCROWDED)")
print("-" * 70)

img = Image.new('RGB', (width, height), color='lightcoral')
draw = ImageDraw.Draw(img)

for i in range(400):
    x = random.randint(0, width)
    y = random.randint(0, height)
    radius = random.randint(2, 6)
    color = (160, 80, 80)
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color)

draw.text((20, height-40), 'OVERCROWDED - 400+ people', fill='white')
img.save('static/uploads/overcrowded.jpg')

with open('static/uploads/overcrowded.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', files={'file': f})
result = response.json()

print(f"  Input: Overcrowded image")
print(f"  Crowd Count: {result['count']} people")
print(f"  Density Level: {result['level']}")
print(f"  Alert: {'⚠️ YES' if result['alert'] else '✅ NO'}")
print(f"  Status: {result['alert_message']}")

print("\n" + "="*70)
print("✅ ALL TEST CASES COMPLETED SUCCESSFULLY!")
print("="*70)

# Show summary
print("\n📊 SUMMARY TABLE")
print("-" * 70)
print(f"{'Test Case':<25} {'Count':<12} {'Level':<15} {'Alert':<10}")
print("-" * 70)

test_cases = [
    ('Sparse', 'static/uploads/sparse_crowd.jpg'),
    ('Moderate', 'static/uploads/moderate_crowd.jpg'),
    ('Dense', 'static/uploads/dense_crowd.jpg'),
    ('Overcrowded', 'static/uploads/overcrowded.jpg'),
]

for name, path in test_cases:
    with open(path, 'rb') as f:
        response = requests.post('http://localhost:5000/api/predict', files={'file': f})
    result = response.json()
    alert_str = '⚠️ YES' if result['alert'] else '✅ NO'
    print(f"{name:<25} {result['count']:<12} {result['level']:<15} {alert_str:<10}")

print("-" * 70)
print("\n✨ Demonstration Complete!")
