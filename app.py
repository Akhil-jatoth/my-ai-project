import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import base64
import time
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Analysis history tracking
analysis_history = []

# Try to import torch, if not available use demo mode
try:
    import torch
    from model.crowd_model import CrowdDensityPredictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = CrowdDensityPredictor(device=device)
    DEMO_MODE = False
    print(f"✅ Model loaded | Using device: {device}")
except ImportError:
    DEMO_MODE = True
    predictor = None
    print("⚠️  Running in DEMO MODE (PyTorch not installed - showing mocked predictions)")


def to_json_safe(value):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def generate_heatmap_overlay(original_image_path, density_map, output_path):
    """
    Generate heatmap overlay on original image
    """
    # Read original image
    img = cv2.imread(original_image_path)
    if img is None:
        return None
    
    # Resize density map to match image dimensions
    height, width = img.shape[:2]
    density_map_resized = cv2.resize(density_map, (width, height))
    
    # Normalize density map to 0-255
    density_normalized = np.uint8(
        255 * (density_map_resized - density_map_resized.min()) / 
        (density_map_resized.max() - density_map_resized.min() + 1e-5)
    )
    
    # Apply colormap (Jet)
    heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
    
    # Blend with original image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Save result
    cv2.imwrite(output_path, overlay)
    
    return output_path


def generate_mock_density_map(image_path):
    """Generate a more realistic mock density map for demo mode"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    img_area = height * width
    
    # Convert to HSV for skin tone detection
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect approximate skin tone regions (rough estimation)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    
    # Also detect orange/red tones which could be people
    lower_skin2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(img_hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
    
    # Calculate percentage of image that's likely people
    skin_pixels = np.count_nonzero(skin_mask)
    coverage_ratio = skin_pixels / img_area if img_area > 0 else 0.3
    
    # Create a density map with Gaussian blobs based on skin detection
    density_map = np.zeros((height, width), dtype=np.float32)
    
    # Number of clusters based on coverage
    num_clusters = max(2, min(8, int(coverage_ratio * 100)))
    
    # Gaussian amplitudes scaled based on coverage
    max_amplitude = max(5, min(30, int(coverage_ratio * 200)))
    
    for _ in range(num_clusters):
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        sigma = np.random.randint(20, 60)
        amplitude = np.random.randint(max(2, max_amplitude - 10), max_amplitude)
        
        y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        density_map += gaussian * amplitude
    
    # Normalize and scale to realistic crowd estimates
    # Typical: 0-100 = sparse, 100-300 = moderate, 300-1000 = dense, 1000+ = overcrowded
    density_map = density_map.astype(np.float32)
    
    # Scale based on image area and coverage
    scale_factor = max(0.5, coverage_ratio * 3)
    density_map = density_map * scale_factor
    
    return density_map


def calculate_confidence_score(density_map, image_path):
    """Calculate AI confidence score for the prediction"""
    img = cv2.imread(image_path)
    
    # Factors affecting confidence:
    # 1. Density map variance (higher = more confident)
    variance = np.var(density_map)
    
    # 2. Image quality
    laplacian = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    image_quality = min(100, np.var(laplacian) / 10)
    
    # 3. Density distribution (well-distributed = more confident)
    histogram = np.histogram(density_map.flatten(), bins=10)[0]
    distribution_score = np.std(histogram) * 2
    
    # Final confidence score (0-100%)
    confidence = min(100, (variance + image_quality + distribution_score) / 3)
    confidence = max(50, confidence)  # Minimum 50% confidence
    
    return int(confidence)


def detect_risk_zones(density_map):
    """Detect high-risk zones in the crowd"""
    # Normalize density map
    normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
    
    # Find hotspots (top 20% density areas)
    threshold = np.percentile(normalized, 80)
    risk_zones = np.where(normalized > threshold)
    
    risk_percentage = len(risk_zones[0]) / density_map.size * 100
    
    return {
        'risk_zones_count': len(np.unique(risk_zones[0])),
        'risk_area_percentage': round(risk_percentage, 2),
        'max_density': float(density_map.max()),
        'avg_density': float(density_map.mean()),
        'risk_level': 'Critical' if risk_percentage > 30 else 'High' if risk_percentage > 15 else 'Moderate'
    }


def get_density_distribution(density_map):
    """Break down image area into low, medium, and high density zones"""
    normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
    total = normalized.size
    return {
        'low': round(float(np.sum(normalized < 0.33) / total * 100), 1),
        'medium': round(float(np.sum((normalized >= 0.33) & (normalized < 0.66)) / total * 100), 1),
        'high': round(float(np.sum(normalized >= 0.66) / total * 100), 1)
    }


def generate_recommendations(crowd_count, level, alert, risk_data, metrics):
    """Generate actionable safety recommendations based on analysis"""
    recommendations = []

    if alert:
        recommendations.append({
            'type': 'critical',
            'title': 'Immediate Action Required',
            'text': 'Crowd exceeds safe threshold. Deploy additional security personnel and consider redirecting foot traffic to alternate routes.'
        })
    elif level in ('High', 'Very High'):
        recommendations.append({
            'type': 'warning',
            'title': 'Elevated Crowd Levels',
            'text': 'Density is approaching unsafe limits. Increase monitoring frequency and prepare crowd-control measures.'
        })

    risk_level = risk_data.get('risk_level', 'Moderate')
    if risk_level == 'Critical':
        recommendations.append({
            'type': 'critical',
            'title': 'Critical Hotspots Detected',
            'text': f'{risk_data.get("risk_area_percentage", 0)}% of the area shows critical density. Focus resources on bottleneck zones immediately.'
        })
    elif risk_level == 'High':
        recommendations.append({
            'type': 'warning',
            'title': 'High-Risk Zones Identified',
            'text': 'Multiple areas are approaching capacity. Prepare contingency plans for crowd redistribution.'
        })

    if metrics.get('growth_potential', 100) < 25:
        recommendations.append({
            'type': 'info',
            'title': 'Near Maximum Capacity',
            'text': f'Only ~{metrics.get("available_space", 0)}% usable space remains. Consider restricting new entries.'
        })

    if metrics.get('crowd_spread_score', 0) > 65:
        recommendations.append({
            'type': 'info',
            'title': 'Dispersed Crowd Pattern',
            'text': 'Crowd is widely scattered across the area. Plan exit routes and guide movement toward designated zones.'
        })

    if not recommendations:
        recommendations.append({
            'type': 'success',
            'title': 'Situation Under Control',
            'text': 'Crowd density is within safe operating limits. Continue routine monitoring at standard intervals.'
        })

    return recommendations


def calculate_advanced_metrics(crowd_count, density_map, image_path):
    """Calculate advanced analytics metrics"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    img_area = height * width
    
    # Density per square meter (approximate)
    density_density = crowd_count / (img_area / 10000) if img_area > 0 else 0
    
    # Crowd spread (how scattered vs concentrated)
    normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
    spread_score = min(100, np.std(normalized) * 100)
    
    # Growth potential (if this area continues filling)
    max_possible = int(crowd_count * 2)
    growth_potential = ((max_possible - crowd_count) / max_possible) * 100 if max_possible > 0 else 0
    
    return {
        'density_per_sqm': round(density_density, 2),
        'crowd_spread_score': round(spread_score, 1),
        'growth_potential': round(growth_potential, 1),
        'max_capacity_estimate': max_possible,
        'available_space': round(growth_potential / 2, 1)
    }


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for crowd density prediction
    Expects multipart form data with 'file' field
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Run prediction
        print(f"Processing image: {filename}")
        start_time = time.time()

        if DEMO_MODE:
            # Generate mock prediction for demo
            density_map = generate_mock_density_map(upload_path)
            
            # Better scaling for more realistic predictions
            base_count = int(np.sum(density_map) / 50)  # Scale to realistic range
            
            # Add some randomness but keep within reasonable bounds
            crowd_count = max(10, int(base_count + np.random.randint(-base_count // 10, base_count // 10)))
            
            # Determine density level with improved thresholds
            if crowd_count < 100:
                level = 'Low'
            elif crowd_count < 500:
                level = 'Medium'
            elif crowd_count < 1500:
                level = 'High'
            else:
                level = 'Very High'
            
            alert = crowd_count > 1200
            alert_msg = '⚠️ OVERCROWDING DETECTED!' if alert else '✅ Crowd Level Normal'
        else:
            # Use actual model prediction
            prediction_result = predictor.predict(upload_path)
            crowd_count = prediction_result['count']
            level = prediction_result['level']
            alert = prediction_result['alert']
            density_map = prediction_result['density_map']
            alert_msg = prediction_result.get('alert_message', '✅ Crowd Level Normal')
        
        # Generate heatmap
        heatmap_filename = filename.rsplit('.', 1)[0] + '_heatmap.png'
        heatmap_path = os.path.join(app.config['OUTPUT_FOLDER'], heatmap_filename)
        
        generate_heatmap_overlay(upload_path, density_map, heatmap_path)
        
        # Convert images to base64
        original_base64 = image_to_base64(upload_path)
        heatmap_base64 = image_to_base64(heatmap_path)
        
        # Calculate advanced metrics
        confidence = calculate_confidence_score(density_map, upload_path)
        risk_data = detect_risk_zones(density_map)
        metrics = calculate_advanced_metrics(crowd_count, density_map, upload_path)
        processing_time = round(time.time() - start_time, 2)
        recommendations = generate_recommendations(crowd_count, level, alert, risk_data, metrics)
        density_distribution = get_density_distribution(density_map)
        occupancy_pct = round(min(100, crowd_count / metrics['max_capacity_estimate'] * 100), 1) if metrics['max_capacity_estimate'] > 0 else 0

        # Add to history
        analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'count': crowd_count,
            'level': level,
            'confidence': confidence,
            'alert': alert,
            'processing_time': processing_time
        })

        # Prepare response with advanced analytics
        response = {
            'success': True,
            'count': crowd_count,
            'level': level,
            'alert': alert,
            'original_image': f"data:image/png;base64,{original_base64}",
            'heatmap_image': f"data:image/png;base64,{heatmap_base64}",
            'alert_message': alert_msg,

            # Advanced Analytics
            'confidence': confidence,
            'risk_zones': risk_data,
            'metrics': metrics,
            'recommendations': recommendations,
            'density_distribution': density_distribution,
            'occupancy_percent': occupancy_pct,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        mode = "DEMO" if DEMO_MODE else "PRODUCTION"
        print(f"[{mode}] Prediction - Count: {crowd_count}, Level: {level}, Confidence: {confidence}%, Alert: {alert}")
        
        return jsonify(to_json_safe(response))
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    return jsonify({
        'history': analysis_history[-10:],  # Last 10 analyses
        'total_analyses': len(analysis_history)
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': 'CPU (Demo Mode)' if DEMO_MODE else 'CUDA/CPU',
        'model_loaded': not DEMO_MODE,
        'mode': 'DEMO' if DEMO_MODE else 'PRODUCTION'
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 50MB'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
