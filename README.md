# 👥 AI-Powered Crowd Density Estimation and Monitoring System

A full-stack web application that automatically estimates the number of people in crowded images using Deep Learning and Computer Vision techniques.

## 📌 Project Overview

This system uses **density map estimation** instead of traditional object detection to predict crowd counts even in highly congested environments where people overlap or are partially visible.

### Key Features

✅ **Crowd Count Estimation** - AI-based people counting
✅ **Density Heatmap Visualization** - Visual representation of crowd density
✅ **Crowd Density Classification** - Low/Medium/High/Very High levels
✅ **Overcrowding Alerts** - Real-time overcrowding detection
✅ **Interactive Web Interface** - User-friendly dashboard

## 🏗️ Architecture

```
User Uploads Image
          ↓
Frontend (HTML/CSS/JS)
          ↓
Flask Backend API
          ↓
Image Preprocessing
          ↓
Crowd Counting Model (MCNN)
          ↓
Density Map Generation
          ↓
Count Calculation
          ↓
Results Returned
          ↓
Display Count + Heatmap
```

## 🛠️ Tech Stack

### Backend
- **Flask** - Web framework
- **PyTorch** - Deep Learning framework
- **OpenCV** - Computer Vision library
- **NumPy** - Numerical computing

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling
- **JavaScript (Vanilla)** - Interactivity

### Machine Learning
- **MCNN** (Multi-Column CNN) - For crowd density estimation
- **CNN Architecture** - Multiple columns with different receptive fields

## 📁 Project Structure

```
crowd-density-estimator/
│
├── app.py                          # Flask application
│
├── model/
│   └── crowd_model.py              # MCNN model & prediction logic
│
├── templates/
│   └── index.html                  # Frontend UI
│
├── static/
│   ├── css/
│   │   └── style.css               # Styling
│   ├── js/
│   │   └── script.js               # Frontend logic
│   ├── uploads/                    # User uploaded images
│   └── outputs/                    # Generated heatmaps
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- 2GB+ available RAM for model
- GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crowd-density-estimator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained model (Optional)**
   ```bash
   # Place downloaded model weights in:
   model/weights.pth
   ```

### Running the Application

1. **Start Flask server**
   ```bash
   python app.py
   ```

2. **Open browser**
   Navigate to: `http://localhost:5000`

3. **Upload image and analyze**
   - Click on upload area or drag & drop image
   - Click "Analyze Image" button
   - View results and density heatmap

## 🧠 How It Works

### Why Density Estimation?

Traditional object detection (YOLO, R-CNN) struggle with:
- People overlapping each other
- Partially visible faces
- Individuals appearing very small
- Occlusion causing missed detections

Density estimation solves these by:
- Creating a density map of the crowd
- Calculating total count by summing pixels
- Working effectively even with heavy occlusion

### Model Architecture (MCNN)

```
Input Image (640×480)
        ↓
┌───────┼───────┐
│       │       │
↓       ↓       ↓
Col1   Col2   Col3     (3 parallel columns)
(9×9)  (7×7)  (5×5)    (different kernel sizes)
│       │       │
└───────┼───────┘
        ↓
  Concatenate
        ↓
Density Map (1 channel)
        ↓
Count = Sum(pixels)
```

### Density Level Classification

| Count    | Level      |
|----------|-----------|
| < 50     | Low       |
| 50-150   | Medium    |
| 150-300  | High      |
| > 300    | Very High |

## 📊 API Endpoints

### 1. Upload & Predict
**POST** `/api/predict`

**Request:**
```
Content-Type: multipart/form-data
Body: file (image file)
```

**Response:**
```json
{
  "success": true,
  "count": 487,
  "level": "High",
  "alert": true,
  "alert_message": "⚠️ OVERCROWDING DETECTED!",
  "original_image": "data:image/png;base64,...",
  "heatmap_image": "data:image/png;base64,..."
}
```

### 2. Health Check
**GET** `/api/health`

**Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "model_loaded": true
}
```

## 🎨 Frontend Features

### Dashboard
- **Crowd Count** - Estimated number of people
- **Density Level** - Classification (Low/Medium/High/Very High)
- **Alert Status** - Overcrowding indicator
- **Alert Banner** - Visual alert for overcrowding

### Image Comparison
- **Original Image** - User uploaded image
- **Density Heatmap** - AI-generated heatmap overlay

### Controls
- **Drag & Drop** - Upload images by dragging
- **Analyze Button** - Start prediction
- **Download Results** - Save analysis as JSON
- **Reset Button** - Clear and start new analysis

## 🔧 Configuration

### app.py Settings

```python
# Upload folder
UPLOAD_FOLDER = 'static/uploads'

# Output folder
OUTPUT_FOLDER = 'static/outputs'

# Max file size
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Model device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Server port
PORT = 5000
```

## 📈 Model Performance

### Training Data Recommendations
- **ShanghaiTech Dataset** - 3,000 images, 1.3M people
- **UCF_CC_50** - 50 images, 63-4453 people
- **WorldExpo Dataset** - 2,415 images

### Expected Accuracy
- **MAE** (Mean Absolute Error): ±15-20 people
- **MSE** (Mean Squared Error): ±30-50 people

## 🚀 Future Enhancements

### Video Processing
- Frame-by-frame analysis
- Real-time crowd tracking
- Temporal density patterns

### Live CCTV Monitoring
- Camera feed integration
- Continuous monitoring
- Event logging

### Advanced Analytics
- Peak hour analysis
- Crowd trend prediction
- Historical comparisons
- Statistical reports

### Model Improvements
- CSRNet architecture
- CANNet for better accuracy
- Multi-scale feature fusion

## 📚 Training (Optional)

To train with your own dataset:

```python
# trainer.py (create this file)
from model.crowd_model import MCNN
import torch
import torch.optim as optim

model = MCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop here
for epoch in range(num_epochs):
    for images, density_maps in train_loader:
        outputs = model(images)
        loss = criterion(outputs, density_maps)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model/weights.pth')
```

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution:** Use CPU mode or reduce batch size
```python
device = 'cpu'
```

### Issue: Model not loading
**Solution:** Ensure `model/weights.pth` exists or initialize new model

### Issue: Images not uploading
**Solution:** Check file size limit and allowed formats

### Issue: Slow predictions
**Solution:** Use GPU acceleration or reduce image size

## 📄 Resume Description

*AI-Powered Crowd Density Estimation and Monitoring System*

Developed a full-stack web application using HTML, CSS, JavaScript, Flask, PyTorch, and OpenCV to estimate crowd counts in dense scenes using deep-learning-based density map estimation. Implemented heatmap visualization, crowd density classification, and overcrowding alerts. Integrated frontend, backend APIs, and machine learning inference into a deployable web platform.

### Skills Demonstrated
- ✅ Machine Learning & Deep Learning
- ✅ Computer Vision & Image Processing
- ✅ Backend Development (Flask, REST APIs)
- ✅ Frontend Development (HTML, CSS, JavaScript)
- ✅ Full-Stack Integration
- ✅ Model Deployment

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: your-email@example.com

## 🎓 Learning Resources

- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Flask Docs**: https://flask.palletsprojects.com/
- **OpenCV Docs**: https://docs.opencv.org/
- **CNN Basics**: https://cs231n.github.io/convolutional-networks/

---

**Built with ❤️ for Computer Vision & Machine Learning**

⭐ If this project helped you, please give it a star!
