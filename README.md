# **🏠 RoofScope: Automatic Roof Detection & Analysis**

**Advanced Deep Learning System for Automated Roof Detection and Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat&logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat&logo=opencv)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-orange?style=flat&logo=flask)](https://flask.palletsprojects.com)

## **🎯 Overview**

RoofScope is a comprehensive computer vision system that combines **deep learning** and **traditional CV techniques** to automatically detect and analyze roof structures from satellite imagery. Built for photovoltaic (PV) system planning, the system achieves **97.65% pixel accuracy** and **88.86% IoU score**.

### **Key Capabilities**
- 🤖 **Semantic Segmentation** using U-Net architecture
- 🔍 **Edge Detection** with Canny algorithm for structural analysis
- 📡 **Obstacle Detection** using MSER for identifying solar panels, chimneys
- 📊 **Comprehensive Analysis** with quantitative metrics and visualizations

## **🌟 Features**

### **Deep Learning Pipeline**
- **U-Net Architecture**: Custom implementation with encoder-decoder design
- **Data Handling**: Automated dataset preprocessing and augmentation
- **Model Training**: Advanced optimization with mixed precision training
- **Inference**: Real-time roof segmentation from satellite images

### **Computer Vision Analysis**
- **Edge Detection**: Canny algorithm with optimized parameters
- **Contour Analysis**: Geometric feature extraction from roof boundaries  
- **Obstacle Detection**: MSER-based identification of rooftop objects
- **Visualization**: Multi-panel analysis results with overlays

### **Web Interface**
- **RESTful API**: Flask backend with comprehensive endpoints
- **React Frontend**: Modern UI with drag-and-drop image upload
- **Real-time Processing**: Instant analysis with progress indicators
- **Results Dashboard**: Interactive visualization of analysis metrics

## **📊 Performance Metrics**

| Metric | Score | Description |
|--------|--------|-------------|
| **Pixel Accuracy** | **97.65%** | Proportion of correctly classified pixels |
| **IoU Score** | **88.86%** | Intersection over Union for segmentation quality |
| **Processing Time** | ~2-3 seconds | Average analysis time per image |
| **Dataset Size** | 1,670 images | Aerial rooftop images with ground truth masks |

### **Model Architecture**

U-Net:
- Encoder: 4 downsampling blocks (3→64→128→256→512)
- Bottleneck: 512→1024 channels  
- Decoder: 4 upsampling blocks with skip connections
- Output: 1×1 conv → binary segmentation mask

Training Configuration:
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001)
- Batch Size: 16
- Epochs: 15
- Hardware: Google Colab T4 GPU


## **🛠 Technology Stack**

### **Backend & AI**
- **Python 3.8+**: Core application logic and deep learning implementation
- **PyTorch**: Deep learning framework for U-Net model development
- **OpenCV**: Computer vision operations and image processing
- **Flask**: RESTful API server with CORS support

### **Frontend & Deployment**  
- **React**: Modern component-based user interface
- **Tailwind CSS**: Utility-first styling framework
- **Railway**: Scalable cloud hosting with automatic deployments
- **Vercel**: Optional frontend deployment platform

### **Libraries & Tools**
- **Deep Learning**: PyTorch, torchvision, torch.nn
- **Computer Vision**: OpenCV, scikit-image, PIL
- **Data Processing**: NumPy, pandas, matplotlib
- **Web Framework**: Flask, Flask-CORS
- **Frontend**: React, Tailwind CSS, Lucide Icons

## **🚀 Quick Start**

### **Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/roofscope-portfolio.git
cd roofscope-portfolio

# Install dependencies
pip install -r requirements.txt

# Run the application
python roofscope_app.py
```

### **API Usage**

```python
import requests

# Analyze satellite image
with open('satellite_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze',
        files={'image': f}
    )

results = response.json()
print(f"Roof Coverage: {results['results']['roof_coverage_percent']}%")
print(f"Obstacles Found: {results['results']['obstacles_detected']}")
```
### **Project Structure**

**Main Files:**
- `roofscope_app.py` - Flask API server
- `UNET.py` - U-Net model architecture
- `UNET_blocks.py` - Neural network components
- `Satellite_Dataset.py` - Dataset handling utilities
- `inference.py` - Model inference pipeline
- `model.py` - Model training script
- `requirements.txt` - Python dependencies

**Directories:**
- `frontend/src/` - React frontend component
- `static/examples/` - Sample satellite images  
- `docs/` - Academic paper and documentation
