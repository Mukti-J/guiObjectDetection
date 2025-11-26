# Object Detection Desktop Application 🎯

A powerful desktop application for object detection using YOLOv5 and YOLOv11 models with GPU acceleration support and real-time processing capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## 📋 Table of Contents

- [Features](#features)
- [Supported Classes](#supported-classes)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Image Enhancement Options](#image-enhancement-options)
- [GPU Support](#gpu-support)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Capabilities
- 🖼️ **Image & Video Detection** - Support for multiple image formats (JPG, PNG, BMP, TIFF, WEBP) and video formats (MP4, AVI, MOV, MKV, FLV, WMV)
- 🚀 **GPU Acceleration** - CUDA support for faster processing with automatic CPU fallback
- 🎨 **Image Enhancement** - Pre-processing options to improve detection accuracy
- 📊 **Real-time Object Counting** - Live counter display with responsive scaling
- 🎬 **Video Preview** - Frame-by-frame playback with pause/resume controls
- 💾 **Export Results** - Save annotated images and videos with detection results
- 🎯 **Confidence Threshold** - Adjustable detection sensitivity (0.01 - 1.0)
- 📦 **Multiple Model Support** - YOLOv5 (Nano, Small, Medium, Large) and YOLOv11 (Nano, Small, Medium)

### User Interface
- Modern PyQt5-based GUI with intuitive controls
- Real-time progress tracking and logging
- Responsive layout with scrollable preview area
- Device selection (CPU/GPU) with CUDA information
- Model information display with size and performance metrics

## 🎯 Supported Classes

The application can detect the following animal classes:
- 🐔 Ayam (Chicken)
- 🐷 Babi (Pig)
- 🐐 Kambing (Goat)
- 🦆 Bebek (Duck)
- 🐄 Sapi (Cow)

## 📋 Requirements

### System Requirements
- **Operating System**: Windows, Linux, or macOS
- **Python Version**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended)
- **GPU** (Optional): NVIDIA GPU with CUDA support for acceleration

### Python Dependencies
```
PyQt5>=5.15.0
opencv-python>=4.5.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
numpy>=1.19.0
Pillow>=8.0.0
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/guiObjectDetection.git
cd guiObjectDetection
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
Place your trained model files (`.pt` format) in the `model/` directory. The application supports:
- YOLOv5: `5n.pt`, `5s.pt`, `5m.pt`, `5l.pt`
- YOLOv11: `11n.pt`, `11s.pt`, `11m.pt`

### 5. Run the Application
```bash
python main.py
```

## 📖 Usage

### Using the GUI Application

1. **Select Model**
   - Choose from available YOLO models in the dropdown
   - Model information (size, speed, accuracy) is displayed

2. **Upload File**
   - Click "📁 Pilih Gambar/Video" button
   - Select an image or video file
   - Preview will be shown automatically

3. **Configure Settings**
   - **Image Enhancement**: Select pre-processing option if needed
   - **Confidence Threshold**: Adjust slider (default: 0.25)
   - **Device**: Choose CPU or GPU (if available)

4. **Start Detection**
   - Click "Mulai Deteksi" button
   - Wait for processing to complete
   - View results in the preview area

5. **Download Results**
   - Click "💾 Download Hasil" button
   - Choose save location
   - Results include bounding boxes and object counts

### Command Line Usage

For advanced users, detection can be run directly:

```bash
python detect.py --weights model/5m.pt --source path/to/image.jpg --conf-thres 0.25 --device cpu
```

**Arguments:**
- `--weights`: Path to model file
- `--source`: Path to input image/video
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--enhancement`: Enhancement type (optional)
- `--output-dir`: Output directory (default: runs/detect)
- `--device`: Device to use (cpu, cuda, 0, 1, etc.)

## 📁 Project Structure

```
guiObjectDetection/
├── main.py                 # Main entry point with dependency checks
├── GUI.py                  # PyQt5 GUI implementation
├── detect.py               # Core detection logic
├── config.py               # Configuration and utilities
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model/                 # Model files directory
│   ├── 5n.pt
│   ├── 5s.pt
│   ├── 5m.pt
│   └── ...
└── runs/                  # Output directory
    └── detect/            # Detection results
```

## 🎯 Model Information

### YOLOv5 Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| 5n.pt | 1.9 MB | Very Fast | Low-Medium | Real-time applications |
| 5s.pt | 7.2 MB | Fast | Medium | Balanced performance |
| 5m.pt | 21.2 MB | Medium | Medium-High | **Recommended for most cases** |
| 5l.pt | 46.5 MB | Slow | High | High accuracy requirements |

### YOLOv11 Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| 11n.pt | ~3 MB | Very Fast | Medium | Real-time applications |
| 11s.pt | ~10 MB | Fast | Medium-High | Balanced performance |
| 11m.pt | ~20 MB | Medium | High | High accuracy with reasonable speed |

## 🎨 Image Enhancement Options

Pre-processing options to improve detection accuracy in challenging conditions:

1. **Histogram Equalization**
   - Improves global contrast
   - Best for: Images with uneven lighting

2. **CLAHE (Adaptive Histogram)**
   - Adaptive contrast enhancement
   - Best for: Images with unclear details

3. **Brightness & Contrast**
   - Adjusts brightness and contrast
   - Best for: Images that are too dark or bright

4. **Sharpening**
   - Enhances edges and details
   - Best for: Blurry or soft images

5. **Denoising**
   - Reduces noise and grain
   - Best for: Noisy images from low-light conditions

## 🚀 GPU Support

### Checking GPU Availability
The application automatically detects CUDA-capable GPUs. GPU information is displayed in the device selection panel.

### GPU Requirements
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.0 or higher
- cuDNN library

### Performance Comparison
- **CPU Processing**: ~1-5 FPS (depends on hardware)
- **GPU Processing**: ~20-60 FPS (depends on GPU model)

### Troubleshooting GPU Issues
If GPU is not detected:
1. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Install appropriate PyTorch version with CUDA support
3. Check NVIDIA driver version

## 📸 Screenshots

### Main Interface
*(Add screenshot of main GUI here)*

### Detection Results
*(Add screenshot of detection results here)*

### Video Processing
*(Add screenshot of video processing here)*

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors
- Solo Wandika Putra Manurung @SoloWPM23-
- Mukti Jaenal @Mukti-J-
- Rafael Dito Sihaloho-

**Tugas Akhir Deep Learning G**

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- YOLOv11 by Ultralytics
- PyQt5 for GUI framework
- OpenCV for image processing

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is an educational project for Deep Learning coursework. Model performance may vary depending on training data and use case.
