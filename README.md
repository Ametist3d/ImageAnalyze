# 🎨 Image Analysis System

Advanced image analysis system with automatic color extraction and object detection, featuring an interactive web UI.

## 🏗️ Architecture

```
📁 Project Structure
├── 🎨 auto_color_extractor.py    # Intelligent color extraction with visualization
├── 🔍 object_detector.py         # Multi-method object detection  
├── 🌐 gradio_ui.py              # Interactive web interface
├── 🚀 app.py                    # Simple launcher
├── 🚀 main.py                   # Command-line interface
├── 📋 requirements.txt          # Dependencies
└── 📖 README.md                # This file
```

## ✨ Features

### 🌐 **Web UI (New!)**
- **Interactive Gradio interface** - Upload images via drag & drop
- **Real-time color scheme visualization** - Proportional color bars
- **Dynamic n_colors slider** - Adjust extraction on the fly
- **Tabbed results** - Separate color analysis and object detection
- **Auto-analysis** - Instant results when uploading images
- **Responsive design** - Works on desktop and mobile

### 🎨 **Enhanced Color Extraction**
- **Proportional color schemes** - Visual representation of color frequencies
- **Multiple output formats** - RGB, HEX, color names, percentages
- **Automatic image analysis** - Detects image type and selects optimal strategy
- **Adaptive K-means clustering** - Different approaches for different image types
- **Smart color naming** - Improved RGB→name conversion

### 🔍 **Object Detection**
- **Multiple detection methods** with automatic fallback
- **Confidence filtering** - Adjustable detection threshold
- **Rich output** - Object names, confidence scores, bounding boxes

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Launch Web UI** (Recommended)
```bash
python app.py
```
Then open http://localhost:7860 in your browser

### 3. **Command Line Usage**
```bash
python main.py
```

## 📱 **Web UI Usage**

1. **Upload Image**: Drag & drop or click to upload
2. **Adjust Settings**: Use the n_colors slider (2-10 colors)
3. **Extract Colors**: Click "Extract Colors" or it happens automatically
4. **Detect Objects**: Click "Detect Objects" for AI object detection
5. **Analyze Everything**: Click "Analyze Everything" for complete analysis

### **UI Features:**
- 🎨 **Proportional Color Scheme**: Visual bars showing color frequency
- 📊 **Detailed Analysis**: Image type, strategy used, confidence scores
- 🔍 **Object Detection Results**: Objects found with confidence percentages
- ⚡ **Real-time Updates**: Changes when you adjust the n_colors slider

## 🛠️ Installation Options

### **Option A: Complete Installation (Recommended)**
```bash
pip install gradio numpy opencv-python scikit-learn pillow ultralytics matplotlib
```

### **Option B: Step by Step**
```bash
# Core dependencies
pip install numpy opencv-python pillow scikit-learn

# Web UI
pip install gradio

# Object detection (choose one)
pip install ultralytics                    # YOLOv8 (recommended)
# OR
pip install transformers torch torchvision  # DETR/Transformers
# OR  
pip install torch torchvision              # Faster R-CNN only
```

### **Option C: Requirements File**
```bash
pip install -r requirements.txt
```

## 📊 **API Usage**

### **Color Extraction with Visualization**
```python
from auto_color_extractor import AutoColorExtractor

extractor = AutoColorExtractor(n_colors=5)

# Get color scheme image + data
color_scheme_img, results = extractor.build_color_scheme("image.jpg")

# Get detailed palette data
palette_data = extractor.get_color_palette_data("image.jpg")

for color in palette_data["colors"]:
    print(f"{color['name']}: {color['hex']} ({color['frequency']:.1f}%)")
```

### **Object Detection**
```python
from object_detector import ObjectDetector

detector = ObjectDetector(confidence_threshold=0.5)
results = detector.detect_objects("image.jpg")

for obj in results["detections"]:
    print(f"{obj['object']}: {obj['confidence']:.1f}%")
```

### **Combined Analysis**
```python
from gradio_ui import ImageAnalysisUI

ui = ImageAnalysisUI()
color_scheme, color_analysis, object_analysis = ui.analyze_complete(image, n_colors=6)
```

## 🎯 **Color Scheme Visualization**

The system now creates **proportional color bars** showing the relative frequency of each dominant color:

```python
# Create a color scheme visualization
extractor = AutoColorExtractor(n_colors=5)
color_scheme_image, results = extractor.build_color_scheme("image.jpg", width=400, height=100)

# Returns:
# - PIL Image with proportional color bars
# - Complete analysis results with percentages
```

**Output Example:**
- Red: 35% → Takes up 35% of the color bar width
- Blue: 25% → Takes up 25% of the color bar width  
- Green: 20% → Takes up 20% of the color bar width
- etc.

## 🔧 **Customization**

### **Web UI Settings**
```python
# In app.py, modify launch settings:
launch_ui(
    share=True,   # Create public link
    debug=True    # Enable debug mode
)
```

### **Color Extraction Settings**
```python
extractor = AutoColorExtractor(n_colors=8)  # More colors

# Custom color scheme size
color_scheme, results = extractor.build_color_scheme(
    "image.jpg", 
    width=600,   # Wider scheme
    height=150   # Taller bars
)
```

### **Object Detection Settings**
```python
detector = ObjectDetector(confidence_threshold=0.3)  # Lower threshold
```

## 📊 **Example Output**

### **Web UI Results:**
```
🎨 COLOR ANALYSIS

Image Type: artistic_gradient
Strategy: filtered_vibrant  
Confidence: 95.0%

Dominant Colors:
1. 🟠 orange - #f49f86 (23.4%)
2. 🔵 teal - #5b8fae (21.8%) 
3. 🔴 red - #90281b (19.2%)
4. 🟠 orange - #d66b36 (18.9%)
5. ⚫ dark_gray - #20293b (16.7%)
```

```
🔍 OBJECT DETECTION (YOLOv8)

Found 2 objects:
1. 👜 handbag - 78.5% confidence
2. 🪑 chair - 65.2% confidence
```

## 🌟 **Key Improvements**

| Feature | Before | Now |
|---------|--------|-----|
| **Interface** | Command line only | Interactive web UI |
| **Color Visualization** | Text list | Proportional color bars |
| **Usability** | Technical users | Anyone can use |
| **Real-time** | Static analysis | Dynamic slider updates |
| **Output Format** | Console text | Visual + formatted results |

## 🎮 **Usage Scenarios**

- **🎨 Designers**: Extract color palettes from inspiration images
- **🏠 Interior Design**: Analyze room colors and objects
- **📸 Photography**: Understand color composition
- **🎯 Marketing**: Analyze brand colors in images
- **🔬 Research**: Automated image content analysis
- **🎓 Education**: Learn about color theory and AI

## 🤝 **Contributing**

The modular design makes it easy to improve:
- **Enhanced UI**: Improve the Gradio interface in `gradio_ui.py`
- **New color algorithms**: Add methods in `auto_color_extractor.py`
- **Additional object detectors**: Extend `object_detector.py`
- **Better visualization**: Enhance color scheme generation

## 📝 **License**

Open source - feel free to use and modify!

---

**🚀 Ready to analyze your images?**
```bash
pip install -r requirements.txt
python app.py
```
Then visit http://localhost:7860 and start uploading images! 🎉