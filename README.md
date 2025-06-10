# 🎨 Image Analysis System

Extract colors from specific objects in images using AI-powered object detection and segmentation.

## ✨ Key Features

### 🎯 **Object-Focused Color Extraction**
- **Extract colors from specific objects** - Get colors from just the car, person, or any detected object
- **AI-powered segmentation** - Uses YOLOv11 for precise object boundaries (not just bounding boxes)
- **Interactive object selection** - Choose which objects to analyze via checkboxes
- **Real-time preview** - See exactly which objects are selected with colored masks

### 🎨 **Smart Color Analysis**
- **Proportional color schemes** - Visual bars showing color frequency
- **Multiple extraction modes** - Whole image vs object-focused analysis
- **Adaptive algorithms** - Automatically chooses best extraction strategy
- **Smart color naming** - RGB values converted to descriptive names (cyan, forest_green, etc.)

### 🔧 **Interactive Controls**
- **Confidence threshold slider** - Adjust object detection sensitivity (0.1-1.0)
- **Color count slider** - Extract 2-10 dominant colors
- **Toggle modes** - Switch between whole image and object analysis
- **Real-time updates** - Changes apply instantly

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install gradio numpy opencv-python scikit-learn pillow ultralytics
```

### 2. Launch Web Interface
```bash
python app.py
```
Open http://localhost:7860 in your browser

### 3. Basic Usage
1. **Upload an image** (drag & drop)
2. **Click "Detect Objects"** to see what's detected
3. **Select specific objects** using checkboxes
4. **Enable "Extract colors from detected objects"** toggle
5. **Click "Extract Colors"** to get object-specific colors

## 📊 Example Results

### Object-Focused Analysis
```
🎨 COLOR ANALYSIS (Car Object)
1. cyan - #4fa8b8 (31.2%)      ← Actual car color
2. dark_gray - #2a2a2a (28.5%)  ← Car details
3. red - #b42d1a (17.2%)       ← Brake lights
```

### vs Whole Image Analysis
```
🎨 COLOR ANALYSIS (Whole Image)
1. green - #7ba05b (45.3%)     ← Trees/background
2. blue - #87ceeb (22.1%)      ← Sky
3. cyan - #4fa8b8 (12.2%)      ← Car (diluted)
```

## 🎮 Use Cases

### **Design & Fashion**
- Extract exact colors from clothing without background interference
- Get brand colors from product photos
- Analyze furniture colors separately from room décor

### **Automotive**
- True vehicle paint colors without road/sky contamination
- Compare multiple car colors in one image
- Paint matching for repairs

### **Photography & Art**
- Subject-focused color palettes
- Analyze color relationships between objects
- Remove background color bias

## 🛠️ API Usage

### Object-Focused Extraction
```python
from auto_color_extractor import AutoColorExtractor
from object_detector import ObjectDetector

extractor = AutoColorExtractor(n_colors=5)
detector = ObjectDetector(confidence_threshold=0.5)

# Extract colors from specific objects
results = extractor.extract_object_focused_colors(
    "image.jpg",
    object_detector=detector,
    selected_object_indices=[0, 2]  # First and third detected objects
)

for color in results["dominant_colors"]:
    print(f"{color['color_name']}: {color['frequency']:.1f}%")
```

### Command Line
```bash
# Simple analysis with custom settings
python simple_color-obj_detection_seg.py image.jpg --n_colors 6

# Adjust detection sensitivity
python simple_color-obj_detection_seg.py image.jpg --confidence 0.3
```

## 🎯 Key Advantages

| Traditional Tools | This System |
|------------------|-------------|
| Whole image only | Object-specific analysis |
| Background interference | Pure object colors |
| Static analysis | Interactive selection |
| Generic results | Context-aware extraction |

## 📁 Project Structure

```
├── 🎨 auto_color_extractor.py    # Smart color extraction with object focus
├── 🔍 object_detector.py         # YOLOv11 segmentation detection
├── 🌐 gradio_ui.py              # Interactive web interface
├── 🚀 app.py                    # Launch script
└── 📊 simple_color-obj_detection_seg.py  # Command line version
```

## 🔧 Configuration

### Detection Sensitivity
```python
# More objects (less confident detections)
detector = ObjectDetector(confidence_threshold=0.3)

# Fewer objects (only high-confidence detections)  
detector = ObjectDetector(confidence_threshold=0.8)
```

### Color Analysis
```python
# More detailed color breakdown
extractor = AutoColorExtractor(n_colors=8)

# Custom visualization
color_scheme, results = extractor.build_color_scheme(
    "image.jpg",
    extract_from_object=True,
    width=600,
    height=150
)
```

## 🚀 Installation Options

### Quick Install
```bash
pip install -r requirements.txt
python app.py
```

### Manual Install
```bash
pip install gradio numpy opencv-python scikit-learn pillow ultralytics
```

### Docker (Optional)
```bash
docker build -t image-analysis .
docker run -p 7860:7860 image-analysis
```

## 💡 Pro Tips

- **Lower confidence threshold (0.3)** to detect more objects
- **Higher confidence threshold (0.8)** for only obvious objects
- **Try "All Objects" first** to see what's detected
- **Select specific objects** for precise color analysis
- **Compare object vs whole image** modes to see the difference

## 🎯 Perfect For

- **Designers** extracting color palettes from inspiration images
- **Researchers** analyzing object-specific color properties  
- **Developers** building color-aware applications
- **Artists** understanding color relationships in compositions
- **Anyone** who needs precise color analysis without background interference

---

**🚀 Ready to extract precise colors?**
```bash
pip install -r requirements.txt
python app.py
```

Visit http://localhost:7860 and try object-focused color extraction! 

**Example**: Upload a photo of a red car → Detect Objects → Select "1. car" → Enable object extraction → Get true car colors without road/sky interference! 🎉