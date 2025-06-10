import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from ultralytics import YOLO

class AdvancedImageAnalyzer:
    def __init__(self, n_colors=5, detection_method="yolo"):
        """
        Initialize the analyzer
        
        Args:
            n_colors (int): Number of dominant colors to extract
            detection_method (str): "yolo", "detr", or "torchvision"
        """
        self.n_colors = n_colors
        self.detection_method = detection_method
        self.detector = YOLO('yolov8n.pt')
    
    def extract_dominant_colors(self, image_path):
        """Extract dominant colors using K-means clustering"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Reshape to pixels
        pixels = image_array.reshape(-1, 3)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        # Create color frequency list
        color_freq = []
        for i in range(self.n_colors):
            color = tuple(colors[i])
            frequency = label_counts[i] / len(labels)
            color_name = self.rgb_to_color_name(color)
            color_freq.append({
                "color_name": color_name,
                "rgb": color,
                "frequency": round(frequency * 100, 2)
            })
        
        # Sort by frequency
        color_freq.sort(key=lambda x: x['frequency'], reverse=True)
        return color_freq
    
    def detect_objects_yolo(self, image_path):
        """Object detection using YOLOv8"""
        
        if not self.detector:
            return {"error": "YOLO model not available"}
        print("Using YOLOv8 for object detection")
        
        results = self.detector(image_path)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.detector.names[class_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        "object": class_name,
                        "confidence": round(confidence * 100, 2),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return {"detections": detections, "method": "YOLOv8"}
    
    def rgb_to_color_name(self, rgb):
        """Convert RGB to color name"""
        r, g, b = rgb
        
        # Enhanced color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 180 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 180 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 180:
            return "blue"
        elif r > 180 and g > 180 and b < 100:
            return "yellow"
        elif r > 180 and g > 100 and b < 100:
            return "orange"
        elif r > 150 and g < 150 and b > 150:
            return "purple"
        elif r < 150 and g > 150 and b > 150:
            return "cyan"
        elif r > 100 and g > 60 and b < 60:
            return "brown"
        elif abs(r-g) < 30 and abs(g-b) < 30 and abs(r-b) < 30:
            if r > 150:
                return "light gray"
            else:
                return "dark gray"
        else:
            return f"mixed color"
    
    def analyze_image(self, image_path):
        """Complete image analysis"""
        print(f"Analyzing image: {image_path}")
        
        # Extract colors
        print("Extracting dominant colors...")
        colors = self.extract_dominant_colors(image_path)
        
        # Detect objects
        print("Detecting objects...")
        objects = self.detect_objects_yolo(image_path)
        
        return {
            "dominant_colors": colors,
            "object_detection": objects
        }

# Example usage
def main():
    
    analyzer = AdvancedImageAnalyzer(n_colors=8, detection_method="yolo")
    # Analyze image
    image_path = "images/fabric.png"  # Replace with actual image path
    
    try:
        results = analyzer.analyze_image(image_path)
        
        # Print results
        print("\n" + "="*50)
        print("DOMINANT COLORS:")
        print("="*50)
        for i, color in enumerate(results["dominant_colors"], 1):
            print(f"{i}. {color['color_name']} - RGB{color['rgb']} ({color['frequency']}%)")
        
        print("\n" + "="*50)
        print("OBJECT DETECTION:")
        print("="*50)
        if "error" in results["object_detection"]:
            print(f"Error: {results['object_detection']['error']}")
        else:
            method = results["object_detection"]["method"]
            detections = results["object_detection"]["detections"]
            print(f"Method: {method}")
            
            if detections:
                for detection in detections:
                    print(f"- {detection['object']} ({detection['confidence']}% confidence)")
            else:
                print("No objects detected with high confidence")
    
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()