import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
from ultralytics import YOLO
import argparse
from tqdm import tqdm


class AdvancedImageAnalyzer:
    def __init__(self, n_colors=5):
        """
        Initialize the analyzer

        Args:
            n_colors (int): Number of dominant colors to extract
        """
        self.n_colors = n_colors
        self.detector = YOLO("yolov8n-seg.pt")

    def extract_dominant_colors(self, image_path):
        """Extract dominant colors using K-means clustering"""
        # Load image
        image = Image.open(image_path).convert("RGB")
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
        for i in tqdm(range(self.n_colors)):
            color = tuple(colors[i])
            frequency = label_counts[i] / len(labels)
            color_name = self.rgb_to_color_name(color)
            color_freq.append(
                {
                    "color_name": color_name,
                    "rgb": color,
                    "frequency": round(frequency * 100, 2),
                }
            )

        # Sort by frequency
        color_freq.sort(key=lambda x: x["frequency"], reverse=True)
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
            masks = result.masks  # Get masks
            if boxes is not None:
                for i, box in tqdm(enumerate(boxes)):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.detector.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Extract mask if available
                    mask_data = None
                    if masks is not None and i < len(masks.data):
                        mask_data = masks.data[i].cpu().numpy()

                    detections.append(
                        {
                            "object": class_name,
                            "confidence": round(confidence * 100, 2),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "mask": mask_data,  # Add mask
                        }
                    )

        return {"detections": detections}

    def extract_colors_from_mask(self, image_path, mask):
        """Extract colors from masked region only"""
        import cv2

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        h, w = image_array.shape[:2]

        # Resize mask to match image
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
            mask_bool = mask_resized > 0.5
        else:
            mask_bool = mask > 0.5

        # Extract object pixels only
        object_pixels = image_array[mask_bool]

        if len(object_pixels) < self.n_colors * 10:
            return self.extract_dominant_colors(image_path)  # Fallback

        # K-means on object pixels
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(object_pixels)

        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)

        color_freq = []
        for i in tqdm(range(self.n_colors)):
            color = tuple(colors[i])
            frequency = label_counts[i] / len(labels)
            color_name = self.rgb_to_color_name(color)
            color_freq.append(
                {
                    "color_name": color_name,
                    "rgb": color,
                    "frequency": round(frequency * 100, 2),
                }
            )

        color_freq.sort(key=lambda x: x["frequency"], reverse=True)
        return color_freq

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
        elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            if r > 150:
                return "light gray"
            else:
                return "dark gray"
        else:
            return f"mixed color"

    def analyze_image(self, image_path):
        """Complete image analysis"""
        print(f"Analyzing image: {image_path}")

        # Detect objects
        print("Detecting objects...")
        objects = self.detect_objects_yolo(image_path)

        if objects.get("detections"):
            main_detection = objects["detections"][0]  # Highest confidence
            if main_detection.get("mask") is not None:
                print(f"Extracting colors from {main_detection['object']} mask...")
                colors = self.extract_colors_from_mask(
                    image_path, main_detection["mask"]
                )
            else:
                print("No mask available, using whole image...")
                colors = self.extract_dominant_colors(image_path)
        else:
            print("Extracting colors from whole image...")
            colors = self.extract_dominant_colors(image_path)

        return {"dominant_colors": colors, "object_detection": objects}


# Example usage
def main():

    parser = argparse.ArgumentParser(
        description="Analyze image colors and detect objects"
    )
    parser.add_argument("--image_path", help="Path to the image file")
    parser.add_argument(
        "--n_colors",
        type=int,
        default=5,
        help="Number of colors to extract (default: 8)",
    )

    args = parser.parse_args()

    analyzer = AdvancedImageAnalyzer(n_colors=args.n_colors)

    try:
        results = analyzer.analyze_image(args.image_path)

        # Print results
        print("\n" + "=" * 50)
        print("DOMINANT COLORS:")
        print("=" * 50)
        for i, color in enumerate(results["dominant_colors"], 1):
            print(
                f"{i}. {color['color_name']} - RGB{color['rgb']} ({color['frequency']}%)"
            )

        print("\n" + "=" * 50)
        print("OBJECT DETECTION:")
        print("=" * 50)
        if "error" in results["object_detection"]:
            print(f"Error: {results['object_detection']['error']}")
        else:
            detections = results["object_detection"]["detections"]

            if detections:
                for detection in detections:
                    print(
                        f"- {detection['object']} ({detection['confidence']}% confidence)"
                    )
            else:
                print("No objects detected with high confidence")

    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
