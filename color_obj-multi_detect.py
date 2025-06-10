"""
Advanced Object Detection with Multiple State-of-the-Art Models
Supports YOLOv11, RT-DETR, and other cutting-edge detectors
"""

# Import with graceful fallbacks
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection, RTDetrForObjectDetection
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from PIL import Image
import numpy as np


class AdvancedObjectDetector:
    def __init__(self, model_type="auto", confidence_threshold=0.5):
        """
        Initialize advanced object detector with multiple model options
        
        Args:
            model_type: "auto", "yolov11", "yolov8", "rt-detr", "detr", "clip"
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.detector = None
        self.processor = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the best available detection method"""
        
        if self.model_type == "auto":
            # Try methods in order of preference (newest first)
            methods_to_try = ["yolov11", "rt-detr", "yolov8", "detr"]
        else:
            methods_to_try = [self.model_type]
        
        for method in methods_to_try:
            try:
                if method == "yolov11" and ULTRALYTICS_AVAILABLE:
                    # Try YOLOv11 first (latest)
                    self.detector = YOLO('yolo11n.pt')  # nano version
                    self.model_type = "yolov11"
                    print("✅ Using YOLOv11 (Latest YOLO)")
                    return
                    
                elif method == "yolov8" and ULTRALYTICS_AVAILABLE:
                    self.detector = YOLO('yolov8n.pt')
                    self.model_type = "yolov8"
                    print("✅ Using YOLOv8")
                    return
                    
                elif method == "rt-detr" and HF_AVAILABLE:
                    # RT-DETR (Real-Time Detection Transformer)
                    self.processor = DetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
                    self.detector = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
                    self.model_type = "rt-detr"
                    print("✅ Using RT-DETR (Advanced Transformer)")
                    return
                    
                elif method == "detr" and HF_AVAILABLE:
                    self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                    self.detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                    self.model_type = "detr"
                    print("✅ Using DETR")
                    return
                    
            except Exception as e:
                print(f"⚠️ Failed to initialize {method}: {e}")
                continue
        
        print("❌ No advanced detection method available!")
        print("💡 Install: pip install ultralytics transformers torch")
    
    def detect_objects(self, image_path, custom_prompt=None):
        """
        Advanced object detection with multiple model support
        
        Args:
            image_path: Path to image or PIL Image
            custom_prompt: Optional text prompt for advanced models
            
        Returns:
            dict: Detection results with model info
        """
        if not self.detector:
            return {
                "error": "No detection model available",
                "detections": [],
                "method": "none"
            }
        
        try:
            if self.model_type in ["yolov11", "yolov8"]:
                return self._detect_yolo(image_path)
            elif self.model_type == "rt-detr":
                return self._detect_rt_detr(image_path)
            elif self.model_type == "detr":
                return self._detect_detr(image_path)
            else:
                return {"error": f"Unknown model type: {self.model_type}"}
                
        except Exception as e:
            return {
                "error": f"Detection failed: {str(e)}",
                "detections": [],
                "method": self.model_type
            }
    
    def _detect_yolo(self, image_path):
        """YOLOv8/v11 detection"""
        # Handle both file paths and PIL Images
        if isinstance(image_path, str):
            results = self.detector(image_path, verbose=False)
        else:
            # PIL Image - save temporarily
            temp_path = "temp_detection.jpg"
            image_path.save(temp_path)
            results = self.detector(temp_path, verbose=False)
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        detections = []
        all_detections = []  # For debug
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.detector.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection_info = {
                        "object": class_name,
                        "confidence": round(confidence * 100, 2),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    }
                    
                    all_detections.append(detection_info)
                    
                    if confidence >= self.confidence_threshold:
                        detections.append(detection_info)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        model_name = "YOLOv11" if self.model_type == "yolov11" else "YOLOv8"
        
        return {
            "detections": detections,
            "method": model_name,
            "total_found": len(detections),
            "total_raw": len(all_detections),
            "model_type": self.model_type
        }
    
    def _detect_rt_detr(self, image_path):
        """RT-DETR detection (Advanced transformer model)"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.detector(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = score.item()
            
            if confidence < self.confidence_threshold:
                continue
            
            box_coords = [round(i) for i in box.tolist()]
            class_name = self.detector.config.id2label[label.item()]
            
            detections.append({
                "object": class_name,
                "confidence": round(confidence * 100, 2),
                "bbox": box_coords
            })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "detections": detections,
            "method": "RT-DETR (Transformer)",
            "total_found": len(detections),
            "model_type": "rt-detr"
        }
    
    def _detect_detr(self, image_path):
        """Standard DETR detection"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.detector(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = score.item()
            
            if confidence < self.confidence_threshold:
                continue
            
            box_coords = [round(i) for i in box.tolist()]
            class_name = self.detector.config.id2label[label.item()]
            
            detections.append({
                "object": class_name,
                "confidence": round(confidence * 100, 2),
                "bbox": box_coords
            })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "detections": detections,
            "method": "DETR",
            "total_found": len(detections),
            "model_type": "detr"
        }
    
    def get_model_info(self):
        """Get information about the current model"""
        model_info = {
            "yolov11": {
                "name": "YOLOv11",
                "year": "2024",
                "type": "CNN-based",
                "speed": "Very Fast",
                "accuracy": "High",
                "description": "Latest YOLO with improved architecture"
            },
            "yolov8": {
                "name": "YOLOv8", 
                "year": "2023",
                "type": "CNN-based",
                "speed": "Very Fast",
                "accuracy": "High",
                "description": "Popular and reliable YOLO variant"
            },
            "rt-detr": {
                "name": "RT-DETR",
                "year": "2023-2024", 
                "type": "Transformer",
                "speed": "Fast",
                "accuracy": "Very High",
                "description": "Real-time detection transformer, no NMS needed"
            },
            "detr": {
                "name": "DETR",
                "year": "2020-2021",
                "type": "Transformer", 
                "speed": "Medium",
                "accuracy": "High",
                "description": "Original detection transformer"
            }
        }
        
        return model_info.get(self.model_type, {"name": "Unknown"})
    
    def benchmark_models(self, image_path, models_to_test=None):
        """
        Benchmark multiple models on the same image
        
        Args:
            image_path: Image to test
            models_to_test: List of models to test (default: all available)
            
        Returns:
            dict: Benchmark results
        """
        if models_to_test is None:
            models_to_test = ["yolov11", "rt-detr", "yolov8", "detr"]
        
        results = {}
        
        for model_type in models_to_test:
            print(f"🧪 Testing {model_type}...")
            
            # Initialize model
            temp_detector = AdvancedObjectDetector(model_type=model_type)
            
            if temp_detector.detector:
                import time
                start_time = time.time()
                
                # Run detection
                detection_results = temp_detector.detect_objects(image_path)
                
                end_time = time.time()
                
                results[model_type] = {
                    "detections": detection_results.get("detections", []),
                    "total_found": detection_results.get("total_found", 0),
                    "inference_time": round(end_time - start_time, 3),
                    "method": detection_results.get("method", model_type),
                    "available": True
                }
            else:
                results[model_type] = {
                    "available": False,
                    "error": "Model not available"
                }
        
        return results
    
    def is_available(self):
        """Check if detector is available"""
        return self.detector is not None


# Example usage and comparison
def main():
    """Demo of advanced object detection"""
    print("🚀 Advanced Object Detection Demo")
    print("=" * 50)
    
    # Test image
    test_image = "images/panda_graffiti.png"
    
    # Initialize with auto-selection (will pick best available)
    detector = AdvancedObjectDetector(model_type="auto", confidence_threshold=0.5)
    
    if detector.is_available():
        # Show model info
        model_info = detector.get_model_info()
        print(f"📊 Using: {model_info['name']} ({model_info['year']})")
        print(f"   Type: {model_info['type']}")
        print(f"   Speed: {model_info['speed']}")
        print(f"   Accuracy: {model_info['accuracy']}")
        
        # Run detection
        results = detector.detect_objects(test_image)
        
        print(f"\n🔍 Detection Results:")
        if results["detections"]:
            for obj in results["detections"]:
                print(f"   - {obj['object']}: {obj['confidence']}%")
        else:
            print("   No objects detected")
        
        # Optional: Benchmark multiple models
        print(f"\n🧪 Benchmarking available models...")
        benchmark = detector.benchmark_models(test_image)
        
        print(f"\n📊 Model Comparison:")
        for model, result in benchmark.items():
            if result.get("available", False):
                count = result["total_found"]
                time_ms = result["inference_time"] * 1000
                print(f"   {model}: {count} objects, {time_ms:.1f}ms")
            else:
                print(f"   {model}: Not available")
    
    else:
        print("❌ No detection models available")
        print("💡 Install: pip install ultralytics transformers torch")


if __name__ == "__main__":
    main()