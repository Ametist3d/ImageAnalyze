from ultralytics import YOLO
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the object detector

        Args:
            confidence_threshold (float): Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.detector = YOLO("yolo11x-seg.pt")
        logger.info(
            f"YOLOv11 initialized with {confidence_threshold*100}% confidence threshold"
        )

    def is_available(self):
        """Check if the YOLO detector is available and ready"""
        return self.detector is not None

    def detect_objects(self, image_path):
        """Object detection using YOLOv11 - keeping original YOLO order"""

        if not self.detector:
            return {"error": "YOLO model not available", "detections": []}

        try:
            # Handle both file paths and PIL Image objects
            if isinstance(image_path, str):
                # File path - use directly
                results = self.detector(image_path)
            else:
                # PIL Image - save temporarily
                temp_path = "temp_yolo_image.jpg"
                image_path.save(temp_path)
                results = self.detector(temp_path)
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            detections = []

            for result in results:
                boxes = result.boxes
                masks = result.masks
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        confidence = float(box.conf[0])

                        # Filter by confidence threshold
                        if confidence < self.confidence_threshold:
                            continue

                        # Get class info
                        class_id = int(box.cls[0])
                        class_name = self.detector.names[class_id]

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        mask_data = None
                        if masks is not None and i < len(masks.data):
                            mask_data = masks.data[i].cpu().numpy()

                        detections.append(
                            {
                                "object": class_name,
                                "confidence": round(confidence * 100, 2),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "mask": mask_data,
                                "yolo_index": i,
                            }
                        )

            return {
                "detections": detections,
                "method": "YOLOv11",
                "total_found": len(detections),
            }

        except Exception as e:
            return {
                "error": f"Detection failed: {str(e)}",
                "detections": [],
                "method": "YOLOv11",
            }
