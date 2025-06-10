import numpy as np
import cv2
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from collections import Counter


class AutoColorExtractor:
    def __init__(self, n_colors=5):
        self.n_colors = n_colors

    def analyze_image_complexity(self, image_array):
        """
        Analyze key image characteristics for classification
        Only calculates metrics that are actually used

        Returns:
            dict: Essential image analysis metrics
        """
        try:
            # Convert to HSV for better color analysis
            hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv_image)

            # Metric 1: High saturation ratio (colorful vs muted)
            high_saturation_ratio = np.sum(s > 100) / s.size

            # Metric 2: Dark areas ratio (shadows/contrast)
            very_dark_ratio = np.sum(v < 50) / v.size

            # Metric 3: Gradient smoothness (artistic vs photographic)
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude))

            return {
                "high_saturation_ratio": high_saturation_ratio,
                "very_dark_ratio": very_dark_ratio,
                "gradient_smoothness": gradient_smoothness,
            }
        except Exception as e:
            print(f"⚠️ Analysis warning: {e}")
            return {
                "high_saturation_ratio": 0.5,
                "very_dark_ratio": 0.1,
                "gradient_smoothness": 0.05,
            }

    def classify_image_type(self, metrics):
        """
        Automatically classify image type and choose extraction strategy
        """
        high_sat = metrics["high_saturation_ratio"]
        dark_ratio = metrics["very_dark_ratio"]
        gradient_smooth = metrics["gradient_smoothness"]

        if high_sat > 0.7:
            if gradient_smooth > 0.05 or dark_ratio > 0.08:
                image_type = "artistic_gradient"
                strategy = "filtered_vibrant"
                params = {
                    "filter_dark": True,
                    "min_saturation": 40,
                    "min_value": 60,
                    "use_hsv_clustering": True,
                }
            else:
                image_type = "colorful_distinct"
                strategy = "enhanced_standard"
                params = {
                    "filter_dark": False,
                    "min_saturation": 30,
                    "enhance_vibrant": True,
                }
        elif high_sat < 0.15:
            image_type = "simple_muted"
            strategy = "standard"
            params = {
                "filter_dark": False,
                "min_saturation": 5,
                "focus_on_dominant": True,
            }
        elif dark_ratio > 0.2:
            image_type = "high_contrast"
            strategy = "shadow_filtered"
            params = {"filter_dark": True, "min_saturation": 15, "min_value": 50}
        else:
            image_type = "balanced"
            strategy = "enhanced_standard"
            params = {
                "filter_dark": dark_ratio > 0.1,
                "min_saturation": 25,
                "enhance_vibrant": False,
            }

        return {
            "image_type": image_type,
            "strategy": strategy,
            "params": params,
            "confidence": self._calculate_confidence(metrics, image_type),
        }

    def _calculate_confidence(self, metrics, image_type):
        """Calculate confidence in classification"""
        high_sat = metrics["high_saturation_ratio"]
        dark_ratio = metrics["very_dark_ratio"]

        if image_type == "artistic_gradient":
            return 0.95 if high_sat > 0.8 else 0.85
        elif image_type == "simple_muted":
            return 0.95 if high_sat < 0.1 else 0.8
        elif image_type == "colorful_distinct":
            return 0.9
        elif image_type == "high_contrast":
            return 0.85 if dark_ratio > 0.25 else 0.7
        else:
            return 0.75

    def extract_colors_adaptive(self, image_path, debug=False):
        """Main function: automatically analyze image and extract colors with best method"""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path.convert("RGB")

            image_array = np.array(image)

            # Step 1: Analyze image
            metrics = self.analyze_image_complexity(image_array)
            classification = self.classify_image_type(metrics)

            if debug:
                self._print_analysis(metrics, classification)

            # Step 2: Extract colors using chosen strategy
            colors = self._extract_with_strategy(
                image_array, classification["strategy"], classification["params"]
            )

            # Step 3: Improve color naming
            for color in colors:
                color["color_name"] = self._get_smart_color_name(color["rgb"])

            return {
                "dominant_colors": colors,
                "image_analysis": {
                    "type": classification["image_type"],
                    "strategy_used": classification["strategy"],
                    "confidence": classification["confidence"],
                },
                "extraction_method": "whole_image",
                "metrics": metrics if debug else None,
            }
        except Exception as e:
            return {
                "error": f"Color extraction failed: {str(e)}",
                "dominant_colors": [],
                "image_analysis": {
                    "type": "error",
                    "strategy_used": "none",
                    "confidence": 0.0,
                },
            }

    def _combine_masks(self, image_array, masks):
        """Combine multiple masks into one"""
        h, w = image_array.shape[:2]
        combined = np.zeros((h, w), dtype=bool)
        
        for mask in masks:
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                mask_bool = mask_resized > 0.5
            else:
                mask_bool = mask > 0.5
            combined |= mask_bool
        
        return combined

    # NEW: Object-focused extraction method
    def extract_object_focused_colors(
        self, image_path, object_detector=None, use_all_objects=False, debug=False
    ):
        """Extract colors focusing on the main detected object"""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path.convert("RGB")

            image_array = np.array(image)

            # Try object detection first
            if object_detector and object_detector.is_available():
                detection_results = object_detector.detect_objects(image)

                if detection_results.get("detections"):
                    # Combine all object masks
                    main_object = detection_results["detections"][0]
                    object_name = main_object["object"]
                    confidence = main_object["confidence"]
                    mask = main_object.get("mask")
                    bbox = main_object["bbox"] 
                    if use_all_objects:
                        # Multi-object extraction logic
                        all_masks = [d.get("mask") for d in detection_results["detections"] if d.get("mask") is not None]
                        if all_masks:
                            combined_mask = self._combine_masks(image_array, all_masks)
                            object_colors = self._extract_from_object_mask(image_array, combined_mask)
                        else:
                            object_colors = self._extract_from_object_mask(image_array, mask)
                    # Object detected - extract colors from object region
                    else:
                        # Extract colors from object region with expanded context
                        object_colors = self._extract_from_object_mask(image_array, mask)

                    if debug:
                        print(
                            f"🎯 Extracting colors from detected {object_name} (confidence: {confidence}%)"
                        )


                    # Add color names
                    for color in object_colors:
                        color["color_name"] = self._get_smart_color_name(color["rgb"])

                    return {
                        "dominant_colors": object_colors,
                        "extraction_method": "object_focused",
                        "main_object": {
                            "name": object_name,
                            "confidence": confidence,
                            "bbox": bbox,
                        },
                        "image_analysis": {
                            "type": "object_focused",
                            "strategy_used": "object_region",
                            "confidence": 0.9,
                        },
                    }
                else:
                    # No objects detected - fallback
                    if debug:
                        print("⚠️ No objects detected, falling back to whole image")

                    fallback_result = self.extract_colors_adaptive(
                        image_path, debug=debug
                    )
                    fallback_result["extraction_method"] = "object_focused_fallback"
                    fallback_result["fallback_reason"] = "no_objects_detected"
                    return fallback_result
            else:
                # Object detection not available - fallback
                if debug:
                    print(
                        "⚠️ Object detection not available, falling back to whole image"
                    )

                fallback_result = self.extract_colors_adaptive(image_path, debug=debug)
                fallback_result["extraction_method"] = "object_focused_fallback"
                fallback_result["fallback_reason"] = "no_object_detector"
                return fallback_result

        except Exception as e:
            return {
                "error": f"Object-focused extraction failed: {str(e)}",
                "dominant_colors": [],
                "extraction_method": "error",
                "image_analysis": {
                    "type": "error",
                    "strategy_used": "none",
                    "confidence": 0.0,
                },
            }

    def _extract_from_object_mask(self, image_array, mask):
        """Extract colors from segmentation mask (actual object pixels only)"""
        try:
            # Convert mask to boolean and resize to image dimensions
            h, w = image_array.shape[:2]
            
            # Resize mask to match image size if needed
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                mask_bool = mask_resized > 0.5  # Convert to boolean
            else:
                mask_bool = mask > 0.5  # Convert to boolean
            
            # Extract object pixels
            object_pixels = image_array[mask_bool]
            
            if len(object_pixels) < self.n_colors * 10:
                # Fallback if mask too small
                return self._extract_standard(image_array)
            
            # Cluster only the actual object pixels
            n_colors = min(self.n_colors, len(object_pixels) // 10)
            if n_colors < 2:
                n_colors = 2
                
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(object_pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            return self._calculate_frequencies_with_labels(colors, labels)
            
        except Exception as e:
            print(f"⚠️ Mask extraction failed: {e}, falling back to standard")
            return self._extract_standard(image_array)

    # UPDATED: Build color scheme with object focus option
    def build_color_scheme(
        self,
        image_path,
        object_detector=None,
        extract_from_object=False,
        use_all_objects=False,
        width=400,
        height=100,
    ):
        """Build a proportional color scheme visualization"""
        try:
            # Choose extraction method
            if extract_from_object:
                results = self.extract_object_focused_colors(
                    image_path, object_detector, use_all_objects, debug=False
                )
            else:
                results = self.extract_colors_adaptive(image_path, debug=False)

            if "error" in results:
                error_image = Image.new("RGB", (width, height), (255, 200, 200))
                return error_image, results

            colors = results["dominant_colors"]
            if not colors:
                error_image = Image.new("RGB", (width, height), (255, 200, 200))
                return error_image, results

            # Create color scheme
            scheme_image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(scheme_image)

            current_x = 0
            for color_info in colors:
                rgb = color_info["rgb"]
                frequency = color_info["frequency"] / 100.0
                section_width = int(width * frequency)

                if current_x + section_width > width:
                    section_width = width - current_x

                draw.rectangle(
                    [current_x, 0, current_x + section_width, height], fill=rgb
                )
                current_x += section_width

                if current_x >= width:
                    break

            return scheme_image, results

        except Exception as e:
            error_image = Image.new("RGB", (width, height), (255, 200, 200))
            error_results = {
                "error": f"Scheme generation failed: {str(e)}",
                "dominant_colors": [],
            }
            return error_image, error_results

    # UPDATED: Get palette data with object focus option
    def get_color_palette_data(
        self, image_path, object_detector=None, extract_from_object=False, use_all_objects=False
    ):
        """Get color palette data formatted for UI display"""
        try:
            # Choose extraction method
            if extract_from_object:
                results = self.extract_object_focused_colors(
                    image_path, object_detector, use_all_objects, debug=False
                )
            else:
                results = self.extract_colors_adaptive(image_path, debug=False)

            if "error" in results:
                return {"error": results["error"]}

            palette_data = {
                "colors": [],
                "image_analysis": results["image_analysis"],
                "extraction_method": results.get("extraction_method", "whole_image"),
                "main_object": results.get("main_object"),
                "fallback_reason": results.get("fallback_reason"),
            }

            for i, color_info in enumerate(results["dominant_colors"], 1):
                palette_data["colors"].append(
                    {
                        "rank": i,
                        "name": color_info["color_name"],
                        "rgb": color_info["rgb"],
                        "hex": f"#{color_info['rgb'][0]:02x}{color_info['rgb'][1]:02x}{color_info['rgb'][2]:02x}",
                        "frequency": color_info["frequency"],
                    }
                )

            return palette_data

        except Exception as e:
            return {"error": f"Palette generation failed: {str(e)}"}

    # ... rest of your existing methods (keeping them the same) ...

    def _extract_with_strategy(self, image_array, strategy, params):
        """Apply the chosen extraction strategy"""
        if strategy == "filtered_vibrant":
            return self._extract_filtered_vibrant(image_array, params)
        elif strategy == "enhanced_standard":
            return self._extract_enhanced_standard(image_array, params)
        elif strategy == "shadow_filtered":
            return self._extract_shadow_filtered(image_array, params)
        elif strategy == "adaptive_standard":
            return self._extract_adaptive_standard(image_array, params)
        else:
            return self._extract_standard(image_array)

    def _extract_filtered_vibrant(self, image_array, params):
        """For artistic/gradient images with lots of vibrant colors"""
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        mask = (s > params["min_saturation"]) & (v > params.get("min_value", 50))

        if params.get("use_hsv_clustering", False):
            hsv_pixels = hsv_image.reshape(-1, 3)
            filtered_hsv = hsv_pixels[mask.flatten()]

            if len(filtered_hsv) < self.n_colors * 50:
                mask = (s > params["min_saturation"] // 2) & (v > 30)
                filtered_hsv = hsv_pixels[mask.flatten()]
            if len(filtered_hsv) < self.n_colors * 20:
                filtered_hsv = hsv_pixels

            kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
            kmeans.fit(filtered_hsv)

            hsv_centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            hsv_centers_image = hsv_centers.reshape(1, -1, 3).astype(np.uint8)
            rgb_centers = cv2.cvtColor(hsv_centers_image, cv2.COLOR_HSV2RGB)[0]
            colors = rgb_centers

            return self._calculate_frequencies_with_labels(colors, labels)
        else:
            rgb_pixels = image_array.reshape(-1, 3)
            filtered_rgb = rgb_pixels[mask.flatten()]
            if len(filtered_rgb) < self.n_colors * 50:
                filtered_rgb = rgb_pixels
            kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
            kmeans.fit(filtered_rgb)
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            return self._calculate_frequencies_with_labels(colors, labels)

    def _extract_enhanced_standard(self, image_array, params):
        """For colorful images with distinct regions"""
        pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        if params.get("enhance_vibrant", False):
            hsv_colors = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
            hsv_colors[:, 1] = np.minimum(255, hsv_colors[:, 1] * 1.1)
            colors = cv2.cvtColor(hsv_colors.reshape(1, -1, 3), cv2.COLOR_HSV2RGB)[0]

        return self._calculate_frequencies_with_labels(colors, labels)

    def _extract_shadow_filtered(self, image_array, params):
        """For images with significant shadows"""
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        mask = v > params.get("min_value", 60)
        rgb_pixels = image_array.reshape(-1, 3)
        filtered_pixels = rgb_pixels[mask.flatten()]
        if len(filtered_pixels) < self.n_colors * 20:
            filtered_pixels = rgb_pixels
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(filtered_pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        return self._calculate_frequencies_with_labels(colors, labels)

    def _extract_adaptive_standard(self, image_array, params):
        """Adaptive approach that adjusts based on image"""
        pixels = image_array.reshape(-1, 3)
        original_pixel_count = len(pixels)
        if params.get("filter_dark", False):
            hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            v = hsv_image[:, :, 2]
            mask = v > 50
            pixels = pixels[mask.flatten()]
            if len(pixels) < original_pixel_count * 0.3:
                pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        return self._calculate_frequencies_with_labels(colors, labels)

    def _extract_standard(self, image_array):
        """Standard K-means extraction"""
        pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        return self._calculate_frequencies_with_labels(colors, labels)

    def _calculate_frequencies_with_labels(self, colors, labels):
        """Calculate frequencies when we have cluster labels"""
        label_counts = Counter(labels)
        total = len(labels)
        results = []
        for i, color in enumerate(colors):
            frequency = (label_counts[i] / total) * 100
            results.append({"rgb": tuple(color), "frequency": round(frequency, 2)})
        results.sort(key=lambda x: x["frequency"], reverse=True)
        return results

    def _get_smart_color_name(self, rgb):
        """Improved color naming with better accuracy"""
        r, g, b = rgb
        brightness = (r + g + b) / 3
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0

        if saturation < 0.15:
            if brightness > 200:
                return "white"
            elif brightness < 50:
                return "black"
            elif brightness > 150:
                return "light_gray"
            elif brightness > 100:
                return "gray"
            else:
                return "dark_gray"

        if max_val == r:
            if g > b + 20:
                if g > r * 0.7:
                    return "yellow" if g > 150 else "olive"
                else:
                    return "orange" if g > 80 else "red"
            elif b > g + 20:
                return "pink" if b > 120 else "crimson"
            else:
                if r > 180:
                    return "red"
                else:
                    return "dark_red"
        elif max_val == g:
            if r > b + 20:
                if r > g * 0.8:
                    return "lime" if g > 150 else "olive"
                else:
                    return "green" if g > 100 else "forest_green"
            elif b > r + 20:
                if b > g * 0.7:
                    return "cyan" if b > 150 else "teal"
                else:
                    return "green" if g > 120 else "dark_green"
            else:
                return "green"
        else:
            if r > g + 20:
                if r > b * 0.7:
                    return "purple" if r > 120 else "indigo"
                else:
                    return "blue" if b > 120 else "navy"
            elif g > r + 20:
                if g > b * 0.6:
                    return "turquoise" if g > 120 else "teal"
                else:
                    return "blue" if b > 120 else "navy"
            else:
                if b > 150:
                    return "blue"
                else:
                    return "dark_blue"

    def _print_analysis(self, metrics, classification):
        """Print detailed analysis for debugging"""
        print("=== AUTOMATIC IMAGE ANALYSIS ===")
        print(f"Image Type: {classification['image_type']}")
        print(f"Strategy: {classification['strategy']}")
        print(f"Confidence: {classification['confidence']:.1%}")
        print(f"\nKey Metrics:")
        print(f"  High Saturation Ratio: {metrics['high_saturation_ratio']:.1%}")
        print(f"  Dark Areas Ratio: {metrics['very_dark_ratio']:.1%}")
        print(f"  Gradient Smoothness: {metrics['gradient_smoothness']:.3f}")
        print("=" * 35)
