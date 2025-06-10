import gradio as gr
import numpy as np
from PIL import Image
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from auto_color_extractor import AutoColorExtractor

    COLOR_EXTRACTION_AVAILABLE = True
except ImportError:
    COLOR_EXTRACTION_AVAILABLE = False
    logging.error("Color extraction not available")

try:
    from object_detector import ObjectDetector

    OBJECT_DETECTION_AVAILABLE = True
except ImportError:
    OBJECT_DETECTION_AVAILABLE = False
    logging.error("Object detection not available")


class ImageAnalysisUI:
    def __init__(self):
        """Initialize the UI with available modules"""
        self.color_extractor = None
        self.object_detector = None
        self.last_detection_results = None
        self.last_scaled_image = None

        # Initialize color extractor
        if COLOR_EXTRACTION_AVAILABLE:
            self.color_extractor = AutoColorExtractor()
            logging.info("Color extraction ready")

        # Initialize object detector
        if OBJECT_DETECTION_AVAILABLE:
            try:
                self.object_detector = ObjectDetector(confidence_threshold=0.5)
                logging.info("Object detection ready")
            except Exception as e:
                logging.exception(f"Object detection failed to initialize: {e}")

    def scale_image(self, image):
        """Scale down image to 1024 on largest side if it's larger than 1024"""
        if image is None:
            return None

        # Make sure we have a PIL Image object
        if not isinstance(image, Image.Image):
            logging.info(f"Warning: Expected PIL Image, got {type(image)}")
            return image

        # Get current dimensions
        width, height = image.size
        max_dimension = max(width, height)

        # Only scale down if largest side is > 1024
        if max_dimension > 1024:
            # Calculate scaling factor
            scale_factor = 1024 / max_dimension

            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image with high quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            logging.info(
                f"Image scaled from {width}x{height} to {new_width}x{new_height}"
            )

        return image

    def extract_colors(self, image, n_colors, extract_from_object, selected_objects):
        """Extract colors with object selection support"""
        if not self.color_extractor or image is None:
            return None, "Color extraction not available or no image provided"

        try:
            # Use stored scaled image if available, otherwise scale
            if (
                hasattr(self, "last_scaled_image")
                and self.last_scaled_image is not None
                and extract_from_object
            ):
                scaled_image = self.last_scaled_image
            else:
                scaled_image = self.scale_image(image)

            self.color_extractor.n_colors = n_colors

            logging.info(
                f"DEBUG: extract_from_object={extract_from_object}, Selected objects: {selected_objects}"
            )

            if not extract_from_object:
                color_scheme, results = self.color_extractor.build_color_scheme(
                    scaled_image, width=550, height=180
                )
            else:
                # Parse selected objects
                use_all_objects = (
                    "All Objects" in selected_objects if selected_objects else False
                )
                selected_indices = []

                if selected_objects and not use_all_objects:
                    for item in selected_objects:
                        if item.startswith(
                            ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")
                        ):
                            try:
                                idx_str = item.split(".")[0]
                                idx = int(idx_str) - 1
                                selected_indices.append(idx)
                            except Exception as e:
                                logging.exception(f"Error parsing '{item}': {e}")

                # Check if we have valid object selection
                if not selected_objects or (
                    not use_all_objects and not selected_indices
                ):
                    color_scheme, results = self.color_extractor.build_color_scheme(
                        scaled_image, width=400, height=100
                    )
                else:
                    # Extract from selected objects
                    color_scheme, results = self.color_extractor.build_color_scheme(
                        scaled_image,
                        object_detector=self.object_detector,
                        extract_from_object=True,
                        use_all_objects=use_all_objects,
                        selected_object_indices=selected_indices,
                        width=400,
                        height=100,
                    )

            # Handle errors
            if "error" in results:
                error_message = f"{results['error']}\n\n **Possible solutions:**\n"
                error_message += "• Try a different image format (JPG, PNG)\n"
                error_message += "• Ensure image is not corrupted\n"
                error_message += "• Check if image has sufficient color variation"
                return color_scheme, error_message

            # Get detailed color information
            palette_data = self.color_extractor.get_color_palette_data(
                scaled_image,
                object_detector=self.object_detector,
                extract_from_object=extract_from_object and bool(selected_objects),
                use_all_objects=use_all_objects if extract_from_object else False,
                selected_object_indices=selected_indices
                if extract_from_object
                else None,
            )

            if "error" in palette_data or not palette_data.get("colors"):
                error_message = "Failed to extract color palette from image"
                return color_scheme, error_message

            # Format analysis text
            analysis_text = self._format_color_analysis(palette_data)

            return color_scheme, analysis_text

        except Exception as e:
            error_message = f"Unexpected error in color extraction: {str(e)}\n\n"
            error_message += f"Debug info: extract_from_object={extract_from_object}, selected_objects={selected_objects}"
            return None, error_message

    def analyze_complete(
        self,
        image,
        n_colors,
        extract_from_object,
        selected_objects,
        confidence_threshold=0.5,
    ):
        """Complete analysis: colors, objects, and masked image"""
        # Detect objects first (this will scale and store the image)
        object_analysis, masked_image, object_selection_update, initial_preview = (
            self.detect_objects_with_masks(image, confidence_threshold)
        )

        # Extract colors (this will use the stored scaled image)
        color_scheme, color_analysis = self.extract_colors(
            image, n_colors, extract_from_object, selected_objects
        )

        if selected_objects:
            selected_preview = self.update_mask_preview(image, selected_objects)
        else:
            selected_preview = initial_preview

        return (
            color_scheme,
            color_analysis,
            object_analysis,
            masked_image,
            object_selection_update,
            selected_preview,
        )

    def _format_color_analysis(self, palette_data):
        """Format color analysis for display"""
        if not palette_data:
            return "No color data available"

        # Check for errors
        if "error" in palette_data:
            return f"Error: {palette_data['error']}"

        analysis = palette_data.get("image_analysis", {})
        colors = palette_data.get("colors", [])

        if not colors:
            return "No colors extracted from image"

        # Header with image analysis
        text = f"**COLOR ANALYSIS**\n\n"
        text += f"**Image Type:** {analysis.get('type', 'unknown')}\n"
        text += f"**Strategy:** {analysis.get('strategy_used', 'unknown')}\n"
        text += f"**Confidence:** {analysis.get('confidence', 0):.1%}\n\n"

        # Color breakdown
        text += f"**Dominant Colors:**\n"
        for color in colors:
            text += f"{color['rank']}. **{color['name']}** - {color['hex']} ({color['frequency']:.1f}%)\n"

        return text

    def _format_object_detection(self, results):
        """Format object detection results for display"""
        if "error" in results:
            return f"{results['error']}"

        detections = results.get("detections", [])
        method = results.get("method", "Unknown")

        if not detections:
            return f"**OBJECT DETECTION** ({method})\n\nNo objects detected with sufficient confidence"

        text = f"**OBJECT DETECTION** ({method})\n\n"
        text += f"**Found {len(detections)} objects:**\n"

        for i, obj in enumerate(detections, 1):
            text += f"{i}. **{obj['object']}** - {obj['confidence']:.1f}% confidence\n"

        return text

    def create_selected_mask_preview(self, image, detection_results, selected_objects):
        """Create preview showing only selected object masks"""
        import cv2
        from PIL import ImageDraw, ImageFont

        if not detection_results.get("detections") or not selected_objects:
            return image

        # Convert PIL to numpy array
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Create a darker version of the original image
        preview = (img_array * 0.3).astype(np.uint8)  # Darken background

        # Colors for different objects
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 100),  # Orange
            (150, 100, 255),  # Purple
        ]

        # Determine which objects are selected
        use_all_objects = "All Objects" in selected_objects
        selected_indices = []

        if use_all_objects:
            selected_indices = list(range(len(detection_results["detections"])))
        else:
            for item in selected_objects:
                if item.startswith(
                    ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")
                ):
                    try:
                        idx_str = item.split(".")[0]
                        idx = int(idx_str) - 1  # Convert to 0-based
                        selected_indices.append(idx)
                    except Exception as e:
                        logging.exception(f"PREVIEW: Error parsing '{item}': {e}")

        # Highlight only selected masks
        for i, detection in enumerate(detection_results["detections"]):
            if i in selected_indices:
                mask = detection.get("mask")
                if mask is not None:
                    # Resize mask to image size
                    if mask.shape != (h, w):
                        mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                        mask_bool = mask_resized > 0.5
                    else:
                        mask_bool = mask > 0.5

                    # Get color for this object
                    color = colors[i % len(colors)]

                    # Restore original brightness for selected areas and add color overlay
                    preview[mask_bool] = (
                        img_array[mask_bool] * 0.8 + np.array(color) * 0.2
                    )

        # Convert back to PIL and add labels for selected objects
        preview_pil = Image.fromarray(preview.astype(np.uint8))
        draw = ImageDraw.Draw(preview_pil)

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        # Draw labels only for selected objects
        for i, detection in enumerate(detection_results["detections"]):
            if i in selected_indices:
                bbox = detection["bbox"]
                obj_name = detection["object"]
                confidence = detection["confidence"]

                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

                # Create label
                label = f"✓ {i+1}. {obj_name} ({confidence:.1f}%)"

                # Draw label background
                if font:
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                else:
                    text_width = len(label) * 10
                    text_height = 20

                label_bg_coords = [x1, y1 - text_height - 5, x1 + text_width + 10, y1]
                draw.rectangle(label_bg_coords, fill=(0, 0, 0, 180))

                # Draw label text
                if font:
                    draw.text(
                        (x1 + 5, y1 - text_height - 2), label, fill=color, font=font
                    )
                else:
                    draw.text((x1 + 5, y1 - text_height - 2), label, fill=color)

        return preview_pil

    def update_mask_preview(self, image, selected_objects):
        """Update the mask preview when selection changes"""
        if (
            not hasattr(self, "last_detection_results")
            or not self.last_detection_results
        ):
            return None

        if hasattr(self, "last_scaled_image") and self.last_scaled_image is not None:
            image_to_use = self.last_scaled_image
        else:
            image_to_use = self.scale_image(image) if image else None

        if image_to_use is None:
            return None

        return self.create_selected_mask_preview(
            image_to_use, self.last_detection_results, selected_objects
        )

    def create_masked_image(self, image, detection_results):
        """Create image with colored masks and bounding boxes overlaid on detected objects"""
        import cv2
        from PIL import ImageDraw, ImageFont

        if not detection_results.get("detections"):
            return image

        # Convert PIL to numpy array
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Create overlay
        overlay = img_array.copy()

        # Colors for different objects (bright, distinct colors)
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 100),  # Orange
            (150, 100, 255),  # Purple
        ]

        # Apply masks first
        for i, detection in enumerate(detection_results["detections"]):
            mask = detection.get("mask")
            if mask is not None:
                # Resize mask to image size
                if mask.shape != (h, w):
                    mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                    mask_bool = mask_resized > 0.5
                else:
                    mask_bool = mask > 0.5

                # Get color for this object
                color = colors[i % len(colors)]

                # Apply colored mask (semi-transparent)
                overlay[mask_bool] = overlay[mask_bool] * 0.7 + np.array(color) * 0.3

        # Convert back to PIL to draw bounding boxes and text
        overlay_pil = Image.fromarray(overlay.astype(np.uint8))
        draw = ImageDraw.Draw(overlay_pil)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
            except:
                font = None
                font_small = None

        # Draw bounding boxes and labels
        for i, detection in enumerate(detection_results["detections"]):
            bbox = detection["bbox"]
            obj_name = detection["object"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]

            # Draw bounding box (thick border)
            line_width = 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Create label text
            label = f"{i+1}. {obj_name} ({confidence:.1f}%)"

            # Get text size
            if font:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                # Estimate text size if no font available
                text_width = len(label) * 8
                text_height = 15

            # Draw label background (semi-transparent)
            label_bg_coords = [x1, y1 - text_height - 5, x1 + text_width + 10, y1]
            draw.rectangle(label_bg_coords, fill=(*color, 200))  # Semi-transparent

            # Draw label text
            if font:
                draw.text(
                    (x1 + 5, y1 - text_height - 2), label, fill=(0, 0, 0), font=font
                )
            else:
                draw.text((x1 + 5, y1 - text_height - 2), label, fill=(0, 0, 0))

            # Also draw index number in corner for easy reference
            index_text = str(i + 1)
            circle_radius = 15
            circle_center = (x1 + circle_radius, y1 + circle_radius)

            # Draw circle background
            circle_bbox = [
                circle_center[0] - circle_radius,
                circle_center[1] - circle_radius,
                circle_center[0] + circle_radius,
                circle_center[1] + circle_radius,
            ]
            draw.ellipse(circle_bbox, fill=color, outline=(0, 0, 0), width=2)

            # Draw index number
            if font_small:
                # Center the text in the circle
                text_bbox = draw.textbbox((0, 0), index_text, font=font_small)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                text_x = circle_center[0] - text_w // 2
                text_y = circle_center[1] - text_h // 2
                draw.text((text_x, text_y), index_text, fill=(0, 0, 0), font=font_small)
            else:
                draw.text(
                    (circle_center[0] - 5, circle_center[1] - 8),
                    index_text,
                    fill=(0, 0, 0),
                )

        return overlay_pil

    def detect_objects_with_masks(self, image, confidence_threshold=0.5):
        """Detect objects and return analysis, masked image, and object choices"""
        if not self.object_detector or image is None:
            return (
                "Object detection not available or no image provided",
                None,
                gr.CheckboxGroup(choices=[], value=[], visible=False),
                None,
            )

        try:
            self.object_detector.confidence_threshold = confidence_threshold
            scaled_image = self.scale_image(image)
            self.last_scaled_image = scaled_image

            # Save temp image
            temp_path = "temp_image.jpg"
            scaled_image.save(temp_path)

            # Detect objects
            results = self.object_detector.detect_objects(temp_path)
            self.last_detection_results = results

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # DEBUG: logging.info detection results
            # logging.info(f"DEBUG: Detection results:")
            # for i, detection in enumerate(results.get("detections", [])):
            #     obj_name = detection["object"]
            #     confidence = detection["confidence"]
            #     bbox = detection["bbox"]
            #     has_mask = detection.get("mask") is not None
            #     if has_mask:
            #         mask_shape = detection["mask"].shape
            #         mask_pixels = np.sum(detection["mask"] > 0.5)
            #     else:
            #         mask_shape = "None"
            #         mask_pixels = 0
            #     logging.info(
            #         f"  [{i}] {obj_name} ({confidence:.1f}%) - BBox: {bbox} - Mask: {mask_shape} ({mask_pixels} pixels)"
            #     )

            for i, detection in enumerate(results.get("detections", [])):
                obj_name = detection["object"]
                confidence = detection["confidence"]

            # Create masked image with labels
            masked_image = self.create_masked_image(scaled_image, results)

            # Create object choices for checkboxes
            choices = []
            default_selection = []
            if results.get("detections"):
                choices.append("All Objects")
                default_selection = ["All Objects"]

                for i, detection in enumerate(results["detections"]):
                    obj_name = detection["object"]
                    confidence = detection["confidence"]
                    choices.append(f"{i+1}. {obj_name} ({confidence:.1f}%)")

            initial_preview = None
            if choices and scaled_image:
                initial_preview = self.create_selected_mask_preview(
                    scaled_image, results, default_selection
                )

            # Format text results
            text_analysis = self._format_object_detection(results)

            # Return updated checkbox group
            object_selection_update = gr.CheckboxGroup(
                choices=choices,
                value=["All Objects"] if choices else [],  # Default to "All Objects"
                visible=len(choices) > 0,
                label="Select Objects for Color Extraction",
                info="Select which detected objects to include in color analysis",
            )

            return text_analysis, masked_image, object_selection_update, initial_preview

        except Exception as e:
            logging.exception(f"Detection error: {str(e)}")
            return (
                f"Object detection failed: {str(e)}",
                None,
                gr.CheckboxGroup(choices=[], value=[], visible=False),
                None,
            )

    def create_interface(self):
        """Create and return the Gradio interface"""

        # Define the interface layout
        with gr.Blocks(
            title="Image Analysis System",
            theme=gr.themes.Soft(),
            css="""
            .color-scheme-container {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
            }
            .analysis-text {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.0;
            }
            .gradient-style {
                background: linear-gradient(90deg, #5542e3 0%, #34288f 100%) !important;
                border-radius: 8px !important;
                padding: 5px !important;
            }

            """,
        ) as interface:
            gr.Markdown("Upload an image to extract dominant colors and detect objects")

            with gr.Row():
                # Left column: Image input and controls
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Load Image",
                        placeholder="Click to upload an image or drag and drop",
                        height=500,
                    )

                    n_colors_slider = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=5,
                        label="Number of Colors",
                        info="Adjust how many dominant colors to extract",
                    )
                    confidence_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                        label="Object detection threshold",
                        info="Lower values detect more objects (less confident)",
                    )

                    extract_from_object_toggle = gr.Checkbox(
                        label="Extract colors from detected objects (vs whole image)",
                        value=False,
                        info="When OFF: Always extracts from whole image. When ON: Extracts from selected objects below.",
                        elem_classes=["gradient-style"],
                    )

                    with gr.Row():
                        extract_colors_btn = gr.Button(
                            "Extract Colors", variant="primary", size="lg"
                        )
                        detect_objects_btn = gr.Button(
                            "Detect Objects", variant="secondary", size="lg"
                        )

                    analyze_all_btn = gr.Button(
                        "Analyze Everything", variant="stop", size="lg"
                    )

                # Right column: Results
                with gr.Column(scale=1):
                    # Masked image output
                    masked_image_output = gr.Image(
                        label="Object Detection Masks",
                        height=300,
                        elem_classes=["color-scheme-container"],
                    )

                    selected_mask_preview = gr.Image(
                        label="Selected Objects Preview",
                        height=300,
                        elem_classes=["color-scheme-container"],
                    )

                    # Color sceheme
                    color_scheme_output = gr.Image(
                        label="Color Scheme (Proportional)",
                        height=250,
                        elem_classes=["color-scheme-container"],
                    )

                with gr.Tabs():
                    with gr.Tab("Color Analysis"):
                        color_analysis_output = gr.Markdown(
                            value="*Extract colors to see analysis*",
                            elem_classes=["analysis-text"],
                        )

                    with gr.Tab("Object Detection"):
                        object_selection = gr.CheckboxGroup(
                            label="Select Objects for Color Extraction",
                            choices=[],
                            value=[],
                            visible=False,
                            info="Select which detected objects to include in color analysis",
                        )

                        object_analysis_output = gr.Markdown(
                            value="*Detect objects to see results*",
                            elem_classes=["analysis-text"],
                        )

            # Usage info
            with gr.Accordion(
                "💡 How to use Object-Focused Color Extraction", open=False
            ):
                gr.Markdown(
                    """
                **🎯 Object-Focused Mode:**
                - Extracts colors primarily from the detected main object
                - Perfect for challenge-style analysis: "red bus", "black and white zebra"
                - Automatically falls back to whole image if no object is detected
                
                **🖼️ Whole Image Mode:**
                - Analyzes colors from the entire image 
                - Good for overall composition analysis
                - Includes background, sky, surroundings
                
                **💡 Tips:**
                - Enable object detection first to see what objects are found
                - Try both modes to see the difference
                - Object-focused works best with clear, prominent subjects
                """
                )

            # Status and info
            gr.Markdown("---")
            with gr.Row():
                gr.Markdown(
                    f"**Status:** "
                    + f"{'✅ Color Extraction' if COLOR_EXTRACTION_AVAILABLE else '❌ Color Extraction'} | "
                    + f"{'✅ Object Detection' if OBJECT_DETECTION_AVAILABLE else '❌ Object Detection'}"
                )

            # Event handlers - REMOVED auto-change events
            extract_colors_btn.click(
                fn=self.extract_colors,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    # all_objects_toggle,
                    object_selection,
                ],
                outputs=[color_scheme_output, color_analysis_output],
            )

            detect_objects_btn.click(
                fn=self.detect_objects_with_masks,
                inputs=[image_input, confidence_threshold],
                outputs=[
                    object_analysis_output,
                    masked_image_output,
                    object_selection,
                    selected_mask_preview,
                ],
            )

            analyze_all_btn.click(
                fn=self.analyze_complete,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    # all_objects_toggle,
                    object_selection,
                    confidence_threshold,
                ],
                outputs=[
                    color_scheme_output,
                    color_analysis_output,
                    object_analysis_output,
                    masked_image_output,
                    object_selection,
                    selected_mask_preview,
                ],
            )

            object_selection.change(
                fn=self.update_mask_preview,
                inputs=[image_input, object_selection],
                outputs=[selected_mask_preview],
            )

        return interface


def launch_ui(share=False, debug=False, port=7860):
    """
    Launch the Gradio interface

    Args:
        share: Whether to create a public link
        debug: Whether to enable debug mode
        port: Port number to use (default: 7860)
    """
    logging.info("🚀 Launching Image Analysis UI...")

    # Check system status
    logging.info("\n🔧 System Status:")
    logging.info(
        f"   Color Extraction: {'✅ Available' if COLOR_EXTRACTION_AVAILABLE else '❌ Not Available'}"
    )
    logging.info(
        f"   Object Detection: {'✅ Available' if OBJECT_DETECTION_AVAILABLE else '❌ Not Available'}"
    )

    if not COLOR_EXTRACTION_AVAILABLE and not OBJECT_DETECTION_AVAILABLE:
        logging.info("\n❌ No analysis modules available!")
        logging.info(
            "Install dependencies: pip install numpy opencv-python scikit-learn pillow ultralytics"
        )
        return

    # Create and launch interface
    ui = ImageAnalysisUI()
    interface = ui.create_interface()

    interface.launch(
        share=share,
        debug=debug,
        server_name="0.0.0.0" if share else "127.0.0.1",
        server_port=port,  # Configurable port
        show_error=True,
        quiet=not debug,
    )


# Example usage and main function
if __name__ == "__main__":
    # You can customize these settings
    launch_ui(
        share=False,  # Set to True to create a public link
        debug=True,  # Set to True for development
        port=7860,  # Default Gradio port
    )
