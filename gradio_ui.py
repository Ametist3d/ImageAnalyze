"""
Gradio UI for Image Analysis System
Provides an interactive web interface for color extraction and object detection
"""

import gradio as gr
import numpy as np
from PIL import Image
import os

# Import our modules
try:
    from auto_color_extractor import AutoColorExtractor

    COLOR_EXTRACTION_AVAILABLE = True
except ImportError:
    COLOR_EXTRACTION_AVAILABLE = False
    print("❌ Color extraction not available")

try:
    from object_detector import ObjectDetector

    OBJECT_DETECTION_AVAILABLE = True
except ImportError:
    OBJECT_DETECTION_AVAILABLE = False
    print("❌ Object detection not available")


class ImageAnalysisUI:
    def __init__(self):
        """Initialize the UI with available modules"""
        self.color_extractor = None
        self.object_detector = None

        # Initialize color extractor
        if COLOR_EXTRACTION_AVAILABLE:
            self.color_extractor = AutoColorExtractor()
            print("✅ Color extraction ready")

        # Initialize object detector
        if OBJECT_DETECTION_AVAILABLE:
            try:
                self.object_detector = ObjectDetector(confidence_threshold=0.5)
                print("✅ Object detection ready")
            except Exception as e:
                print(f"⚠️ Object detection failed to initialize: {e}")

    def scale_image(self, image):
        """Scale down image to 1024 on largest side if it's larger than 1024"""
        if image is None:
            return None

        # Make sure we have a PIL Image object
        if not isinstance(image, Image.Image):
            print(f"Warning: Expected PIL Image, got {type(image)}")
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

            print(f"Image scaled from {width}x{height} to {new_width}x{new_height}")

        return image

    def extract_colors(self, image, n_colors, extract_from_object, use_all_objects):
        """
        Extract colors and create color scheme visualization

        Args:
            image: PIL Image from Gradio
            n_colors: Number of colors to extract
            extract_from_object: Boolean - whether to extract from object or whole image

        Returns:
            tuple: (color_scheme_image, analysis_text)
        """
        if not self.color_extractor or image is None:
            return None, "❌ Color extraction not available or no image provided"

        try:
            # Scale the image first
            image = self.scale_image(image)

            # Update number of colors
            self.color_extractor.n_colors = n_colors

            # Build color scheme visualization with object focus option
            color_scheme, results = self.color_extractor.build_color_scheme(
                image,
                object_detector=self.object_detector,
                extract_from_object=extract_from_object,
                use_all_objects=use_all_objects,
                width=400,
                height=100,
            )

            # Check if there was an error in color extraction
            if "error" in results:
                error_message = f"❌ {results['error']}\n\n💡 **Possible solutions:**\n"
                error_message += "• Try a different image format (JPG, PNG)\n"
                error_message += "• Ensure image is not corrupted\n"
                error_message += "• Check if image has sufficient color variation\n"
                error_message += "• Try reducing the number of colors"
                return color_scheme, error_message

            # Get detailed color information
            palette_data = self.color_extractor.get_color_palette_data(
                image,
                object_detector=self.object_detector,
                extract_from_object=extract_from_object,
                use_all_objects=use_all_objects,
            )

            # Check if palette extraction also failed
            if "error" in palette_data or not palette_data.get("colors"):
                error_message = "❌ Failed to extract color palette from image"
                return color_scheme, error_message

            # Format analysis text
            analysis_text = self._format_color_analysis(palette_data)

            return color_scheme, analysis_text

        except Exception as e:
            error_message = f"❌ Unexpected error in color extraction: {str(e)}\n\n"
            error_message += "💡 **Try:**\n• Different image format\n• Smaller image size\n• Different number of colors"
            return None, error_message

    def detect_objects(self, image):
        """
        Detect objects in the image

        Args:
            image: PIL Image from Gradio

        Returns:
            str: Formatted object detection results
        """
        if not self.object_detector or image is None:
            return "❌ Object detection not available or no image provided"

        try:
            # Scale the image first
            image = self.scale_image(image)

            # Save image temporarily for object detection
            temp_path = "temp_image.jpg"
            image.save(temp_path)

            # Detect objects
            results = self.object_detector.detect_objects(temp_path)

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Format results
            return self._format_object_detection(results)

        except Exception as e:
            return f"❌ Object detection failed: {str(e)}"

    def analyze_complete(self, image, n_colors, extract_from_object, use_all_objects):
        """
        Complete analysis: both colors and objects

        Args:
            image: PIL Image from Gradio
            n_colors: Number of colors to extract
            extract_from_object: Boolean - whether to extract from object or whole image

        Returns:
            tuple: (color_scheme_image, color_analysis, object_analysis)
        """
        # Scale the image first (once)
        if image is not None:
            image = self.scale_image(image)

        # Extract colors
        color_scheme, color_analysis = self.extract_colors(
            image, n_colors, extract_from_object, use_all_objects
        )

        # Detect objects
        object_analysis = self.detect_objects(image)

        return color_scheme, color_analysis, object_analysis

    def _format_color_analysis(self, palette_data):
        """Format color analysis for display"""
        if not palette_data:
            return "❌ No color data available"

        # Check for errors
        if "error" in palette_data:
            return f"❌ Error: {palette_data['error']}"

        analysis = palette_data.get("image_analysis", {})
        colors = palette_data.get("colors", [])

        if not colors:
            return "❌ No colors extracted from image"

        # Header with image analysis
        text = f"🎨 **COLOR ANALYSIS**\n\n"
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
            return f"❌ {results['error']}"

        detections = results.get("detections", [])
        method = results.get("method", "Unknown")

        if not detections:
            return f"🔍 **OBJECT DETECTION** ({method})\n\nNo objects detected with sufficient confidence"

        text = f"🔍 **OBJECT DETECTION** ({method})\n\n"
        text += f"**Found {len(detections)} objects:**\n"

        for i, obj in enumerate(detections, 1):
            text += f"{i}. **{obj['object']}** - {obj['confidence']:.1f}% confidence\n"

        return text

    def create_interface(self):
        """Create and return the Gradio interface"""

        # Define the interface layout
        with gr.Blocks(
            title="🎨 Image Analysis System",
            theme=gr.themes.Soft(),
            css="""
            .color-scheme-container {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
            }
            .analysis-text {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
            }
            .gradient-style {
                background: linear-gradient(90deg, #5542e3 0%, #34288f 100%) !important;
                border-radius: 8px !important;
                padding: 5px !important;
            }
            .gradient-style-invert {
                background: linear-gradient(90deg, #34288f 0%, #5542e3 100%) !important;
                border-radius: 8px !important;
                padding: 5px !important;
            }
            """,
        ) as interface:

            gr.Markdown("# 🎨 Image Analysis System")
            gr.Markdown(
                "Upload an image to extract dominant colors and detect objects using AI"
            )

            with gr.Row():
                # Left column: Image input and controls
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="📁 Load Image",
                        placeholder="Click to upload an image or drag and drop",
                        height=500
                    )

                    n_colors_slider = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=5,
                        label="🎨 Number of Colors (n_colors)",
                        info="Adjust how many dominant colors to extract",
                    )

                    extract_from_object_toggle = gr.Checkbox(
                        label="🎯 Extract colors from detected object (vs whole image)",
                        value=False,
                        info="When enabled, focuses on the main object's colors. Falls back to whole image if no object detected.",
                        elem_classes=["gradient-style-invert"],
                    )

                    all_objects_toggle = gr.Checkbox(
                        label="🔀 Use all detected objects (vs main object only)",
                        value=False,
                        info="When enabled with object extraction, combines colors from all detected objects",
                        elem_classes=["gradient-style"],
                    )

                    with gr.Row():
                        extract_colors_btn = gr.Button(
                            "🎨 Extract Colors", variant="primary", size="lg"
                        )
                        detect_objects_btn = gr.Button(
                            "🔍 Detect Objects", variant="secondary", size="lg"
                        )

                    analyze_all_btn = gr.Button(
                        "🚀 Analyze Everything", variant="stop", size="lg"
                    )

                # Right column: Results
                with gr.Column(scale=1):
                    color_scheme_output = gr.Image(
                        label="🌈 Color Scheme (Proportional)",
                        height=120,
                        elem_classes=["color-scheme-container"],
                    )

                    with gr.Tabs():
                        with gr.Tab("🎨 Color Analysis"):
                            color_analysis_output = gr.Markdown(
                                value="*Extract colors to see analysis*",
                                elem_classes=["analysis-text"],
                            )

                        with gr.Tab("🔍 Object Detection"):
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

            # Event handlers - all now include the toggle parameter
            extract_colors_btn.click(
                fn=self.extract_colors,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    all_objects_toggle,
                ],
                outputs=[color_scheme_output, color_analysis_output],
            )

            detect_objects_btn.click(
                fn=self.detect_objects,
                inputs=[image_input],
                outputs=[object_analysis_output],
            )

            analyze_all_btn.click(
                fn=self.analyze_complete,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    all_objects_toggle,
                ],
                outputs=[
                    color_scheme_output,
                    color_analysis_output,
                    object_analysis_output,
                ],
            )

            # Auto-update when n_colors changes
            n_colors_slider.change(
                fn=self.extract_colors,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    all_objects_toggle,
                ],
                outputs=[color_scheme_output, color_analysis_output],
            )

            # Auto-update when extraction method toggle changes
            extract_from_object_toggle.change(
                fn=self.extract_colors,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    all_objects_toggle,
                ],
                outputs=[color_scheme_output, color_analysis_output],
            )

            all_objects_toggle.change(
                fn=self.extract_colors,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    all_objects_toggle,
                ],
                outputs=[color_scheme_output, color_analysis_output],
            )

            # Auto-analyze when image is uploaded
            image_input.change(
                fn=self.analyze_complete,
                inputs=[
                    image_input,
                    n_colors_slider,
                    extract_from_object_toggle,
                    all_objects_toggle,
                ],
                outputs=[
                    color_scheme_output,
                    color_analysis_output,
                    object_analysis_output,
                ],
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
    print("🚀 Launching Image Analysis UI...")

    # Check system status
    print("\n🔧 System Status:")
    print(
        f"   Color Extraction: {'✅ Available' if COLOR_EXTRACTION_AVAILABLE else '❌ Not Available'}"
    )
    print(
        f"   Object Detection: {'✅ Available' if OBJECT_DETECTION_AVAILABLE else '❌ Not Available'}"
    )

    if not COLOR_EXTRACTION_AVAILABLE and not OBJECT_DETECTION_AVAILABLE:
        print("\n❌ No analysis modules available!")
        print(
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
