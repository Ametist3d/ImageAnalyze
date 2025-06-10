"""
App Launcher for Image Analysis System
Simple script to launch the Gradio UI
"""

import sys
import os

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gradio_ui import launch_ui
    
    if __name__ == "__main__":
        print("🎨 Image Analysis System")
        print("=" * 40)
        
        # Launch the UI
        launch_ui(
            share=True,  # Change to True for public sharing
            debug=False   # Change to True for development
        )
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n📦 Missing dependencies. Install with:")
    print("   pip install -r requirements.txt")
    print("\nOr install individually:")
    print("   pip install gradio numpy opencv-python scikit-learn pillow ultralytics")
except Exception as e:
    print(f"❌ Error launching app: {e}")
    print("\n🔧 Check that all required files are present:")
    print("   - auto_color_extractor.py")
    print("   - object_detector.py") 
    print("   - gradio_ui.py")