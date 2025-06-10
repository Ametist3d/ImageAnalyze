from auto_color_extractor import AutoColorExtractor
from object_detector import ObjectDetector


def main():
    extractor = AutoColorExtractor(n_colors=5)
    detector = ObjectDetector(confidence_threshold=0.6)


    test_images = [
        "images/anime.jpg",
        "images/fabric.png",  
        "images/FrenchBuldog.png", 
        "images/panda_graffiti.png"   
    ]

    for image_path in test_images:
        try:
            print(f"\n🖼️  ANALYZING: {image_path}")
            print("=" * 50)

            # Automatic extraction
            results = extractor.extract_colors_adaptive(image_path, debug=True)
            objects = detector.detect_objects(image_path)

            print(f"\n🔍 OBJECTS DETECTED for {image_path}:")
            if objects["detections"]:
                for i, obj in enumerate(objects["detections"], 1):  # Fixed: obj not object
                    print(f"{i}. {obj['object']} - {obj['confidence']}% confidence")
            else:
                print("No objects detected")

            print(f"\n🎨 DOMINANT COLORS:")
            for i, color in enumerate(results["dominant_colors"], 1):
                print(
                    f"{i}. {color['color_name']} - RGB{color['rgb']} ({color['frequency']}%)"
                )

            print(f"\n📊 IMAGE ANALYSIS:")
            analysis = results["image_analysis"]
            print(f"Type: {analysis['type']}")
            print(f"Strategy: {analysis['strategy_used']}")
            print(f"Confidence: {analysis['confidence']:.1%}")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()