import os
import sys
sys.path.append('.')
from ai_image_detector import AIImageDetector
import glob


def test_single_image(detector, image_path, expected_type="unknown"):
    print(f"\n{'='*60}")
    print(f"📸 Testing: {os.path.basename(image_path)}")
    print(f"Expected: {expected_type}")
    print(f"{'='*60}")
    
    try:
        result = detector.predict(image_path)
        
        print(f"\n🎯 RESULTS:")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Verdict: {result['verdict']}")
        print(f"   Method: {result['method']}")
        
        if result.get('metrics'):
            print(f"\n📊 Technical Metrics:")
            for key, value in result['metrics'].items():
                print(f"   {key}: {value:.4f}")
        
        if result.get('explanation'):
            print(f"\n💡 Explanation:")
            print(f"   {result['explanation']}")
        
        # Check if prediction matches expected
        is_correct = (expected_type == "unknown") or \
                    (expected_type == "real" and not result['is_ai_generated']) or \
                    (expected_type == "ai" and result['is_ai_generated'])
        
        if expected_type != "unknown":
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"\n{status}")
        
        return is_correct
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_folder(detector, folder_path, expected_type):
    """Test all images in a folder"""
    print(f"\n{'='*70}")
    print(f"📁 Testing folder: {folder_path}")
    print(f"Expected type: {expected_type.upper()}")
    print(f"{'='*70}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not images:
        print(f"⚠️ No images found in {folder_path}")
        return 0, 0
    
    print(f"Found {len(images)} images")
    
    correct = 0
    total = len(images)
    
    for img_path in images:
        is_correct = test_single_image(detector, img_path, expected_type)
        if is_correct:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*70}")
    print(f"📊 FOLDER RESULTS:")
    print(f"   Correct: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}")
    
    return correct, total


def run_full_test(sensitivity='medium'):
    """Run complete test suite"""
    print("\n" + "="*70)
    print("🧪 AI IMAGE DETECTOR - FULL TEST SUITE")
    print("="*70)
    
    # Initialize detector
    print("\n🔄 Initializing detector...")
    detector = AIImageDetector(method='artifact', sensitivity=sensitivity)  # Change to 'huggingface' or 'hybrid' if available
    print(f"✅ Detector initialized (sensitivity={sensitivity})")

    
    # Test folders
    test_cases = [
        ('data/images/test/real_images', 'real'),
        ('data/images/test/ai_generated', 'ai'),
    ]
    
    total_correct = 0
    total_images = 0
    
    for folder, expected_type in test_cases:
        if os.path.exists(folder):
            correct, count = test_folder(detector, folder, expected_type)
            total_correct += correct
            total_images += count
        else:
            print(f"\n⚠️ Folder not found: {folder}")
            print(f"   Please create it and add test images")
    
    # Final summary
    if total_images > 0:
        overall_accuracy = (total_correct / total_images) * 100
        print(f"\n{'='*70}")
        print(f"📈 OVERALL RESULTS:")
        print(f"   Total Images: {total_images}")
        print(f"   Correct: {total_correct}")
        print(f"   Wrong: {total_images - total_correct}")
        print(f"   Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"{'='*70}\n")
    else:
        print(f"\n⚠️ No test images found!")
        print(f"\nTo test the detector:")
        print(f"1. Create folders:")
        print(f"   - data/images/test/real_images/")
        print(f"   - data/images/test/ai_generated/")
        print(f"2. Add test images to each folder")
        print(f"3. Run this script again")


def quick_test():
    """Quick test on a single image"""
    print("\n" + "="*70)
    print("🧪 AI IMAGE DETECTOR - QUICK TEST")
    print("="*70)
    
    # Initialize detector
    print("\n🔄 Initializing detector...")
    detector = AIImageDetector(method='artifact')
    print("✅ Detector initialized")
    
    # Test on first available image
    test_folders = [
        'data/images/test/real_images',
        'data/images/test/ai_generated',
        'data/images/test/laptop',
        'data/images/test/mobile',
        'data/images/train/laptop',
    ]
    
    test_image = None
    for folder in test_folders:
        if os.path.exists(folder):
            images = glob.glob(os.path.join(folder, '*.jpg')) + \
                    glob.glob(os.path.join(folder, '*.png'))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        test_single_image(detector, test_image)
    else:
        print("\n❌ No test images found!")
        print("Please add images to one of these folders:")
        for folder in test_folders:
            print(f"   - {folder}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AI Image Detector')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Test mode: quick (single image) or full (all test folders)')
    parser.add_argument('--image', type=str, help='Path to specific image to test')
    parser.add_argument('--sensitivity', choices=['low','medium','high'], default='medium', help='Sensitivity level for artifact detector')
    
    args = parser.parse_args()
    
    if args.image:
        # Test specific image
        detector = AIImageDetector(method='artifact', sensitivity=args.sensitivity)
        test_single_image(detector, args.image)
    elif args.mode == 'full':
        run_full_test(sensitivity=args.sensitivity)
    else:
        quick_test()