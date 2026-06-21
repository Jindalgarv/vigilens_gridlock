import cv2
import glob
import os
from vehicle_detector import VehicleDetector

def test_on_dataset(dataset_path):
    print(f"Testing on images from {dataset_path}...")
    
    # Grab a few images
    images = glob.glob(f"{dataset_path}/**/*.jpg", recursive=True) + \
             glob.glob(f"{dataset_path}/**/*.png", recursive=True)
             
    if not images:
        print(f"No images found in {dataset_path}")
        return

    # Initialize the detector
    detector = VehicleDetector(model_size='l', confidence_threshold=0.3)
    
    os.makedirs('problem/output_samples', exist_ok=True)

    # Test on up to 2 images
    for i, img_path in enumerate(images[:2]):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"Processing: {os.path.basename(img_path)}")
        
        # In a real pipeline, we'd run preprocessing first. 
        # For this test, we just pass the raw image to verify detection works.
        detections, _ = detector.detect(img)
        
        print(f"Found {len(detections)} traffic objects.")
        for d in detections:
            print(f"  - {d['class']} ({d['confidence']:.2f})")
            
        annotated_img = detector.draw_detections(img, detections)
        
        out_path = f"problem/output_samples/yolo_test_{os.path.basename(img_path)}"
        cv2.imwrite(out_path, annotated_img)
        print(f"Saved annotated image to {out_path}\n")

if __name__ == "__main__":
    # Test on the DAWN dataset (which usually has cars/trucks)
    test_on_dataset("../../dataset/DAWN")
    # If the relative path above fails, try the other one
    test_on_dataset("dataset/DAWN")
