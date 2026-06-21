import cv2
import glob
import os
import sys
import numpy as np

# Add src to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import SmartPreprocessor
from detection.vehicle_detector import VehicleDetector

def evaluate_pipeline(dataset_path):
    print(f"--- Testing Combined Pipeline on {dataset_path} ---")
    images = glob.glob(f"{dataset_path}/**/*.jpg", recursive=True) + \
             glob.glob(f"{dataset_path}/**/*.png", recursive=True)
             
    if not images:
        print("No images found.\n")
        return

    # Initialize modules
    preprocessor = SmartPreprocessor()
    detector = VehicleDetector(model_size='l', confidence_threshold=0.25)
    
    os.makedirs('problem/output_samples/combined', exist_ok=True)

    # Test on a specific dark or bad weather image if possible, else first available
    img_path = images[0]
    for p in images:
        if 'rain_storm-172.jpg' in p:
            img_path = p
            break

    if not img_path:
        print("Could not find the target image.")
        return

    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print("Failed to read image.")
        return

    print(f"\nProcessing: {os.path.basename(img_path)}")

    # --- Test 1: Raw Image straight to YOLO ---
    # Need to letterbox raw image so sizes match for fair comparison
    raw_resized = preprocessor.letterbox(raw_img)
    raw_dets, _ = detector.detect(raw_resized)
    raw_annotated = detector.draw_detections(raw_resized, raw_dets)
    
    print(f"[Baseline] Raw Image Detections: {len(raw_dets)}")
    raw_conf_sum = sum([d['confidence'] for d in raw_dets])
    print(f"[Baseline] Total Confidence Sum: {raw_conf_sum:.2f}")

    # --- Test 2: Combined Pipeline (Preprocess -> YOLO) ---
    enhanced_img, metrics = preprocessor.process(raw_img)
    print(f"[Pipeline] Applied Enhancements based on metrics: {metrics}")
    
    enh_dets, _ = detector.detect(enhanced_img)
    enh_annotated = detector.draw_detections(enhanced_img, enh_dets)
    
    print(f"[Pipeline] Enhanced Image Detections: {len(enh_dets)}")
    enh_conf_sum = sum([d['confidence'] for d in enh_dets])
    print(f"[Pipeline] Total Confidence Sum: {enh_conf_sum:.2f}")

    # --- Compare and Save ---
    # Stack side by side for visual comparison
    comparison = np.hstack((raw_annotated, enh_annotated))
    
    # Add titles
    cv2.putText(comparison, f"RAW (Dets: {len(raw_dets)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, f"ENHANCED (Dets: {len(enh_dets)})", (raw_annotated.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out_path = f"problem/output_samples/combined/eval_{os.path.basename(img_path)}"
    cv2.imwrite(out_path, comparison)
    print(f"\nSaved comparison image to: {out_path}\n")

if __name__ == "__main__":
    evaluate_pipeline("dataset/ExDark")
    evaluate_pipeline("dataset/DAWN")

