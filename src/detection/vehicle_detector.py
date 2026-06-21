import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

class VehicleDetector:
    """
    Handles the detection of vehicles, riders, drivers, and pedestrians using YOLO11.
    Filters raw YOLO predictions to focus strictly on traffic participants.
    """
    def __init__(self, model_size='l', confidence_threshold=0.3):
        # We are using YOLO11 Large (l) as specified for maximum accuracy with dedicated GPU
        model_name = f'yolo11{model_size}.pt'
        
        print(f"[Init] Loading {model_name}...")
        
        # Load the YOLO model (Ultralytics handles downloading the weights automatically)
        self.model = YOLO(model_name)
        
        # Explicitly set device (uses CUDA if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"[Init] YOLO Model loaded on {self.device}.")

        self.confidence_threshold = confidence_threshold

        # COCO Class IDs relevant to traffic violations
        # We ignore classes like 'dog', 'train' (unless needed), 'stop sign', etc.
        self.target_classes = {
            0: 'person',       # To detect riders/pedestrians
            1: 'bicycle',      
            2: 'car',
            3: 'motorcycle',   # Crucial for helmet/triple riding
            5: 'bus',
            7: 'truck'
        }
        self.target_class_ids = list(self.target_classes.keys())

    def detect(self, image):
        """
        Runs the image through YOLO and filters the output.
        Expects a preprocessed RGB image.
        """
        # Run inference
        # verbose=False prevents the console from being flooded with print statements
        results = self.model.predict(source=image, 
                                     conf=self.confidence_threshold, 
                                     classes=self.target_class_ids,
                                     device=self.device,
                                     verbose=False)
        
        # Ultralytics returns a list of Results objects (one per image)
        result = results[0]
        
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Bounding box coordinates (x_center, y_center, width, height) or (x1, y1, x2, y2)
                # We extract xyxy (top-left, bottom-right) for easy drawing
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Confidence score
                conf = float(box.conf[0])
                
                # Class ID
                cls_id = int(box.cls[0])
                class_name = self.target_classes[cls_id]
                
                detections.append({
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                
        return detections, result

    def draw_detections(self, image, detections):
        """
        Utility function to draw bounding boxes and labels on the image for visual verification.
        """
        img_copy = image.copy()
        
        # Color map for different classes
        colors = {
            'person': (0, 0, 255),       # Red
            'bicycle': (255, 0, 0),      # Blue
            'car': (0, 255, 0),          # Green
            'motorcycle': (255, 255, 0), # Cyan
            'bus': (0, 255, 255),        # Yellow
            'truck': (255, 0, 255)       # Magenta
        }

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']
            color = colors.get(cls_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw text
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return img_copy

# Quick test stub
if __name__ == "__main__":
    detector = VehicleDetector(model_size='l')
    dummy_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    dets, raw_result = detector.detect(dummy_img)
    print(f"Pipeline test successful. Detected {len(dets)} traffic objects in dummy image.")
