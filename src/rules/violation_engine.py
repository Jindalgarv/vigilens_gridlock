import cv2
import numpy as np

# Handle imports whether run as a script or a module
try:
    from .plugins.overcrowding import OvercrowdingRule
    from .plugins.stop_line import StopLineRule
    from .plugins.hazardous_load import HazardousLoadRule
    from .plugins.helmet import HelmetNonComplianceRule
    from .plugins.red_light import RedLightViolationRule
    from .plugins.illegal_parking import IllegalParkingRule
    from .plugins.wrong_side import WrongSideDrivingRule
    from .plugins.seatbelt import SeatbeltNonComplianceRule
    from .plugins.speeding import SpeedingRule
except ImportError:
    from plugins.overcrowding import OvercrowdingRule
    from plugins.stop_line import StopLineRule
    from plugins.hazardous_load import HazardousLoadRule
    from plugins.helmet import HelmetNonComplianceRule
    from plugins.red_light import RedLightViolationRule
    from plugins.illegal_parking import IllegalParkingRule
    from plugins.wrong_side import WrongSideDrivingRule
    from plugins.seatbelt import SeatbeltNonComplianceRule
    from plugins.speeding import SpeedingRule

class ViolationEngine:
    """
    The main orchestrator for traffic rules.
    It loads modular rule plugins and runs them against YOLO detections.
    """
    def __init__(self):
        # Load active rule plugins
        self.rules = [
            OvercrowdingRule(),
            HelmetNonComplianceRule(),
            SeatbeltNonComplianceRule(),
            StopLineRule(),
            RedLightViolationRule(),
            IllegalParkingRule(),
            WrongSideDrivingRule(),
            SpeedingRule(),
            HazardousLoadRule()
        ]

    def analyze(self, detections, **kwargs):
        """
        Runs all registered rules against the detections.
        Pass kwargs (like stop_zone_polygon) to rules that need them.
        """
        all_violations = []
        for rule in self.rules:
            violations = rule.evaluate(detections, **kwargs)
            all_violations.extend(violations)
            
        return all_violations

    def draw_violations(self, image, violations, stop_zone=None):
        """
        Draws the flagged violations prominently on the image.
        """
        img_copy = image.copy()
        
        # Draw Stop Zone if provided
        if stop_zone:
            poly = np.array(stop_zone, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [poly], True, (0, 0, 255), 2)
            # Create a transparent overlay for the zone
            overlay = img_copy.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)

        for viol in violations:
            x1, y1, x2, y2 = viol['vehicle_bbox']
            
            # Draw a thick RED box for violations
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            label = f"VIOLATION: {viol['type']}"
            if 'rider_count' in viol:
                label += f" ({viol['rider_count']} riders)"
            if 'aspect_ratio' in viol:
                label += f" (Ratio: {viol['aspect_ratio']})"
                
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - 25), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return img_copy

# Quick test stub
if __name__ == "__main__":
    engine = ViolationEngine()
    
    mock_detections = [
        # A bike with 4 people
        {"class": "motorcycle", "bbox": [100, 100, 200, 300]}, 
        {"class": "person", "bbox": [110, 80, 150, 250]},     
        {"class": "person", "bbox": [130, 80, 170, 250]},     
        {"class": "person", "bbox": [150, 80, 190, 250]},     
        {"class": "person", "bbox": [170, 80, 210, 250]},     
        
        # A bike carrying a very long rod (Width 300, Height 50 -> Ratio 6.0)
        {"class": "motorcycle", "bbox": [100, 400, 400, 450]},
        
        # A car in the stop zone
        {"class": "car", "bbox": [300, 300, 400, 400]}        
    ]
    
    mock_stop_zone = [(280, 380), (420, 380), (420, 450), (280, 450)]
    
    # Run engine with **kwargs
    results = engine.analyze(mock_detections, stop_zone_polygon=mock_stop_zone)
    
    print(f"Modular Engine found {len(results)} violations:")
    for r in results:
        print(f" - {r['type']}")
