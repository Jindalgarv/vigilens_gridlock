from .base_rule import BaseViolationRule

class HazardousLoadRule(BaseViolationRule):
    """
    Detects if a vehicle is carrying dangerously long items (rods, pipes).
    Logic: If the bounding box of a motorcycle/car has an extreme aspect ratio 
    (e.g., extremely wide compared to its height) AND exceeds typical dimensions.
    """
    def __init__(self, extreme_aspect_ratio=3.5):
        super().__init__()
        self.name = "Hazardous/Long Load"
        # If width is 3.5x the height, it's flagged as potentially carrying rods/long items
        self.extreme_aspect_ratio = extreme_aspect_ratio

    def evaluate(self, detections, **kwargs):
        violations = []
        # Mostly applies to two-wheelers and small cars in traffic contexts
        targets = [d for d in detections if d['class'] in ['motorcycle', 'bicycle', 'car']]

        for target in targets:
            x1, y1, x2, y2 = target['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Avoid division by zero
            if height == 0: continue

            aspect_ratio = width / float(height)

            if aspect_ratio > self.extreme_aspect_ratio:
                violations.append({
                    "type": self.name,
                    "vehicle_class": target['class'],
                    "vehicle_bbox": target['bbox'],
                    "aspect_ratio": round(aspect_ratio, 2)
                })

        return violations
