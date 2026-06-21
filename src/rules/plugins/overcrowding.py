from .base_rule import BaseViolationRule

class OvercrowdingRule(BaseViolationRule):
    """
    Detects if >2 persons are riding a single motorcycle.
    """
    def __init__(self, iou_threshold=0.3):
        super().__init__()
        self.name = "Overcrowding / Triple Riding"
        self.iou_threshold = iou_threshold

    def evaluate(self, detections, **kwargs):
        violations = []
        motorcycles = [d for d in detections if d['class'] == 'motorcycle']
        persons = [d for d in detections if d['class'] == 'person']

        for moto in motorcycles:
            riders_count = 0
            rider_boxes = []
            
            for person in persons:
                iou = self._calculate_iou(moto['bbox'], person['bbox'])
                
                px_center = (person['bbox'][0] + person['bbox'][2]) // 2
                py_bottom = person['bbox'][3]
                mx1, my1, mx2, my2 = moto['bbox']
                
                if iou > self.iou_threshold or (mx1 < px_center < mx2 and my1 < py_bottom < my2):
                    riders_count += 1
                    rider_boxes.append(person['bbox'])

            # As requested: > 2 handles 3, 4, 5+ riders dynamically
            if riders_count > 2:
                violations.append({
                    "type": self.name,
                    "vehicle_bbox": moto['bbox'],
                    "rider_count": riders_count,
                    "evidence_boxes": rider_boxes
                })

        return violations
