import numpy as np
import cv2

from .base_rule import BaseViolationRule
from .rule_utils import canonical_class, detection_confidence


class StopLineRule(BaseViolationRule):
    """
    Checks if a vehicle's tires (bottom of bbox) cross into a restricted polygon zone.
    """
    def __init__(self):
        super().__init__()
        self.name = "Stop-Line / Wrong Way"

    def evaluate(self, detections, stop_zone_polygon=None, **kwargs):
        if not stop_zone_polygon:
            return []

        violations = []
        vehicles = [
            (idx, d)
            for idx, d in enumerate(detections)
            if d.get("class") in ["car", "bus", "truck", "motorcycle"]
        ]
        poly = np.array(stop_zone_polygon, np.int32)

        for idx, veh in vehicles:
            bbox = veh.get("bbox", [0, 0, 0, 0])
            x_center = (bbox[0] + bbox[2]) // 2
            y_bottom = bbox[3]
            
            dist = cv2.pointPolygonTest(poly, (x_center, y_bottom), False)
            
            if dist >= 0:
                violations.append({
                    "type": self.name,
                    "vehicle_class": canonical_class(veh),
                    "vehicle_bbox": bbox,
                    "confidence": detection_confidence(veh),
                    "detection_index": idx,
                    "evidence_point": (x_center, y_bottom),
                    "details": "Vehicle tire contact point is inside the configured stop-line zone.",
                })

        return violations
