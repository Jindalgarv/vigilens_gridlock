from .base_rule import BaseViolationRule
from .rule_utils import (
    bbox_bottom_center,
    canonical_class,
    detection_confidence,
    is_vehicle_detection,
    point_inside_polygon,
)


class RedLightViolationRule(BaseViolationRule):
    """
    Flags vehicles crossing a stop/restricted zone while the signal is red.

    Required runtime metadata:
    - traffic_light_status="RED"
    - red_light_zone_polygon or stop_zone_polygon
    """

    def __init__(self):
        super().__init__()
        self.name = "Red-Light Violation"

    def evaluate(self, detections, traffic_light_status=None, red_light_zone_polygon=None, stop_zone_polygon=None, **kwargs):
        if str(traffic_light_status or "").strip().upper() != "RED":
            return []

        zone_polygon = red_light_zone_polygon or stop_zone_polygon
        if not zone_polygon:
            return []

        violations = []
        for idx, detection in enumerate(detections):
            if not is_vehicle_detection(detection):
                continue

            bbox = detection.get("bbox", [0, 0, 0, 0])
            tire_point = bbox_bottom_center(bbox)
            if not point_inside_polygon(tire_point, zone_polygon):
                continue

            violations.append(
                {
                    "type": self.name,
                    "vehicle_class": canonical_class(detection),
                    "vehicle_bbox": bbox,
                    "confidence": detection_confidence(detection),
                    "detection_index": idx,
                    "signal_status": "RED",
                    "evidence_point": tire_point,
                    "details": "Vehicle tire contact point entered the stop/restricted zone while signal status was RED.",
                }
            )

        return violations
