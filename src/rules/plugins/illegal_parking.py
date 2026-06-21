from .base_rule import BaseViolationRule
from .rule_utils import (
    bbox_bottom_center,
    canonical_class,
    detection_confidence,
    is_vehicle_detection,
    point_inside_polygon,
)


class IllegalParkingRule(BaseViolationRule):
    """
    Flags vehicles inside a configured no-parking zone.

    For still images, the rule emits a suspected violation. If stationary time
    is supplied from a tracker, it can enforce a minimum stationary duration.
    """

    def __init__(self):
        super().__init__()
        self.name = "Illegal Parking"

    def evaluate(
        self,
        detections,
        no_parking_zone_polygon=None,
        parking_zone_polygon=None,
        stationary_seconds=None,
        min_stationary_seconds=0,
        require_stationary=False,
        **kwargs,
    ):
        zone_polygon = no_parking_zone_polygon or parking_zone_polygon
        if not zone_polygon:
            return []

        violations = []
        stationary_lookup = stationary_seconds or {}

        for idx, detection in enumerate(detections):
            if not is_vehicle_detection(detection, include_bicycle=False):
                continue

            bbox = detection.get("bbox", [0, 0, 0, 0])
            tire_point = bbox_bottom_center(bbox)
            if not point_inside_polygon(tire_point, zone_polygon):
                continue

            dwell_time = self._stationary_time_for_detection(detection, idx, stationary_lookup)
            if require_stationary or float(min_stationary_seconds or 0) > 0:
                if dwell_time is None or dwell_time < float(min_stationary_seconds):
                    continue

            is_confirmed = dwell_time is not None and dwell_time >= float(min_stationary_seconds or 0)
            violation_type = self.name if is_confirmed else "Suspected Illegal Parking"

            violations.append(
                {
                    "type": violation_type,
                    "vehicle_class": canonical_class(detection),
                    "vehicle_bbox": bbox,
                    "confidence": detection_confidence(detection),
                    "detection_index": idx,
                    "stationary_seconds": dwell_time,
                    "evidence_point": tire_point,
                    "details": (
                        "Vehicle is inside a configured no-parking zone."
                        if is_confirmed
                        else "Vehicle is inside a configured no-parking zone in a still image; tracker duration can confirm parking."
                    ),
                }
            )

        return violations

    @staticmethod
    def _stationary_time_for_detection(detection, index, stationary_lookup):
        for key in ("stationary_seconds", "dwell_seconds", "parked_seconds"):
            if detection.get(key) is not None:
                return float(detection[key])

        candidate_keys = [index, str(index)]
        for key in ("track_id", "id", "detection_id"):
            if detection.get(key) is not None:
                candidate_keys.append(detection[key])
                candidate_keys.append(str(detection[key]))

        for key in candidate_keys:
            if key in stationary_lookup:
                return float(stationary_lookup[key])

        return None
