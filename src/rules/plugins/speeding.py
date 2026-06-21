from .base_rule import BaseViolationRule
from .rule_utils import (
    canonical_class,
    detection_confidence,
    get_detection_speed,
    is_vehicle_detection,
    speed_limit_for_class,
)


class SpeedingRule(BaseViolationRule):
    """
    Flags vehicles exceeding class-specific speed limits.

    This rule does not estimate speed from a single image. It consumes speed
    metadata from tracking, radar, camera calibration, or demo-provided values.
    """

    DEFAULT_SPEED_LIMITS_KMPH = {
        "motorcycle": 50,
        "bicycle": 25,
        "car": 60,
        "bus": 50,
        "truck": 40,
        "default": 50,
    }

    def __init__(self, tolerance=0.0):
        super().__init__()
        self.name = "Speeding"
        self.tolerance = tolerance

    def evaluate(
        self,
        detections,
        speed_estimates=None,
        class_speed_limits=None,
        default_speed_limit=None,
        speed_unit="kmph",
        speed_tolerance=None,
        **kwargs,
    ):
        limits = class_speed_limits or self.DEFAULT_SPEED_LIMITS_KMPH
        tolerance = self.tolerance if speed_tolerance is None else float(speed_tolerance)
        violations = []

        for idx, detection in enumerate(detections):
            if not is_vehicle_detection(detection):
                continue

            vehicle_class = canonical_class(detection)
            speed_value, speed_source = get_detection_speed(
                detection=detection,
                index=idx,
                speed_estimates=speed_estimates,
                speed_unit=speed_unit,
            )
            if speed_value is None:
                continue

            limit = speed_limit_for_class(
                vehicle_class,
                class_speed_limits=limits,
                default_speed_limit=default_speed_limit,
            )
            if limit is None:
                continue

            excess = speed_value - limit
            if excess <= tolerance:
                continue

            bbox = detection.get("bbox", [0, 0, 0, 0])
            violations.append(
                {
                    "type": self.name,
                    "vehicle_class": vehicle_class,
                    "vehicle_bbox": bbox,
                    "confidence": detection_confidence(detection),
                    "detection_index": idx,
                    "speed": round(float(speed_value), 2),
                    "speed_limit": round(float(limit), 2),
                    "speed_excess": round(float(excess), 2),
                    "speed_unit": speed_unit,
                    "speed_source": speed_source,
                    "details": (
                        f"{vehicle_class} speed {speed_value:.1f} {speed_unit} exceeds "
                        f"class limit {limit:.1f} {speed_unit}."
                    ),
                }
            )

        return violations
