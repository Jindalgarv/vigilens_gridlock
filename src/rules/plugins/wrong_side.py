from .base_rule import BaseViolationRule
from .rule_utils import (
    bbox_bottom_center,
    canonical_class,
    cosine_similarity,
    detection_confidence,
    is_vehicle_detection,
    point_inside_polygon,
    vector_from_detection,
)


class WrongSideDrivingRule(BaseViolationRule):
    """
    Flags wrong-side driving using lane metadata.

    Supported modes:
    1. wrong_side_zone_polygon: still-image restricted zone fallback.
    2. lane_configs + motion vectors: true direction check using tracking.
    """

    def __init__(self, wrong_direction_cosine_threshold=-0.25):
        super().__init__()
        self.name = "Wrong-Side Driving"
        self.wrong_direction_cosine_threshold = wrong_direction_cosine_threshold

    def evaluate(
        self,
        detections,
        wrong_side_zone_polygon=None,
        lane_configs=None,
        motion_vectors=None,
        wrong_direction_cosine_threshold=None,
        **kwargs,
    ):
        threshold = (
            self.wrong_direction_cosine_threshold
            if wrong_direction_cosine_threshold is None
            else float(wrong_direction_cosine_threshold)
        )
        violations = []

        for idx, detection in enumerate(detections):
            if not is_vehicle_detection(detection):
                continue

            bbox = detection.get("bbox", [0, 0, 0, 0])
            tire_point = bbox_bottom_center(bbox)

            if wrong_side_zone_polygon and point_inside_polygon(tire_point, wrong_side_zone_polygon):
                violations.append(
                    self._build_violation(
                        detection=detection,
                        index=idx,
                        bbox=bbox,
                        tire_point=tire_point,
                        confidence=detection_confidence(detection),
                        details="Vehicle is inside a configured wrong-side/restricted lane zone in a still image.",
                        suspected=True,
                    )
                )
                continue

            lane = self._containing_lane(tire_point, lane_configs or [])
            if not lane:
                continue

            allowed_direction = lane.get("allowed_direction") or lane.get("direction")
            if not allowed_direction:
                continue

            motion_vector, vector_source = vector_from_detection(detection, idx, motion_vectors)
            if motion_vector is None:
                continue

            cosine = cosine_similarity(motion_vector, allowed_direction)
            if cosine is None or cosine > threshold:
                continue

            violations.append(
                self._build_violation(
                    detection=detection,
                    index=idx,
                    bbox=bbox,
                    tire_point=tire_point,
                    confidence=detection_confidence(detection),
                    details=(
                        "Vehicle motion vector opposes the configured lane direction "
                        f"(cosine={cosine:.3f}, source={vector_source})."
                    ),
                    suspected=False,
                    lane_id=lane.get("id") or lane.get("name"),
                    direction_cosine=round(float(cosine), 4),
                    motion_vector=motion_vector,
                    allowed_direction=allowed_direction,
                )
            )

        return violations

    @staticmethod
    def _containing_lane(point, lane_configs):
        for lane in lane_configs:
            polygon = lane.get("polygon") or lane.get("zone_polygon")
            if polygon and point_inside_polygon(point, polygon):
                return lane
        return None

    def _build_violation(
        self,
        detection,
        index,
        bbox,
        tire_point,
        confidence,
        details,
        suspected=False,
        **extra,
    ):
        payload = {
            "type": "Suspected Wrong-Side Driving" if suspected else self.name,
            "vehicle_class": canonical_class(detection),
            "vehicle_bbox": bbox,
            "confidence": confidence,
            "detection_index": index,
            "evidence_point": tire_point,
            "details": details,
        }
        payload.update(extra)
        return payload
