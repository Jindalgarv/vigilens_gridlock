from .base_rule import BaseViolationRule
from .rule_utils import (
    average_confidence,
    bbox_center,
    canonical_class,
    detection_class,
    expand_bbox,
    is_four_wheeler_detection,
    normalize_class_name,
    point_inside_bbox,
    upper_region_bbox,
)


class SeatbeltNonComplianceRule(BaseViolationRule):
    """
    Flags no-seatbelt detections associated with a four-wheeler cabin.

    The base YOLO11 COCO model does not detect seatbelts. This rule requires a
    specialized detector that emits no_seatbelt / seatbelt_missing classes.
    """

    NO_SEATBELT_NAMES = {
        "no_seatbelt",
        "no seatbelt",
        "no-seatbelt",
        "without_seatbelt",
        "without seatbelt",
        "seatbelt_missing",
        "seatbelt missing",
        "seat_belt_missing",
        "no_seat_belt",
    }
    OCCUPANT_NAMES = {"person", "driver", "passenger", "occupant"}

    def __init__(self, occupant_iou_threshold=0.01):
        super().__init__()
        self.name = "Seatbelt Non-Compliance"
        self.occupant_iou_threshold = occupant_iou_threshold

    def evaluate(self, detections, **kwargs):
        vehicles = [
            (idx, det)
            for idx, det in enumerate(detections)
            if is_four_wheeler_detection(det)
        ]
        no_seatbelts = [
            (idx, det)
            for idx, det in enumerate(detections)
            if self._is_no_seatbelt(detection_class(det))
        ]
        occupants = [
            (idx, det)
            for idx, det in enumerate(detections)
            if self._is_occupant(detection_class(det))
        ]

        if not vehicles or not no_seatbelts:
            return []

        violations = []
        used_no_seatbelt_indexes = set()

        for vehicle_idx, vehicle in vehicles:
            vehicle_bbox = vehicle.get("bbox", [0, 0, 0, 0])
            cabin_bbox = self._cabin_region(vehicle_bbox)
            associated_occupants = self._associated_occupants(cabin_bbox, occupants)

            for seatbelt_idx, seatbelt in no_seatbelts:
                if seatbelt_idx in used_no_seatbelt_indexes:
                    continue

                seatbelt_bbox = seatbelt.get("bbox", [0, 0, 0, 0])
                seatbelt_center = bbox_center(seatbelt_bbox)
                matched_occupant = self._matched_occupant(seatbelt_bbox, associated_occupants)

                if associated_occupants:
                    if matched_occupant is None:
                        continue
                elif occupants:
                    continue
                elif not point_inside_bbox(seatbelt_center, cabin_bbox):
                    continue

                used_no_seatbelt_indexes.add(seatbelt_idx)
                evidence_boxes = [seatbelt_bbox]
                related_indexes = [vehicle_idx, seatbelt_idx]

                if matched_occupant is not None:
                    occupant_idx, occupant = matched_occupant
                    evidence_boxes.append(occupant.get("bbox", [0, 0, 0, 0]))
                    related_indexes.append(occupant_idx)

                violations.append(
                    {
                        "type": self.name,
                        "vehicle_class": canonical_class(vehicle),
                        "vehicle_bbox": vehicle_bbox,
                        "confidence": average_confidence(vehicle, seatbelt),
                        "evidence_boxes": evidence_boxes,
                        "no_seatbelt_bbox": seatbelt_bbox,
                        "detection_index": vehicle_idx,
                        "related_detection_index": seatbelt_idx,
                        "related_detection_indexes": related_indexes,
                        "details": (
                            "No-seatbelt detection is associated with the cabin/occupant "
                            "of a four-wheeler. Absence of a seatbelt detection alone is not used."
                        ),
                    }
                )

        return violations

    @classmethod
    def _is_no_seatbelt(cls, class_name):
        name = normalize_class_name(class_name)
        readable = name.replace("_", " ")
        return name in cls.NO_SEATBELT_NAMES or readable in cls.NO_SEATBELT_NAMES

    @classmethod
    def _is_occupant(cls, class_name):
        name = normalize_class_name(class_name)
        readable = name.replace("_", " ")
        return name in cls.OCCUPANT_NAMES or readable in cls.OCCUPANT_NAMES

    @staticmethod
    def _cabin_region(vehicle_bbox):
        cabin = upper_region_bbox(vehicle_bbox, height_ratio=0.62)
        return expand_bbox(cabin, x_ratio=0.04, y_top_ratio=0.02, y_bottom_ratio=0.02)

    def _associated_occupants(self, cabin_bbox, occupants):
        associated = []
        for occupant_idx, occupant in occupants:
            occupant_bbox = occupant.get("bbox", [0, 0, 0, 0])
            occupant_center = bbox_center(occupant_bbox)
            occupant_iou = self._calculate_iou(cabin_bbox, occupant_bbox)

            if point_inside_bbox(occupant_center, cabin_bbox) or occupant_iou >= self.occupant_iou_threshold:
                associated.append((occupant_idx, occupant))

        return associated

    def _matched_occupant(self, seatbelt_bbox, occupants):
        seatbelt_center = bbox_center(seatbelt_bbox)
        for occupant_idx, occupant in occupants:
            occupant_upper = upper_region_bbox(occupant.get("bbox", [0, 0, 0, 0]), height_ratio=0.75)
            if point_inside_bbox(seatbelt_center, occupant_upper):
                return occupant_idx, occupant
            if self._calculate_iou(seatbelt_bbox, occupant_upper) >= self.occupant_iou_threshold:
                return occupant_idx, occupant
        return None
