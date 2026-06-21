import math

from .base_rule import BaseViolationRule


class HelmetNonComplianceRule(BaseViolationRule):
    """
    Detects no-helmet violations for motorcycle riders.

    This rule expects detections from a helmet/no-helmet capable detector. The
    COCO YOLO11 vehicle model does not produce no_helmet classes by itself, so
    this plugin stays inactive until those detections are present in the shared
    detection list.
    """

    MOTORCYCLE_NAMES = {
        "motorcycle",
        "motorbike",
        "motobike",
        "motor_bike",
        "motor_cycle",
        "bike",
    }
    NO_HELMET_NAMES = {
        "no_helmet",
        "no helmet",
        "no-helmet",
        "without_helmet",
        "without helmet",
        "helmet_missing",
        "helmet missing",
    }
    RIDER_NAMES = {
        "person",
        "pedestrian",
        "rider",
        "driver",
        "pillion",
    }

    def __init__(self, motorcycle_iou_threshold=0.08, rider_iou_threshold=0.02):
        super().__init__()
        self.name = "Helmet Non-Compliance"
        self.motorcycle_iou_threshold = motorcycle_iou_threshold
        self.rider_iou_threshold = rider_iou_threshold

    def evaluate(self, detections, **kwargs):
        violations = []

        motorcycles = [
            (idx, det)
            for idx, det in enumerate(detections)
            if self._is_motorcycle(self._class_name(det))
        ]
        no_helmets = [
            (idx, det)
            for idx, det in enumerate(detections)
            if self._is_no_helmet(self._class_name(det))
        ]
        riders = [
            (idx, det)
            for idx, det in enumerate(detections)
            if self._is_rider(self._class_name(det))
        ]

        if not motorcycles or not no_helmets:
            return violations

        used_no_helmet_indexes = set()

        for motorcycle_idx, motorcycle in motorcycles:
            motorcycle_bbox = motorcycle.get("bbox", [0, 0, 0, 0])
            associated_riders = self._associated_riders(motorcycle_bbox, riders)

            for no_helmet_idx, no_helmet in no_helmets:
                if no_helmet_idx in used_no_helmet_indexes:
                    continue

                no_helmet_bbox = no_helmet.get("bbox", [0, 0, 0, 0])
                matched_rider = self._matched_rider(no_helmet_bbox, associated_riders)

                if associated_riders:
                    if matched_rider is None:
                        continue
                elif riders:
                    # Rider detections exist, but none belong to this bike. Avoid
                    # flagging a nearby standing pedestrian as a motorcycle rider.
                    continue
                elif not self._no_helmet_belongs_to_motorcycle(no_helmet_bbox, motorcycle_bbox):
                    continue

                used_no_helmet_indexes.add(no_helmet_idx)
                evidence_boxes = [no_helmet_bbox]
                related_detection_indexes = [motorcycle_idx, no_helmet_idx]

                if matched_rider is not None:
                    rider_idx, rider = matched_rider
                    evidence_boxes.append(rider.get("bbox", [0, 0, 0, 0]))
                    related_detection_indexes.append(rider_idx)

                violations.append(
                    {
                        "type": self.name,
                        "vehicle_class": self._class_name(motorcycle),
                        "vehicle_bbox": motorcycle_bbox,
                        "confidence": self._average_confidence(motorcycle, no_helmet),
                        "evidence_boxes": evidence_boxes,
                        "no_helmet_bbox": no_helmet_bbox,
                        "detection_index": motorcycle_idx,
                        "related_detection_index": no_helmet_idx,
                        "related_detection_indexes": related_detection_indexes,
                        "details": (
                            "No-helmet detection is associated with a motorcycle rider. "
                            "Standalone pedestrians are ignored."
                        ),
                    }
                )

        return violations

    @classmethod
    def _normalize_name(cls, class_name):
        name = str(class_name or "").strip().lower()
        return name.replace("-", "_")

    @classmethod
    def _class_name(cls, detection):
        return str(detection.get("class") or detection.get("class_name") or "")

    @classmethod
    def _is_motorcycle(cls, class_name):
        name = cls._normalize_name(class_name)
        readable = name.replace("_", " ")
        return name in cls.MOTORCYCLE_NAMES or readable in cls.MOTORCYCLE_NAMES

    @classmethod
    def _is_no_helmet(cls, class_name):
        name = cls._normalize_name(class_name)
        readable = name.replace("_", " ")
        return name in cls.NO_HELMET_NAMES or readable in cls.NO_HELMET_NAMES

    @classmethod
    def _is_rider(cls, class_name):
        name = cls._normalize_name(class_name)
        readable = name.replace("_", " ")
        return name in cls.RIDER_NAMES or readable in cls.RIDER_NAMES

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_size(bbox):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(1.0, x2 - x1), max(1.0, y2 - y1)

    @staticmethod
    def _point_inside_box(point, box):
        px, py = point
        x1, y1, x2, y2 = [float(v) for v in box]
        return x1 <= px <= x2 and y1 <= py <= y2

    def _expanded_motorcycle_head_area(self, motorcycle_bbox):
        x1, y1, x2, y2 = [float(v) for v in motorcycle_bbox]
        width, height = self._bbox_size(motorcycle_bbox)
        return [
            x1 - 0.35 * width,
            y1 - 0.85 * height,
            x2 + 0.35 * width,
            y2 + 0.10 * height,
        ]

    def _expanded_motorcycle_rider_area(self, motorcycle_bbox):
        x1, y1, x2, y2 = [float(v) for v in motorcycle_bbox]
        width, height = self._bbox_size(motorcycle_bbox)
        return [
            x1 - 0.30 * width,
            y1 - 1.15 * height,
            x2 + 0.30 * width,
            y2 + 0.20 * height,
        ]

    def _associated_riders(self, motorcycle_bbox, riders):
        associated = []
        rider_area = self._expanded_motorcycle_rider_area(motorcycle_bbox)

        for rider_idx, rider in riders:
            rider_bbox = rider.get("bbox", [0, 0, 0, 0])
            rider_center = self._bbox_center(rider_bbox)
            rider_bottom_center = (rider_center[0], float(rider_bbox[3]))
            rider_iou = self._calculate_iou(motorcycle_bbox, rider_bbox)

            if (
                rider_iou >= self.motorcycle_iou_threshold
                or self._point_inside_box(rider_bottom_center, rider_area)
                or self._point_inside_box(rider_center, rider_area)
            ):
                associated.append((rider_idx, rider))

        return associated

    def _matched_rider(self, no_helmet_bbox, associated_riders):
        if not associated_riders:
            return None

        no_helmet_center = self._bbox_center(no_helmet_bbox)
        best_match = None
        best_score = -1.0

        for rider_idx, rider in associated_riders:
            rider_bbox = rider.get("bbox", [0, 0, 0, 0])
            rider_upper = self._upper_body_box(rider_bbox)
            rider_iou = self._calculate_iou(no_helmet_bbox, rider_upper)

            if self._point_inside_box(no_helmet_center, rider_upper):
                score = 1.0 + rider_iou
            elif rider_iou >= self.rider_iou_threshold:
                score = rider_iou
            else:
                continue

            if score > best_score:
                best_score = score
                best_match = (rider_idx, rider)

        return best_match

    @staticmethod
    def _upper_body_box(rider_bbox):
        x1, y1, x2, y2 = [float(v) for v in rider_bbox]
        height = max(1.0, y2 - y1)
        return [x1, y1, x2, y1 + 0.45 * height]

    def _no_helmet_belongs_to_motorcycle(self, no_helmet_bbox, motorcycle_bbox):
        no_helmet_center = self._bbox_center(no_helmet_bbox)
        head_area = self._expanded_motorcycle_head_area(motorcycle_bbox)

        if self._point_inside_box(no_helmet_center, head_area):
            return True

        allowed_distance = max(self._bbox_size(motorcycle_bbox)) * 0.95
        return self._center_distance(no_helmet_bbox, motorcycle_bbox) <= allowed_distance

    def _center_distance(self, box_a, box_b):
        ax, ay = self._bbox_center(box_a)
        bx, by = self._bbox_center(box_b)
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    @staticmethod
    def _average_confidence(*detections):
        values = [float(det.get("confidence", 0.0)) for det in detections]
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)
