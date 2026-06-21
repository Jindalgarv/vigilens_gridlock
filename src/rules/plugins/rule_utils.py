import math

import cv2
import numpy as np


VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
MOTORCYCLE_CLASSES = {"motorcycle", "motorbike", "motobike", "motor_bike", "motor_cycle", "bike"}
FOUR_WHEELER_CLASSES = {"car", "bus", "truck", "vehicle"}


def normalize_class_name(class_name):
    name = str(class_name or "").strip().lower()
    return name.replace("-", "_")


def readable_class_name(class_name):
    return normalize_class_name(class_name).replace("_", " ")


def detection_class(detection):
    return str(detection.get("class") or detection.get("class_name") or "")


def canonical_class(detection):
    name = normalize_class_name(detection_class(detection))
    readable = name.replace("_", " ")
    if name in MOTORCYCLE_CLASSES or readable in MOTORCYCLE_CLASSES:
        return "motorcycle"
    return name


def is_vehicle_detection(detection, include_bicycle=True):
    class_name = canonical_class(detection)
    if include_bicycle:
        return class_name in VEHICLE_CLASSES
    return class_name in VEHICLE_CLASSES - {"bicycle"}


def is_four_wheeler_detection(detection):
    class_name = canonical_class(detection)
    return class_name in FOUR_WHEELER_CLASSES


def bbox_center(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_bottom_center(bbox):
    x1, _, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, y2


def bbox_width_height(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def point_inside_bbox(point, bbox):
    px, py = point
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= px <= x2 and y1 <= py <= y2


def point_inside_polygon(point, polygon):
    if not polygon:
        return False
    poly = np.array(polygon, np.int32)
    return cv2.pointPolygonTest(poly, tuple(map(float, point)), False) >= 0


def expand_bbox(bbox, x_ratio=0.0, y_top_ratio=0.0, y_bottom_ratio=0.0):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width, height = bbox_width_height(bbox)
    return [
        x1 - x_ratio * width,
        y1 - y_top_ratio * height,
        x2 + x_ratio * width,
        y2 + y_bottom_ratio * height,
    ]


def upper_region_bbox(bbox, height_ratio=0.50):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    height = max(1.0, y2 - y1)
    return [x1, y1, x2, y1 + height_ratio * height]


def center_distance(box_a, box_b):
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def detection_confidence(detection, default=0.0):
    return float(detection.get("confidence", default) or default)


def average_confidence(*detections):
    values = [detection_confidence(det) for det in detections]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def get_detection_speed(detection, index=None, speed_estimates=None, speed_unit="kmph"):
    keys = [
        f"speed_{speed_unit}",
        "speed_kmph",
        "speed_kph",
        "speed_mph",
        "speed",
    ]

    for key in keys:
        if detection.get(key) is not None:
            return float(detection[key]), key

    if not speed_estimates:
        return None, ""

    candidate_keys = []
    if index is not None:
        candidate_keys.append(index)
        candidate_keys.append(str(index))
    for key in ("track_id", "id", "detection_id"):
        if detection.get(key) is not None:
            candidate_keys.append(detection[key])
            candidate_keys.append(str(detection[key]))

    for key in candidate_keys:
        if key in speed_estimates:
            return float(speed_estimates[key]), "speed_estimates"

    return None, ""


def speed_limit_for_class(class_name, class_speed_limits=None, default_speed_limit=None):
    limits = class_speed_limits or {}
    normalized = normalize_class_name(class_name)
    readable = normalized.replace("_", " ")

    for key in (normalized, readable, "default"):
        if key in limits:
            return float(limits[key])

    if default_speed_limit is not None:
        return float(default_speed_limit)

    return None


def vector_from_detection(detection, index=None, motion_vectors=None):
    if detection.get("motion_vector") is not None:
        return detection["motion_vector"], "motion_vector"
    if detection.get("direction_vector") is not None:
        return detection["direction_vector"], "direction_vector"

    if not motion_vectors:
        return None, ""

    candidate_keys = []
    if index is not None:
        candidate_keys.append(index)
        candidate_keys.append(str(index))
    for key in ("track_id", "id", "detection_id"):
        if detection.get(key) is not None:
            candidate_keys.append(detection[key])
            candidate_keys.append(str(detection[key]))

    for key in candidate_keys:
        if key in motion_vectors:
            return motion_vectors[key], "motion_vectors"

    return None, ""


def cosine_similarity(vector_a, vector_b):
    ax, ay = [float(v) for v in vector_a]
    bx, by = [float(v) for v in vector_b]
    denom = math.sqrt(ax * ax + ay * ay) * math.sqrt(bx * bx + by * by)
    if denom == 0:
        return None
    return (ax * bx + ay * by) / denom
