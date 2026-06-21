import cv2

from .plate_detector import LicensePlateDetector
from .plate_ocr import PlateOCR


class LicensePlatePipeline:
    """
    Runs plate detection and OCR for a single violating vehicle.

    The pipeline never runs plate detection on the full frame. It crops the
    violating vehicle first, detects a plate inside that crop, then maps the
    plate coordinates back to the source image.
    """

    def __init__(
        self,
        plate_detector=None,
        ocr_reader=None,
        crop_padding=8,
    ):
        self.plate_detector = plate_detector or LicensePlateDetector()
        self.ocr_reader = ocr_reader or PlateOCR(engine="auto")
        self.crop_padding = int(crop_padding)

    def process_violation(self, image, violation):
        vehicle_bbox = self._vehicle_bbox_from_violation(violation)
        vehicle_crop, padded_vehicle_bbox = self.crop_bbox(image, vehicle_bbox, self.crop_padding)

        if vehicle_crop is None or vehicle_crop.size == 0:
            return self._empty_result(vehicle_bbox, padded_vehicle_bbox, "empty_vehicle_crop")

        plate_detections = self.plate_detector.detect(vehicle_crop)
        if not plate_detections:
            return self._empty_result(
                vehicle_bbox,
                padded_vehicle_bbox,
                "plate_not_detected",
                vehicle_crop=vehicle_crop,
            )

        best_result = None
        best_score = -1.0

        for plate_detection in plate_detections:
            candidate = self._process_plate_candidate(
                vehicle_crop=vehicle_crop,
                padded_vehicle_bbox=padded_vehicle_bbox,
                plate_detection=plate_detection,
                vehicle_bbox=vehicle_bbox,
                candidate_count=len(plate_detections),
            )
            score = self._plate_candidate_score(candidate)
            if score > best_score:
                best_result = candidate
                best_score = score

            if candidate["valid_plate_format"] and not candidate["needs_manual_review"]:
                return candidate

        return best_result or self._empty_result(
            vehicle_bbox,
            padded_vehicle_bbox,
            "plate_candidate_ocr_failed",
            vehicle_crop=vehicle_crop,
        )

    def _process_plate_candidate(
        self,
        vehicle_crop,
        padded_vehicle_bbox,
        plate_detection,
        vehicle_bbox,
        candidate_count,
    ):
        best_plate = plate_detection
        plate_bbox_crop = best_plate["bbox_crop"]
        plate_crop, clipped_plate_bbox_crop = self.crop_bbox(vehicle_crop, plate_bbox_crop, padding=8)
        plate_bbox_original = self.crop_bbox_to_original(clipped_plate_bbox_crop, padded_vehicle_bbox)

        ocr_result = self.ocr_reader.read(plate_crop)

        return {
            "vehicle_bbox": [int(v) for v in vehicle_bbox],
            "vehicle_bbox_padded": [int(v) for v in padded_vehicle_bbox],
            "plate_bbox_crop": [round(float(v), 2) for v in clipped_plate_bbox_crop],
            "plate_bbox_original": [round(float(v), 2) for v in plate_bbox_original],
            "plate_detection_confidence": float(best_plate.get("confidence", 0.0)),
            "plate_detector": best_plate.get("detector", "unknown"),
            "plate_class_name": best_plate.get("class_name", "license_plate"),
            "plate_candidates_evaluated": int(candidate_count),
            "plate_text": ocr_result["text"],
            "ocr_raw_text": ocr_result["raw_text"],
            "ocr_confidence": ocr_result["confidence"],
            "ocr_engine": ocr_result["engine"],
            "valid_plate_format": ocr_result["valid_format"],
            "needs_manual_review": ocr_result["needs_manual_review"],
            "error": ocr_result.get("error", ""),
            "vehicle_crop": vehicle_crop,
            "plate_crop": plate_crop,
        }

    @staticmethod
    def _plate_candidate_score(candidate):
        if not candidate:
            return -1.0
        valid_bonus = 2.0 if candidate.get("valid_plate_format") else 0.0
        review_penalty = -0.5 if candidate.get("needs_manual_review") else 0.0
        text_bonus = 0.25 if candidate.get("plate_text") else 0.0
        return (
            valid_bonus
            + text_bonus
            + float(candidate.get("ocr_confidence", 0.0) or 0.0)
            + 0.25 * float(candidate.get("plate_detection_confidence", 0.0) or 0.0)
            + review_penalty
        )

    @staticmethod
    def _vehicle_bbox_from_violation(violation):
        bbox = (
            violation.get("vehicle_bbox")
            or violation.get("bbox")
            or violation.get("vehicle_box")
            or [0, 0, 0, 0]
        )
        return [int(round(float(v))) for v in bbox]

    @staticmethod
    def crop_bbox(image, bbox, padding=0):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        if x2 <= x1 or y2 <= y1:
            return None, [x1, y1, x2, y2]

        return image[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]

    @staticmethod
    def crop_bbox_to_original(crop_bbox, crop_origin_bbox):
        ox1, oy1, _, _ = [float(v) for v in crop_origin_bbox]
        x1, y1, x2, y2 = [float(v) for v in crop_bbox]
        return [ox1 + x1, oy1 + y1, ox1 + x2, oy1 + y2]

    @staticmethod
    def attach_to_violation(violation, plate_result):
        enriched = dict(violation)
        enriched["license_plate"] = plate_result.get("plate_text") or "N/A"
        enriched["ocr_plate_number"] = enriched["license_plate"]
        enriched["ocr_confidence"] = plate_result.get("ocr_confidence", 0.0)
        enriched["ocr_engine"] = plate_result.get("ocr_engine", "")
        enriched["plate_bbox"] = plate_result.get("plate_bbox_original")
        enriched["plate_detection_confidence"] = plate_result.get("plate_detection_confidence", 0.0)
        enriched["plate_needs_manual_review"] = plate_result.get("needs_manual_review", True)
        return enriched

    @staticmethod
    def _empty_result(vehicle_bbox, padded_vehicle_bbox, error, vehicle_crop=None):
        return {
            "vehicle_bbox": [int(v) for v in vehicle_bbox],
            "vehicle_bbox_padded": [int(v) for v in padded_vehicle_bbox],
            "plate_bbox_crop": None,
            "plate_bbox_original": None,
            "plate_detection_confidence": 0.0,
            "plate_detector": "none",
            "plate_class_name": "license_plate",
            "plate_text": "",
            "ocr_raw_text": "",
            "ocr_confidence": 0.0,
            "ocr_engine": "none",
            "valid_plate_format": False,
            "needs_manual_review": True,
            "error": error,
            "vehicle_crop": vehicle_crop,
            "plate_crop": None,
        }


def draw_plate_debug(image, plate_result):
    output = image.copy()
    bbox = plate_result.get("plate_bbox_original")
    if bbox:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return output
