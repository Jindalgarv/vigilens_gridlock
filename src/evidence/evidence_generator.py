import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2


class EvidenceGenerator:
    """
    Writes per-violation evidence artifacts.

    Each violation gets:
    - annotated full frame
    - violating vehicle crop
    - plate crop, if available
    - structured metadata JSON
    """

    def __init__(self, output_dir="problem/output_evidence"):
        self.output_dir = Path(output_dir)
        self.annotated_dir = self.output_dir / "annotated"
        self.vehicle_dir = self.output_dir / "vehicles"
        self.plate_dir = self.output_dir / "plates"
        self.metadata_dir = self.output_dir / "metadata"

        for directory in (
            self.annotated_dir,
            self.vehicle_dir,
            self.plate_dir,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def generate(self, image, violation, plate_result=None, source_image_path=""):
        violation_id = violation.get("violation_id") or self._new_violation_id()
        plate_result = plate_result or {}

        annotated = self.draw_evidence(image, violation, plate_result, violation_id)

        annotated_path = self.annotated_dir / f"{violation_id}.jpg"
        vehicle_path = self.vehicle_dir / f"{violation_id}.jpg"
        plate_path = self.plate_dir / f"{violation_id}.jpg"
        metadata_path = self.metadata_dir / f"{violation_id}.json"

        cv2.imwrite(str(annotated_path), annotated)

        vehicle_crop = plate_result.get("vehicle_crop")
        if vehicle_crop is not None and vehicle_crop.size > 0:
            cv2.imwrite(str(vehicle_path), vehicle_crop)
        else:
            vehicle_path = None

        plate_crop = plate_result.get("plate_crop")
        if plate_crop is not None and plate_crop.size > 0:
            plate_to_save = plate_crop
            if min(plate_crop.shape[:2]) < 80:
                plate_to_save = cv2.resize(
                    plate_crop,
                    None,
                    fx=4.0,
                    fy=4.0,
                    interpolation=cv2.INTER_CUBIC,
                )
            cv2.imwrite(str(plate_path), plate_to_save)
        else:
            plate_path = None

        metadata = self.build_metadata(
            violation_id=violation_id,
            violation=violation,
            plate_result=plate_result,
            source_image_path=source_image_path,
            annotated_path=annotated_path,
            vehicle_path=vehicle_path,
            plate_path=plate_path,
        )

        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "violation_id": violation_id,
            "annotated_image_path": str(annotated_path),
            "vehicle_crop_path": str(vehicle_path) if vehicle_path else "",
            "plate_crop_path": str(plate_path) if plate_path else "",
            "metadata_path": str(metadata_path),
            "metadata": metadata,
        }

    def draw_evidence(self, image, violation, plate_result, violation_id):
        output = image.copy()
        overlay = output.copy()

        vehicle_bbox = violation.get("vehicle_bbox") or violation.get("bbox") or [0, 0, 0, 0]
        self._draw_box(overlay, vehicle_bbox, (0, 0, 255), thickness=4)

        for evidence_box in violation.get("evidence_boxes", []):
            self._draw_box(overlay, evidence_box, (0, 165, 255), thickness=2)

        plate_bbox = plate_result.get("plate_bbox_original")
        if plate_bbox:
            self._draw_box(overlay, plate_bbox, (0, 255, 255), thickness=3)

        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

        label = self._label_text(violation, plate_result, violation_id)
        self._draw_banner(output, label)
        return output

    @staticmethod
    def _draw_box(image, bbox, color, thickness=2):
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def _draw_banner(image, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        margin = 8
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            image,
            (0, 0),
            (min(image.shape[1], text_w + 2 * margin), text_h + 2 * margin + 6),
            (0, 255, 255),
            -1,
        )
        cv2.putText(
            image,
            text,
            (margin, text_h + margin),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _label_text(violation, plate_result, violation_id):
        violation_type = violation.get("type") or violation.get("violation_type") or "Violation"
        confidence = float(violation.get("confidence", 0.0) or 0.0)
        plate = plate_result.get("plate_text") or violation.get("license_plate") or "N/A"
        return f"{violation_id} | {violation_type} | Plate: {plate} | Conf: {confidence:.2f}"

    @staticmethod
    def build_metadata(
        violation_id,
        violation,
        plate_result,
        source_image_path,
        annotated_path,
        vehicle_path,
        plate_path,
    ):
        return {
            "violation_id": violation_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_image_path": str(source_image_path),
            "violation_type": violation.get("type") or violation.get("violation_type"),
            "vehicle_class": violation.get("vehicle_class") or violation.get("vehicle_type"),
            "confidence": violation.get("confidence", 0.0),
            "vehicle_bbox": violation.get("vehicle_bbox") or violation.get("bbox"),
            "evidence_boxes": violation.get("evidence_boxes", []),
            "details": violation.get("details", ""),
            "plate": {
                "text": plate_result.get("plate_text") or "",
                "ocr_raw_text": plate_result.get("ocr_raw_text") or "",
                "ocr_confidence": plate_result.get("ocr_confidence", 0.0),
                "ocr_engine": plate_result.get("ocr_engine", ""),
                "valid_format": plate_result.get("valid_plate_format", False),
                "needs_manual_review": plate_result.get("needs_manual_review", True),
                "bbox_crop": plate_result.get("plate_bbox_crop"),
                "bbox_original": plate_result.get("plate_bbox_original"),
                "detection_confidence": plate_result.get("plate_detection_confidence", 0.0),
                "detector": plate_result.get("plate_detector", ""),
            },
            "artifacts": {
                "annotated_image_path": str(annotated_path),
                "vehicle_crop_path": str(vehicle_path) if vehicle_path else "",
                "plate_crop_path": str(plate_path) if plate_path else "",
            },
        }

    @staticmethod
    def _new_violation_id():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid4().hex[:8].upper()
        return f"TVAI-{timestamp}-{suffix}"
