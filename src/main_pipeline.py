import os
from pathlib import Path

import cv2

try:
    from .detection.vehicle_detector import VehicleDetector
    from .evidence import EvidenceGenerator
    from .ocr import LicensePlatePipeline
    from .preprocessing.pipeline import SmartPreprocessor
    from .rules.violation_engine import ViolationEngine
except ImportError:
    from detection.vehicle_detector import VehicleDetector
    from evidence import EvidenceGenerator
    from ocr import LicensePlatePipeline
    from preprocessing.pipeline import SmartPreprocessor
    from rules.violation_engine import ViolationEngine


class TrafficViolationPipeline:
    """
    End-to-end backend pipeline for Phases 1-4.

    Flow:
        image -> preprocessing -> YOLO traffic detections -> violation rules
        -> plate detection/OCR on violating vehicle crops -> evidence artifacts
    """

    def __init__(
        self,
        preprocessor=None,
        detector=None,
        violation_engine=None,
        plate_pipeline=None,
        evidence_generator=None,
        use_preprocessing=True,
        detector_model_size="l",
        detector_confidence=0.3,
    ):
        self.preprocessor = preprocessor
        self.detector = detector
        self.violation_engine = violation_engine or ViolationEngine()
        self.plate_pipeline = plate_pipeline or LicensePlatePipeline()
        self.evidence_generator = evidence_generator or EvidenceGenerator()
        self.use_preprocessing = bool(use_preprocessing)
        self.detector_model_size = detector_model_size
        self.detector_confidence = detector_confidence

    def process_image(
        self,
        image_path,
        rule_context=None,
        extra_detections=None,
        generate_evidence=True,
    ):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.process_frame(
            image=image,
            source_image_path=str(image_path),
            rule_context=rule_context,
            extra_detections=extra_detections,
            generate_evidence=generate_evidence,
        )

    def process_frame(
        self,
        image,
        source_image_path="",
        rule_context=None,
        extra_detections=None,
        generate_evidence=True,
    ):
        rule_context = rule_context or {}

        processed_image, preprocessing_metrics = self._preprocess(image)
        detections = self._detect(processed_image)
        original_shape = list(image.shape)

        if extra_detections:
            detections.extend(self._normalize_detections(extra_detections, source="extra"))

        violations = self.violation_engine.analyze(detections, **rule_context)

        enriched_violations = []
        plate_results = []
        evidence_records = []
        preprocessing_transform = preprocessing_metrics.get("letterbox", {}) if preprocessing_metrics else {}

        for violation in violations:
            working_violation = dict(violation)
            if self.use_preprocessing and preprocessing_transform:
                working_violation = self._map_violation_to_original(
                    working_violation,
                    preprocessing_transform,
                    original_shape,
                )

            plate_source_image = processed_image
            plate_violation = violation

            if self.use_preprocessing and preprocessing_transform:
                plate_source_image = image
                plate_violation = working_violation
                plate_result = self.plate_pipeline.process_violation(plate_source_image, plate_violation)
            else:
                plate_result = self.plate_pipeline.process_violation(plate_source_image, plate_violation)

            enriched = self.plate_pipeline.attach_to_violation(working_violation, plate_result)
            plate_results.append(self._strip_image_payloads(plate_result))

            if generate_evidence:
                evidence = self.evidence_generator.generate(
                    image=image if self.use_preprocessing else processed_image,
                    violation=enriched,
                    plate_result=plate_result,
                    source_image_path=source_image_path,
                )
                enriched.update(
                    {
                        "violation_id": evidence["violation_id"],
                        "annotated_image_path": evidence["annotated_image_path"],
                        "vehicle_crop_path": evidence["vehicle_crop_path"],
                        "plate_crop_path": evidence["plate_crop_path"],
                        "metadata_path": evidence["metadata_path"],
                    }
                )
                evidence_records.append(evidence)

            enriched_violations.append(enriched)

        return {
            "source_image_path": source_image_path,
            "original_image_shape": original_shape,
            "processed_image_shape": list(processed_image.shape),
            "preprocessing_metrics": preprocessing_metrics,
            "preprocessing_transform": preprocessing_transform,
            "detections": detections,
            "violations": enriched_violations,
            "plate_results": plate_results,
            "evidence_records": evidence_records,
        }

    def _preprocess(self, image):
        if not self.use_preprocessing:
            return image.copy(), {}

        if self.preprocessor is None:
            # Keep Zero-DCE enabled when its weights are available; the class
            # already falls back safely to CLAHE if not.
            self.preprocessor = SmartPreprocessor()

        return self.preprocessor.process(image)

    def _detect(self, image):
        if self.detector is None:
            self.detector = VehicleDetector(
                model_size=self.detector_model_size,
                confidence_threshold=self.detector_confidence,
            )

        detections, _ = self.detector.detect(image)
        return self._normalize_detections(detections, source="vehicle_detector")

    @staticmethod
    def _normalize_detections(detections, source="unknown"):
        normalized = []
        for index, detection in enumerate(detections):
            item = dict(detection)
            class_name = item.get("class_name") or item.get("class") or "unknown"
            item["class"] = class_name
            item["class_name"] = class_name
            item.setdefault("confidence", 0.0)
            item.setdefault("bbox", [0, 0, 0, 0])
            item.setdefault("source", source)
            item.setdefault("detection_index", index)
            normalized.append(item)
        return normalized

    @staticmethod
    def _strip_image_payloads(plate_result):
        cleaned = dict(plate_result)
        cleaned.pop("vehicle_crop", None)
        cleaned.pop("plate_crop", None)
        return cleaned

    @staticmethod
    def _map_bbox_to_original(bbox, transform, original_shape):
        if not bbox or not transform:
            return [int(round(float(v))) for v in bbox] if bbox else [0, 0, 0, 0]

        scale = float(transform.get("scale", 1.0) or 1.0)
        pad_left = float(transform.get("pad_left", 0.0) or 0.0)
        pad_top = float(transform.get("pad_top", 0.0) or 0.0)
        height = int(original_shape[0])
        width = int(original_shape[1])

        x1, y1, x2, y2 = [float(v) for v in bbox]
        mapped = [
            int(round((x1 - pad_left) / scale)),
            int(round((y1 - pad_top) / scale)),
            int(round((x2 - pad_left) / scale)),
            int(round((y2 - pad_top) / scale)),
        ]
        mapped[0] = max(0, min(width - 1, mapped[0]))
        mapped[1] = max(0, min(height - 1, mapped[1]))
        mapped[2] = max(mapped[0] + 1, min(width, mapped[2]))
        mapped[3] = max(mapped[1] + 1, min(height, mapped[3]))
        return mapped

    def _map_violation_to_original(self, violation, transform, original_shape):
        mapped = dict(violation)
        mapped["vehicle_bbox"] = self._map_bbox_to_original(mapped.get("vehicle_bbox") or mapped.get("bbox"), transform, original_shape)
        if mapped.get("bbox"):
            mapped["bbox"] = self._map_bbox_to_original(mapped.get("bbox"), transform, original_shape)

        evidence_boxes = []
        for box in mapped.get("evidence_boxes", []):
            evidence_boxes.append(self._map_bbox_to_original(box, transform, original_shape))
        if evidence_boxes:
            mapped["evidence_boxes"] = evidence_boxes
        return mapped

    def _map_plate_result_to_original(self, plate_result, transform, original_shape):
        mapped = dict(plate_result)
        mapped["vehicle_bbox"] = self._map_bbox_to_original(mapped.get("vehicle_bbox"), transform, original_shape)
        mapped["vehicle_bbox_padded"] = self._map_bbox_to_original(
            mapped.get("vehicle_bbox_padded"),
            transform,
            original_shape,
        )
        if mapped.get("plate_bbox_original"):
            mapped["plate_bbox_original"] = self._map_bbox_to_original(
                mapped.get("plate_bbox_original"),
                transform,
                original_shape,
            )
        if mapped.get("plate_bbox_crop"):
            mapped["plate_bbox_crop"] = [int(round(float(v))) for v in mapped["plate_bbox_crop"]]
        return mapped


def process_image(image_path, **kwargs):
    return TrafficViolationPipeline().process_image(image_path, **kwargs)


if __name__ == "__main__":
    image_path = os.environ.get("GRIDLOCK_TEST_IMAGE", "")
    if not image_path:
        raise SystemExit("Set GRIDLOCK_TEST_IMAGE=/path/to/image to run the pipeline.")
    result = process_image(image_path)
    print(f"Detections: {len(result['detections'])}")
    print(f"Violations: {len(result['violations'])}")
