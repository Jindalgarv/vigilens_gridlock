import shutil
from pathlib import Path

import cv2
import numpy as np

from evidence import EvidenceGenerator
from main_pipeline import TrafficViolationPipeline


class StubVehicleDetector:
    def detect(self, image):
        return [
            {
                "class": "car",
                "class_name": "car",
                "class_id": 2,
                "confidence": 0.94,
                "bbox": [160, 150, 500, 330],
            }
        ], None


def make_synthetic_scene():
    image = np.full((400, 640, 3), 245, dtype=np.uint8)

    # Car body
    cv2.rectangle(image, (160, 150), (500, 330), (80, 80, 80), -1)
    cv2.rectangle(image, (190, 175), (470, 245), (40, 40, 40), -1)

    # License plate region
    cv2.rectangle(image, (270, 265), (395, 292), (235, 235, 235), -1)
    cv2.rectangle(image, (270, 265), (395, 292), (20, 20, 20), 2)
    for x in range(282, 380, 14):
        cv2.rectangle(image, (x, 271), (x + 7, 286), (10, 10, 10), -1)

    return image


def main():
    output_dir = Path("problem/output_evidence/test_phase4")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    image = make_synthetic_scene()
    engine_context = {
        "no_parking_zone_polygon": [(140, 300), (520, 300), (520, 360), (140, 360)],
    }

    pipeline = TrafficViolationPipeline(
        detector=StubVehicleDetector(),
        evidence_generator=EvidenceGenerator(output_dir=output_dir),
        use_preprocessing=False,
    )

    result = pipeline.process_frame(
        image=image,
        source_image_path="synthetic_phase4.jpg",
        rule_context=engine_context,
        generate_evidence=True,
    )

    assert result["detections"], "Expected stub vehicle detection"
    assert result["violations"], "Expected at least one violation"

    violation = result["violations"][0]
    assert violation.get("license_plate"), "Expected attached license plate field"
    assert "plate_bbox" in violation, "Expected original-frame plate bbox field"
    assert result["plate_results"], "Expected plate result"
    assert result["evidence_records"], "Expected evidence record"

    evidence = result["evidence_records"][0]
    assert Path(evidence["annotated_image_path"]).exists(), "Annotated evidence missing"
    assert Path(evidence["vehicle_crop_path"]).exists(), "Vehicle crop missing"
    assert Path(evidence["metadata_path"]).exists(), "Metadata JSON missing"

    print("Phase 4 synthetic pipeline test passed")
    print(f"Violation type: {violation.get('type') or violation.get('violation_type')}")
    print(f"Plate text: {violation.get('license_plate')}")
    print(f"Plate bbox: {violation.get('plate_bbox')}")
    print(f"Evidence: {evidence['metadata_path']}")


if __name__ == "__main__":
    main()
