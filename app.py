from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from src.context import AREA_PROFILES, build_rule_context
from src.evidence import EvidenceGenerator
from src.main_pipeline import TrafficViolationPipeline
from src.ocr import LicensePlatePipeline
from src.ocr.plate_ocr import PlateOCR


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "app_data"
UPLOAD_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"
DB_PATH = DATA_DIR / "traffic_violations.db"

for directory in (DATA_DIR, UPLOAD_DIR, EVIDENCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


LOCATIONS = {
    "Vastrapur Crossroad, Ahmedabad": "23.0389 N, 72.5298 E",
    "Lal Darwaza, Ahmedabad": "23.0225 N, 72.5714 E",
    "Hazratganj Junction, Lucknow": "26.8467 N, 80.9462 E",
    "MG Road, Bangalore": "12.9716 N, 77.6094 E",
    "Connaught Place, Delhi": "28.6315 N, 77.2167 E",
    "Custom Location": "",
}


SEVERITY = {
    "Helmet Non-Compliance": 7,
    "Suspected Helmet Non-Compliance": 6,
    "Overcrowding / Triple Riding": 8,
    "Red-Light Violation": 9,
    "Stop-Line / Wrong Way": 6,
    "Seatbelt Non-Compliance": 7,
    "Suspected Illegal Parking": 5,
    "Illegal Parking": 6,
    "Suspected Wrong-Side Driving": 7,
    "Wrong-Side Driving": 8,
    "Speeding": 8,
    "Hazardous/Long Load": 7,
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id TEXT UNIQUE,
            timestamp TEXT NOT NULL,
            image_filename TEXT,
            image_hash TEXT,
            location_name TEXT,
            gps_coordinates TEXT,
            area_type TEXT,
            vehicle_type TEXT,
            violation_type TEXT,
            confidence REAL,
            area_prior REAL,
            review_priority TEXT,
            ocr_plate_number TEXT,
            ocr_confidence REAL,
            ocr_engine TEXT,
            manual_review INTEGER,
            details TEXT,
            metadata_json TEXT,
            original_image_path TEXT,
            annotated_image_path TEXT,
            vehicle_crop_path TEXT,
            plate_crop_path TEXT,
            review_status TEXT NOT NULL DEFAULT 'Pending Review'
        )
        """
    )
    conn.commit()
    conn.close()


def load_records():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM violations ORDER BY id DESC", conn)
    conn.close()
    return df


def insert_record(record):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    columns = list(record.keys())
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        f"INSERT INTO violations ({', '.join(columns)}) VALUES ({placeholders})",
        [record[col] for col in columns],
    )
    conn.commit()
    conn.close()


def update_review_status(row_id, status):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE violations SET review_status = ? WHERE id = ?", (status, int(row_id)))
    conn.commit()
    conn.close()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def save_upload(uploaded_file, data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    path.write_bytes(data)
    return path


def load_pil(path):
    return Image.open(path).convert("RGB")


def crop_pil_from_bbox(image_path, bbox, padding=8):
    if not bbox:
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    width, height = image.size
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def priority_for_violation(violation, area_prior):
    confidence = float(violation.get("confidence", 0.0) or 0.0)
    needs_review = bool(violation.get("plate_needs_manual_review", True))
    score = confidence + 0.35 * float(area_prior or 0.0)
    if needs_review:
        score += 0.15
    if score >= 0.85:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def bbox_center(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_motorcycle_rider_area(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return [
        x1 - 0.45 * width,
        y1 - 1.30 * height,
        x2 + 0.45 * width,
        y2 + 0.25 * height,
    ]


def union_bbox(boxes, padding=12, image_shape=None):
    if not boxes:
        return None

    xs1 = [float(box[0]) for box in boxes]
    ys1 = [float(box[1]) for box in boxes]
    xs2 = [float(box[2]) for box in boxes]
    ys2 = [float(box[3]) for box in boxes]

    x1 = min(xs1) - padding
    y1 = min(ys1) - padding
    x2 = max(xs2) + padding
    y2 = max(ys2) + padding

    if image_shape:
        height, width = int(image_shape[0]), int(image_shape[1])
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(x1 + 1, min(width, int(round(x2))))
        y2 = max(y1 + 1, min(height, int(round(y2))))
        return [x1, y1, x2, y2]

    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def map_bbox_to_original(bbox, transform, original_shape):
    if not bbox:
        return None
    if not transform:
        return [int(round(float(v))) for v in bbox]

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


def select_suspected_helmet_candidate(result):
    detections = result.get("detections", [])
    image_shape = result.get("processed_image_shape") or result.get("original_image_shape") or [0, 0, 0]
    image_center = (float(image_shape[1]) / 2.0, float(image_shape[0]) / 2.0)

    motorcycles = [d for d in detections if d.get("class") == "motorcycle"]
    persons = [d for d in detections if d.get("class") == "person"]
    candidates = []

    for index, motorcycle in enumerate(motorcycles):
        motorcycle_bbox = motorcycle.get("bbox", [0, 0, 0, 0])
        rider_area = expand_motorcycle_rider_area(motorcycle_bbox)
        riders = []
        for person in persons:
            person_bbox = person.get("bbox", [0, 0, 0, 0])
            center = bbox_center(person_bbox)
            bottom_center = (center[0], float(person_bbox[3]))
            if point_in_bbox(center, rider_area) or point_in_bbox(bottom_center, rider_area):
                riders.append(person)

        if not riders:
            continue

        avg_person_conf = sum(float(r.get("confidence", 0.0) or 0.0) for r in riders) / len(riders)
        confidence = round((float(motorcycle.get("confidence", 0.0) or 0.0) + avg_person_conf) / 2.0, 4)
        bike_center = bbox_center(motorcycle_bbox)
        center_distance = abs(bike_center[0] - image_center[0]) + abs(bike_center[1] - image_center[1])
        bike_area = max(1.0, (float(motorcycle_bbox[2]) - float(motorcycle_bbox[0])) * (float(motorcycle_bbox[3]) - float(motorcycle_bbox[1])))
        score = (
            0.55 * confidence
            + 0.15 * len(riders)
            + 0.15 * (bike_area / max(1.0, float(image_shape[0]) * float(image_shape[1])))
            + 0.15 * (1.0 / (1.0 + center_distance / max(1.0, float(image_shape[1]))))
        )
        candidates.append(
            {
                "motorcycle": motorcycle,
                "riders": riders,
                "confidence": confidence,
                "score": score,
                "detection_index": index,
            }
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["score"])


def add_suspected_helmet_review(result, image_path, plate_pipeline, evidence_generator):
    """
    COCO YOLO cannot detect helmet state. This assist creates one manual-review
    violation when a motorcycle is spatially associated with rider detections.
    It does not pretend to be a confirmed no-helmet classifier.
    """
    candidate = select_suspected_helmet_candidate(result)
    if not candidate:
        return result

    motorcycle = candidate["motorcycle"]
    riders = candidate["riders"]
    transform = result.get("preprocessing_transform") or {}
    original_shape = result.get("original_image_shape") or result.get("processed_image_shape") or [0, 0, 0]

    vehicle_bbox_processed = union_bbox(
        [motorcycle.get("bbox", [0, 0, 0, 0])] + [r.get("bbox", [0, 0, 0, 0]) for r in riders],
        padding=18,
        image_shape=result.get("processed_image_shape"),
    )
    vehicle_bbox_original = map_bbox_to_original(vehicle_bbox_processed, transform, original_shape)
    rider_boxes_original = [
        map_bbox_to_original(r.get("bbox", [0, 0, 0, 0]), transform, original_shape)
        for r in riders
    ]

    violation = {
        "type": "Suspected Helmet Non-Compliance",
        "violation_type": "Suspected Helmet Non-Compliance",
        "vehicle_class": "motorcycle",
        "vehicle_bbox": vehicle_bbox_original,
        "confidence": candidate["confidence"],
        "detection_index": candidate["detection_index"],
        "rider_count": len(riders),
        "evidence_boxes": rider_boxes_original,
        "license_plate": "N/A",
        "ocr_confidence": 0.0,
        "ocr_engine": "none",
        "plate_needs_manual_review": True,
        "details": (
            "COCO YOLO detected a motorcycle with associated rider(s). "
            "Helmet status is marked for manual review because no specialized "
            "helmet/no-helmet model is active."
        ),
    }

    image = cv2.imread(str(image_path))
    if image is None:
        result.setdefault("violations", []).append(violation)
        return result

    plate_result = plate_pipeline.process_violation(image, violation)
    enriched = plate_pipeline.attach_to_violation(violation, plate_result)
    evidence = evidence_generator.generate(
        image=image,
        violation=enriched,
        plate_result=plate_result,
        source_image_path=str(image_path),
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

    result.setdefault("violations", []).append(enriched)
    result.setdefault("evidence_records", []).append(evidence)
    result.setdefault("plate_results", []).append(
        {k: v for k, v in plate_result.items() if k not in {"vehicle_crop", "plate_crop"}}
    )
    return result


def enrich_violations(result, area_type):
    priors = AREA_PROFILES.get(area_type, {}).get("priors", {})
    enriched = []
    for violation in result.get("violations", []):
        item = dict(violation)
        vtype = item.get("type") or item.get("violation_type") or "Violation"
        prior = float(priors.get(vtype, 0.50))
        item["violation_type"] = vtype
        item["area_prior"] = prior
        item["review_priority"] = priority_for_violation(item, prior)
        item["severity"] = SEVERITY.get(vtype, 5)
        enriched.append(item)
    result["violations"] = enriched
    return result


def run_backend(
    image_path,
    area_type,
    location,
    traffic_light,
    stop_line_ratio,
    confidence,
    ocr_mode,
    google_key,
    use_preprocessing,
    enable_stop_zone,
    enable_no_parking_zone,
    enable_wrong_side_zone,
    wrong_side,
):
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"Could not read uploaded image: {image_path}")

    context_shape = (640, 640, 3) if use_preprocessing else raw.shape
    rule_context = build_rule_context(
        area_type=area_type,
        image_shape=context_shape,
        traffic_light_status=traffic_light,
        stop_line_ratio=stop_line_ratio,
        enable_stop_zone=enable_stop_zone,
        enable_no_parking_zone=enable_no_parking_zone,
        enable_wrong_side_zone=enable_wrong_side_zone,
        wrong_side=wrong_side,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    evidence = EvidenceGenerator(output_dir=EVIDENCE_DIR / run_id)
    ocr = PlateOCR(
        engine=ocr_mode,
        easyocr_gpu=False,
        google_api_key=google_key or None,
        online_timeout=5.0,
    )
    plate_pipeline = LicensePlatePipeline(ocr_reader=ocr)
    pipeline = TrafficViolationPipeline(
        plate_pipeline=plate_pipeline,
        evidence_generator=evidence,
        use_preprocessing=use_preprocessing,
        detector_model_size="l",
        detector_confidence=confidence,
    )

    result = pipeline.process_image(
        image_path,
        rule_context=rule_context,
        generate_evidence=True,
    )
    result = add_suspected_helmet_review(result, image_path, plate_pipeline, evidence)
    result["rule_context"] = rule_context
    result["area_type"] = area_type
    result["location"] = location
    return enrich_violations(result, area_type)


def save_violations_to_db(result, image_path, image_hash, location, gps):
    saved = 0
    for violation in result.get("violations", []):
        violation_id = violation.get("violation_id") or f"TVAI-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8].upper()}"
        record = {
            "violation_id": violation_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_filename": Path(image_path).name,
            "image_hash": image_hash,
            "location_name": location,
            "gps_coordinates": gps,
            "area_type": result.get("area_type", ""),
            "vehicle_type": violation.get("vehicle_class") or violation.get("vehicle_type") or "unknown",
            "violation_type": violation.get("violation_type") or violation.get("type") or "Violation",
            "confidence": float(violation.get("confidence", 0.0) or 0.0),
            "area_prior": float(violation.get("area_prior", 0.0) or 0.0),
            "review_priority": violation.get("review_priority", "Medium"),
            "ocr_plate_number": violation.get("license_plate") or "N/A",
            "ocr_confidence": float(violation.get("ocr_confidence", 0.0) or 0.0),
            "ocr_engine": violation.get("ocr_engine", ""),
            "manual_review": int(bool(violation.get("plate_needs_manual_review", True))),
            "details": violation.get("details", ""),
            "metadata_json": json.dumps(violation, default=str),
            "original_image_path": str(image_path),
            "annotated_image_path": violation.get("annotated_image_path", ""),
            "vehicle_crop_path": violation.get("vehicle_crop_path", ""),
            "plate_crop_path": violation.get("plate_crop_path", ""),
            "review_status": "Pending Review",
        }
        insert_record(record)
        saved += 1
    return saved


def image_or_notice(path, caption):
    if path and Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"{caption}: not available")


def vehicle_crop_or_notice(violation, caption):
    if violation.get("vehicle_crop_path") and Path(violation["vehicle_crop_path"]).exists():
        st.image(str(violation["vehicle_crop_path"]), caption=caption, use_container_width=True)
        return

    source_image = st.session_state.get("gridlock_image_path")
    fallback = crop_pil_from_bbox(
        source_image,
        violation.get("vehicle_bbox") or violation.get("bbox"),
    )
    if fallback is not None:
        st.image(fallback, caption=f"{caption} (computed)", use_container_width=True)
    else:
        st.caption(f"{caption}: not available")


st.set_page_config(
    page_title="Gridlock Traffic Vision AI",
    page_icon="GV",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1240px; }
    .metric-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.75rem; background: #fff; }
    .small-muted { color: #6b7280; font-size: 0.85rem; }
    .priority-high { color: #991b1b; font-weight: 700; }
    .priority-medium { color: #92400e; font-weight: 700; }
    .priority-low { color: #166534; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gridlock Traffic Vision AI")
st.caption("Unified backend: preprocessing, YOLO11 detection, modular violation rules, hybrid OCR, and evidence review.")

records = load_records()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Saved Violations", len(records))
k2.metric("Pending Review", int((records["review_status"] == "Pending Review").sum()) if not records.empty else 0)
k3.metric("Approved", int((records["review_status"] == "Approved").sum()) if not records.empty else 0)
k4.metric("Unique Plates", int(records["ocr_plate_number"].replace("N/A", pd.NA).dropna().nunique()) if not records.empty else 0)

tab_detect, tab_review, tab_analytics = st.tabs(["Detect", "Human Review", "Analytics"])

with tab_detect:
    with st.sidebar:
        st.header("Runtime Context")
        location = st.selectbox("Deployment Location", list(LOCATIONS.keys()))
        gps = LOCATIONS[location]
        area_type = st.selectbox("Area Profile", list(AREA_PROFILES.keys()))
        st.caption(AREA_PROFILES[area_type]["description"])

        traffic_light = st.selectbox("Traffic Signal", ["RED", "GREEN", "YELLOW"])
        confidence = st.slider("YOLO Confidence", 0.15, 0.80, 0.25, 0.05)
        stop_line_ratio = st.slider("Stop/Zone Line Position", 0.25, 0.90, 0.60, 0.05)

        st.subheader("Rule Zones")
        st.caption("Enable zone rules only for calibrated camera views or drawn demo zones.")
        enable_stop_zone = st.checkbox("Enable calibrated stop/red-light zone", value=False)
        enable_no_parking_zone = st.checkbox("Enable no-parking zone", value=area_type in {"Market / No-Parking Zone"})
        enable_wrong_side_zone = st.checkbox("Enable wrong-side zone", value=False)
        wrong_side = st.selectbox("Wrong-side zone", ["left", "right"])

        st.subheader("OCR")
        ocr_mode = st.selectbox("OCR Mode", ["auto", "hybrid", "online", "offline", "easyocr"], index=0)
        google_key = st.text_input(
            "Gemini / Google OCR API Key",
            value=os.getenv("GRIDLOCK_GOOGLE_OCR_API_KEY") or os.getenv("GEMINI_API_KEY", ""),
            type="password",
        )
        use_preprocessing = st.checkbox("Use SmartPreprocessor", value=False)
        st.caption("Keep preprocessing off for original-image coordinates during demos.")

    upload = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.info("Upload an image to run the unified backend.")
    else:
        data = upload.getvalue()
        image_hash = sha256_bytes(data)
        image_path = save_upload(upload, data)
        pil = Image.open(BytesIO(data)).convert("RGB")

        left, right = st.columns([1, 1])
        with left:
            st.image(pil, caption=f"Original upload ({pil.width} x {pil.height})", use_container_width=True)
            st.caption(f"SHA-256: `{image_hash[:48]}...`")

        if st.button("Run Unified Backend", type="primary", use_container_width=True):
            with st.spinner("Running YOLO11, rules engine, plate OCR, and evidence generation..."):
                result = run_backend(
                    image_path=image_path,
                    area_type=area_type,
                    location=location,
                    traffic_light=traffic_light,
                    stop_line_ratio=stop_line_ratio,
                    confidence=confidence,
                    ocr_mode=ocr_mode,
                    google_key=google_key,
                    use_preprocessing=use_preprocessing,
                    enable_stop_zone=enable_stop_zone,
                    enable_no_parking_zone=enable_no_parking_zone,
                    enable_wrong_side_zone=enable_wrong_side_zone,
                    wrong_side=wrong_side,
                )
            st.session_state["gridlock_result"] = result
            st.session_state["gridlock_image_path"] = str(image_path)
            st.session_state["gridlock_image_hash"] = image_hash
            st.session_state["gridlock_location"] = location
            st.session_state["gridlock_gps"] = gps
            st.rerun()

        result = st.session_state.get("gridlock_result")
        if result:
            violations = result.get("violations", [])
            detections = result.get("detections", [])

            with right:
                first_annotated = ""
                for violation in violations:
                    if violation.get("annotated_image_path"):
                        first_annotated = violation["annotated_image_path"]
                        break
                if first_annotated:
                    image_or_notice(first_annotated, "Annotated evidence")
                else:
                    st.info("No annotated evidence yet.")

            st.subheader("Detection Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Detections", len(detections))
            s2.metric("Violations", len(violations))
            s3.metric("Plate Results", len(result.get("plate_results", [])))
            s4.metric("Area Profile", result.get("area_type", area_type))

            if result.get("rule_context", {}).get("area_profile"):
                st.caption("Area focus: " + ", ".join(result["rule_context"]["area_profile"].get("focus", [])))

            if not violations:
                st.success("No violations detected under the current rule context.")
            else:
                st.subheader("Violation Evidence")
                for idx, violation in enumerate(violations, start=1):
                    with st.container(border=True):
                        h1, h2, h3, h4 = st.columns([2.3, 1, 1, 1])
                        h1.markdown(f"**{idx}. {violation.get('violation_type')}**")
                        h2.metric("Vision Conf.", f"{float(violation.get('confidence', 0.0) or 0.0):.2f}")
                        h3.metric("Area Prior", f"{float(violation.get('area_prior', 0.0) or 0.0):.2f}")
                        priority = violation.get("review_priority", "Medium")
                        h4.markdown(f"Priority<br><span class='priority-{priority.lower()}'>{priority}</span>", unsafe_allow_html=True)

                        c1, c2, c3 = st.columns([1, 1, 1.4])
                        with c1:
                            vehicle_crop_or_notice(violation, "Vehicle crop")
                        with c2:
                            image_or_notice(violation.get("plate_crop_path"), "Plate crop")
                        with c3:
                            st.markdown(f"**Vehicle:** `{violation.get('vehicle_class', 'unknown')}`")
                            st.markdown(f"**Plate:** `{violation.get('license_plate', 'N/A')}`")
                            st.markdown(f"**OCR:** {violation.get('ocr_engine', 'none')} ({float(violation.get('ocr_confidence', 0.0) or 0.0):.2f})")
                            st.markdown(f"**Manual review:** {'Yes' if violation.get('plate_needs_manual_review') else 'No'}")
                            if violation.get("details"):
                                st.caption(violation["details"])

                if st.button("Save Violations for Human Review", type="primary", use_container_width=True):
                    saved = save_violations_to_db(
                        result=result,
                        image_path=st.session_state["gridlock_image_path"],
                        image_hash=st.session_state["gridlock_image_hash"],
                        location=st.session_state["gridlock_location"],
                        gps=st.session_state["gridlock_gps"],
                    )
                    st.success(f"Saved {saved} violation(s).")
                    st.rerun()

with tab_review:
    st.subheader("Human Review Queue")
    df = load_records()
    if df.empty:
        st.info("No saved violations yet.")
    else:
        f1, f2, f3 = st.columns(3)
        status = f1.selectbox("Status", ["All", "Pending Review", "Approved", "Rejected"])
        area = f2.selectbox("Area", ["All"] + sorted(df["area_type"].dropna().unique().tolist()))
        f3.metric("Rows", len(df))

        filtered = df.copy()
        if status != "All":
            filtered = filtered[filtered["review_status"] == status]
        if area != "All":
            filtered = filtered[filtered["area_type"] == area]

        st.dataframe(
            filtered[
                [
                    "id",
                    "violation_id",
                    "timestamp",
                    "violation_type",
                    "vehicle_type",
                    "ocr_plate_number",
                    "review_priority",
                    "review_status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered.empty:
            selected = st.selectbox(
                "Select record",
                filtered["id"].tolist(),
                format_func=lambda row_id: filtered[filtered["id"] == row_id].iloc[0]["violation_id"],
            )
            row = filtered[filtered["id"] == selected].iloc[0]
            r1, r2 = st.columns([1.5, 1])
            with r1:
                image_or_notice(row.get("annotated_image_path"), "Annotated evidence")
            with r2:
                st.markdown(f"**Violation:** {row.get('violation_type')}")
                st.markdown(f"**Plate:** `{row.get('ocr_plate_number')}`")
                st.markdown(f"**Vehicle:** {row.get('vehicle_type')}")
                st.markdown(f"**Location:** {row.get('location_name')}")
                st.markdown(f"**Priority:** {row.get('review_priority')}")
                b1, b2, b3 = st.columns(3)
                if b1.button("Approve"):
                    update_review_status(selected, "Approved")
                    st.rerun()
                if b2.button("Reject"):
                    update_review_status(selected, "Rejected")
                    st.rerun()
                if b3.button("Pending"):
                    update_review_status(selected, "Pending Review")
                    st.rerun()

with tab_analytics:
    st.subheader("Analytics")
    df = load_records()
    if df.empty:
        st.info("No analytics yet. Save violations from the Detect tab first.")
    else:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Violations by Type**")
            st.bar_chart(df["violation_type"].value_counts())
        with a2:
            st.markdown("**Violations by Area**")
            st.bar_chart(df["area_type"].value_counts())

        st.markdown("**Repeat Plates**")
        plate_df = (
            df[df["ocr_plate_number"].fillna("N/A") != "N/A"]
            .groupby("ocr_plate_number")
            .size()
            .reset_index(name="violations")
            .sort_values("violations", ascending=False)
        )
        st.dataframe(plate_df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV Report", csv, "gridlock_violations.csv", "text/csv")
