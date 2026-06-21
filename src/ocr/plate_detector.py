import os
from pathlib import Path

import cv2
import numpy as np


class LicensePlateDetector:
    """
    Detects license plates inside a vehicle crop.

    If a YOLO plate model path is provided, that model is used. Otherwise the
    detector falls back to a conservative OpenCV rectangular-region heuristic.
    This keeps the backend runnable offline while still supporting a stronger
    trained plate detector when weights are available.
    """

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.25,
        max_detections=6,
        use_yolo=True,
    ):
        self.model_path = model_path or os.getenv("GRIDLOCK_PLATE_MODEL_PATH", "")
        self.confidence_threshold = float(confidence_threshold)
        self.max_detections = int(max_detections)
        self.use_yolo = bool(use_yolo)
        self._model = None
        self._model_load_error = ""

    def detect(self, vehicle_crop):
        """
        Return license plate detections in vehicle-crop coordinates.

        Each result contains:
            bbox_crop: [x1, y1, x2, y2]
            confidence: float
            class_name: "license_plate"
            detector: "yolo" or "opencv_heuristic"
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        if self.use_yolo and self._can_use_yolo():
            detections = self._detect_yolo(vehicle_crop)
            if detections:
                return detections[: self.max_detections]

        heuristic_detections = self._non_max_suppression(
            self._detect_heuristic(vehicle_crop),
            iou_threshold=0.35,
        )
        texture_detections = self._non_max_suppression(
            self._detect_texture_windows(vehicle_crop),
            iou_threshold=0.35,
        )

        prioritized = heuristic_detections[: min(3, self.max_detections)]
        for detection in texture_detections:
            if len(prioritized) >= self.max_detections:
                break
            if any(self._bbox_iou(detection["bbox_crop"], item["bbox_crop"]) >= 0.35 for item in prioritized):
                continue
            prioritized.append(detection)

        return prioritized[: self.max_detections]

    def _can_use_yolo(self):
        if not self.model_path:
            return False
        return Path(self.model_path).exists()

    def _load_yolo(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            return self._model
        except Exception as exc:
            self._model_load_error = str(exc)
            return None

    def _detect_yolo(self, vehicle_crop):
        model = self._load_yolo()
        if model is None:
            return []

        results = model.predict(
            source=vehicle_crop,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            names = result.names or {}
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                class_name = str(names.get(cls_id, "license_plate"))
                confidence = float(box.conf[0])

                if "plate" not in class_name.lower():
                    continue

                detections.append(
                    {
                        "bbox_crop": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "confidence": round(confidence, 4),
                        "class_id": cls_id,
                        "class_name": class_name,
                        "detector": "yolo",
                    }
                )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _detect_heuristic(self, vehicle_crop):
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        if width < 24 or height < 16:
            return []

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

        # Plates are high-contrast horizontal rectangles. Black-hat and Sobel-X
        # emphasize dark text strokes on a light rectangular background.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_x = np.absolute(grad_x)
        min_val, max_val = float(grad_x.min()), float(grad_x.max())
        if max_val > min_val:
            grad_x = ((grad_x - min_val) / (max_val - min_val) * 255).astype("uint8")
        else:
            grad_x = np.zeros_like(gray)

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        image_area = width * height
        min_area = max(80, image_area * 0.002)
        max_area = image_area * 0.25

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0:
                continue

            area = w * h
            aspect = w / float(h)
            if not (2.0 <= aspect <= 6.5):
                continue
            if not (min_area <= area <= max_area):
                continue
            if w < max(25, width * 0.08) or h < max(8, height * 0.025):
                continue

            # Prefer lower/middle regions on vehicles and plate-like aspect.
            y_center = y + h / 2.0
            vertical_score = 1.0 - abs((y_center / max(1.0, height)) - 0.62)
            aspect_score = 1.0 - min(abs(aspect - 4.0) / 4.0, 1.0)
            area_score = min(area / max(min_area, 1.0), 4.0) / 4.0
            score = 0.50 * aspect_score + 0.30 * vertical_score + 0.20 * area_score

            candidates.append(
                {
                    "bbox_crop": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": round(max(0.05, min(score, 0.75)), 4),
                    "class_id": 0,
                    "class_name": "license_plate",
                    "detector": "opencv_heuristic",
                }
            )

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return candidates

    def _detect_texture_windows(self, vehicle_crop):
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width < 40 or height < 40:
            return []

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(clahe, 40, 140)

        candidates = []
        window_widths = [
            int(width * ratio)
            for ratio in (0.18, 0.24, 0.30, 0.38)
            if int(width * ratio) >= 28
        ]
        seen = set()

        for win_w in window_widths:
            for aspect in (2.2, 2.8, 3.4, 4.2):
                win_h = int(round(win_w / aspect))
                if win_h < 10 or win_h > height * 0.20:
                    continue

                step_x = max(6, win_w // 4)
                step_y = max(5, win_h // 3)
                y_start = int(height * 0.32)
                y_end = min(height - win_h, int(height * 0.82))

                for y in range(y_start, max(y_start + 1, y_end + 1), step_y):
                    for x in range(0, max(1, width - win_w + 1), step_x):
                        key = (x, y, win_w, win_h)
                        if key in seen:
                            continue
                        seen.add(key)

                        roi_gray = clahe[y : y + win_h, x : x + win_w]
                        roi_edges = edges[y : y + win_h, x : x + win_w]
                        score = self._texture_window_score(
                            roi_gray=roi_gray,
                            roi_edges=roi_edges,
                            x=x,
                            y=y,
                            w=win_w,
                            h=win_h,
                            image_width=width,
                            image_height=height,
                            aspect=win_w / float(win_h),
                        )
                        if score < 0.38:
                            continue

                        candidates.append(
                            {
                                "bbox_crop": [
                                    float(x),
                                    float(y),
                                    float(x + win_w),
                                    float(y + win_h),
                                ],
                                "confidence": round(min(score, 0.82), 4),
                                "class_id": 0,
                                "class_name": "license_plate",
                                "detector": "texture_window",
                            }
                        )

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return candidates

    @staticmethod
    def _texture_window_score(
        roi_gray,
        roi_edges,
        x,
        y,
        w,
        h,
        image_width,
        image_height,
        aspect,
    ):
        area = max(1, w * h)
        edge_density = float(np.count_nonzero(roi_edges)) / area
        contrast = float(np.std(roi_gray)) / 80.0
        mean = float(np.mean(roi_gray)) / 255.0

        aspect_score = 1.0 - min(abs(aspect - 3.2) / 3.2, 1.0)
        y_center = (y + h / 2.0) / max(1.0, image_height)
        vertical_score = 1.0 - min(abs(y_center - 0.58) / 0.40, 1.0)
        x_center = (x + w / 2.0) / max(1.0, image_width)
        horizontal_score = 1.0 - min(abs(x_center - 0.55) / 0.65, 1.0)
        edge_score = min(edge_density / 0.38, 1.0)
        contrast_score = min(contrast / 1.45, 1.0)

        return (
            0.20 * edge_score
            + 0.18 * contrast_score
            + 0.12 * aspect_score
            + 0.28 * vertical_score
            + 0.22 * horizontal_score
        )

    def _non_max_suppression(self, detections, iou_threshold=0.35):
        selected = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            bbox = detection["bbox_crop"]
            if any(self._bbox_iou(bbox, chosen["bbox_crop"]) >= iou_threshold for chosen in selected):
                continue
            selected.append(detection)
        return selected

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom
