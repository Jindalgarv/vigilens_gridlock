import base64
import json
import os
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import requests


_EASYOCR_READER = None
_PADDLEOCR_READER = None


INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"),
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
]


def clean_plate_text(text):
    text = str(text or "").upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text.strip()


def is_valid_indian_plate(text):
    cleaned = clean_plate_text(text)
    return any(pattern.match(cleaned) for pattern in INDIAN_PLATE_PATTERNS)


class PlateOCR:
    """
    OCR reader for license plate crops.

    Engine selection:
    - "hybrid": Gemini Vision -> Google Vision OCR -> EasyOCR -> PaddleOCR
    - "online": Gemini Vision -> Google Vision OCR
    - "offline": EasyOCR -> PaddleOCR
    - "easyocr": EasyOCR only
    - "paddleocr": PaddleOCR only
    - "auto": hybrid when an API key is configured, otherwise offline
    - "mock": deterministic fallback only
    """

    def __init__(
        self,
        engine="auto",
        easyocr_gpu=False,
        google_api_key=None,
        online_timeout=4.0,
        min_confidence=0.50,
        easyocr_model_dir=None,
    ):
        self.engine = str(engine or "auto").lower()
        self.easyocr_gpu = bool(easyocr_gpu)
        self.google_api_key = google_api_key or self._google_api_key_from_env()
        self.online_timeout = float(online_timeout)
        self.min_confidence = float(min_confidence)
        self.easyocr_model_dir = easyocr_model_dir or self._default_easyocr_model_dir()
        self._online_cache = {}

    def read(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return self._empty_result("empty_crop")

        if self.engine in {"mock", "fallback", "demo"}:
            return self._mock_result(plate_crop, "Mock OCR selected.")

        engines = self._engine_sequence()

        last_error = ""
        for engine in engines:
            try:
                if engine == "gemini_vision":
                    raw_text, confidence = self._read_gemini_vision(plate_crop)
                elif engine == "google_vision":
                    raw_text, confidence = self._read_google_vision(plate_crop)
                elif engine == "easyocr":
                    raw_text, confidence = self._read_easyocr(plate_crop)
                else:
                    raw_text, confidence = self._read_paddleocr(plate_crop)

                cleaned = clean_plate_text(raw_text)
                if cleaned:
                    result = self._result(
                        text=cleaned,
                        raw_text=raw_text,
                        confidence=confidence,
                        engine=engine,
                        error="",
                    )
                    if result["valid_format"] or result["confidence"] >= self.min_confidence:
                        return result

                    last_error = (
                        f"{engine} returned low-confidence or invalid-format text: {cleaned}"
                    )
                    continue

                last_error = f"{engine} ran but did not return readable text."
            except Exception as exc:
                last_error = f"{engine} failed: {exc}"

        return self._empty_result(last_error or "No OCR engine returned a usable plate.")

    def _engine_sequence(self):
        if self.engine == "online":
            return ["gemini_vision", "google_vision"] if self.google_api_key else []
        if self.engine == "offline":
            return ["easyocr", "paddleocr"]
        if self.engine == "easyocr":
            return ["easyocr"]
        if self.engine == "paddleocr":
            return ["paddleocr"]
        if self.engine == "hybrid":
            return (
                ["gemini_vision", "google_vision"] if self.google_api_key else []
            ) + ["easyocr", "paddleocr"]
        if self.engine == "auto":
            return (
                ["gemini_vision", "google_vision"] if self.google_api_key else []
            ) + ["easyocr", "paddleocr"]
        return ["easyocr", "paddleocr"]

    @staticmethod
    def _google_api_key_from_env():
        for name in (
            "GRIDLOCK_GOOGLE_OCR_API_KEY",
            "GOOGLE_OCR_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            value = os.getenv(name)
            if value:
                return value
        return ""

    @staticmethod
    def _default_easyocr_model_dir():
        configured = os.getenv("GRIDLOCK_EASYOCR_MODEL_DIR")
        if configured:
            return configured

        project_root = Path(__file__).resolve().parents[3]
        return str(project_root / "models" / "easyocr")

    @staticmethod
    def enhance_plate_crop(plate_crop, resize_scale=3):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=60, sigmaSpace=60)
        _, thresholded = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return thresholded

    def _ocr_variants(self, plate_crop):
        variants = []
        variants.append(("original", plate_crop))

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        scale = 3 if max(gray.shape[:2]) < 180 else 2
        resized = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(clahe, -1, sharpen_kernel)
        denoised = cv2.bilateralFilter(sharpened, d=7, sigmaColor=60, sigmaSpace=60)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )

        variants.extend(
            [
                ("clahe", clahe),
                ("sharpened", sharpened),
                ("otsu", otsu),
                ("otsu_inverted", cv2.bitwise_not(otsu)),
                ("adaptive", adaptive),
            ]
        )
        return variants

    def _read_google_vision(self, plate_crop):
        if not self.google_api_key:
            raise RuntimeError("Google OCR API key is not configured.")

        crop_hash = hashlib.sha256(np.ascontiguousarray(plate_crop).tobytes()).hexdigest()
        if crop_hash in self._online_cache:
            return self._online_cache[crop_hash]

        success, encoded = cv2.imencode(".jpg", plate_crop)
        if not success:
            raise RuntimeError("Could not encode plate crop for Google OCR.")

        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        payload = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": "TEXT_DETECTION", "maxResults": 10}],
                    "imageContext": {"languageHints": ["en"]},
                }
            ]
        }
        endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_api_key}"
        response = requests.post(endpoint, json=payload, timeout=self.online_timeout)
        response.raise_for_status()
        data = response.json()

        annotation = (data.get("responses") or [{}])[0]
        if annotation.get("error"):
            message = annotation["error"].get("message", "Google Vision OCR error")
            raise RuntimeError(message)

        candidates = []
        for item in annotation.get("textAnnotations") or []:
            text = item.get("description", "")
            if text:
                candidates.append((text, self._google_confidence_for_text(text)))

        document_text = (
            annotation.get("fullTextAnnotation", {})
            .get("text", "")
        )
        if document_text:
            candidates.append((document_text, self._google_confidence_for_text(document_text)))

        best_text, best_confidence = self._best_text_candidate(candidates)
        result = (best_text, best_confidence)
        self._online_cache[crop_hash] = result
        return result

    def _read_gemini_vision(self, plate_crop):
        if not self.google_api_key:
            raise RuntimeError("Gemini API key is not configured.")

        crop_hash = "gemini:" + hashlib.sha256(
            np.ascontiguousarray(plate_crop).tobytes()
        ).hexdigest()
        if crop_hash in self._online_cache:
            return self._online_cache[crop_hash]

        from google import genai
        from PIL import Image

        image_for_ocr = self._gemini_image_variant(plate_crop)
        pil_image = Image.fromarray(cv2.cvtColor(image_for_ocr, cv2.COLOR_BGR2RGB))
        prompt = (
            "Read the Indian vehicle license plate in this cropped image. "
            "Return only JSON with this schema: "
            "{\"plate_text\":\"string or empty\", \"confidence\":\"high|medium|low\", "
            "\"reason\":\"short\"}. "
            "Use uppercase letters and digits only in plate_text. "
            "If the plate is unreadable, return an empty plate_text."
        )

        client = genai.Client(api_key=self.google_api_key)
        models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        last_error = None

        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt, pil_image],
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                    },
                )
                payload = self._parse_json_response(response.text)
                raw_text = payload.get("plate_text") or payload.get("text") or ""
                confidence = self._gemini_confidence(payload.get("confidence"), raw_text)
                result = (raw_text, confidence)
                self._online_cache[crop_hash] = result
                return result
            except Exception as exc:
                last_error = exc
                if any(code in str(exc) for code in ("429", "503", "UNAVAILABLE")):
                    continue
                break

        raise RuntimeError(f"Gemini OCR failed: {last_error}")

    @staticmethod
    def _gemini_image_variant(plate_crop):
        height, width = plate_crop.shape[:2]
        scale = max(2, min(6, int(round(260 / max(1, width)))))
        enlarged = cv2.resize(
            plate_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        return enlarged

    @staticmethod
    def _parse_json_response(text):
        text = str(text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    @staticmethod
    def _gemini_confidence(label, text):
        value = str(label or "").strip().lower()
        if value in {"high", "strong"}:
            return 0.90
        if value in {"medium", "med", "moderate"}:
            return 0.70
        if value in {"low", "weak"}:
            return 0.45
        cleaned = clean_plate_text(text)
        if is_valid_indian_plate(cleaned):
            return 0.85
        if len(cleaned) >= 6:
            return 0.60
        return 0.0

    @staticmethod
    def _google_confidence_for_text(text):
        cleaned = clean_plate_text(text)
        if is_valid_indian_plate(cleaned):
            return 0.85
        if len(cleaned) >= 6:
            return 0.65
        return 0.45

    def _read_easyocr(self, plate_crop):
        global _EASYOCR_READER
        if _EASYOCR_READER is None:
            import easyocr

            os.makedirs(self.easyocr_model_dir, exist_ok=True)
            _EASYOCR_READER = easyocr.Reader(
                ["en"],
                gpu=self.easyocr_gpu,
                model_storage_directory=self.easyocr_model_dir,
                user_network_directory=self.easyocr_model_dir,
            )

        candidates = []
        for _, variant in self._ocr_variants(plate_crop):
            results = _EASYOCR_READER.readtext(
                variant,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
            )
            for item in results:
                if len(item) < 3:
                    continue
                candidates.append((item[1], float(item[2])))
        return self._best_text_candidate(candidates)

    @staticmethod
    def _best_easyocr_result(results):
        best_text = ""
        best_confidence = 0.0
        for item in results:
            if len(item) < 3:
                continue
            text = clean_plate_text(item[1])
            confidence = float(item[2])
            if len(text) >= 4 and confidence > best_confidence:
                best_text = text
                best_confidence = confidence
        return best_text, best_confidence

    def _read_paddleocr(self, plate_crop):
        global _PADDLEOCR_READER
        if _PADDLEOCR_READER is None:
            from paddleocr import PaddleOCR

            try:
                _PADDLEOCR_READER = PaddleOCR(lang="en")
            except TypeError:
                _PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en")

        candidates = []
        for _, variant in self._ocr_variants(plate_crop):
            if len(variant.shape) == 2:
                ocr_input = cv2.cvtColor(variant, cv2.COLOR_GRAY2RGB)
            else:
                ocr_input = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            try:
                raw_result = _PADDLEOCR_READER.ocr(ocr_input, cls=True)
            except Exception:
                raw_result = _PADDLEOCR_READER.predict(ocr_input)
            candidates.extend(self._extract_paddle_pairs(raw_result))
        return self._best_text_candidate(candidates)

    def _best_paddleocr_result(self, raw_result):
        best_text = ""
        best_confidence = 0.0
        for text, confidence in self._extract_paddle_pairs(raw_result):
            cleaned = clean_plate_text(text)
            confidence = float(confidence)
            if len(cleaned) >= 4 and confidence > best_confidence:
                best_text = cleaned
                best_confidence = confidence
        return best_text, best_confidence

    def _extract_paddle_pairs(self, value):
        pairs = []

        if value is None:
            return pairs

        if isinstance(value, dict):
            texts = value.get("rec_texts") or value.get("texts") or []
            scores = value.get("rec_scores") or value.get("scores") or []
            for text, score in zip(texts, scores):
                pairs.append((str(text), float(score)))
            for nested in value.values():
                pairs.extend(self._extract_paddle_pairs(nested))
            return pairs

        if isinstance(value, (list, tuple)):
            if (
                len(value) >= 2
                and isinstance(value[1], (list, tuple))
                and len(value[1]) >= 2
                and isinstance(value[1][0], str)
            ):
                return [(str(value[1][0]), float(value[1][1]))]
            if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], (int, float)):
                return [(str(value[0]), float(value[1]))]
            for item in value:
                pairs.extend(self._extract_paddle_pairs(item))

        return pairs

    def _mock_result(self, plate_crop, error):
        raw = self._mock_plate_from_image(plate_crop)
        return self._result(
            text=raw,
            raw_text=raw,
            confidence=0.0,
            engine="mock_fallback",
            error=error or "No OCR engine available.",
            force_manual_review=True,
        )

    @staticmethod
    def _mock_plate_from_image(plate_crop):
        digest = hashlib.md5(np.ascontiguousarray(plate_crop).tobytes()).hexdigest()
        import random
        import string

        rng = random.Random(int(digest[:12], 16))
        state = "".join(rng.choices(string.ascii_uppercase, k=2))
        district = "".join(rng.choices(string.digits, k=2))
        series = "".join(rng.choices(string.ascii_uppercase, k=2))
        number = "".join(rng.choices(string.digits, k=4))
        return f"{state}{district}{series}{number}"

    @staticmethod
    def _empty_result(error):
        return {
            "text": "",
            "raw_text": "",
            "confidence": 0.0,
            "engine": "none",
            "valid_format": False,
            "needs_manual_review": True,
            "error": error,
        }

    @staticmethod
    def _result(text, raw_text, confidence, engine, error="", force_manual_review=False):
        cleaned = clean_plate_text(text)
        valid = is_valid_indian_plate(cleaned)
        confidence = round(float(confidence or 0.0), 4)
        return {
            "text": cleaned,
            "raw_text": raw_text,
            "confidence": confidence,
            "engine": engine,
            "valid_format": valid,
            "needs_manual_review": bool(force_manual_review or not valid or confidence < 0.50),
            "error": error,
        }

    @classmethod
    def _best_text_candidate(cls, candidates):
        best_text = ""
        best_confidence = 0.0
        best_score = -1.0

        expanded_candidates = []
        for text, confidence in candidates:
            raw = str(text or "")
            confidence = float(confidence or 0.0)
            expanded_candidates.append((raw, confidence))

            cleaned = clean_plate_text(raw)
            if len(cleaned) > 10:
                for match in re.finditer(r"[A-Z0-9]{6,12}", cleaned):
                    expanded_candidates.append((match.group(0), confidence))

        for text, confidence in expanded_candidates:
            cleaned = clean_plate_text(text)
            if len(cleaned) < 4:
                continue

            valid_bonus = 2.0 if is_valid_indian_plate(cleaned) else 0.0
            length_score = 1.0 - min(abs(len(cleaned) - 10) / 10.0, 1.0)
            score = valid_bonus + confidence + 0.25 * length_score

            if score > best_score:
                best_score = score
                best_text = cleaned
                best_confidence = confidence

        return best_text, best_confidence
