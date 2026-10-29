import os
import sys
import logging
import base64
import io
import re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps
import pytesseract
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Configure pytesseract binary path
_tesseract_path = getattr(config, "TESSERACT_CMD_PATH", r"C:/Program Files/Tesseract-OCR/tesseract.exe")
_tesseract_path = os.path.normpath(_tesseract_path)
if os.path.exists(_tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path
    logger.info(f"Tesseract OCR path configured: {_tesseract_path}")
else:
    logger.warning(f"Tesseract OCR binary not found at '{_tesseract_path}'.")


class OCRQualityResult:
    def __init__(self, text: str, confidence: float, word_count: int, pass_name: str):
        self.text = text
        self.confidence = confidence
        self.word_count = word_count
        self.pass_name = pass_name

    @property
    def score(self) -> float:
        if not self.text.strip():
            return 0.0
        # Text completeness and alpha ratio
        alpha_chars = len(re.findall(r'[a-zA-Z0-9]', self.text))
        alpha_ratio = alpha_chars / max(1, len(self.text))
        # Composite score balancing confidence, length, and readability
        return self.confidence * min(1.0, self.word_count / 15.0) * alpha_ratio


class OCRManager:
    _rapidocr_engine = None

    @staticmethod
    def is_tesseract_available() -> bool:
        """Check if Tesseract binary is installed and executable."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def is_rapidocr_available() -> bool:
        """Check if rapidocr-onnxruntime package is installed and importable."""
        try:
            import rapidocr_onnxruntime
            return True
        except ImportError:
            return False

    @staticmethod
    def get_available_languages() -> List[str]:
        """Return list of available Tesseract OCR language codes."""
        try:
            if OCRManager.is_tesseract_available():
                langs = pytesseract.get_languages()
                return [l for l in langs if l != "osd"]
        except Exception:
            pass
        return ["eng"]

    @staticmethod
    def preprocess_image(image: Image.Image, variant: str = "standard") -> Image.Image:
        """Apply high-quality OpenCV and Pillow image preprocessing for OCR without excessive scaling.
        
        Variants:
        - 'standard': EXIF correction, moderate scaling, Grayscale + CLAHE contrast + mild median denoise
        - 'adaptive_thresh': Gaussian adaptive thresholding (for shadow/uneven lighting removal)
        - 'deskew': Skew angle estimation and rotation correction
        """
        try:
            import cv2

            # 1. Correct EXIF orientation
            image = ImageOps.exif_transpose(image)

            # 2. Resize within reasonable bounds
            max_dim = getattr(config, "OCR_MAX_IMAGE_DIM", 2000)
            w, h = image.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            elif max(w, h) < 600:
                scale = min(2.0, 1000.0 / float(max(1, max(w, h))))
                image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            # Convert PIL to OpenCV BGR/Gray
            rgb_img = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

            if variant == "adaptive_thresh":
                # Adaptive threshold for documents with uneven lighting/shadows
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9
                )
                return Image.fromarray(thresh)

            elif variant == "deskew":
                # Estimate skew angle from text coordinates
                pts = np.column_stack(np.where(gray < 220))
                if len(pts) > 100:
                    rect = cv2.minAreaRect(pts)
                    angle = rect[-1]
                    if angle < -45:
                        angle = 90 + angle
                    elif angle > 45:
                        angle = angle - 90

                    if 0.5 < abs(angle) < 45:
                        (gh, gw) = gray.shape[:2]
                        M = cv2.getRotationMatrix2D((gw // 2, gh // 2), angle, 1.0)
                        deskewed = cv2.warpAffine(
                            gray, M, (gw, gh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                        )
                        return Image.fromarray(deskewed)
                return Image.fromarray(gray)

            else:
                # Default 'standard': CLAHE contrast enhancement + median blur
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                contrast = clahe.apply(gray)
                denoised = cv2.medianBlur(contrast, 3)
                return Image.fromarray(denoised)

        except Exception as e:
            logger.warning(f"OpenCV preprocessing error ({variant}), using Pillow fallback: {e}")
            try:
                from PIL import ImageEnhance
                image = ImageOps.exif_transpose(image)
                gray = ImageOps.grayscale(image)
                enhancer = ImageEnhance.Contrast(gray)
                return enhancer.enhance(1.8)
            except Exception:
                return image

    @classmethod
    def _run_tesseract_with_data(cls, image: Image.Image, psm: int = 3, lang: str = "eng") -> OCRQualityResult:
        """Run Tesseract image_to_data to capture confidence and bounding box statistics."""
        psm_config = f"--psm {psm}"
        try:
            data = pytesseract.image_to_data(
                image, 
                lang=lang, 
                config=psm_config, 
                output_type=pytesseract.Output.DICT
            )
            texts = data.get("text", [])
            confs = data.get("conf", [])

            valid_words = []
            valid_confs = []
            lines = {}

            for i, (word, conf) in enumerate(zip(texts, confs)):
                clean_w = str(word).strip()
                try:
                    conf_f = float(conf)
                except (ValueError, TypeError):
                    conf_f = -1.0

                if clean_w and conf_f >= 0:
                    line_num = data.get("line_num", [0])[i] if "line_num" in data else 0
                    block_num = data.get("block_num", [0])[i] if "block_num" in data else 0
                    key = (block_num, line_num)
                    if key not in lines:
                        lines[key] = []
                    lines[key].append(clean_w)
                    valid_words.append(clean_w)
                    valid_confs.append(conf_f)

            # Reconstruct text preserving line breaks
            formatted_lines = [" ".join(words) for words in lines.values() if words]
            full_text = "\n".join(formatted_lines).strip()
            avg_conf = (sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0

            return OCRQualityResult(
                text=full_text, 
                confidence=avg_conf, 
                word_count=len(valid_words), 
                pass_name=f"Tesseract(psm={psm})"
            )
        except Exception as e:
            logger.debug(f"Tesseract run failed (psm={psm}): {e}")
            return OCRQualityResult(text="", confidence=0.0, word_count=0, pass_name=f"Failed(psm={psm})")

    @classmethod
    def extract_text_via_rapidocr(cls, image: Image.Image) -> OCRQualityResult:
        """Optional local OCR extraction using RapidOCR (ONNXRuntime)."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            if cls._rapidocr_engine is None:
                cls._rapidocr_engine = RapidOCR()

            # Ensure EXIF orientation is corrected
            img = ImageOps.exif_transpose(image).convert("RGB")
            np_img = np.array(img)
            result, _ = cls._rapidocr_engine(np_img)

            if not result:
                return OCRQualityResult("", 0.0, 0, "RapidOCR")

            lines = []
            confs = []
            total_words = 0

            for item in result:
                if len(item) >= 3:
                    text_line = str(item[1]).strip()
                    conf_val = float(item[2]) * 100.0  # RapidOCR returns 0.0-1.0
                    if text_line:
                        lines.append(text_line)
                        confs.append(conf_val)
                        total_words += len(text_line.split())

            full_text = "\n".join(lines).strip()
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0
            return OCRQualityResult(full_text, avg_conf, total_words, "RapidOCR")
        except Exception as e:
            logger.warning(f"RapidOCR execution failed: {e}")
            return OCRQualityResult("", 0.0, 0, "RapidOCR-Failed")

    @staticmethod
    def extract_text_via_ollama_vision(image: Image.Image) -> str:
        """Fallback: attempt text extraction via local Ollama vision model if available."""
        try:
            import requests
            tags_res = requests.get(f"{config.OLLAMA_API_URL}/api/tags", timeout=3)
            if tags_res.status_code != 200:
                return ""
            models = [m['name'] for m in tags_res.json().get('models', [])]
            vision_models = [m for m in models if any(v in m.lower() for v in ['vision', 'llava', 'bakllava', 'moondream', 'minicpm'])]
            if not vision_models:
                return ""

            vis_model = vision_models[0]
            logger.info(f"Attempting local Vision OCR fallback using {vis_model}...")

            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {
                "model": vis_model,
                "prompt": "Transcribe all visible text from this image accurately and completely. Return only the extracted text.",
                "images": [img_b64],
                "stream": False
            }
            res = requests.post(f"{config.OLLAMA_API_URL}/api/generate", json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama Vision OCR error: {e}")
        return ""

    @classmethod
    def extract_text_from_pil(
        cls, 
        image: Image.Image, 
        preprocess: bool = True, 
        lang: Optional[str] = None,
        preferred_engine: Optional[str] = None,
        return_warning: bool = False
    ) -> str:
        """Extract text from a PIL Image using high-confidence multi-pass OCR and quality ranking.
        
        Follows a small-number-of-passes workflow:
        1. Pass 1: Standard preprocessed image with Tesseract.
        2. Early exit if confidence >= 70% and words >= 8.
        3. Pass 2 (Retry only if Pass 1 is poor): Adaptive thresholding or deskewing.
        4. Optional RapidOCR fallback if selected or if Tesseract is missing.
        """
        ocr_lang = lang or getattr(config, "OCR_LANG", "eng")
        selected_engine = preferred_engine or getattr(config, "DEFAULT_OCR_ENGINE", "tesseract")

        # 1. If RapidOCR is explicitly selected
        if selected_engine == "rapidocr" and cls.is_rapidocr_available():
            rapid_res = cls.extract_text_via_rapidocr(image)
            if rapid_res.text:
                return rapid_res.text

        # 2. Check Tesseract availability
        if not cls.is_tesseract_available():
            # If RapidOCR is available as local alternative, use it
            if cls.is_rapidocr_available():
                rapid_res = cls.extract_text_via_rapidocr(image)
                if rapid_res.text:
                    return rapid_res.text

            # Vision LLM fallback
            vision_text = cls.extract_text_via_ollama_vision(image)
            if vision_text:
                return f"[Vision LLM Extracted Text]\n{vision_text}"

            if return_warning:
                return (
                    "⚠️ Tesseract OCR is not installed or configured on your system.\n\n"
                    "Install Tesseract OCR for Windows to 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'."
                )
            return ""

        # --- PASS 1: Standard Preprocessing with Page Segmentation Mode 3 ---
        prep_standard = cls.preprocess_image(image, variant="standard") if preprocess else image
        pass1_res = cls._run_tesseract_with_data(prep_standard, psm=3, lang=ocr_lang)

        # Early exit: high confidence extraction with sufficient words
        if pass1_res.confidence >= 70.0 and pass1_res.word_count >= 8:
            return pass1_res.text

        # --- PASS 2: Adaptive Threshold / Deskew (Only when Pass 1 is poor) ---
        candidates: List[OCRQualityResult] = [pass1_res]

        prep_thresh = cls.preprocess_image(image, variant="adaptive_thresh")
        pass2_res = cls._run_tesseract_with_data(prep_thresh, psm=6, lang=ocr_lang)
        candidates.append(pass2_res)

        # Early exit if pass2 solved it
        if pass2_res.confidence >= 75.0 and pass2_res.word_count >= 8:
            return pass2_res.text

        # Pass 3 (Only if text still very poor): RapidOCR if available
        if cls.is_rapidocr_available():
            rapid_res = cls.extract_text_via_rapidocr(image)
            candidates.append(rapid_res)

        # Pick best candidate by overall quality score
        best_candidate = max(candidates, key=lambda c: c.score)
        if best_candidate.text and len(best_candidate.text.strip()) > 3:
            return best_candidate.text.strip()

        # Final fallback: Ollama vision model
        vision_text = cls.extract_text_via_ollama_vision(image)
        if vision_text:
            return f"[Vision LLM Extracted Text]\n{vision_text}"

        return "No readable text detected in the image." if return_warning else ""

    @classmethod
    def extract_text_from_image(
        cls, 
        image_path: str, 
        preprocess: bool = True, 
        lang: Optional[str] = None,
        return_warning: bool = False
    ) -> str:
        """Extract text from an image file on disk."""
        try:
            if not os.path.exists(image_path):
                return "Error: Image file does not exist." if return_warning else ""
            with Image.open(image_path) as image:
                return cls.extract_text_from_pil(image, preprocess=preprocess, lang=lang, return_warning=return_warning)
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return f"Error during OCR execution: {str(e)}" if return_warning else ""

    @classmethod
    def extract_text_from_bytes(
        cls, 
        image_bytes, 
        preprocess: bool = True, 
        lang: Optional[str] = None,
        return_warning: bool = True
    ) -> str:
        """Extract text from raw image bytes (e.g. from st.file_uploader)."""
        try:
            image = Image.open(image_bytes)
            return cls.extract_text_from_pil(image, preprocess=preprocess, lang=lang, return_warning=return_warning)
        except Exception as e:
            logger.error(f"OCR extraction failed from bytes: {e}")
            return f"Error during OCR execution: {str(e)}" if return_warning else ""
