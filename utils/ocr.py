import os
import sys
import logging
from PIL import Image
import pytesseract

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Configure pytesseract binary path from environment or default Windows installation path
tesseract_cmd = os.environ.get("TESSERACT_CMD_PATH", r"C:/Program Files/Tesseract-OCR/tesseract.exe")
# Normalize path formatting
tesseract_cmd = os.path.normpath(tesseract_cmd)
if os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    logger.info(f"Tesseract OCR path set to: {tesseract_cmd}")
else:
    # Handle the case where backslash-t was parsed as a tab character in Windows strings
    repaired_path = tesseract_cmd.replace("\t", "\\t")
    if os.path.exists(repaired_path):
        pytesseract.pytesseract.tesseract_cmd = repaired_path
        logger.info(f"Tesseract OCR path set (repaired) to: {repaired_path}")
    else:
        logger.warning(f"Tesseract OCR binary not found at '{tesseract_cmd}'. OCR functions might fail until Tesseract is installed and path is configured.")

import os
import sys
import logging
import base64
import io
import requests
from typing import List
from PIL import Image
import pytesseract

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Configure pytesseract binary path from environment or default Windows installation path
tesseract_cmd = os.environ.get("TESSERACT_CMD_PATH", r"C:/Program Files/Tesseract-OCR/tesseract.exe")
tesseract_cmd = os.path.normpath(tesseract_cmd)
if os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    logger.info(f"Tesseract OCR path set to: {tesseract_cmd}")
else:
    repaired_path = tesseract_cmd.replace("\t", "\\t")
    if os.path.exists(repaired_path):
        pytesseract.pytesseract.tesseract_cmd = repaired_path
        logger.info(f"Tesseract OCR path set (repaired) to: {repaired_path}")
    else:
        logger.warning(f"Tesseract OCR binary not found at '{tesseract_cmd}'. Ensure Tesseract-OCR is installed.")


class OCRManager:
    @staticmethod
    def is_tesseract_available() -> bool:
        """Check if Tesseract binary is installed and responsive."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def preprocess_variants(image: Image.Image) -> List[Image.Image]:
        """Generate multiple image preprocessing variants to maximize OCR text extraction."""
        variants = []
        try:
            from PIL import ImageEnhance, ImageOps, ImageFilter
            
            # Ensure RGB
            if image.mode not in ("L", "RGB"):
                rgb_img = image.convert("RGB")
            else:
                rgb_img = image

            # Upscale small images to improve DPI for OCR
            width, height = rgb_img.size
            if width < 1000 or height < 1000:
                scale = max(2.0, 1200.0 / float(max(1, min(width, height))))
                new_w = min(int(width * scale), 3000)
                new_h = min(int(height * scale), 3000)
                rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Variant 1: Original / Upscaled
            variants.append(rgb_img)

            # Variant 2: Grayscale + High Contrast
            gray = ImageOps.grayscale(rgb_img)
            enhancer = ImageEnhance.Contrast(gray)
            high_contrast = enhancer.enhance(2.0)
            variants.append(high_contrast)

            # Variant 3: Sharpened High Contrast Binarized
            sharpened = high_contrast.filter(ImageFilter.SHARPEN)
            variants.append(sharpened)

        except Exception as e:
            logger.warning(f"Error preparing image preprocessing variants: {e}")
            variants = [image]

        return variants

    @staticmethod
    def extract_text_via_ollama_vision(image: Image.Image) -> str:
        """Fallback: attempt to extract text using a local Ollama vision model if available."""
        try:
            tags_res = requests.get(f"{config.OLLAMA_API_URL}/api/tags", timeout=3)
            if tags_res.status_code != 200:
                return ""
            models = [m['name'] for m in tags_res.json().get('models', [])]
            vision_models = [m for m in models if any(v in m.lower() for v in ['vision', 'llava', 'bakllava', 'moondream', 'minicpm'])]
            if not vision_models:
                return ""

            vis_model = vision_models[0]
            logger.info(f"Attempting Vision OCR using Ollama model: {vis_model}")

            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {
                "model": vis_model,
                "prompt": "Extract and transcribe all text present in this image accurately. Return only the extracted text.",
                "images": [img_b64],
                "stream": False
            }
            res = requests.post(f"{config.OLLAMA_API_URL}/api/generate", json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama Vision OCR fallback error: {e}")
        return ""

    @staticmethod
    def get_installation_instructions() -> str:
        return (
            "⚠️ Tesseract OCR is not installed or configured on your system.\n\n"
            "To extract text from images & scanned PDFs:\n"
            "1. Download Tesseract OCR installer for Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Install to 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n"
            "3. Set the executable path in the Settings page."
        )

    @classmethod
    def extract_text_from_pil(cls, image: Image.Image, preprocess: bool = True, return_warning: bool = False) -> str:
        """Extract text from a PIL Image using multi-variant preprocessing, multi-PSM OCR, and Vision fallback."""
        if not cls.is_tesseract_available():
            vision_text = cls.extract_text_via_ollama_vision(image)
            if vision_text:
                return f"[Vision LLM Extracted Text]\n{vision_text}"
            return cls.get_installation_instructions() if return_warning else ""

        psm_configs = ["--psm 3", "--psm 6", "--psm 11"]
        variants = cls.preprocess_variants(image) if preprocess else [image]

        for var_img in variants:
            for psm in psm_configs:
                try:
                    text = pytesseract.image_to_string(var_img, config=psm)
                    if text and len(text.strip()) > 3:
                        return text.strip()
                except Exception:
                    continue

        # If Tesseract produced empty text, attempt Ollama Vision fallback
        vision_text = cls.extract_text_via_ollama_vision(image)
        if vision_text:
            return f"[Vision LLM Extracted Text]\n{vision_text}"

        return "No readable text found in the image. Try uploading a clearer photo." if return_warning else ""


    @classmethod
    def extract_text_from_image(cls, image_path: str, preprocess: bool = True, return_warning: bool = False) -> str:
        """Extract text from an image file on disk."""
        try:
            if not os.path.exists(image_path):
                return "Error: Image file does not exist." if return_warning else ""
            image = Image.open(image_path)
            return cls.extract_text_from_pil(image, preprocess=preprocess, return_warning=return_warning)
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return f"Error during OCR execution: {str(e)}" if return_warning else ""

    @classmethod
    def extract_text_from_bytes(cls, image_bytes, preprocess: bool = True, return_warning: bool = True) -> str:
        """Extract text from raw image bytes (e.g. from st.file_uploader)."""
        try:
            image = Image.open(image_bytes)
            return cls.extract_text_from_pil(image, preprocess=preprocess, return_warning=return_warning)
        except Exception as e:
            logger.error(f"OCR extraction failed from bytes: {e}")
            return f"Error during OCR execution: {str(e)}" if return_warning else ""



