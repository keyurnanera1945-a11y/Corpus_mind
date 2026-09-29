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

class OCRManager:
    @staticmethod
    def extract_text_from_image(image_path: str) -> str:
        """Extract text from an image on disk."""
        try:
            if not os.path.exists(image_path):
                return "Error: Image file does not exist."
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"Error during OCR execution: {str(e)}\n\nMake sure Tesseract-OCR is installed on your system and the executable path is correctly set in Settings."

    @staticmethod
    def extract_text_from_bytes(image_bytes) -> str:
        """Extract text from raw image bytes (e.g. from st.file_uploader)."""
        try:
            image = Image.open(image_bytes)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed from bytes: {e}")
            return f"Error during OCR execution: {str(e)}\n\nMake sure Tesseract-OCR is installed on your system and the executable path is correctly set in Settings."

    @staticmethod
    def is_tesseract_available() -> bool:
        """Check if Tesseract binary is responsive."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
