import os
import json
import logging
import csv
from pathlib import Path
import pypdf
import pdfplumber
import docx
from PIL import Image

import config
from utils.ocr import OCRManager

logger = logging.getLogger(__name__)

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF using pypdf, pdfplumber, and OCR fallback for scanned pages."""
        text = ""
        try:
            # 1. Try digital text extraction with pypdf
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"--- Page {page_num} ---\n{page_text.strip()}\n\n"
        except Exception as e:
            logger.warning(f"pypdf extraction failed, trying pdfplumber: {e}")
            text = ""
            
        # 2. Fallback to pdfplumber if pypdf returns empty or very little text
        if not text.strip() or len(text.strip()) < 30:
            text = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"--- Page {page_num} ---\n{page_text.strip()}\n\n"
            except Exception as e:
                logger.error(f"pdfplumber digital text extraction failed: {e}")
                
        # 3. OCR Fallback for scanned/image PDFs
        if not text.strip() or len(text.strip()) < 30:
            logger.info(f"Digital text empty or minimal for {file_path}. Initiating OCR scanning for PDF pages...")
            ocr_text_acc = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            # Render page to PIL image
                            im = page.to_image(resolution=200)
                            pil_img = im.original
                            page_ocr = OCRManager.extract_text_from_pil(pil_img, preprocess=True, return_warning=False)
                            if page_ocr and not page_ocr.startswith("⚠️") and not page_ocr.startswith("Error"):
                                ocr_text_acc += f"--- Page {page_num} (OCR Scanned) ---\n{page_ocr}\n\n"
                        except Exception as page_err:
                            logger.error(f"Failed OCR on PDF page {page_num}: {page_err}")
                if ocr_text_acc.strip():
                    text = ocr_text_acc
            except Exception as ocr_err:
                logger.error(f"Scanned PDF OCR failed for {file_path}: {ocr_err}")

        return text.strip()

    @staticmethod
    def load_image(file_path: str) -> str:
        """Extract text from an image file using Tesseract OCR."""
        file_name = Path(file_path).name
        ocr_text = OCRManager.extract_text_from_image(file_path, preprocess=True, return_warning=False)
        if ocr_text and not ocr_text.startswith("Error") and not ocr_text.startswith("⚠️"):
            return f"--- OCR Extracted Text from Image: {file_name} ---\n{ocr_text}"
        else:
            logger.warning(f"Failed or empty OCR result for image {file_name}")
            return ""


    @staticmethod
    def load_docx(file_path: str) -> str:
        """Extract text from Word Document using docx package."""
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        text += " | ".join(row_text) + "\n"
        except Exception as e:
            logger.error(f"python-docx extraction failed: {e}")
        return text.strip()

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Extract text from TXT with encoding error handling."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        logger.error(f"Failed to decode text file: {file_path}")
        return ""

    @staticmethod
    def load_md(file_path: str) -> str:
        """Extract text from Markdown file."""
        return DocumentLoader.load_txt(file_path)

    @staticmethod
    def load_csv(file_path: str) -> str:
        """Extract tabular text from CSV file."""
        lines = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    lines.append(f"Columns: {', '.join(headers)}")
                for i, row in enumerate(reader, 1):
                    if row:
                        row_str = " | ".join([cell.strip() for cell in row])
                        lines.append(f"Row {i}: {row_str}")
        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
        return "\n".join(lines)

    @staticmethod
    def load_json(file_path: str) -> str:
        """Extract text from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return DocumentLoader.load_txt(file_path)

    @classmethod
    def load_document(cls, file_path: str) -> str:
        """Central router for document & image text extraction based on file extension."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return ""

        ext = path.suffix.lower()
        if ext == ".pdf":
            return cls.load_pdf(file_path)
        elif ext == ".docx":
            return cls.load_docx(file_path)
        elif ext == ".txt":
            return cls.load_txt(file_path)
        elif ext == ".md":
            return cls.load_md(file_path)
        elif ext == ".csv":
            return cls.load_csv(file_path)
        elif ext == ".json":
            return cls.load_json(file_path)
        elif ext in config.SUPPORTED_IMAGE_EXTENSIONS:
            return cls.load_image(file_path)
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return ""

