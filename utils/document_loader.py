import os
import pypdf
import pdfplumber
import docx
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF using pypdf and pdfplumber as a fallback."""
        text = ""
        try:
            # Try pypdf first
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pypdf extraction failed, trying pdfplumber: {e}")
            text = ""
            
        # Fallback to pdfplumber if pypdf returns empty or errors
        if not text.strip():
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                logger.error(f"pdfplumber extraction failed: {e}")
                
        return text

    @staticmethod
    def load_docx(file_path: str) -> str:
        """Extract text from Word Document using docx package."""
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    text += " | ".join(row_text) + "\n"
        except Exception as e:
            logger.error(f"python-docx extraction failed: {e}")
        return text

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Extract text from TXT with encoding error handling."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        logger.error(f"Failed to decode text file: {file_path}")
        return ""

    @classmethod
    def load_document(cls, file_path: str) -> str:
        """Central router for document text extraction based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return cls.load_pdf(file_path)
        elif ext == ".docx":
            return cls.load_docx(file_path)
        elif ext == ".txt":
            return cls.load_txt(file_path)
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return ""
        path = Path(file_path)
        if path.exists():
            return cls.load_pdf(str(path))
        return ""
