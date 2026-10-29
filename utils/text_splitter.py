import re
import os
import sys
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class RecursiveTextSplitter:
    """Enhanced recursive text splitter that preserves paragraphs, headings, definitions, and page metadata."""

    def __init__(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or getattr(config, "CHUNK_SIZE", 500)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else getattr(config, "CHUNK_OVERLAP", 50)
        # Ensure overlap is strictly less than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 5)

    def split_text(self, text: str) -> List[str]:
        """Split raw text recursively into semantic chunks conforming strictly to chunk_size limits."""
        if not text or not text.strip():
            return []

        separators = [
            "\n# ", "\n## ", "\n### ", "\n#### ",
            "\n---", "\n___",
            "\n\n",
            "\n| ", "\n- ", "\n* ", "\n",
            ":\n", ". ", "? ", "! ", "; ", 
            " ", ""
        ]
        raw_chunks = self._recursive_split(text.strip(), separators)

        # Post-process: clean up whitespace and filter micro-noise
        cleaned = []
        for c in raw_chunks:
            s = c.strip()
            if len(s) > 5:
                # Ensure no chunk exceeds the configured limit
                if len(s) > self.chunk_size:
                    for i in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        sub = s[i:i + self.chunk_size].strip()
                        if len(sub) > 5:
                            cleaned.append(sub)
                else:
                    cleaned.append(s)
        return cleaned

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of semantic delimiters."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Force hard split conforming strictly to size and overlap
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunks.append(text[i:i + self.chunk_size])
            return chunks

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            if not split:
                continue

            # Determine piece to attach
            piece = (separator + split) if (current_chunk and separator not in (" ", "")) else split

            # If this individual piece exceeds chunk_size, process recursively
            if len(piece) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                sub_chunks = self._recursive_split(split, next_separators)
                chunks.extend(sub_chunks)
                continue

            test_chunk = (
                current_chunk + 
                ("\n" if current_chunk and separator in ("\n\n", "\n") else (" " if current_chunk and separator == " " else "")) + 
                split
            )

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Overlap calculation from prior chunk
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_chars = current_chunk[-self.chunk_overlap:]
                    # Try to break on whitespace in overlap
                    last_space = overlap_chars.find(" ")
                    if last_space != -1:
                        overlap_chars = overlap_chars[last_space + 1:]
                    current_chunk = (overlap_chars + " " + split).strip()
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def _extract_section_header(text: str) -> Optional[str]:
        """Extract nearest markdown header or title if present."""
        match = re.search(r'^(?:#{1,4}\s+|Chapter\s+\d+|Section\s+\d+[:\s]+)(.+)$', text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def split_pages(
        self, 
        pages: List[Dict[str, Any]], 
        doc_id: int, 
        user_id: int, 
        file_name: str, 
        file_type: str = "Document"
    ) -> List[Dict[str, Any]]:
        """Split a list of document pages into structured chunk objects with full metadata."""
        all_chunks = []
        global_chunk_idx = 0
        current_section = "General"

        for page_obj in pages:
            page_num = page_obj.get("page_number", 1)
            page_text = page_obj.get("text", "")
            if not page_text or not page_text.strip():
                continue

            # Detect section if present in page text
            detected_sec = self._extract_section_header(page_text)
            if detected_sec:
                current_section = detected_sec

            text_splits = self.split_text(page_text)
            for split_idx, split_text in enumerate(text_splits):
                # Update section if sub-chunk has header
                sub_sec = self._extract_section_header(split_text)
                if sub_sec:
                    current_section = sub_sec

                all_chunks.append({
                    "chunk_id": global_chunk_idx,
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "file_name": file_name,
                    "page_number": page_num,
                    "section": current_section,
                    "file_type": file_type,
                    "text": split_text
                })
                global_chunk_idx += 1

        return all_chunks
