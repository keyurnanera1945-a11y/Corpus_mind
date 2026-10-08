import re
from typing import List

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text recursively into chunks based on semantic separators."""
        if not text or not text.strip():
            return []
            
        separators = [
            "\n# ", "\n## ", "\n### ", "\n---", 
            "\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""
        ]
        raw_chunks = self._recursive_split(text.strip(), separators)
        
        # Post-process: clean up whitespace
        cleaned = [c.strip() for c in raw_chunks if c and len(c.strip()) > 5]
        return cleaned

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators until pieces fit chunk_size."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Force chunking by size
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunks.append(text[i:i + self.chunk_size])
            return chunks

        separator = separators[0]
        next_separators = separators[1:]
        
        # Split the text by current separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
            
        chunks = []
        current_chunk = ""
        
        for i, split in enumerate(splits):
            if not split:
                continue

            # Re-attach separator context except for whitespace splitting
            piece = (separator + split) if (current_chunk and separator not in (" ", "")) else split

            # If the single split piece is larger than chunk_size, process it recursively with finer separators
            if len(piece) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                sub_chunks = self._recursive_split(split, next_separators)
                chunks.extend(sub_chunks)
                continue
                
            test_chunk = current_chunk + ("\n" if current_chunk and separator in ("\n\n", "\n") else (" " if current_chunk and separator == " " else "")) + split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Apply overlap from previous chunk end
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + split
                else:
                    current_chunk = split
                        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

