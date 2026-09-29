import re
from typing import List

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text recursively into chunks based on semantic separators."""
        if not text:
            return []
            
        separators = ["\n\n", "\n", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators until pieces fit chunk_size."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Force chunking by size
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])
            return chunks

        separator = separators[0]
        next_separators = separators[1:]
        
        # Split the text
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
            
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If the single split is larger than chunk_size, process it recursively
            if len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(self._recursive_split(split, next_separators))
                continue
                
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle overlap: backtrack by scanning split parts or slicing
                current_chunk = split
                if self.chunk_overlap > 0:
                    # Keep overlap from last chunk if possible
                    overlap_source = chunks[-1] if chunks else ""
                    if len(overlap_source) > self.chunk_overlap:
                        current_chunk = overlap_source[-self.chunk_overlap:] + (separator if separator else "") + split
                        
        if current_chunk:
            chunks.append(current_chunk)
            
        # Clean chunks
        return [c.strip() for c in chunks if c.strip()]
