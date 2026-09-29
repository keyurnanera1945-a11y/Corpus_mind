import streamlit as st
import numpy as np
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    @staticmethod
    @st.cache_resource
    def load_model():
        """Load and cache the SentenceTransformer model using Streamlit's cache_resource."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer model: {config.DEFAULT_EMBEDDING_MODEL}")
            # This downloads or loads from cache (~420MB)
            model = SentenceTransformer(config.DEFAULT_EMBEDDING_MODEL)
            return model
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            return None

    @classmethod
    def get_embedding(cls, text: str) -> np.ndarray:
        """Get embedding vector for a single text."""
        model = cls.load_model()
        if model is None:
            # Fallback to dummy random embedding vector of length 768 if library load failed
            logger.warning("Using mock embedding because sentence-transformers is not available.")
            return np.random.rand(768).astype(np.float32)
        
        # sentence-transformers returns ndarray
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding

    @classmethod
    def get_embeddings(cls, texts: list) -> np.ndarray:
        """Get embedding vectors for a list of texts."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
            
        model = cls.load_model()
        if model is None:
            logger.warning("Using mock embeddings because sentence-transformers is not available.")
            return np.random.rand(len(texts), 768).astype(np.float32)
            
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
