import logging
import os
import sys
from typing import List, Optional, Dict, Any
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Known model dimensions for fast sanity validation
KNOWN_MODEL_DIMENSIONS: Dict[str, int] = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L6-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-mpnet-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class EmbeddingModelError(Exception):
    """Raised when the embedding model cannot be loaded or is unavailable."""
    pass


class EmbeddingDimensionMismatchError(Exception):
    """Raised when embedding dimensions do not match the expected vector index dimension."""
    pass


class InvalidEmbeddingError(Exception):
    """Raised when embedding output is empty, malformed, non-finite, or invalid."""
    pass


def _load_model_cached_streamlit(model_name: str):
    """Internal function wrapped with Streamlit's cache_resource if Streamlit is running."""
    try:
        import streamlit as st
        # Only use st.cache_resource if inside an active Streamlit execution context
        @st.cache_resource(show_spinner=False)
        def _cached_loader(name: str):
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(name)
        return _cached_loader(model_name)
    except Exception:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)


class EmbeddingManager:
    """Manages sentence-transformer embeddings with strict validation and caching."""

    # In-process singleton cache for non-Streamlit contexts (tests, background tasks, scripts)
    _in_memory_cache: Dict[str, Any] = {}

    @classmethod
    def get_model_name(cls, model_name: Optional[str] = None) -> str:
        """Resolve model name from argument, config, or fallback."""
        return model_name or getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @classmethod
    def load_model(cls, model_name: Optional[str] = None):
        """Load and cache the SentenceTransformer model without repeated disk/download operations.
        
        Never falls back to random/dummy vectors. Raises EmbeddingModelError on failure.
        """
        resolved_name = cls.get_model_name(model_name)

        # Check in-memory cache first
        if resolved_name in cls._in_memory_cache:
            return cls._in_memory_cache[resolved_name]

        try:
            logger.info(f"Loading embedding model: {resolved_name}")
            model = _load_model_cached_streamlit(resolved_name)
            if model is None:
                raise EmbeddingModelError(
                    f"Embedding model '{resolved_name}' failed to load (returned None). "
                    "Please verify PyTorch and sentence-transformers installation."
                )
            cls._in_memory_cache[resolved_name] = model
            return model
        except Exception as e:
            logger.error(f"Critical failure loading embedding model '{resolved_name}': {e}", exc_info=True)
            raise EmbeddingModelError(
                f"Embedding model '{resolved_name}' could not be loaded: {str(e)}. "
                "Ensure local sentence-transformers model files or internet connectivity are available."
            ) from e

    @classmethod
    def get_model_dimension(cls, model_name: Optional[str] = None) -> int:
        """Get the embedding vector dimension for a model."""
        resolved_name = cls.get_model_name(model_name)
        if resolved_name in KNOWN_MODEL_DIMENSIONS:
            return KNOWN_MODEL_DIMENSIONS[resolved_name]

        model = cls.load_model(resolved_name)
        dim = model.get_sentence_embedding_dimension()
        if dim:
            KNOWN_MODEL_DIMENSIONS[resolved_name] = dim
            return dim
        return 384

    @classmethod
    def get_embedding(cls, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """Get embedding vector for a single text with strict validation.
        
        Never returns random mock vectors.
        """
        if text is None or not isinstance(text, str):
            raise InvalidEmbeddingError(f"Input text must be a valid string, got {type(text)}.")

        clean_text = text.strip()
        if not clean_text:
            raise InvalidEmbeddingError("Cannot generate embedding for empty or whitespace-only text.")

        model = cls.load_model(model_name)
        resolved_name = cls.get_model_name(model_name)
        expected_dim = cls.get_model_dimension(resolved_name)

        try:
            embedding = model.encode(clean_text, convert_to_numpy=True)
        except Exception as e:
            raise EmbeddingModelError(f"Model inference failed for embedding generation: {e}") from e

        embedding = np.asarray(embedding, dtype=np.float32)

        # Validate vector shape & dimension
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.flatten()

        if embedding.ndim != 1:
            raise InvalidEmbeddingError(
                f"Expected 1D embedding array for single text, got shape {embedding.shape}."
            )

        if embedding.shape[0] != expected_dim:
            raise EmbeddingDimensionMismatchError(
                f"Generated embedding dimension {embedding.shape[0]} does not match expected {expected_dim} "
                f"for model '{resolved_name}'."
            )

        # Validate numbers are finite (no NaNs or Infs)
        if not np.all(np.isfinite(embedding)):
            raise InvalidEmbeddingError("Generated embedding contains NaN or Infinite values.")

        return embedding

    @classmethod
    def get_embeddings(
        cls, 
        texts: List[str], 
        model_name: Optional[str] = None, 
        batch_size: int = 32
    ) -> np.ndarray:
        """Get embedding vectors for a list of texts in batches with validation.
        
        Never returns random mock vectors.
        """
        resolved_name = cls.get_model_name(model_name)
        expected_dim = cls.get_model_dimension(resolved_name)

        if not texts:
            return np.empty((0, expected_dim), dtype=np.float32)

        model = cls.load_model(resolved_name)

        # Sanitize texts (replace empty texts with a single placeholder token so batch indexing stays 1:1)
        sanitized_texts = [t.strip() if (t and isinstance(t, str) and t.strip()) else "empty" for t in texts]

        try:
            embeddings = model.encode(
                sanitized_texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            raise EmbeddingModelError(f"Batch embedding generation failed: {e}") from e

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Validate dimensions
        if embeddings.shape != (len(texts), expected_dim):
            raise EmbeddingDimensionMismatchError(
                f"Batch embedding output shape {embeddings.shape} does not match expected "
                f"({len(texts)}, {expected_dim}) for model '{resolved_name}'."
            )

        if not np.all(np.isfinite(embeddings)):
            raise InvalidEmbeddingError("Batch embedding contains NaN or Infinite values.")

        return embeddings

    @classmethod
    def validate_compatibility(
        cls, 
        index_model_name: str, 
        index_dimension: int, 
        target_model_name: Optional[str] = None
    ) -> bool:
        """Verify that a target embedding model matches the model and dimension used to build an index.
        
        Raises EmbeddingDimensionMismatchError if incompatible.
        """
        current_model = cls.get_model_name(target_model_name)
        current_dim = cls.get_model_dimension(current_model)

        if index_dimension != current_dim:
            raise EmbeddingDimensionMismatchError(
                f"Index vector dimension ({index_dimension}) from model '{index_model_name}' "
                f"is incompatible with current model '{current_model}' (dimension {current_dim}). "
                "The index must be rebuilt or switched to the matching model."
            )

        if index_model_name and index_model_name != current_model:
            logger.warning(
                f"Embedding model name differs: Index built with '{index_model_name}', "
                f"querying with '{current_model}'. Dimension matches ({current_dim}), but embeddings may drift."
            )

        return True
