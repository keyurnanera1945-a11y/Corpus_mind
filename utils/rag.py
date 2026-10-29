import os
import sys
import pickle
import math
import re
import time
import hashlib
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.embeddings import EmbeddingManager, EmbeddingDimensionMismatchError
from utils.database import db

logger = logging.getLogger(__name__)

# Attempt to import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library detected and available.")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not found. Native optimized NumPy vector store will be used.")


class BM25Retriever:
    """High-performance Okapi BM25 retriever with inverted index postings."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_texts = corpus_texts
        self.N = len(corpus_texts)
        self.doc_tokens = [self._tokenize(t) for t in corpus_texts]
        self.doc_lens = np.array([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) > 0 else 1.0

        # Build Inverted Index: token -> list of (doc_idx, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, tf))

        # Precompute IDF for all terms in vocabulary
        self.idf: Dict[str, float] = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def search(self, query: str, top_k: int = 20) -> List[Tuple[float, int]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or self.N == 0:
            return []

        scores = np.zeros(self.N, dtype=np.float32)

        # Inverted index traversal: only touch documents containing query words
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            token_idf = self.idf[token]
            postings = self.inverted_index[token]

            for doc_idx, tf in postings:
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                num = tf * (self.k1 + 1.0)
                scores[doc_idx] += token_idf * (num / denom)

        # Retrieve top scores
        nonzero_indices = np.where(scores > 0)[0]
        if len(nonzero_indices) == 0:
            return []

        if len(nonzero_indices) > top_k:
            part_idx = np.argpartition(scores[nonzero_indices], -top_k)[-top_k:]
            sorted_candidates = nonzero_indices[part_idx[np.argsort(scores[nonzero_indices[part_idx]])[::-1]]]
        else:
            sorted_candidates = nonzero_indices[np.argsort(scores[nonzero_indices])[::-1]]

        results = [(float(scores[idx]), int(idx)) for idx in sorted_candidates if scores[idx] > 0]
        return results


class NumpyVectorStore:
    """Pre-normalized 2D matrix vector store with vectorized cosine search."""

    def __init__(self):
        self.matrix: Optional[np.ndarray] = None  # shape: (N, D), pre-normalized float32
        self.metadata: List[Dict[str, Any]] = []

    def set_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.matrix = embeddings / norms
        self.metadata = metadata

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[float, int]]:
        if self.matrix is None or len(self.matrix) == 0:
            return []

        q = np.asarray(query_emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_unit = q / q_norm

        # Cosine similarity via single matrix dot product
        similarities = np.dot(self.matrix, q_unit)
        top_k = min(k, len(similarities))

        if len(similarities) > top_k:
            part_idx = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = part_idx[np.argsort(similarities[part_idx])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        return [(float(similarities[idx]), int(idx)) for idx in top_indices]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.matrix, self.metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.matrix, self.metadata = pickle.load(f)


class CachedRetrievalState:
    """In-memory cache capsule for a user's index."""

    def __init__(
        self,
        user_id: int,
        version: str,
        embedding_model: str,
        dimension: int,
        metadata: List[Dict[str, Any]],
        bm25: Optional[BM25Retriever],
        faiss_index: Any,
        numpy_store: Optional[NumpyVectorStore],
        file_mtime: float
    ):
        self.user_id = user_id
        self.version = version
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.metadata = metadata
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.numpy_store = numpy_store
        self.file_mtime = file_mtime
        self.last_accessed = time.time()


class RAGManager:
    """Manages multi-tenant RAG indexing, in-memory caching, and hybrid retrieval."""

    # In-memory retrieval state cache keyed strictly by user_id
    _user_cache: Dict[int, CachedRetrievalState] = {}

    @classmethod
    def get_user_index_path(cls, user_id: int) -> str:
        return str(config.get_user_vector_store_dir(user_id) / "vector_index.bin")

    @classmethod
    def get_user_metadata_path(cls, user_id: int) -> str:
        return str(config.get_user_vector_store_dir(user_id) / "vector_metadata.pkl")

    @classmethod
    def invalidate_cache(cls, user_id: Optional[int] = None):
        """Invalidate in-memory cached state for a specific user or all users."""
        if user_id is not None:
            cls._user_cache.pop(user_id, None)
            logger.info(f"Invalidated in-memory retrieval cache for user_id={user_id}.")
        else:
            cls._user_cache.clear()
            logger.info("Cleared all in-memory retrieval caches.")

    @classmethod
    def load_user_retrieval_state(cls, user_id: int) -> Optional[CachedRetrievalState]:
        """Load or retrieve the in-memory cached state for a specific user.
        
        Loads once per index version. Detects updates and reloads safely.
        """
        meta_path = cls.get_user_metadata_path(user_id)
        index_path = cls.get_user_index_path(user_id)

        if not os.path.exists(meta_path) or not os.path.exists(index_path):
            cls._user_cache.pop(user_id, None)
            return None

        current_mtime = max(os.path.getmtime(meta_path), os.path.getmtime(index_path))

        # Check in-memory cache validity
        cached = cls._user_cache.get(user_id)
        if cached is not None and cached.file_mtime >= current_mtime:
            cached.last_accessed = time.time()
            return cached

        # Load from disk
        logger.info(f"Loading retrieval index state from disk into memory for user_id={user_id}...")
        try:
            with open(meta_path, 'rb') as f:
                meta_payload = pickle.load(f)

            if isinstance(meta_payload, dict):
                metadata = meta_payload.get("chunks", [])
                stored_model = meta_payload.get("embedding_model", config.DEFAULT_EMBEDDING_MODEL)
                dimension = meta_payload.get("dimension", 384)
                version = meta_payload.get("version", str(current_mtime))
            else:
                # Legacy metadata list format
                metadata = meta_payload
                stored_model = getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                dimension = EmbeddingManager.get_model_dimension(stored_model)
                version = str(current_mtime)

            if not metadata:
                return None

            # Validate compatibility with currently configured embedding model
            EmbeddingManager.validate_compatibility(stored_model, dimension)

            # Build BM25 retriever once
            corpus_texts = [m['text'] for m in metadata]
            bm25 = BM25Retriever(corpus_texts)

            # Load Dense vector index
            faiss_index = None
            numpy_store = None

            if FAISS_AVAILABLE:
                try:
                    faiss_index = faiss.read_index(index_path)
                except Exception as fe:
                    logger.warning(f"Failed to read FAISS index for user {user_id}, falling back to numpy: {fe}")
                    faiss_index = None

            if faiss_index is None:
                numpy_store = NumpyVectorStore()
                numpy_store.load(index_path)

            cached_state = CachedRetrievalState(
                user_id=user_id,
                version=version,
                embedding_model=stored_model,
                dimension=dimension,
                metadata=metadata,
                bm25=bm25,
                faiss_index=faiss_index,
                numpy_store=numpy_store,
                file_mtime=current_mtime
            )

            cls._user_cache[user_id] = cached_state
            return cached_state

        except Exception as e:
            logger.error(f"Error loading retrieval state for user_id={user_id}: {e}", exc_info=True)
            return None

    @classmethod
    def index_user_chunks(cls, user_id: int, chunks: List[Dict[str, Any]]) -> bool:
        """Create/rebuild vector and BM25 index strictly for user_id with batching and validation."""
        if not chunks:
            logger.warning(f"No chunks provided to index for user {user_id}.")
            return False

        try:
            texts = [c['text'] for c in chunks]
            model_name = getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            expected_dim = EmbeddingManager.get_model_dimension(model_name)

            # Generate dense embeddings in batches with strict validation
            embeddings = EmbeddingManager.get_embeddings(texts, model_name=model_name, batch_size=32)
            if embeddings.shape[0] != len(chunks) or embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Generated embeddings shape {embeddings.shape} does not match ({len(chunks)}, {expected_dim})."
                )

            # Assign sequential chunk_id and enforce user_id in every metadata record
            metadata_records = []
            for idx, c in enumerate(chunks):
                record = {
                    "chunk_id": idx,
                    "doc_id": c.get("doc_id"),
                    "user_id": user_id,
                    "file_name": c.get("file_name", "unknown"),
                    "page_number": c.get("page_number", 1),
                    "section": c.get("section", "General"),
                    "file_type": c.get("file_type", "Document"),
                    "text": c["text"]
                }
                metadata_records.append(record)

            index_path = cls.get_user_index_path(user_id)
            meta_path = cls.get_user_metadata_path(user_id)

            if FAISS_AVAILABLE:
                norm_embeddings = embeddings.copy()
                faiss.normalize_L2(norm_embeddings)
                index = faiss.IndexFlatIP(expected_dim)
                index.add(norm_embeddings)
                faiss.write_index(index, index_path)
            else:
                store = NumpyVectorStore()
                store.set_vectors(embeddings, metadata_records)
                store.save(index_path)

            # Save versioned metadata bundle
            version_id = hashlib.sha256(f"{user_id}_{time.time()}_{len(chunks)}".encode()).hexdigest()[:16]
            meta_payload = {
                "version": version_id,
                "user_id": user_id,
                "embedding_model": model_name,
                "dimension": expected_dim,
                "chunks": metadata_records
            }
            with open(meta_path, "wb") as f:
                pickle.dump(meta_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Invalidate in-memory cache to force immediate reload
            cls.invalidate_cache(user_id)
            logger.info(f"User {user_id}: successfully indexed {len(chunks)} chunks with model '{model_name}'.")
            return True

        except Exception as e:
            logger.error(f"Failed to index documents for user {user_id}: {e}", exc_info=True)
            return False

    @classmethod
    def rebuild_user_index(cls, user_id: int) -> Tuple[bool, int, int]:
        """Rebuild vector & BM25 index exclusively for a specific authenticated user."""
        from utils.document_loader import DocumentLoader
        from utils.text_splitter import RecursiveTextSplitter

        user_docs = db.get_documents_by_user(user_id)
        if not user_docs:
            for path in [cls.get_user_index_path(user_id), cls.get_user_metadata_path(user_id)]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            cls.invalidate_cache(user_id)
            return True, 0, 0

        splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        all_chunks = []

        for doc in user_docs:
            f_path = doc['file_path']
            if not f_path or not os.path.exists(f_path):
                continue

            # Load page-structured data
            pages = DocumentLoader.load_document_pages(f_path)
            ext = os.path.splitext(doc['file_name'])[1].lower()
            file_type = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"

            chunks = splitter.split_pages(
                pages=pages,
                doc_id=doc['id'],
                user_id=user_id,
                file_name=doc['file_name'],
                file_type=file_type
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            cls.invalidate_cache(user_id)
            return True, len(user_docs), 0

        success = cls.index_user_chunks(user_id, all_chunks)
        return success, len(user_docs), len(all_chunks)

    @classmethod
    def rebuild_index_from_db(cls, user_id: Optional[int] = None) -> Tuple[bool, int, int]:
        """Rebuild index with user-scoping.
        
        If user_id is provided, rebuilds that user's store.
        If user_id is None (admin context), rebuilds each user's store independently without mixing.
        """
        if user_id is not None:
            return cls.rebuild_user_index(user_id)

        # Admin global rebuild: iterate each user and build isolated user index
        users = db.get_all_users()
        total_docs = 0
        total_chunks = 0
        overall_success = True

        for u in users:
            s, d_c, c_c = cls.rebuild_user_index(u['id'])
            if not s:
                overall_success = False
            total_docs += d_c
            total_chunks += c_c

        return overall_success, total_docs, total_chunks

    @staticmethod
    def _deduplicate_chunks(chunks: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
        """Deduplicate passages with high string overlap or identical content."""
        deduped = []
        seen_texts = set()

        for chunk in chunks:
            # Normalized fingerprint: lowercase alphanumeric only
            fingerprint = re.sub(r'\W+', '', chunk['text'].lower())
            if not fingerprint or fingerprint in seen_texts:
                continue

            # Check if this fingerprint is a high-overlap substring of an existing chunk
            is_redundant = False
            for seen in seen_texts:
                if len(fingerprint) > 40 and len(seen) > 40:
                    if fingerprint in seen or seen in fingerprint:
                        is_redundant = True
                        break

            if not is_redundant:
                seen_texts.add(fingerprint)
                deduped.append(chunk)

            if len(deduped) >= max_items:
                break

        return deduped

    @classmethod
    def retrieve_context(
        cls, 
        query: str, 
        user_id: int,
        top_k: int = 4, 
        similarity_threshold: float = 0.15,
        search_mode: str = "hybrid",
        doc_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks strictly for the authenticated user_id using cached state."""
        if user_id is None:
            logger.error("Security violation: retrieve_context called without authenticated user_id.")
            return []

        if not query or not query.strip():
            return []

        state = cls.load_user_retrieval_state(user_id)
        if not state or not state.metadata:
            logger.info(f"No indexed documents found in state for user_id={user_id}.")
            return []

        doc_filter_set = set(doc_filter) if doc_filter else None
        candidate_k = max(top_k * 4, 20)

        # 1. Sparse BM25 Search (using cached BM25 index)
        bm25_results = state.bm25.search(query, top_k=candidate_k) if state.bm25 else []

        # 2. Dense Vector Search (using cached FAISS index or NumPy vector store)
        dense_results: List[Tuple[float, int]] = []
        try:
            query_emb = EmbeddingManager.get_embedding(query, model_name=state.embedding_model)

            if state.faiss_index is not None:
                q_emb = query_emb.reshape(1, -1).copy()
                faiss.normalize_L2(q_emb)
                scores, indices = state.faiss_index.search(q_emb, candidate_k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx < len(state.metadata):
                        dense_results.append((float(score), int(idx)))
            elif state.numpy_store is not None:
                dense_results = state.numpy_store.search(query_emb, candidate_k)

        except Exception as dense_err:
            logger.error(f"Dense vector search failed: {dense_err}")

        # 3. Hybrid RRF / Search Mode Selection
        raw_candidates = []

        if search_mode == "bm25":
            for score, idx in bm25_results:
                item = state.metadata[idx]
                if doc_filter_set and item['file_name'] not in doc_filter_set:
                    continue
                res = item.copy()
                res['score'] = score
                res['search_mode'] = 'BM25 Sparse'
                raw_candidates.append(res)

        elif search_mode == "dense":
            for score, idx in dense_results:
                if score < similarity_threshold:
                    continue
                item = state.metadata[idx]
                if doc_filter_set and item['file_name'] not in doc_filter_set:
                    continue
                res = item.copy()
                res['score'] = score
                res['search_mode'] = 'Dense Vector'
                raw_candidates.append(res)

        else:
            # Default: Hybrid Search with Reciprocal Rank Fusion (RRF)
            rrf_k = getattr(config, 'RRF_K', 60)
            rrf_scores: Dict[int, float] = {}
            dense_map = {idx: s for s, idx in dense_results}
            bm25_map = {idx: s for s, idx in bm25_results}

            for rank, (score, idx) in enumerate(dense_results, start=1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

            for rank, (score, idx) in enumerate(bm25_results, start=1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            for idx, rrf_score in sorted_rrf:
                item = state.metadata[idx]
                if doc_filter_set and item['file_name'] not in doc_filter_set:
                    continue
                res = item.copy()
                res['score'] = rrf_score
                res['dense_score'] = dense_map.get(idx, 0.0)
                res['bm25_score'] = bm25_map.get(idx, 0.0)
                res['search_mode'] = 'Hybrid RRF'
                raw_candidates.append(res)

        # 4. Deduplicate redundant passages and return top_k
        return cls._deduplicate_chunks(raw_candidates, max_items=top_k)
