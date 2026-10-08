import os
import sys
import pickle
import math
import re
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.embeddings import EmbeddingManager
from utils.database import db

logger = logging.getLogger(__name__)

# Attempt to import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library detected and imported successfully.")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not found. Falling back to native NumPy vector store.")


class BM25Retriever:
    """Native Okapi BM25 implementation for sparse keyword retrieval."""
    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_texts = corpus_texts
        self.doc_tokens = [self._tokenize(t) for t in corpus_texts]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.N = len(corpus_texts)
        self.idf = self._calc_idf()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _calc_idf(self) -> Dict[str, float]:
        df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] = df.get(t, 0) + 1
        idf: Dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
        return idf

    def search(self, query: str, top_k: int = 15) -> List[Tuple[float, int]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.corpus_texts:
            return []

        scores = np.zeros(self.N, dtype=np.float32)
        for t in query_tokens:
            if t not in self.idf:
                continue
            t_idf = self.idf[t]
            for doc_idx, tokens in enumerate(self.doc_tokens):
                term_freq = tokens.count(t)
                if term_freq == 0:
                    continue
                denom = term_freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avgdl))
                num = term_freq * (self.k1 + 1.0)
                scores[doc_idx] += t_idf * (num / denom)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((score, int(idx)))
        return results


class NumpyVectorStore:
    """A native NumPy-based cosine-similarity vector store."""
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        for emb in embeddings:
            self.embeddings.append(emb)
        self.metadata.extend(metadata)

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[float, int]]:
        if not self.embeddings:
            return []
            
        embeddings_matrix = np.array(self.embeddings, dtype=np.float32) # shape: (N, D)
        query_vector = np.array(query_emb, dtype=np.float32) # shape: (D,)
        
        dot_products = np.dot(embeddings_matrix, query_vector)
        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        matrix_norms[matrix_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10
            
        similarities = dot_products / (matrix_norms * query_norm)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            results.append((score, int(idx)))
            
        return results

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.embeddings, self.metadata), f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.embeddings, self.metadata = pickle.load(f)


class RAGManager:
    @staticmethod
    def get_index_path() -> str:
        return str(config.VECTOR_STORE_DIR / "vector_index.bin")

    @staticmethod
    def get_metadata_path() -> str:
        return str(config.VECTOR_STORE_DIR / "vector_metadata.pkl")

    @classmethod
    def index_documents(cls, chunks: List[Dict[str, Any]]) -> bool:
        """Create/rebuild vector store index from chunks.
        
        chunks is a list of dicts: {'text': str, 'file_name': str, 'doc_id': int, 'file_type': str}
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return False

        try:
            texts = [c['text'] for c in chunks]
            metadata = []
            for idx, c in enumerate(chunks):
                metadata.append({
                    'chunk_id': idx,
                    'text': c['text'],
                    'file_name': c['file_name'],
                    'doc_id': c['doc_id'],
                    'file_type': c.get('file_type', 'doc')
                })
            
            # Generate dense embeddings
            embeddings = EmbeddingManager.get_embeddings(texts)
            embeddings = np.array(embeddings, dtype=np.float32)

            if FAISS_AVAILABLE:
                dimension = embeddings.shape[1]
                faiss.normalize_L2(embeddings)
                
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                
                # Save FAISS index
                faiss.write_index(index, cls.get_index_path())
                with open(cls.get_metadata_path(), 'wb') as f:
                    pickle.dump(metadata, f)
            else:
                store = NumpyVectorStore()
                store.add(embeddings, metadata)
                store.save(cls.get_index_path())
                with open(cls.get_metadata_path(), 'wb') as f:
                    pickle.dump(metadata, f)

            logger.info(f"Indexed {len(chunks)} chunks successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False

    @classmethod
    def retrieve_context(
        cls, 
        query: str, 
        top_k: int = 4, 
        similarity_threshold: float = 0.15,
        search_mode: str = "hybrid",
        doc_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks for a given query using Dense, BM25, or Hybrid RRF.
        
        Optional doc_filter restricts search results strictly to the provided file names.
        """
        try:
            meta_path = cls.get_metadata_path()
            if not os.path.exists(meta_path):
                logger.warning("Metadata index file not found.")
                return []
                
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)

            if not metadata:
                return []

            doc_filter_set = set(doc_filter) if doc_filter else None
            corpus_texts = [m['text'] for m in metadata]
            candidate_k = max(top_k * 4, 20)

            # 1. Sparse BM25 Search
            bm25_retriever = BM25Retriever(corpus_texts)
            bm25_results = bm25_retriever.search(query, top_k=candidate_k) # list of (score, idx)

            # 2. Dense Vector Search
            query_emb = EmbeddingManager.get_embedding(query)
            query_emb = np.array(query_emb, dtype=np.float32)

            dense_results: List[Tuple[float, int]] = []
            
            if FAISS_AVAILABLE:
                index_path = cls.get_index_path()
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)
                    q_emb = query_emb.reshape(1, -1)
                    faiss.normalize_L2(q_emb)
                    scores, indices = index.search(q_emb, candidate_k)
                    for score, idx in zip(scores[0], indices[0]):
                        if idx != -1 and idx < len(metadata):
                            dense_results.append((float(score), int(idx)))
            else:
                store_path = cls.get_index_path()
                if os.path.exists(store_path):
                    store = NumpyVectorStore()
                    store.load(store_path)
                    dense_results = store.search(query_emb, candidate_k)

            # 3. Mode Routing & Ranking with Doc Filtering
            if search_mode == "bm25":
                results = []
                for score, idx in bm25_results:
                    item = metadata[idx]
                    if doc_filter_set and item['file_name'] not in doc_filter_set:
                        continue
                    res_item = item.copy()
                    res_item['score'] = score
                    res_item['search_mode'] = 'BM25 Sparse'
                    results.append(res_item)
                    if len(results) >= top_k:
                        break
                return results

            elif search_mode == "dense":
                results = []
                for score, idx in dense_results:
                    item = metadata[idx]
                    if doc_filter_set and item['file_name'] not in doc_filter_set:
                        continue
                    if score >= similarity_threshold:
                        res_item = item.copy()
                        res_item['score'] = score
                        res_item['search_mode'] = 'Dense Vector'
                        results.append(res_item)
                        if len(results) >= top_k:
                            break
                return results

            else:
                # Default: Hybrid Search with Reciprocal Rank Fusion (RRF)
                rrf_k = getattr(config, 'RRF_K', 60)
                rrf_scores: Dict[int, float] = {}
                dense_scores_map: Dict[int, float] = {idx: score for score, idx in dense_results}
                bm25_scores_map: Dict[int, float] = {idx: score for score, idx in bm25_results}

                # Dense ranks contribution
                for rank, (score, idx) in enumerate(dense_results, start=1):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

                # BM25 ranks contribution
                for rank, (score, idx) in enumerate(bm25_results, start=1):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

                sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

                results = []
                for idx, rrf_score in sorted_candidates:
                    item = metadata[idx]
                    if doc_filter_set and item['file_name'] not in doc_filter_set:
                        continue
                    res_item = item.copy()
                    res_item['score'] = rrf_score
                    res_item['dense_score'] = dense_scores_map.get(idx, 0.0)
                    res_item['bm25_score'] = bm25_scores_map.get(idx, 0.0)
                    res_item['search_mode'] = 'Hybrid RRF'
                    results.append(res_item)
                    if len(results) >= top_k:
                        break

                return results


        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            return []

    @classmethod
    def rebuild_index_from_db(cls) -> Tuple[bool, int, int]:
        """Rebuild the vector & BM25 index from all active documents in database."""
        from utils.document_loader import DocumentLoader
        from utils.text_splitter import RecursiveTextSplitter
        
        all_docs = db.get_all_documents()
        if not all_docs:
            for path in [cls.get_index_path(), cls.get_metadata_path()]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            return True, 0, 0
            
        splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        all_chunks = []
        
        for doc in all_docs:
            if not os.path.exists(doc['file_path']):
                continue
            text = DocumentLoader.load_document(doc['file_path'])
            if text:
                chunks = splitter.split_text(text)
                ext = os.path.splitext(doc['file_name'])[1].lower()
                file_type = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"
                for chunk in chunks:
                    all_chunks.append({
                        'text': chunk,
                        'file_name': doc['file_name'],
                        'doc_id': doc['id'],
                        'file_type': file_type
                    })
                    
        success = cls.index_documents(all_chunks)
        return success, len(all_docs), len(all_chunks)

