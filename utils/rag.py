import os
import sys
import pickle
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


class NumpyVectorStore:
    """A native NumPy-based cosine-similarity vector store."""
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        for emb in embeddings:
            self.embeddings.append(emb)
        self.metadata.extend(metadata)

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.embeddings:
            return []
            
        embeddings_matrix = np.array(self.embeddings, dtype=np.float32) # shape: (N, D)
        query_vector = np.array(query_emb, dtype=np.float32) # shape: (D,)
        
        # Calculate dot products
        dot_products = np.dot(embeddings_matrix, query_vector)
        
        # Calculate norms
        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Avoid division by zero
        matrix_norms[matrix_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10
            
        # Cosine similarities
        similarities = dot_products / (matrix_norms * query_norm)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            results.append((score, self.metadata[idx]))
            
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
        
        chunks is a list of dicts: {'text': str, 'file_name': str, 'doc_id': int}
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return False

        try:
            texts = [c['text'] for c in chunks]
            metadata = [{'text': c['text'], 'file_name': c['file_name'], 'doc_id': c['doc_id']} for c in chunks]
            
            # Generate embeddings
            embeddings = EmbeddingManager.get_embeddings(texts)
            embeddings = np.array(embeddings, dtype=np.float32)

            if FAISS_AVAILABLE:
                # FAISS L2 or Inner Product Index
                dimension = embeddings.shape[1]
                # Normalize for cosine similarity (Inner Product on normalized vectors)
                faiss.normalize_L2(embeddings)
                
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                
                # Save FAISS
                faiss.write_index(index, cls.get_index_path())
                with open(cls.get_metadata_path(), 'wb') as f:
                    pickle.dump(metadata, f)
            else:
                # Numpy store
                store = NumpyVectorStore()
                store.add(embeddings, metadata)
                store.save(cls.get_index_path()) # Save as pickle to same bin path

            logger.info(f"Indexed {len(chunks)} chunks successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False

    @classmethod
    def retrieve_context(cls, query: str, top_k: int = 4, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks for a given query."""
        try:
            query_emb = EmbeddingManager.get_embedding(query)
            query_emb = np.array(query_emb, dtype=np.float32)

            results = []
            
            if FAISS_AVAILABLE:
                index_path = cls.get_index_path()
                meta_path = cls.get_metadata_path()
                
                if not (os.path.exists(index_path) and os.path.exists(meta_path)):
                    logger.warning("Index files not found.")
                    return []
                    
                index = faiss.read_index(index_path)
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                    
                # Normalize query embedding for cosine similarity search
                query_emb = query_emb.reshape(1, -1)
                faiss.normalize_L2(query_emb)
                
                # Search
                scores, indices = index.search(query_emb, top_k)
                
                # Check results
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx < len(metadata) and score >= similarity_threshold:
                        item = metadata[idx].copy()
                        item['score'] = float(score)
                        results.append(item)
            else:
                store_path = cls.get_index_path()
                if not os.path.exists(store_path):
                    logger.warning("Numpy store files not found.")
                    return []
                    
                store = NumpyVectorStore()
                store.load(store_path)
                
                search_results = store.search(query_emb, top_k)
                for score, meta in search_results:
                    if score >= similarity_threshold:
                        item = meta.copy()
                        item['score'] = score
                        results.append(item)

            return results
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            return []

    @classmethod
    def rebuild_index_from_db(cls) -> Tuple[bool, int, int]:
        """Rebuild the vector index from all active documents in database."""
        from utils.document_loader import DocumentLoader
        from utils.text_splitter import RecursiveTextSplitter
        
        all_docs = db.get_all_documents()
        if not all_docs:
            # Delete index files if no docs
            for path in [cls.get_index_path(), cls.get_metadata_path()]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
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
                for chunk in chunks:
                    all_chunks.append({
                        'text': chunk,
                        'file_name': doc['file_name'],
                        'doc_id': doc['id']
                    })
                    
        success = cls.index_documents(all_chunks)
        return success, len(all_docs), len(all_chunks)
