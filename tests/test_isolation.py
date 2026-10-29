import os
import sys
import unittest
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.document_loader import DocumentLoader
from utils.text_splitter import RecursiveTextSplitter
from utils.rag import RAGManager

class TestDocumentIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create two distinct test users
        cls.username_alpha = f"user_alpha_{int(time.time())}"
        cls.username_beta = f"user_beta_{int(time.time())}"
        pw_hash = AuthManager.hash_password("test_pass_123")
        
        cls.user_alpha_id = db.create_user(cls.username_alpha, pw_hash, role="user")
        cls.user_beta_id = db.create_user(cls.username_beta, pw_hash, role="user")
        
        assert cls.user_alpha_id is not None
        assert cls.user_beta_id is not None

        cls.splitter = RecursiveTextSplitter(chunk_size=300, chunk_overlap=30)

        # Create identically named files in each user's isolated directory
        cls.dir_alpha = config.get_user_uploads_dir(cls.user_alpha_id)
        cls.dir_beta = config.get_user_uploads_dir(cls.user_beta_id)

        # Alpha document
        cls.alpha_path = cls.dir_alpha / "strategy_brief.txt"
        with open(cls.alpha_path, "w", encoding="utf-8") as f:
            f.write("Project Alpha secret clearance code XK9-ALPHA-7729. The private allocation is 15 million dollars.")

        # Beta document (SAME FILENAME to test collision protection)
        cls.beta_path = cls.dir_beta / "strategy_brief.txt"
        with open(cls.beta_path, "w", encoding="utf-8") as f:
            f.write("Project Beta confidential memorandum. Target company for acquisition is Zeta Corporation code ZQ4-BETA-3381.")

        # Index Alpha document
        cls.alpha_doc_id = db.add_document(
            user_id=cls.user_alpha_id,
            file_name="strategy_brief.txt",
            file_size=cls.alpha_path.stat().st_size,
            file_path=str(cls.alpha_path),
            chunk_count=1,
            doc_hash=DocumentLoader.compute_file_hash(str(cls.alpha_path))
        )
        s_alpha, _, _ = RAGManager.rebuild_user_index(cls.user_alpha_id)
        assert s_alpha, "Alpha index build failed"

        # Index Beta document
        cls.beta_doc_id = db.add_document(
            user_id=cls.user_beta_id,
            file_name="strategy_brief.txt",
            file_size=cls.beta_path.stat().st_size,
            file_path=str(cls.beta_path),
            chunk_count=1,
            doc_hash=DocumentLoader.compute_file_hash(str(cls.beta_path))
        )
        s_beta, _, _ = RAGManager.rebuild_user_index(cls.user_beta_id)
        assert s_beta, "Beta index build failed"

    @classmethod
    def tearDownClass(cls):
        # Cleanup documents and users
        if hasattr(cls, 'alpha_doc_id') and cls.alpha_doc_id:
            db.delete_document(cls.alpha_doc_id, user_id=cls.user_alpha_id)
        if hasattr(cls, 'beta_doc_id') and cls.beta_doc_id:
            db.delete_document(cls.beta_doc_id, user_id=cls.user_beta_id)

        if hasattr(cls, 'user_alpha_id') and cls.user_alpha_id:
            db.delete_user(cls.user_alpha_id)
        if hasattr(cls, 'user_beta_id') and cls.user_beta_id:
            db.delete_user(cls.user_beta_id)

    def test_alpha_cannot_retrieve_beta_secrets(self):
        """User Alpha querying for Beta's secrets must retrieve 0 results."""
        results = RAGManager.retrieve_context(
            query="Zeta Corporation acquisition target ZQ4-BETA-3381",
            user_id=self.user_alpha_id,
            top_k=5
        )
        for res in results:
            self.assertNotIn("Zeta Corporation", res['text'])
            self.assertNotIn("ZQ4-BETA-3381", res['text'])
            self.assertEqual(res['user_id'], self.user_alpha_id)

    def test_beta_cannot_retrieve_alpha_secrets(self):
        """User Beta querying for Alpha's secrets must retrieve 0 results."""
        results = RAGManager.retrieve_context(
            query="XK9-ALPHA-7729 private allocation 15 million dollars",
            user_id=self.user_beta_id,
            top_k=5
        )
        for res in results:
            self.assertNotIn("XK9-ALPHA-7729", res['text'])
            self.assertNotIn("15 million", res['text'])
            self.assertEqual(res['user_id'], self.user_beta_id)

    def test_alpha_retrieves_own_document(self):
        """User Alpha querying for own document must successfully retrieve own chunk."""
        results = RAGManager.retrieve_context(
            query="What is the clearance code for Project Alpha?",
            user_id=self.user_alpha_id,
            top_k=3
        )
        self.assertGreater(len(results), 0)
        self.assertIn("XK9-ALPHA-7729", results[0]['text'])
        self.assertEqual(results[0]['user_id'], self.user_alpha_id)

    def test_cross_user_deletion_forbidden(self):
        """User Alpha cannot delete User Beta's document via ID injection."""
        success = db.delete_document(self.beta_doc_id, user_id=self.user_alpha_id)
        self.assertFalse(success, "Unauthorized document deletion should return False")
        
        # Verify Beta's doc still exists
        beta_doc = db.get_document_by_id(self.beta_doc_id, user_id=self.user_beta_id)
        self.assertIsNotNone(beta_doc, "Beta's document should not have been deleted")

    def test_missing_user_id_rejected(self):
        """Calling retrieve_context without user_id must be rejected."""
        results = RAGManager.retrieve_context(
            query="test query",
            user_id=None,
            top_k=3
        )
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()
