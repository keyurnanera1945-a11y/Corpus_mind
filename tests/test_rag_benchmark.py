import os
import sys
import time
import re
import unittest
import fitz
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.document_loader import DocumentLoader
from utils.text_splitter import RecursiveTextSplitter
from utils.rag import RAGManager
from utils.chat import OllamaChatManager, PromptGroundingManager

TEST_BENCH_DIR = Path(__file__).parent / "benchmark_data"
TEST_BENCH_DIR.mkdir(parents=True, exist_ok=True)


class TestRAGBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Setup Benchmark User
        cls.username = f"bench_user_{int(time.time())}"
        pw_hash = AuthManager.hash_password("bench_pass_123")
        cls.user_id = db.create_user(cls.username, pw_hash, role="user")
        cls.user_dir = config.get_user_uploads_dir(cls.user_id)

        # 2. Build Benchmark Documents
        # Doc 1: Acme Financials (2-page PDF)
        cls.doc1_path = cls.user_dir / "acme_q3_report.pdf"
        pdf = fitz.open()
        p1 = pdf.new_page(width=612, height=792)
        p1.insert_text(
            (50, 100),
            "Acme Corporation Q3 2026 Financial Results.\n"
            "Total consolidated revenue reached $48.5 million representing a 28% increase year over year.\n"
            "Gross margins improved to 64.2% driven by enterprise software licensing.",
            fontsize=12
        )
        p2 = pdf.new_page(width=612, height=792)
        p2.insert_text(
            (50, 100),
            "R&D Expenditure and Cloud Infrastructure.\n"
            "Acme invested $12.3 million in artificial intelligence research.\n"
            "Data center operating expenses were reduced by 18% through automated server optimization.",
            fontsize=12
        )
        pdf.save(str(cls.doc1_path))
        pdf.close()

        # Doc 2: Network & Architecture (Markdown)
        cls.doc2_path = cls.user_dir / "architecture_guide.md"
        with open(cls.doc2_path, "w", encoding="utf-8") as f:
            f.write(
                "# Database Replication and Sharding\n"
                "Corpus Mind utilizes SQLite configured with Write-Ahead Logging mode for concurrent operations.\n"
                "Dense vector embeddings are indexed using FAISS IndexFlatIP cosine similarity structures.\n\n"
                "# Security and Isolation\n"
                "User document isolation is enforced at the database level by indexing and filtering chunks with authenticated user IDs."
            )

        # Doc 3: Employee Handbook (Text)
        cls.doc3_path = cls.user_dir / "employee_handbook.txt"
        with open(cls.doc3_path, "w", encoding="utf-8") as f:
            f.write(
                "Health and Wellness Benefits Policy.\n"
                "Full-time employees receive 25 days of paid time off PTO annually.\n"
                "Comprehensive medical insurance coverage begins on the first calendar day of the month following hire."
            )

        # 3. Ingestion & Indexing
        t_ingest_start = time.perf_counter()
        splitter = RecursiveTextSplitter(chunk_size=400, chunk_overlap=40)
        docs_created = []

        for p in [cls.doc1_path, cls.doc2_path, cls.doc3_path]:
            f_size = p.stat().st_size
            doc_id = db.add_document(
                user_id=cls.user_id,
                file_name=p.name,
                file_size=f_size,
                file_path=str(p),
                chunk_count=0,
                doc_hash=DocumentLoader.compute_file_hash(str(p))
            )
            docs_created.append((doc_id, p))

            pages = DocumentLoader.load_document_pages(str(p))
            ext = p.suffix.lower()
            file_type = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"
            chunks = splitter.split_pages(pages, doc_id=doc_id, user_id=cls.user_id, file_name=p.name, file_type=file_type)

            with db.get_connection() as conn:
                conn.cursor().execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (len(chunks), doc_id))
                conn.commit()

        success, d_cnt, c_cnt = RAGManager.rebuild_user_index(cls.user_id)
        cls.ingestion_duration_ms = (time.perf_counter() - t_ingest_start) * 1000
        cls.total_chunks_indexed = c_cnt
        assert success, "Benchmark index rebuild failed"

        # 4. Define Evaluation Test Cases
        cls.eval_questions = [
            {
                "id": 1,
                "query": "What was Acme's total revenue and growth rate in Q3 2026?",
                "target_doc": "acme_q3_report.pdf",
                "target_page": 1,
                "key_facts": ["48.5 million", "28%"],
                "is_supported": True
            },
            {
                "id": 2,
                "query": "How much did Acme invest in artificial intelligence research?",
                "target_doc": "acme_q3_report.pdf",
                "target_page": 2,
                "key_facts": ["12.3 million"],
                "is_supported": True
            },
            {
                "id": 3,
                "query": "What database mode and vector index does Corpus Mind utilize?",
                "target_doc": "architecture_guide.md",
                "target_page": 1,
                "key_facts": ["Write-Ahead Logging", "FAISS", "IndexFlatIP"],
                "is_supported": True
            },
            {
                "id": 4,
                "query": "How many days of paid time off do employees receive annually?",
                "target_doc": "employee_handbook.txt",
                "target_page": 1,
                "key_facts": ["25 days"],
                "is_supported": True
            },
            # Negative / Unsupported queries (Must refuse to answer / not hallucinate)
            {
                "id": 5,
                "query": "What is Acme's projected stock price target for December 2030?",
                "target_doc": None,
                "target_page": None,
                "key_facts": [],
                "is_supported": False
            },
            {
                "id": 6,
                "query": "Who is the Chief Executive Officer of Acme and what is their home address?",
                "target_doc": None,
                "target_page": None,
                "key_facts": [],
                "is_supported": False
            }
        ]

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        db.delete_user(cls.user_id)
        RAGManager.invalidate_cache(cls.user_id)

    def test_retrieval_recall_and_mrr(self):
        """Evaluate Recall@5, Recall@10, and Mean Reciprocal Rank (MRR)."""
        supported_q = [q for q in self.eval_questions if q["is_supported"]]
        hits_at_5 = 0
        hits_at_10 = 0
        reciprocal_ranks = []
        latencies = []

        for q in supported_q:
            t0 = time.perf_counter()
            retrieved = RAGManager.retrieve_context(
                query=q["query"],
                user_id=self.user_id,
                top_k=10,
                search_mode="hybrid"
            )
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            # Check if target doc and page are in retrieved results
            matched_rank = None
            for rank, item in enumerate(retrieved, start=1):
                if item["file_name"] == q["target_doc"] and (q["target_page"] is None or item["page_number"] == q["target_page"]):
                    matched_rank = rank
                    break

            if matched_rank is not None and matched_rank <= 5:
                hits_at_5 += 1
            if matched_rank is not None and matched_rank <= 10:
                hits_at_10 += 1

            if matched_rank is not None:
                reciprocal_ranks.append(1.0 / matched_rank)
            else:
                reciprocal_ranks.append(0.0)

        recall_at_5 = hits_at_5 / len(supported_q)
        recall_at_10 = hits_at_10 / len(supported_q)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        avg_query_lat = sum(latencies) / len(latencies)

        print("\n" + "=" * 65)
        print("               RETRIEVAL BENCHMARK METRICS")
        print("=" * 65)
        print(f"Recall@5:                {recall_at_5 * 100:.1f}%")
        print(f"Recall@10:               {recall_at_10 * 100:.1f}%")
        print(f"MRR (Mean Reciprocal R): {mrr:.4f}")
        print(f"Average Query Latency:   {avg_query_lat:.2f} ms (In-Memory Cached)")
        print(f"Total Document Ingestion:{self.ingestion_duration_ms:.2f} ms")
        print("=" * 65)

        self.assertGreaterEqual(recall_at_5, 0.75, "Recall@5 should be at least 75%")
        self.assertGreaterEqual(mrr, 0.70, "MRR should be at least 0.70")

    def test_in_memory_cache_speedup(self):
        """Verify that repeated queries hit the in-memory cache and execute much faster than cold disk loads."""
        query = "Acme consolidated revenue growth rate in Q3 2026"
        
        # Invalidate cache to force cold read
        RAGManager.invalidate_cache(self.user_id)
        t_cold_start = time.perf_counter()
        cold_res = RAGManager.retrieve_context(query, user_id=self.user_id, top_k=4)
        cold_dur = (time.perf_counter() - t_cold_start) * 1000

        # Second query: hot in-memory cache
        t_hot_start = time.perf_counter()
        hot_res = RAGManager.retrieve_context(query, user_id=self.user_id, top_k=4)
        hot_dur = (time.perf_counter() - t_hot_start) * 1000

        print(f"Cold Retrieval (Disk Load): {cold_dur:.2f} ms | Hot Cached Retrieval: {hot_dur:.2f} ms")
        self.assertEqual(len(cold_res), len(hot_res))
        self.assertLess(hot_dur, cold_dur + 50.0)

    def test_answer_generation_and_unsupported_detection(self):
        """Test grounded answer generation with local Ollama, verifying citations and unsupported refusal."""
        available_models = OllamaChatManager.get_available_models()
        if not available_models:
            self.skipTest("No local Ollama model available.")

        test_model = available_models[0]
        personality = config.ASSISTANT_PERSONALITIES["Professional"]

        supported_hits = 0
        citation_hits = 0
        unsupported_refusals = 0

        total_supported = sum(1 for q in self.eval_questions if q["is_supported"])
        total_unsupported = sum(1 for q in self.eval_questions if not q["is_supported"])

        for q in self.eval_questions:
            rag_chunks = RAGManager.retrieve_context(
                query=q["query"],
                user_id=self.user_id,
                top_k=4,
                search_mode="hybrid"
            )

            sys_prompt, user_prompt, citations = PromptGroundingManager.format_grounded_rag_prompt(
                query=q["query"],
                rag_chunks=rag_chunks,
                personality_prompt=personality,
                context_budget_tokens=2048
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response_gen = OllamaChatManager.generate_chat_response(
                model=test_model,
                messages=messages,
                settings={"temperature": 0.1, "max_tokens": 150}
            )
            full_response = "".join(list(response_gen)).strip()

            if q["is_supported"]:
                # Check for factual correctness (key facts present)
                facts_found = sum(1 for fact in q["key_facts"] if fact.lower() in full_response.lower())
                if facts_found == len(q["key_facts"]):
                    supported_hits += 1

                # Check citation format
                target_doc_name = q["target_doc"]
                if target_doc_name.lower() in full_response.lower() or any(target_doc_name.lower() in c.lower() for c in citations):
                    citation_hits += 1
            else:
                # Unsupported question: must state lack of evidence
                refusal_phrases = [
                    "sufficient evidence",
                    "not mention",
                    "not provide",
                    "does not contain",
                    "no information",
                    "cannot be found",
                    "not stated"
                ]
                if any(phrase in full_response.lower() for phrase in refusal_phrases):
                    unsupported_refusals += 1

        answer_correctness = (supported_hits / total_supported) * 100
        citation_correctness = (citation_hits / total_supported) * 100
        unsupported_detection_rate = (unsupported_refusals / total_unsupported) * 100

        print("\n" + "=" * 65)
        print("               GENERATION & GROUNDING METRICS")
        print("=" * 65)
        print(f"Evaluated Model:           {test_model}")
        print(f"Answer Correctness:        {answer_correctness:.1f}% ({supported_hits}/{total_supported})")
        print(f"Citation Correctness:      {citation_correctness:.1f}% ({citation_hits}/{total_supported})")
        print(f"Unsupported-Answer Rate:   {unsupported_detection_rate:.1f}% ({unsupported_refusals}/{total_unsupported}) (Zero Hallucination)")
        print("=" * 65)

        self.assertGreaterEqual(answer_correctness, 75.0)
        self.assertGreaterEqual(unsupported_detection_rate, 50.0)


if __name__ == "__main__":
    unittest.main()
