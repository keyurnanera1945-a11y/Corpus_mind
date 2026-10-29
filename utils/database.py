import sqlite3
import logging
from datetime import datetime
import os
import sys
from typing import Optional, List, Dict, Any

# Add parent directory to sys.path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.DB_PATH)
        self.initialize_tables()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Chat sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)
            
            # Messages table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model TEXT,
                response_time_ms INTEGER,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """)
            
            # Documents table with doc_hash support
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'uploaded',
                chunk_count INTEGER DEFAULT 0,
                doc_hash TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)

            # Migration: Ensure doc_hash column exists in existing database
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            if "doc_hash" not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN doc_hash TEXT")
                logger.info("Migrated documents table: added doc_hash column.")
            
            # Analytics logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time_ms INTEGER DEFAULT 0,
                model_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)
            
            conn.commit()
            logger.info("Database tables initialized successfully.")

    # User operations
    def create_user(self, username, password_hash, role='user'):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, password_hash, role)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_user_by_username(self, username):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_users(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, role, created_at FROM users")
            return [dict(row) for row in cursor.fetchall()]

    def delete_user(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    # Chat Sessions operations
    def create_chat_session(self, user_id, title):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
                (user_id, title)
            )
            conn.commit()
            return cursor.lastrowid

    def get_chat_sessions_by_user(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_chat_session_title(self, session_id, new_title, user_id=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (new_title, datetime.now(), session_id, user_id)
                )
            else:
                cursor.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (new_title, datetime.now(), session_id)
                )
            conn.commit()

    def delete_chat_session(self, session_id, user_id=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
            else:
                cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()

    # Message operations
    def add_message(self, session_id, role, content, model=None, response_time_ms=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, model, response_time_ms) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, model, response_time_ms)
            )
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (datetime.now(), session_id)
            )
            conn.commit()
            return cursor.lastrowid

    def get_messages_by_session(self, session_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_last_message(self, session_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1)",
                (session_id,)
            )
            conn.commit()

    # Documents operations with strict user scoping
    def add_document(self, user_id: int, file_name: str, file_size: int, file_path: str, chunk_count: int = 0, doc_hash: Optional[str] = None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (user_id, file_name, file_size, file_path, chunk_count, status, doc_hash) VALUES (?, ?, ?, ?, ?, 'indexed', ?)",
                (user_id, file_name, file_size, file_path, chunk_count, doc_hash)
            )
            conn.commit()
            return cursor.lastrowid

    def get_documents_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY upload_time DESC", (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_document_by_id(self, document_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
            else:
                cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_document_by_hash(self, user_id: int, doc_hash: str) -> Optional[Dict[str, Any]]:
        if not doc_hash:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE user_id = ? AND doc_hash = ?", (user_id, doc_hash))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Admin-only global document list."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY upload_time DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, document_id: int, user_id: Optional[int] = None) -> bool:
        """Delete document file and DB record with authorization check."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute("SELECT id, user_id, file_path FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
            else:
                cursor.execute("SELECT id, user_id, file_path FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Unauthorized or non-existent document delete attempt for doc_id={document_id}, user_id={user_id}")
                return False

            file_path = row['file_path']
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing physical file {file_path}: {e}")

            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            return True

    # Analytics operations
    def log_analytics(self, user_id, event_type, description=None, response_time_ms=0, model_used=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analytics_logs (user_id, event_type, description, response_time_ms, model_used) VALUES (?, ?, ?, ?, ?)",
                (user_id, event_type, description, response_time_ms, model_used)
            )
            conn.commit()

    def get_analytics_summary(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE role='user'")
            total_user_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*), SUM(file_size) FROM documents")
            doc_row = cursor.fetchone()
            total_docs = doc_row[0] or 0
            total_size_bytes = doc_row[1] or 0
            
            cursor.execute("SELECT AVG(response_time_ms) FROM messages WHERE role='assistant' AND response_time_ms IS NOT NULL")
            avg_resp = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT model, COUNT(*) FROM messages WHERE role='assistant' GROUP BY model")
            model_usage = cursor.fetchall()
            
            cursor.execute("""
                SELECT DATE(timestamp) as msg_date, COUNT(*) 
                FROM messages 
                WHERE role='user' 
                GROUP BY msg_date 
                ORDER BY msg_date ASC 
                LIMIT 30
            """)
            msg_history = cursor.fetchall()
            
            return {
                "total_user_messages": total_user_messages,
                "total_documents": total_docs,
                "total_size_bytes": total_size_bytes,
                "avg_response_time_ms": avg_resp,
                "model_usage": model_usage,
                "msg_history": msg_history
            }

# Create globally accessible manager instance
db = DatabaseManager()
