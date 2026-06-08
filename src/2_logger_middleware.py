"""
Task 2: Logging Middleware
--------------------------
Every LLM request passes through this middleware which:
  - Captures all 11 required fields
  - Saves to CSV (human readable)
  - Saves to SQLite (queryable for dashboard)
  - Handles errors gracefully
  - Works independently from the RAG pipeline
"""

import os
import csv
import json
import sqlite3
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR  = "logs"
CSV_FILE  = os.path.join(LOGS_DIR, "rag_logs.csv")
DB_FILE   = os.path.join(LOGS_DIR, "monitoring.db")

# ── All 11 required fields ─────────────────────────────────────────────────────
CSV_FIELDS = [
    "request_id",
    "timestamp",
    "user_query",
    "retrieved_context",       # JSON string of top-K chunks
    "retrieved_sources",       # JSON list of source filenames
    "model_name",
    "prompt_version",
    "model_response",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "retrieval_latency",
    "llm_latency",
    "end_to_end_latency",
    "top_k",
    "safety_flag",
    "error_code",
]


# ══════════════════════════════════════════════════════════════════════════════
#  LLMLogger Class
# ══════════════════════════════════════════════════════════════════════════════

class LLMLogger:
    """
    Standalone logging middleware for LLM and RAG requests.

    Usage:
        logger = LLMLogger()
        logger.log(result_dict)
        df = logger.get_logs()
        stats = logger.get_summary()
    """

    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._setup_csv()
        self._setup_sqlite()
        print("LLMLogger initialized — logging to CSV and SQLite")

    # ── CSV Setup ──────────────────────────────────────────────────────────────

    def _setup_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        file_exists = os.path.isfile(CSV_FILE)
        self._csv_file   = open(CSV_FILE, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",   # Ignore extra keys not in fieldnames
        )
        if not file_exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    # ── SQLite Setup ───────────────────────────────────────────────────────────

    def _setup_sqlite(self):
        """Create SQLite database and table if they don't exist."""
        self._db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row   # Return rows as dict-like objects
        cursor = self._db_conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_logs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id          TEXT NOT NULL,
                timestamp           TEXT NOT NULL,
                user_query          TEXT,
                retrieved_context   TEXT,
                retrieved_sources   TEXT,
                model_name          TEXT,
                prompt_version      TEXT,
                model_response      TEXT,
                input_tokens        INTEGER DEFAULT 0,
                output_tokens       INTEGER DEFAULT 0,
                total_tokens        INTEGER DEFAULT 0,
                retrieval_latency   REAL DEFAULT 0.0,
                llm_latency         REAL DEFAULT 0.0,
                end_to_end_latency  REAL DEFAULT 0.0,
                top_k               INTEGER DEFAULT 3,
                safety_flag         INTEGER DEFAULT 0,
                error_code          TEXT DEFAULT ''
            )
        """)

        # Index on timestamp for fast time-range queries in dashboard
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON llm_logs(timestamp)
        """)

        # Index on prompt_version for version comparison queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_version
            ON llm_logs(prompt_version)
        """)

        self._db_conn.commit()

    # ── Core Log Method ────────────────────────────────────────────────────────

    def log(self, result: dict):
        """
        Save one complete LLM request log entry to both CSV and SQLite.

        Expects the dict returned by rag_pipeline.rag_query().
        """
        # ── Prepare all fields ─────────────────────────────────────────────────
        retrieved_chunks  = result.get("retrieved_context", [])
        retrieved_sources = [c.get("source", "") for c in retrieved_chunks]

        row = {
            "request_id":         result.get("request_id",         str(uuid.uuid4())),
            "timestamp":          result.get("timestamp",           datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
            "user_query":         result.get("user_query",          ""),
            "retrieved_context":  json.dumps(retrieved_chunks,      ensure_ascii=False),
            "retrieved_sources":  json.dumps(retrieved_sources,     ensure_ascii=False),
            "model_name":         result.get("model_name",          "unknown"),
            "prompt_version":     result.get("prompt_version",      "v1"),
            "model_response":     result.get("model_response",      ""),
            "input_tokens":       result.get("input_tokens",        0),
            "output_tokens":      result.get("output_tokens",       0),
            "total_tokens":       result.get("total_tokens",        0),
            "retrieval_latency":  result.get("retrieval_latency",   0.0),
            "llm_latency":        result.get("llm_latency",         0.0),
            "end_to_end_latency": result.get("end_to_end_latency",  0.0),
            "top_k":              result.get("top_k",               3),
            "safety_flag":        int(result.get("safety_flag",     False)),
            "error_code":         result.get("error_code",          "") or "",
        }

        self._write_csv(row)
        self._write_sqlite(row)

    def _write_csv(self, row: dict):
        """Write one row to CSV and flush immediately."""
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception as e:
            print(f"  CSV write error: {e}")

    def _write_sqlite(self, row: dict):
        """Insert one row into SQLite."""
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                INSERT INTO llm_logs (
                    request_id, timestamp, user_query,
                    retrieved_context, retrieved_sources,
                    model_name, prompt_version, model_response,
                    input_tokens, output_tokens, total_tokens,
                    retrieval_latency, llm_latency, end_to_end_latency,
                    top_k, safety_flag, error_code
                ) VALUES (
                    :request_id, :timestamp, :user_query,
                    :retrieved_context, :retrieved_sources,
                    :model_name, :prompt_version, :model_response,
                    :input_tokens, :output_tokens, :total_tokens,
                    :retrieval_latency, :llm_latency, :end_to_end_latency,
                    :top_k, :safety_flag, :error_code
                )
            """, row)
            self._db_conn.commit()
        except Exception as e:
            print(f"⚠️  SQLite write error: {e}")

    # ── Read Methods ───────────────────────────────────────────────────────────

    def get_logs(self, limit: int = 100, prompt_version: str = None) -> list:
        """
        Retrieve logs from SQLite as a list of dicts.

        Args:
            limit:          Max number of rows to return
            prompt_version: Filter by 'v1' or 'v2' (optional)
        """
        cursor = self._db_conn.cursor()

        if prompt_version:
            cursor.execute("""
                SELECT * FROM llm_logs
                WHERE prompt_version = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (prompt_version, limit))
        else:
            cursor.execute("""
                SELECT * FROM llm_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_summary(self) -> dict:
        """
        Return aggregate statistics across all logged requests.
        Used by the dashboard and for Task 7 evaluation.
        """
        cursor = self._db_conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*)                        AS total_requests,
                SUM(total_tokens)               AS total_tokens,
                AVG(total_tokens)               AS avg_tokens,
                AVG(llm_latency)                AS avg_llm_latency,
                AVG(retrieval_latency)          AS avg_retrieval_latency,
                AVG(end_to_end_latency)         AS avg_e2e_latency,
                MAX(end_to_end_latency)         AS max_latency,
                MIN(end_to_end_latency)         AS min_latency,
                SUM(CASE WHEN safety_flag = 1
                    THEN 1 ELSE 0 END)          AS safety_flag_count,
                SUM(CASE WHEN error_code != ''
                    THEN 1 ELSE 0 END)          AS error_count,
                SUM(CASE WHEN prompt_version = 'v1'
                    THEN 1 ELSE 0 END)          AS v1_count,
                SUM(CASE WHEN prompt_version = 'v2'
                    THEN 1 ELSE 0 END)          AS v2_count
            FROM llm_logs
        """)

        row = cursor.fetchone()
        summary = dict(row) if row else {}

        # Add cost estimate: text-embedding-ada-002 + gpt-4o rough pricing
        total_tokens = summary.get("total_tokens") or 0
        summary["estimated_cost_usd"] = round(total_tokens * 0.000002, 6)

        return summary

    def get_error_logs(self) -> list:
        """Return only the requests that had errors."""
        cursor = self._db_conn.cursor()
        cursor.execute("""
            SELECT request_id, timestamp, user_query, error_code, model_name
            FROM llm_logs
            WHERE error_code != ''
            ORDER BY timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_safety_flagged(self) -> list:
        """Return only the requests that triggered safety flags."""
        cursor = self._db_conn.cursor()
        cursor.execute("""
            SELECT request_id, timestamp, user_query, model_response
            FROM llm_logs
            WHERE safety_flag = 1
            ORDER BY timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self):
        """Close file handles cleanly."""
        try:
            self._csv_file.close()
            self._db_conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
