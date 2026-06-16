"""
Task 5: Embedding Drift Detection
-----------------------------------
Detects drift in vector embeddings over time using cosine similarity.

How it works:
  1. CREATE BASELINE — embed a fixed set of reference queries and save
                       the vectors as the ground-truth reference point.

  2. MEASURE DRIFT   — re-embed the same queries today, compare each
                       new vector against its baseline vector using:

                       drift_score = 1 - cosine_similarity(current, reference)

  3. ALERT           — if drift_score > threshold, raise an anomaly alert.

  4. STORE & REPORT  — save all measurements to SQLite + JSON report.

Run modes:
  python drift_detector.py --mode baseline   (first time setup)
  python drift_detector.py --mode measure    (every monitoring cycle)
  python drift_detector.py                   (auto: baseline if none exists)
"""

import os
import sys
import json
import math
import time
import sqlite3
import argparse
from datetime import datetime
# from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import config


# ══════════════════════════════════════════════════════════════════════════════
#  Reference Queries
#  These 20 queries are FIXED — never change them after baseline is created.
#  They act as anchors. We re-embed these same sentences every monitoring run.
# ══════════════════════════════════════════════════════════════════════════════

REFERENCE_QUERIES = [
    "What are large language models?",
    "How does retrieval augmented generation work?",
    "Explain embedding vectors in natural language processing.",
    "What is cosine similarity used for in AI?",
    "How do transformer models process text?",
    "What is the difference between fine-tuning and RAG?",
    "Explain the concept of semantic search.",
    "What are AI agents and how do they plan tasks?",
    "How does FAISS perform similarity search?",
    "What is the EU AI Act and who does it apply to?",
    "Explain multimodal AI capabilities.",
    "What is hallucination in language models?",
    "How does prompt engineering affect model outputs?",
    "What is the mixture of experts architecture?",
    "Explain context windows in large language models.",
    "What is Constitutional AI developed by Anthropic?",
    "How does vector indexing work in production systems?",
    "What are the main challenges of LLM deployment?",
    "Explain the concept of token usage in LLM APIs.",
    "What is embedding drift and why does it matter?",
]


def save_reference_queries():
    """Save reference queries to disk for reproducibility."""
    os.makedirs("data", exist_ok=True)
    with open(config.REF_QUERY_FILE, "w") as f:
        json.dump(REFERENCE_QUERIES, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Embedding Model
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding_model():
    return AzureOpenAIEmbeddings(
        azure_deployment = config.EMBEDDING_DEPLOYMENT,
        azure_endpoint   = config.AZURE_ENDPOINT,
        api_key          = config.AZURE_API_KEY,
        api_version      = config.AZURE_API_VERSION,
    )


def embed_queries(queries: list, model) -> list:
    """
    Convert a list of text queries into embedding vectors.
    Returns a list of float lists — one vector per query.
    """
    print(f"  Embedding {len(queries)} queries via Azure OpenAI...")
    vectors = []
    for i, query in enumerate(queries, 1):
        vec = model.embed_query(query)
        vectors.append(vec)
        if i % 5 == 0:
            print(f"  [{i}/{len(queries)}] embeddings done...")
        time.sleep(0.1)   # gentle rate limiting
    return vectors


# ══════════════════════════════════════════════════════════════════════════════
#  Cosine Similarity (manual — no extra dependencies needed)
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    Formula:
        cos(θ) = (A · B) / (||A|| × ||B||)

    Returns a value between -1.0 and 1.0:
        1.0  = identical direction (no drift)
        0.0  = perpendicular (90° apart)
       -1.0  = opposite direction (maximum drift)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a      = math.sqrt(sum(a * a for a in vec_a))
    norm_b      = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def drift_score(vec_current: list, vec_reference: list) -> float:
    """
    Calculate drift score using the project formula:

        drift_score = 1 - cosine_similarity(current, reference)

    0.0 = no drift (perfect match)
    1.0 = complete drift (nothing in common)
    """
    sim = cosine_similarity(vec_current, vec_reference)
    return round(1.0 - sim, 6)


def classify_drift(score: float) -> str:
    """Classify a drift score into a status label."""
    if score < config.DRIFT_WARNING_THRESHOLD :
        return "HEALTHY"
    elif score < config.DRIFT_ANOMALY_THRESHOLD:
        return "WARNING"
    else:
        return "ANOMALY"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Create Baseline
# ══════════════════════════════════════════════════════════════════════════════

def create_baseline():
    """
    Embed all reference queries and save as the drift baseline.
    Run this ONCE when you first set up the system.
    """
    print("\n" + "═" * 55)
    print("  PHASE 1: Creating Drift Baseline")
    print("═" * 55)

    if os.path.isfile(config.BASELINE_FILE):
        print(f"\n Baseline already exists at {config.BASELINE_FILE}")
        ans = input("   Overwrite it? (y/n): ").strip().lower()
        if ans != "y":
            print("   Keeping existing baseline.")
            return

    model   = get_embedding_model()
    vectors = embed_queries(REFERENCE_QUERIES, model)

    # Save baseline
    os.makedirs("logs", exist_ok=True)
    baseline = {
        "created_at":        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_model":   config.EMBEDDING_DEPLOYMENT,
        "query_count":       len(REFERENCE_QUERIES),
        "embedding_dim":     len(vectors[0]) if vectors else 0,
        "entries": [
            {
                "query":   query,
                "vector":  vector,
            }
            for query, vector in zip(REFERENCE_QUERIES, vectors)
        ]
    }

    with open(config.BASELINE_FILE, "w") as f:
        json.dump(baseline, f)

    save_reference_queries()

    print(f"\nBaseline saved!")
    print(f"   File     : {config.BASELINE_FILE}")
    print(f"   Queries  : {len(REFERENCE_QUERIES)}")
    print(f"   Emb dim  : {len(vectors[0])} dimensions")
    print(f"   Created  : {baseline['created_at']}")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Measure Drift
# ══════════════════════════════════════════════════════════════════════════════

def measure_drift():
    """
    Re-embed all reference queries and compare against baseline.
    Saves results to SQLite and prints an anomaly report.
    """
    print("\n" + "═" * 55)
    print("  PHASE 2: Measuring Embedding Drift")
    print("═" * 55)

    # ── Load baseline ──────────────────────────────────────────────────────────
    if not os.path.isfile(config.BASELINE_FILE):
        print("No baseline found. Run with --mode baseline first.")
        return None

    with open(config.BASELINE_FILE) as f:
        baseline = json.load(f)

    baseline_entries = {e["query"]: e["vector"] for e in baseline["entries"]}
    baseline_created = baseline["created_at"]
    print(f"\n  Baseline created : {baseline_created}")
    print(f"  Baseline model   : {baseline['embedding_model']}")
    print(f"  Reference queries: {baseline['query_count']}\n")

    # ── Re-embed the same queries now ─────────────────────────────────────────
    model          = get_embedding_model()
    current_vecs   = embed_queries(REFERENCE_QUERIES, model)
    measured_at    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ── Compute drift for each query ───────────────────────────────────────────
    print("\n  Computing drift scores...\n")
    print(f"  {'Query':<48} {'Drift':>6}  Status")
    print("  " + "-" * 65)

    results   = []
    anomalies = []
    warnings  = []

    for query, current_vec in zip(REFERENCE_QUERIES, current_vecs):
        ref_vec = baseline_entries.get(query)
        if ref_vec is None:
            print(f"  Query not in baseline: {query[:40]}")
            continue

        score   = drift_score(current_vec, ref_vec)
        sim     = round(1.0 - score, 6)
        status  = classify_drift(score)

        # Status icon
        icon = {"HEALTHY": "Healthy", "WARNING": "Warn", "ANOMALY": "Anomal"}[status]

        print(f"  {query[:48]:<48} {score:>6.4f}  {icon} {status}")

        entry = {
            "query":              query,
            "drift_score":        score,
            "cosine_similarity":  sim,
            "status":             status,
            "measured_at":        measured_at,
            "baseline_created":   baseline_created,
        }
        results.append(entry)

        if status == "ANOMALY": anomalies.append(entry)
        if status == "WARNING":  warnings.append(entry)

    # ── Aggregate stats ────────────────────────────────────────────────────────
    all_scores   = [r["drift_score"] for r in results]
    avg_drift    = round(sum(all_scores) / len(all_scores), 6) if all_scores else 0
    max_drift    = round(max(all_scores), 6) if all_scores else 0
    min_drift    = round(min(all_scores), 6) if all_scores else 0
    overall_status = (
        "ANOMALY" if anomalies else
        "WARNING" if warnings  else
        "HEALTHY"
    )

    summary = {
        "measured_at":     measured_at,
        "baseline_created":baseline_created,
        "total_queries":   len(results),
        "avg_drift_score": avg_drift,
        "max_drift_score": max_drift,
        "min_drift_score": min_drift,
        "anomaly_count":   len(anomalies),
        "warning_count":   len(warnings),
        "healthy_count":   len(results) - len(anomalies) - len(warnings),
        "overall_status":  overall_status,
        "per_query":       results,
    }

    # ── Save to SQLite ─────────────────────────────────────────────────────────
    _save_drift_to_db(results, summary)

    # ── Save JSON report ───────────────────────────────────────────────────────
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORTS_DIR, "task5_drift_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────────
    icon = {"HEALTHY": "Healthy", "WARNING": "Warn", "ANOMALY": "Anomaly"}[overall_status]
    print(f"\n{'═' * 55}")
    print(f"  DRIFT SUMMARY  {icon}  {overall_status}")
    print(f"{'═' * 55}")
    print(f"  Queries measured : {len(results)}")
    print(f"  Avg drift score  : {avg_drift:.4f}")
    print(f"  Max drift score  : {max_drift:.4f}")
    print(f"  Healthy          : {summary['healthy_count']}")
    print(f"  Warnings         : {len(warnings)}")
    print(f"  Anomalies        : {len(anomalies)}")
    print(f"\n  Threshold Warning : > {config.DRIFT_WARNING_THRESHOLD }")
    print(f"  Threshold Anomaly : > {config.DRIFT_ANOMALY_THRESHOLD }")

    if anomalies:
        print(f"\n ANOMALY QUERIES:")
        for a in anomalies:
            print(f"     drift={a['drift_score']:.4f}  {a['query'][:50]}")

    print(f"\n  Report: {report_path}")
    print(f"{'═' * 55}")
    print("\nTask 5 Complete!")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  SQLite Storage
# ══════════════════════════════════════════════════════════════════════════════

def _save_drift_to_db(results: list, summary: dict):
    """Save drift measurement run to SQLite."""
    conn = sqlite3.connect(config.DB_FILE)

    # Create table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            measured_at       TEXT NOT NULL,
            baseline_created  TEXT,
            query             TEXT,
            drift_score       REAL,
            cosine_similarity REAL,
            status            TEXT,
            avg_drift_run     REAL,
            overall_status    TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_drift_time
        ON drift_logs(measured_at)
    """)

    avg = summary["avg_drift_score"]
    overall = summary["overall_status"]

    for r in results:
        conn.execute("""
            INSERT INTO drift_logs (
                measured_at, baseline_created, query,
                drift_score, cosine_similarity,
                status, avg_drift_run, overall_status
            ) VALUES (?,?,?,?,?,?,?,?)
        """, (
            r["measured_at"],
            r["baseline_created"],
            r["query"],
            r["drift_score"],
            r["cosine_similarity"],
            r["status"],
            avg,
            overall,
        ))

    conn.commit()
    conn.close()
    print(f"\n  Drift results saved to SQLite ({len(results)} rows)")


def get_drift_history() -> list:
    """Return all drift measurement runs grouped by timestamp."""
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT measured_at, overall_status,
               ROUND(avg_drift_run, 6) AS avg_drift,
               COUNT(*) AS query_count,
               SUM(CASE WHEN status='ANOMALY' THEN 1 ELSE 0 END) AS anomalies
        FROM drift_logs
        GROUP BY measured_at
        ORDER BY measured_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Embedding Drift Detector")
    parser.add_argument(
        "--mode",
        choices=["baseline", "measure", "history"],
        default=None,
        help="baseline=create reference | measure=check drift | history=show runs"
    )
    args = parser.parse_args()

    if args.mode == "history":
        history = get_drift_history()
        print(f"\n{'Timestamp':<22} {'Status':<10} {'Avg Drift':>10} {'Anomalies':>10}")
        print("-" * 60)
        for h in history:
            print(f"{h['measured_at']:<22} {h['overall_status']:<10} "
                  f"{h['avg_drift']:>10.4f} {h['anomalies']:>10}")
        return

    # Auto mode: if no baseline exists → create it, else → measure
    if args.mode == "baseline" or not os.path.isfile(config.BASELINE_FILE):
        create_baseline()
    
    if args.mode != "baseline":
        measure_drift()


if __name__ == "__main__":
    main()
