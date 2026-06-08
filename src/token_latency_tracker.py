"""
Task 3: Token Usage, Latency & Prompt Version Tracking
--------------------------------------------------------
Reads from the SQLite database and produces:

  1. TOKEN MONITORING
     - Input / output / total tokens per request
     - Estimated cost per request and overall
     - Daily token usage trends
     - Per-model and per-version token breakdown

  2. LATENCY TRACKING
     - LLM API latency stats  (avg, min, max, p95)
     - RAG retrieval latency stats
     - End-to-end latency stats
     - Slow query detection  (> threshold alert)

  3. PROMPT VERSION CONTROL
     - Side-by-side comparison: v1 vs v2
     - Which version uses fewer tokens (cheaper)?
     - Which version is faster?
     - Regression detection between versions
"""

import os
import json
import sqlite3
import statistics
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_FILE      = "logs/monitoring.db"
REPORTS_DIR  = "logs/reports"

# ── Azure OpenAI Pricing (per 1000 tokens, USD) ────────────────────────────────
PRICING = {
    "gpt-4o": {
        "input_per_1k":  0.005,    # $0.005 per 1K input tokens
        "output_per_1k": 0.015,    # $0.015 per 1K output tokens
    },
    "gpt-35-turbo": {
        "input_per_1k":  0.0005,
        "output_per_1k": 0.0015,
    },
    "default": {
        "input_per_1k":  0.005,
        "output_per_1k": 0.015,
    }
}

# ── Alert Thresholds ───────────────────────────────────────────────────────────
LATENCY_ALERT_THRESHOLD = 10.0   # seconds  — flag if e2e latency > this
TOKEN_ALERT_THRESHOLD   = 2000   # tokens   — flag if single request > this
COST_ALERT_THRESHOLD    = 0.10   # USD      — flag if daily cost > this


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: connect to SQLite
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    if not os.path.isfile(DB_FILE):
        raise FileNotFoundError(
            f"Database not found at {DB_FILE}. "
            "Run simulate_queries.py first to generate logs."
        )
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all(query: str, params: tuple = ()) -> list:
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  1. TOKEN MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def calculate_cost(input_tokens: int,
                   output_tokens: int,
                   model_name: str) -> float:
    """Calculate USD cost for one request based on model pricing."""
    pricing = PRICING.get(model_name, PRICING["default"])
    input_cost  = (input_tokens  / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    return round(input_cost + output_cost, 6)


def get_token_stats() -> dict:
    """
    Compute token statistics across all logged requests.
    Returns counts, averages, totals, and cost breakdown.
    """
    rows = fetch_all("""
        SELECT input_tokens, output_tokens, total_tokens,
               model_name, timestamp
        FROM llm_logs
        WHERE error_code = ''
        ORDER BY timestamp
    """)

    if not rows:
        return {"error": "No successful logs found"}

    input_list  = [r["input_tokens"]  for r in rows]
    output_list = [r["output_tokens"] for r in rows]
    total_list  = [r["total_tokens"]  for r in rows]

    # Cost per request
    costs = [
        calculate_cost(r["input_tokens"], r["output_tokens"], r["model_name"])
        for r in rows
    ]

    # Flag expensive requests
    expensive = [
        {"timestamp": r["timestamp"],
         "total_tokens": r["total_tokens"],
         "cost": calculate_cost(r["input_tokens"], r["output_tokens"], r["model_name"])}
        for r in rows if r["total_tokens"] > TOKEN_ALERT_THRESHOLD
    ]

    return {
        "total_requests":     len(rows),
        "total_input_tokens": sum(input_list),
        "total_output_tokens":sum(output_list),
        "total_tokens":       sum(total_list),
        "avg_input_tokens":   round(statistics.mean(input_list),  2),
        "avg_output_tokens":  round(statistics.mean(output_list), 2),
        "avg_total_tokens":   round(statistics.mean(total_list),  2),
        "max_tokens":         max(total_list),
        "min_tokens":         min(total_list),
        "total_cost_usd":     round(sum(costs), 6),
        "avg_cost_per_req":   round(statistics.mean(costs), 6),
        "expensive_requests": expensive,           # over threshold
        "expensive_count":    len(expensive),
    }


def get_daily_token_usage() -> list:
    """
    Group token usage by date.
    Shows daily cost trend — useful for the dashboard line chart.
    """
    rows = fetch_all("""
        SELECT
            substr(timestamp, 1, 10)    AS date,
            SUM(input_tokens)           AS daily_input,
            SUM(output_tokens)          AS daily_output,
            SUM(total_tokens)           AS daily_total,
            COUNT(*)                    AS request_count
        FROM llm_logs
        WHERE error_code = ''
        GROUP BY substr(timestamp, 1, 10)
        ORDER BY date
    """)

    # Add cost column
    for row in rows:
        row["estimated_cost"] = round(
            (row["daily_input"]  / 1000) * PRICING["default"]["input_per_1k"] +
            (row["daily_output"] / 1000) * PRICING["default"]["output_per_1k"],
            6
        )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  2. LATENCY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def percentile(data: list, pct: float) -> float:
    """Calculate the Nth percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * pct / 100)
    index = min(index, len(sorted_data) - 1)
    return round(sorted_data[index], 4)


def get_latency_stats() -> dict:
    """
    Compute detailed latency statistics.
    Includes percentiles and slow-query alerts.
    """
    rows = fetch_all("""
        SELECT retrieval_latency, llm_latency, end_to_end_latency, timestamp
        FROM llm_logs
        WHERE error_code = ''
        ORDER BY timestamp
    """)

    if not rows:
        return {"error": "No successful logs found"}

    retrieval_list = [r["retrieval_latency"]  for r in rows]
    llm_list       = [r["llm_latency"]        for r in rows]
    e2e_list       = [r["end_to_end_latency"] for r in rows]

    # Flag slow requests
    slow_requests = [
        {"timestamp": r["timestamp"], "e2e_latency": r["end_to_end_latency"]}
        for r in rows if r["end_to_end_latency"] > LATENCY_ALERT_THRESHOLD
    ]

    return {
        # RAG Retrieval latency (FAISS search)
        "retrieval": {
            "avg":  round(statistics.mean(retrieval_list), 4),
            "min":  round(min(retrieval_list),             4),
            "max":  round(max(retrieval_list),             4),
            "p95":  percentile(retrieval_list, 95),
        },
        # LLM API latency (Azure OpenAI call)
        "llm": {
            "avg":  round(statistics.mean(llm_list), 4),
            "min":  round(min(llm_list),             4),
            "max":  round(max(llm_list),             4),
            "p95":  percentile(llm_list, 95),
        },
        # Full end-to-end latency
        "end_to_end": {
            "avg":  round(statistics.mean(e2e_list), 4),
            "min":  round(min(e2e_list),             4),
            "max":  round(max(e2e_list),             4),
            "p95":  percentile(e2e_list, 95),
        },
        "slow_requests":     slow_requests,
        "slow_count":        len(slow_requests),
        "latency_threshold": LATENCY_ALERT_THRESHOLD,
        # Raw lists for time-series plotting in dashboard
        "e2e_series":        [{"timestamp": r["timestamp"],
                               "latency": r["end_to_end_latency"]} for r in rows],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. PROMPT VERSION TRACKING
# ══════════════════════════════════════════════════════════════════════════════

# Stored prompt version definitions
PROMPT_REGISTRY = {
    "v1": "Use provided context to answer the user's question. If the answer is not in the context, say so.",
    "v2": "Be concise and ensure evidence grounding. Cite specific details from the context. If not in context, explicitly state: 'This information is not available in the provided documents.'",
}


def get_version_comparison() -> dict:
    """
    Side-by-side comparison of v1 vs v2 prompt performance.
    Compares tokens, latency, and error rate per version.
    """
    versions = {}

    for version in ["v1", "v2"]:
        rows = fetch_all("""
            SELECT total_tokens, input_tokens, output_tokens,
                   llm_latency, end_to_end_latency, error_code
            FROM llm_logs
            WHERE prompt_version = ?
        """, (version,))

        if not rows:
            versions[version] = {"count": 0}
            continue

        successful = [r for r in rows if r["error_code"] == ""]
        errored    = [r for r in rows if r["error_code"] != ""]

        token_list   = [r["total_tokens"]        for r in successful]
        latency_list = [r["end_to_end_latency"]  for r in successful]
        costs        = [
            calculate_cost(r["input_tokens"], r["output_tokens"], "gpt-4o")
            for r in successful
        ]

        versions[version] = {
            "prompt_text":    PROMPT_REGISTRY.get(version, "Unknown"),
            "total_requests": len(rows),
            "success_count":  len(successful),
            "error_count":    len(errored),
            "error_rate_pct": round(len(errored) / len(rows) * 100, 2) if rows else 0,
            "avg_tokens":     round(statistics.mean(token_list),   2) if token_list   else 0,
            "avg_latency":    round(statistics.mean(latency_list), 4) if latency_list else 0,
            "total_cost":     round(sum(costs), 6),
            "avg_cost":       round(statistics.mean(costs), 6) if costs else 0,
        }

    # Determine winner for each metric
    if "v1" in versions and "v2" in versions:
        v1 = versions["v1"]
        v2 = versions["v2"]
        comparison = {
            "cheaper_version":  "v1" if v1["avg_cost"]    <= v2["avg_cost"]    else "v2",
            "faster_version":   "v1" if v1["avg_latency"] <= v2["avg_latency"] else "v2",
            "lower_error_rate": "v1" if v1["error_rate_pct"] <= v2["error_rate_pct"] else "v2",
        }
        versions["comparison"] = comparison

    return versions


# ══════════════════════════════════════════════════════════════════════════════
#  Save Report to File
# ══════════════════════════════════════════════════════════════════════════════

def save_report(token_stats: dict, latency_stats: dict, version_stats: dict):
    """Save all tracking results to a JSON report file."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "generated_at":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "token_stats":    token_stats,
        "latency_stats":  latency_stats,
        "version_stats":  version_stats,
        "daily_usage":    get_daily_token_usage(),
    }
    path = os.path.join(REPORTS_DIR, "task3_tracking_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Print Report to Terminal
# ══════════════════════════════════════════════════════════════════════════════

def print_report(token_stats: dict, latency_stats: dict, version_stats: dict):

    print("\n" + "═" * 60)
    print("  TASK 3 — TOKEN, LATENCY & PROMPT VERSION REPORT")
    print("═" * 60)

    # ── Token Section ──────────────────────────────────────────────────────────
    print("\n TOKEN USAGE")
    print("─" * 40)
    print(f"  Total requests         : {token_stats.get('total_requests', 0)}")
    print(f"  Total tokens consumed  : {token_stats.get('total_tokens', 0):,}")
    print(f"  Avg input tokens       : {token_stats.get('avg_input_tokens', 0):.1f}")
    print(f"  Avg output tokens      : {token_stats.get('avg_output_tokens', 0):.1f}")
    print(f"  Avg total tokens       : {token_stats.get('avg_total_tokens', 0):.1f}")
    print(f"  Max tokens (1 request) : {token_stats.get('max_tokens', 0):,}")
    print(f"  Total cost (USD)       : ${token_stats.get('total_cost_usd', 0):.4f}")
    print(f"  Avg cost per request   : ${token_stats.get('avg_cost_per_req', 0):.6f}")

    expensive = token_stats.get("expensive_count", 0)
    if expensive:
        print(f"\n ALERT: {expensive} request(s) exceeded {TOKEN_ALERT_THRESHOLD} tokens")

    # ── Latency Section ────────────────────────────────────────────────────────
    print("\n⏱LATENCY TRACKING")
    print("─" * 40)

    for label, key in [("RAG Retrieval", "retrieval"),
                       ("LLM API",       "llm"),
                       ("End-to-End",    "end_to_end")]:
        stats = latency_stats.get(key, {})
        print(f"\n  {label}:")
        print(f"    Avg : {stats.get('avg', 0):.3f}s")
        print(f"    Min : {stats.get('min', 0):.3f}s")
        print(f"    Max : {stats.get('max', 0):.3f}s")
        print(f"    P95 : {stats.get('p95', 0):.3f}s")

    slow = latency_stats.get("slow_count", 0)
    if slow:
        print(f"\n  ALERT: {slow} request(s) exceeded {LATENCY_ALERT_THRESHOLD}s threshold")

    # ── Prompt Version Section ─────────────────────────────────────────────────
    print("\nPROMPT VERSION COMPARISON")
    print("─" * 40)

    for ver in ["v1", "v2"]:
        v = version_stats.get(ver, {})
        if not v.get("total_requests"):
            continue
        print(f"\n  Version {ver}:")
        print(f"    Requests    : {v.get('total_requests', 0)}")
        print(f"    Error rate  : {v.get('error_rate_pct', 0):.1f}%")
        print(f"    Avg tokens  : {v.get('avg_tokens', 0):.1f}")
        print(f"    Avg latency : {v.get('avg_latency', 0):.3f}s")
        print(f"    Avg cost    : ${v.get('avg_cost', 0):.6f}")

    comparison = version_stats.get("comparison", {})
    if comparison:
        print(f"\n  Winner — Cheapest  : {comparison.get('cheaper_version', 'N/A')}")
        print(f"  Winner — Fastest   : {comparison.get('faster_version',  'N/A')}")
        print(f"  Winner — Reliable  : {comparison.get('lower_error_rate','N/A')}")

    print("\n" + "═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def run_tracking():
    print("Running Task 3: Token, Latency & Prompt Version Tracker...")

    token_stats   = get_token_stats()
    latency_stats = get_latency_stats()
    version_stats = get_version_comparison()

    print_report(token_stats, latency_stats, version_stats)
    save_report(token_stats, latency_stats, version_stats)

    print("\nCompleted")
    return token_stats, latency_stats, version_stats


if __name__ == "__main__":
    run_tracking()
