"""
Task 7 Part B: System Evaluation & Final Analysis
---------------------------------------------------
Reads all data from SQLite across Tasks 2–5 and produces a
comprehensive evaluation report covering:

  1. TOKEN & COST TRENDS     — usage over time, expensive requests
  2. LATENCY ANALYSIS        — slow queries, percentile breakdown
  3. RAG SCORE FLUCTUATIONS  — per-query score trends, low quality detection
  4. DRIFT ANOMALIES         — embedding stability summary
  5. SAFETY & ERROR PATTERNS — flag rates, error classification
  6. HALLUCINATION CASES     — faithfulness < 0.4 requests
  7. OVERALL SYSTEM HEALTH   — composite health score

Outputs:
  - Terminal report
  - logs/reports/task7_final_evaluation.json
  - logs/reports/task7_summary.txt  (plain text, easy to copy into report)
"""

import os
import json
import sqlite3
import statistics
from datetime import datetime
import config


# ══════════════════════════════════════════════════════════════════════════════
#  Database helpers
# ══════════════════════════════════════════════════════════════════════════════

def query_db(sql: str, params: tuple = ()) -> list:
    if not os.path.isfile(config.DB_FILE):
        return []
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def table_exists(table: str) -> bool:
    rows = query_db(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return len(rows) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1: Token & Cost Trends
# ══════════════════════════════════════════════════════════════════════════════

def analyse_tokens() -> dict:
    rows = query_db("""
        SELECT input_tokens, output_tokens, total_tokens,
               model_name, prompt_version, timestamp
        FROM llm_logs
        WHERE error_code = ''
    """)
    if not rows:
        return {}

    totals   = [r["total_tokens"]  for r in rows]
    inputs   = [r["input_tokens"]  for r in rows]
    outputs  = [r["output_tokens"] for r in rows]

    # Cost per request (gpt-4o pricing)
    costs = [
        (r["input_tokens"]  / 1000 * 0.005) +
        (r["output_tokens"] / 1000 * 0.015)
        for r in rows
    ]

    # Prompt version breakdown
    v1_rows = [r for r in rows if r["prompt_version"] == "v1"]
    v2_rows = [r for r in rows if r["prompt_version"] == "v2"]

    high_token_requests = [r for r in rows if r["total_tokens"] > config.TOKEN_HIGH_THRESHOLD]

    return {
        "total_requests":     len(rows),
        "total_tokens":       sum(totals),
        "total_input":        sum(inputs),
        "total_output":       sum(outputs),
        "avg_tokens":         round(statistics.mean(totals), 1),
        "max_tokens":         max(totals),
        "min_tokens":         min(totals),
        "std_tokens":         round(statistics.stdev(totals), 1) if len(totals) > 1 else 0,
        "total_cost_usd":     round(sum(costs), 4),
        "avg_cost_per_req":   round(statistics.mean(costs), 6),
        "v1_avg_tokens":      round(statistics.mean([r["total_tokens"] for r in v1_rows]), 1) if v1_rows else 0,
        "v2_avg_tokens":      round(statistics.mean([r["total_tokens"] for r in v2_rows]), 1) if v2_rows else 0,
        "high_token_count":   len(high_token_requests),
        "high_token_pct":     round(len(high_token_requests) / len(rows) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2: Latency Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_latency() -> dict:
    rows = query_db("""
        SELECT retrieval_latency, llm_latency, end_to_end_latency, timestamp
        FROM llm_logs
        WHERE error_code = ''
    """)
    if not rows:
        return {}

    def stats(values: list) -> dict:
        s = sorted(values)
        n = len(s)
        return {
            "avg":  round(statistics.mean(s), 4),
            "min":  round(min(s),             4),
            "max":  round(max(s),             4),
            "p50":  round(s[int(n * 0.50)],   4),
            "p95":  round(s[int(n * 0.95)],   4),
            "p99":  round(s[min(int(n * 0.99), n - 1)], 4),
        }

    e2e  = [r["end_to_end_latency"] for r in rows]
    llm  = [r["llm_latency"]        for r in rows]
    retr = [r["retrieval_latency"]  for r in rows]

    slow_reqs = [r for r in rows if r["end_to_end_latency"] > config.LATENCY_SLOW_THRESHOLD]

    return {
        "end_to_end":   stats(e2e),
        "llm_api":      stats(llm),
        "retrieval":    stats(retr),
        "slow_count":   len(slow_reqs),
        "slow_pct":     round(len(slow_reqs) / len(rows) * 100, 1),
        "llm_share_pct":round(statistics.mean(llm)  / statistics.mean(e2e) * 100, 1),
        "retr_share_pct":round(statistics.mean(retr) / statistics.mean(e2e) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3: RAG Score Fluctuations
# ══════════════════════════════════════════════════════════════════════════════

def analyse_rag_scores() -> dict:
    if not table_exists("rag_evaluations"):
        return {"available": False}

    rows = query_db("SELECT * FROM rag_evaluations")
    if not rows:
        return {"available": False}

    cr    = [r["context_relevance_score"] for r in rows]
    faith = [r["faithfulness_score"]      for r in rows]
    gr    = [r["groundedness_score"]      for r in rows]
    ovr   = [r["overall_score"]           for r in rows]

    low_quality  = [r for r in rows if r["overall_score"]     < config.RAG_LOW_QUALITY_THRESHOLD]
    hallucinated = [r for r in rows if r["faithfulness_score"] < config.RAG_HALLUCINATION_THRESHOLD]
    flagged      = [r for r in rows if r["hallucination_detected"] == 1]

    return {
        "available":        True,
        "total_evaluated":  len(rows),
        "avg_context_rel":  round(statistics.mean(cr),    3),
        "avg_faithfulness": round(statistics.mean(faith), 3),
        "avg_groundedness": round(statistics.mean(gr),    3),
        "avg_overall":      round(statistics.mean(ovr),   3),
        "std_overall":      round(statistics.stdev(ovr),  3) if len(ovr) > 1 else 0,
        "low_quality_count":len(low_quality),
        "low_quality_pct":  round(len(low_quality)  / len(rows) * 100, 1),
        "hallucination_count":  len(hallucinated),
        "hallucination_pct":    round(len(hallucinated) / len(rows) * 100, 1),
        "flagged_hallu_count":  len(flagged),
        "worst_queries": [
            {
                "query":   r["user_query"][:80],
                "overall": r["overall_score"],
                "faith":   r["faithfulness_score"],
            }
            for r in sorted(rows, key=lambda x: x["overall_score"])[:5]
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4: Drift Anomalies
# ══════════════════════════════════════════════════════════════════════════════

def analyse_drift() -> dict:
    if not table_exists("drift_logs"):
        return {"available": False}

    rows = query_db("SELECT * FROM drift_logs")
    if not rows:
        return {"available": False}

    scores    = [r["drift_score"] for r in rows]
    anomalies = [r for r in rows if r["status"] == "ANOMALY"]
    warnings  = [r for r in rows if r["status"] == "WARNING"]
    healthy   = [r for r in rows if r["status"] == "HEALTHY"]

    # Count unique measurement runs
    runs = list(set(r["measured_at"] for r in rows))

    return {
        "available":       True,
        "total_runs":      len(runs),
        "total_queries":   len(rows),
        "avg_drift":       round(statistics.mean(scores), 4),
        "max_drift":       round(max(scores),             4),
        "healthy_count":   len(healthy),
        "warning_count":   len(warnings),
        "anomaly_count":   len(anomalies),
        "anomaly_pct":     round(len(anomalies) / len(rows) * 100, 1),
        "overall_health":  (
            "ANOMALY" if anomalies else
            "WARNING" if warnings  else
            "HEALTHY"
        ),
        "anomalous_queries": [
            {"query": r["query"][:70], "drift": r["drift_score"]}
            for r in anomalies
        ][:5],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5: Safety & Error Patterns
# ══════════════════════════════════════════════════════════════════════════════

def analyse_safety_errors() -> dict:
    all_rows    = query_db("SELECT * FROM llm_logs")
    if not all_rows:
        return {}

    total       = len(all_rows)
    errors      = [r for r in all_rows if r["error_code"]]
    safety_rows = [r for r in all_rows if r["safety_flag"] == 1]

    # Group errors by type
    error_types = {}
    for r in errors:
        ec = r["error_code"]
        error_types[ec] = error_types.get(ec, 0) + 1

    # Group safety flags by prompt version
    safety_by_version = {}
    for r in safety_rows:
        v = r["prompt_version"]
        safety_by_version[v] = safety_by_version.get(v, 0) + 1

    return {
        "total_requests":   total,
        "error_count":      len(errors),
        "error_rate_pct":   round(len(errors)      / total * 100, 2),
        "safety_count":     len(safety_rows),
        "safety_rate_pct":  round(len(safety_rows) / total * 100, 2),
        "error_breakdown":  error_types,
        "safety_by_version":safety_by_version,
        "success_rate_pct": round((total - len(errors)) / total * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Section 6: Overall System Health Score
# ══════════════════════════════════════════════════════════════════════════════

def compute_health_score(tokens, latency, rag, drift, safety) -> dict:
    """
    Composite system health score 0–100.
    Weighted average of 5 dimensions.
    """
    scores = {}

    # Error rate score (lower error rate = higher score)
    err_rate = safety.get("error_rate_pct", 0)
    scores["reliability"] = max(0, 100 - err_rate * 10)

    # Latency score (lower p95 = higher score, baseline 5s)
    p95 = latency.get("end_to_end", {}).get("p95", 5.0)
    scores["latency"] = max(0, min(100, (5.0 / p95) * 100)) if p95 > 0 else 100

    # RAG quality score
    if rag.get("available"):
        scores["rag_quality"] = rag.get("avg_overall", 0) * 100
    else:
        scores["rag_quality"] = 50   # neutral if not evaluated

    # Drift score (lower drift = higher score)
    if drift.get("available"):
        avg_drift = drift.get("avg_drift", 0)
        scores["drift_stability"] = max(0, (1 - avg_drift / 0.20) * 100)
    else:
        scores["drift_stability"] = 50

    # Token efficiency (no runaway requests)
    high_pct = tokens.get("high_token_pct", 0)
    scores["token_efficiency"] = max(0, 100 - high_pct * 2)

    # Weighted composite
    weights = {
        "reliability":     0.30,
        "latency":         0.25,
        "rag_quality":     0.20,
        "drift_stability": 0.15,
        "token_efficiency":0.10,
    }
    composite = sum(scores[k] * weights[k] for k in scores)

    grade = (
        "A" if composite >= 85 else
        "B" if composite >= 70 else
        "C" if composite >= 55 else
        "D"
    )

    return {
        "composite_score": round(composite, 1),
        "grade":           grade,
        "dimension_scores":scores,
        "weights":         weights,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Print & Save Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(tokens, latency, rag, drift, safety, health):
    print("\n" + "═" * 62)
    print("  TASK 7 — COMPREHENSIVE SYSTEM EVALUATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 62)

    # ── Token & Cost ───────────────────────────────────────────────────────────
    print("\nTOKEN & COST ANALYSIS")
    print("─" * 40)
    print(f"  Total requests       : {tokens.get('total_requests', 0):,}")
    print(f"  Total tokens         : {tokens.get('total_tokens', 0):,}")
    print(f"  Avg tokens/request   : {tokens.get('avg_tokens', 0):.1f}")
    print(f"  Max tokens (1 req)   : {tokens.get('max_tokens', 0):,}")
    print(f"  Std dev tokens       : {tokens.get('std_tokens', 0):.1f}")
    print(f"  Total cost (USD)     : ${tokens.get('total_cost_usd', 0):.4f}")
    print(f"  High-token requests  : {tokens.get('high_token_count', 0)} ({tokens.get('high_token_pct', 0):.1f}%)")
    print(f"  v1 avg tokens        : {tokens.get('v1_avg_tokens', 0):.1f}")
    print(f"  v2 avg tokens        : {tokens.get('v2_avg_tokens', 0):.1f}")

    # ── Latency ────────────────────────────────────────────────────────────────
    print("\nLATENCY ANALYSIS")
    print("─" * 40)
    for label, key in [("End-to-End", "end_to_end"),
                       ("LLM API",    "llm_api"),
                       ("Retrieval",  "retrieval")]:
        s = latency.get(key, {})
        print(f"  {label}: avg={s.get('avg',0):.3f}s  "
              f"p50={s.get('p50',0):.3f}s  "
              f"p95={s.get('p95',0):.3f}s  "
              f"max={s.get('max',0):.3f}s")
    print(f"\n LLM share of E2E     : {latency.get('llm_share_pct', 0):.1f}%")
    print(f" Retrieval share      : {latency.get('retr_share_pct', 0):.1f}%")
    print(f" Slow requests (>{config.LATENCY_SLOW_THRESHOLD}s): {latency.get('slow_count', 0)} ({latency.get('slow_pct', 0):.1f}%)")

    # ── RAG Evaluation ─────────────────────────────────────────────────────────
    print("\n RAG EVALUATION METRICS")
    print("─" * 40)
    if rag.get("available"):
        print(f"  Requests evaluated   : {rag.get('total_evaluated', 0)}")
        print(f"  Avg context relevance: {rag.get('avg_context_rel', 0):.3f}")
        print(f"  Avg faithfulness     : {rag.get('avg_faithfulness', 0):.3f}")
        print(f"  Avg groundedness     : {rag.get('avg_groundedness', 0):.3f}")
        print(f"  Avg overall score    : {rag.get('avg_overall', 0):.3f} (±{rag.get('std_overall',0):.3f})")
        print(f"  Low quality (<0.5)   : {rag.get('low_quality_count', 0)} ({rag.get('low_quality_pct', 0):.1f}%)")
        print(f"  Hallucinations found : {rag.get('hallucination_count', 0)} ({rag.get('hallucination_pct', 0):.1f}%)")
        if rag.get("worst_queries"):
            print(f"\n  Worst performing queries:")
            for q in rag["worst_queries"]:
                print(f"    overall={q['overall']:.2f}  faith={q['faith']:.2f}  \"{q['query'][:55]}\"")
    else:
        print("No evaluations found. Run rag_evaluator.py first.")

    # ── Drift ──────────────────────────────────────────────────────────────────
    print("\n EMBEDDING DRIFT")
    print("─" * 40)
    if drift.get("available"):
        status_icon = {"HEALTHY": "Stable", "WARNING": "Needs attention", "ANOMALY": "Critical"}
        print(f"  Measurement runs     : {drift.get('total_runs', 0)}")
        print(f"  Avg drift score      : {drift.get('avg_drift', 0):.4f}")
        print(f"  Max drift score      : {drift.get('max_drift', 0):.4f}")
        print(f"  Healthy queries      : {drift.get('healthy_count', 0)}")
        print(f"  Warning queries      : {drift.get('warning_count', 0)}")
        print(f"  Anomaly queries      : {drift.get('anomaly_count', 0)} ({drift.get('anomaly_pct', 0):.1f}%)")
        h = drift.get("overall_health", "UNKNOWN")
        status_label = status_icon.get(h, "UNKNOWN")
        print(f"  Overall drift status : {h} - {status_label}")
        # print(f"  Overall drift status : {status_icon.get(h, '')} {h}")
    else:
        print(" No drift data. Run drift_detector.py first.")

    # ── Safety & Errors ────────────────────────────────────────────────────────
    print("\nSAFETY & ERROR ANALYSIS")
    print("─" * 40)
    print(f"  Total requests       : {safety.get('total_requests', 0):,}")
    print(f"  Success rate         : {safety.get('success_rate_pct', 0):.1f}%")
    print(f"  Error rate           : {safety.get('error_rate_pct', 0):.2f}%")
    print(f"  Safety flag rate     : {safety.get('safety_rate_pct', 0):.2f}%")
    if safety.get("error_breakdown"):
        print(f"  Error types:")
        for etype, count in safety["error_breakdown"].items():
            print(f"    {etype}: {count}")

    # ── Health Score ───────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  OVERALL SYSTEM HEALTH SCORE")
    print("═" * 62)
    grade = health.get("grade", "N/A")
    score = health.get("composite_score", 0)
    grade_color = {"A": "Excellent", "B": "Good", "C": "Fair", "D": "Poor"}.get(grade, "Unknown")
    # print(f"\n  {grade_color} Grade: {grade}   Score: {score}/100\n")
    print(f"\n  Grade: {grade} ({grade_color})   Score: {score}/100\n")
    for dim, val in health.get("dimension_scores", {}).items():
        bar = "█" * int(val / 5)
        print(f"  {dim:<20} {val:5.1f}/100  {bar}")
    print("\n" + "═" * 62)


def save_reports(tokens, latency, rag, drift, safety, health):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ── Full JSON ──────────────────────────────────────────────────────────────
    full_report = {
        "generated_at":   timestamp,
        "token_analysis": tokens,
        "latency_analysis":latency,
        "rag_evaluation": rag,
        "drift_analysis": drift,
        "safety_errors":  safety,
        "health_score":   health,
    }
    json_path = os.path.join(config.REPORTS_DIR, "task7_final_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    # ── Plain text summary (easy to paste into your report document) ───────────
    txt_path = os.path.join(config.REPORTS_DIR, "task7_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"LLM MONITORING PLATFORM — SYSTEM EVALUATION SUMMARY\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"HEALTH SCORE: {health.get('composite_score', 0)}/100 (Grade {health.get('grade','N/A')})\n\n")
        f.write(f"REQUESTS: {tokens.get('total_requests', 0)} total | "
                f"{safety.get('success_rate_pct', 0):.1f}% success | "
                f"{safety.get('error_rate_pct', 0):.2f}% errors\n\n")
        f.write(f"TOKENS: {tokens.get('total_tokens', 0):,} total | "
                f"avg {tokens.get('avg_tokens', 0):.0f}/req | "
                f"cost ${tokens.get('total_cost_usd', 0):.4f}\n\n")
        f.write(f"LATENCY: avg {latency.get('end_to_end', {}).get('avg', 0):.3f}s | "
                f"p95 {latency.get('end_to_end', {}).get('p95', 0):.3f}s | "
                f"{latency.get('slow_count', 0)} slow requests\n\n")
        if rag.get("available"):
            f.write(f"RAG SCORES: context_rel={rag.get('avg_context_rel',0):.3f} | "
                    f"faithfulness={rag.get('avg_faithfulness',0):.3f} | "
                    f"groundedness={rag.get('avg_groundedness',0):.3f}\n")
            f.write(f"HALLUCINATIONS: {rag.get('hallucination_count',0)} detected\n\n")
        if drift.get("available"):
            f.write(f"DRIFT: avg={drift.get('avg_drift',0):.4f} | "
                    f"status={drift.get('overall_health','N/A')} | "
                    f"{drift.get('anomaly_count',0)} anomalies\n\n")
        f.write(f"SAFETY: {safety.get('safety_count',0)} flagged requests\n")

    print(f"\n  Reports saved:")
    print(f"     {json_path}")
    print(f"     {txt_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    print("Running System Evaluation...")

    tokens  = analyse_tokens()
    latency = analyse_latency()
    rag     = analyse_rag_scores()
    drift   = analyse_drift()
    safety  = analyse_safety_errors()
    health  = compute_health_score(tokens, latency, rag, drift, safety)

    print_report(tokens, latency, rag, drift, safety, health)
    save_reports(tokens, latency, rag, drift, safety, health)

    print("\nTask 7 Complete!")
    return health


if __name__ == "__main__":
    run_evaluation()
