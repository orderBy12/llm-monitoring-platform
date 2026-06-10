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
import config



def _fetch(sql, params=()):
    if not os.path.isfile(config.DB_FILE):
        return []
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def _pct(data, p):
    if not data: return 0.0
    s = sorted(data)
    return round(s[min(int(len(s)*p/100), len(s)-1)], 4)

# ── Token Stats ────────────────────────────────────────────────────────────────
def get_token_stats():
    rows = _fetch("SELECT input_tokens,output_tokens,total_tokens,model_name FROM llm_logs WHERE error_code=''")
    if not rows: return {}
    totals = [r["total_tokens"] for r in rows]
    costs  = [config.calculate_cost(r["input_tokens"], r["output_tokens"], r["model_name"]) for r in rows]
    high   = [r for r in rows if r["total_tokens"] > config.TOKEN_HIGH_THRESHOLD]
    return {
        "total_requests": len(rows),
        "total_tokens":   sum(totals),
        "avg_tokens":     round(statistics.mean(totals), 1),
        "max_tokens":     max(totals),
        "min_tokens":     min(totals),
        "total_cost_usd": round(sum(costs), 4),
        "avg_cost_per_req": round(statistics.mean(costs), 6),
        "high_token_count": len(high),
        "high_token_pct":   round(len(high)/len(rows)*100, 1),
    }

# ── Latency Stats ──────────────────────────────────────────────────────────────
def get_latency_stats():
    rows = _fetch("SELECT retrieval_latency,llm_latency,end_to_end_latency,timestamp FROM llm_logs WHERE error_code=''")
    if not rows: return {}
    def stats(vals):
        return {"avg":round(statistics.mean(vals),4),"min":round(min(vals),4),
                "max":round(max(vals),4),"p95":_pct(vals,95),"p99":_pct(vals,99)}
    e2e  = [r["end_to_end_latency"] for r in rows]
    slow = [r for r in rows if r["end_to_end_latency"] > config.LATENCY_SLOW_THRESHOLD]
    return {
        "retrieval":  stats([r["retrieval_latency"] for r in rows]),
        "llm":        stats([r["llm_latency"]        for r in rows]),
        "end_to_end": stats(e2e),
        "slow_count": len(slow),
        "slow_pct":   round(len(slow)/len(rows)*100, 1),
        "e2e_series": [{"timestamp":r["timestamp"],"latency":r["end_to_end_latency"]} for r in rows],
    }

# ── Version Comparison ─────────────────────────────────────────────────────────
def get_version_comparison():
    versions = {}
    for ver in ["v1","v2"]:
        rows = _fetch("SELECT total_tokens,input_tokens,output_tokens,llm_latency,end_to_end_latency,error_code FROM llm_logs WHERE prompt_version=?", (ver,))
        if not rows:
            versions[ver] = {"count":0}; continue
        ok  = [r for r in rows if r["error_code"]==""]
        err = [r for r in rows if r["error_code"]!=""]
        costs = [config.calculate_cost(r["input_tokens"],r["output_tokens"]) for r in ok]
        versions[ver] = {
            "prompt_text":    config.PROMPT_VERSIONS.get(ver,"")[:80],
            "total_requests": len(rows),
            "success_count":  len(ok),
            "error_count":    len(err),
            "error_rate_pct": round(len(err)/len(rows)*100, 2),
            "avg_tokens":     round(statistics.mean([r["total_tokens"] for r in ok]),2) if ok else 0,
            "avg_latency":    round(statistics.mean([r["end_to_end_latency"] for r in ok]),4) if ok else 0,
            "total_cost":     round(sum(costs),6),
            "avg_cost":       round(statistics.mean(costs),6) if costs else 0,
        }
    if "v1" in versions and "v2" in versions and versions["v1"].get("avg_cost") is not None:
        v1,v2 = versions["v1"], versions["v2"]
        versions["comparison"] = {
            "cheaper_version":  "v1" if v1["avg_cost"]    <= v2["avg_cost"]    else "v2",
            "faster_version":   "v1" if v1["avg_latency"] <= v2["avg_latency"] else "v2",
            "lower_error_rate": "v1" if v1["error_rate_pct"] <= v2["error_rate_pct"] else "v2",
        }
    return versions

def run_tracking():
    print("=" * 55)
    print("  TASK 3: Token, Latency & Version Tracking")
    print("=" * 55)
    tok  = get_token_stats()
    lat  = get_latency_stats()
    ver  = get_version_comparison()

    print(f"\n TOKEN USAGE")
    print(f"  Total: {tok.get('total_tokens',0):,} | Avg: {tok.get('avg_tokens',0):.0f}/req | Cost: ${tok.get('total_cost_usd',0):.4f}")
    if tok.get("high_token_count"):
        print(f"  Alert: {tok['high_token_count']} requests exceeded {config.TOKEN_HIGH_THRESHOLD} tokens")

    print(f"\n LATENCY (End-to-End)")
    e = lat.get("end_to_end",{})
    print(f"  avg={e.get('avg',0):.3f}s | p95={e.get('p95',0):.3f}s | max={e.get('max',0):.3f}s")
    if lat.get("slow_count"):
        print(f"  Alert: {lat['slow_count']} slow requests (>{config.LATENCY_SLOW_THRESHOLD}s)")

    print(f"\n PROMPT VERSION COMPARISON")
    for v in ["v1","v2"]:
        d = ver.get(v,{})
        if d.get("total_requests"):
            print(f"  {v}: requests={d['total_requests']} | avg_tokens={d['avg_tokens']:.0f} | avg_latency={d['avg_latency']:.3f}s | avg_cost=${d['avg_cost']:.6f}")
    if ver.get("comparison"):
        c = ver["comparison"]
        print(f"  Cheaper: {c['cheaper_version']} | Faster: {c['faster_version']}")

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = os.path.join(config.REPORTS_DIR, "task3_tracking_report.json")
    import json
    with open(path,"w") as f:
        json.dump({"token_stats":tok,"latency_stats":lat,"version_stats":ver,
                   "generated_at":datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    print(f"\n  Report: {path}")
    print("\Done.")

if __name__ == "__main__":
    run_tracking()
