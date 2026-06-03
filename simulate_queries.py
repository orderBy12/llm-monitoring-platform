"""
--------------------------------------
This script:
1. Reads all 105 prompts from data/prompts.csv
2. Fires each one through the RAG pipeline
3. Saves all results to logs/rag_logs.csv
4. Prints a progress summary
"""

import os
import csv
import json
import time
import random
import pandas as pd
from dotenv import load_dotenv
from rag_pipeline import load_vectorstore, load_llm, rag_query

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROMPTS_FILE = "data/prompts.csv"
LOGS_DIR     = "logs"
LOGS_FILE    = os.path.join(LOGS_DIR, "rag_logs.csv")

# ── Settings ───────────────────────────────────────────────────────────────────
DELAY_BETWEEN_REQUESTS = 1.0   # Seconds between API calls (avoids rate limiting)
PROMPT_VERSIONS        = ["v1", "v2"]  # Alternate between prompt versions


def save_log_entry(log_entry: dict, writer, file_handle):
    """Write one log entry to the CSV file."""
    # Flatten retrieved_context (list of dicts) → JSON string for CSV storage
    row = {
        "request_id":         log_entry["request_id"],
        "timestamp":          log_entry["timestamp"],
        "user_query":         log_entry["user_query"],
        "retrieved_sources":  json.dumps([c["source"] for c in log_entry["retrieved_context"]]),
        "retrieved_scores":   json.dumps([c["score"]  for c in log_entry["retrieved_context"]]),
        "model_name":         log_entry["model_name"],
        "prompt_version":     log_entry["prompt_version"],
        "model_response":     log_entry["model_response"],
        "input_tokens":       log_entry["input_tokens"],
        "output_tokens":      log_entry["output_tokens"],
        "total_tokens":       log_entry["total_tokens"],
        "retrieval_latency":  log_entry["retrieval_latency"],
        "llm_latency":        log_entry["llm_latency"],
        "end_to_end_latency": log_entry["end_to_end_latency"],
        "top_k":              log_entry["top_k"],
        "safety_flag":        log_entry["safety_flag"],
        "error_code":         log_entry["error_code"] or "",
    }
    writer.writerow(row)
    file_handle.flush()   # Write immediately so we don't lose data on crash


def run_simulation():
    print("=" * 65)
    print("  SIMULATION: Firing 105 prompts through the RAG Pipeline")
    print("=" * 65)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\n📦 Loading FAISS index and Azure OpenAI models...")
    vectorstore = load_vectorstore()
    llm         = load_llm()
    print("Models loaded\n")

    # ── Load prompts ───────────────────────────────────────────────────────────
    # prompts_df = pd.read_csv(PROMPTS_FILE)
    prompts_df = pd.read_csv(
    PROMPTS_FILE,
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip'
)

    total      = len(prompts_df)
    print(f"Loaded {total} prompts from {PROMPTS_FILE}")

    # ── Prepare log file ───────────────────────────────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    fieldnames = [
        "request_id", "timestamp", "user_query",
        "retrieved_sources", "retrieved_scores",
        "model_name", "prompt_version", "model_response",
        "input_tokens", "output_tokens", "total_tokens",
        "retrieval_latency", "llm_latency", "end_to_end_latency",
        "top_k", "safety_flag", "error_code",
    ]

    # Counters for the summary
    success_count = 0
    error_count   = 0
    total_tokens  = 0

    print(f"\n Starting simulation — logs saved to: {LOGS_FILE}\n")
    print("-" * 65)

    with open(LOGS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in prompts_df.iterrows():
            prompt_id    = row["id"]
            query        = row["prompt"]

            # Alternate prompt versions: odd prompts use v1, even use v2
            prompt_version = "v1" if (i % 2 == 0) else "v2"

            print(f"[{i+1:3d}/{total}] Querying: {query[:65]}...")

            # ── Run the RAG query ──────────────────────────────────────────────
            result = rag_query(
                query          = query,
                vectorstore    = vectorstore,
                llm            = llm,
                top_k          = 3,
                prompt_version = prompt_version,
            )

            # ── Save to CSV ────────────────────────────────────────────────────
            save_log_entry(result, writer, f)

            # ── Update counters ────────────────────────────────────────────────
            if result["error_code"]:
                error_count += 1
                print(f"         Error: {result['error_code']}")
            else:
                success_count += 1
                total_tokens  += result["total_tokens"]
                print(f"         Tokens: {result['total_tokens']} | "
                      f"Latency: {result['end_to_end_latency']}s | "
                      f"Prompt: {prompt_version}")

            # ── Rate limiting: wait between requests ───────────────────────────
            if i < total - 1:   # Don't wait after last request
                time.sleep(DELAY_BETWEEN_REQUESTS)

    # ── Print Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SIMULATION COMPLETE")
    print("=" * 65)
    print(f"  Total prompts processed : {total}")
    print(f"  Successful responses    : {success_count}")
    print(f"  Errors                  : {error_count}")
    print(f"  Total tokens consumed   : {total_tokens:,}")
    print(f"  Estimated cost (approx) : ${total_tokens * 0.000002:.4f}")
    print(f"  Log file saved at       : {LOGS_FILE}")
    print("=" * 65)
    print("\nDone")


if __name__ == "__main__":
    run_simulation()
