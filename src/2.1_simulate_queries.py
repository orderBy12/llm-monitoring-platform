"""
Simulate Queries - Task 1 Final Step
--------------------------------------
This script:
1. Reads all 105 prompts from data/prompts.csv
2. Fires each one through the RAG pipeline
3. Saves all results to logs/rag_logs.csv
4. Prints a progress summary

This generates the data that Tasks 2–7 will monitor and analyze.
"""
"""
Simulate Queries - Updated for Task 2
--------------------------------------
Same as Task 1 version BUT now uses LLMLogger middleware.
Manual CSV writing is removed — the middleware handles everything.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from rag_pipeline import load_vectorstore, load_llm, rag_query
from logger_middleware import LLMLogger        # Task 2 addition

load_dotenv()

# Paths
PROMPTS_FILE = "data/prompts.csv"

# Settings
DELAY_BETWEEN_REQUESTS = 1.0   # seconds between API calls (avoids rate limiting)


def run_simulation():
    print("=" * 65)
    print("  SIMULATION: Firing prompts through the RAG Pipeline")
    print("=" * 65)

    # Load models
    print("\n Loading FAISS index and Azure OpenAI models...")
    vectorstore = load_vectorstore()
    llm         = load_llm()
    print("Models loaded")

    # Initialize middleware logger
    logger = LLMLogger()      # All logging handled here from now on

    # Load prompts
    prompts_df = pd.read_csv(
        PROMPTS_FILE,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip'
    )
    total = len(prompts_df)
    print(f"\n Loaded {total} prompts\n")
    print("-" * 65)

    # Counters for summary
    success_count = 0
    error_count   = 0

    for i, row in prompts_df.iterrows():
        query          = row["prompt"]
        prompt_version = "v1" if (i % 2 == 0) else "v2"

        print(f"[{i+1:3d}/{total}] {query[:65]}...")

        # Run RAG query
        result = rag_query(
            query          = query,
            vectorstore    = vectorstore,
            llm            = llm,
            top_k          = 3,
            prompt_version = prompt_version,
        )

        # Log via middleware
        logger.log(result)

        # Print progress
        if result["error_code"]:
            error_count += 1
            print(f"         Error: {result['error_code']}")
        else:
            success_count += 1
            print(f"         Tokens: {result['total_tokens']} | "
                  f"Latency: {result['end_to_end_latency']}s | "
                  f"Prompt: {prompt_version}")

        if i < total - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Final summary from middleware
    stats = logger.get_summary()

    print("\n" + "=" * 65)
    print("  SIMULATION COMPLETE - LOG SUMMARY")
    print("=" * 65)
    print(f"  Total requests      : {stats.get('total_requests', 0)}")
    print(f"  Successful          : {success_count}")
    print(f"  Errors              : {stats.get('error_count', 0)}")
    print(f"  Safety flags        : {stats.get('safety_flag_count', 0)}")
    print(f"  Total tokens        : {stats.get('total_tokens', 0):,}")
    print(f"  Avg tokens/request  : {stats.get('avg_tokens', 0):.1f}")
    print(f"  Avg LLM latency     : {stats.get('avg_llm_latency', 0):.3f}s")
    print(f"  Avg E2E latency     : {stats.get('avg_e2e_latency', 0):.3f}s")
    print(f"  Estimated cost      : ${stats.get('estimated_cost_usd', 0):.4f}")
    print(f"  Prompt v1 requests  : {stats.get('v1_count', 0)}")
    print(f"  Prompt v2 requests  : {stats.get('v2_count', 0)}")
    print("=" * 65)

    logger.close()
    print("\nTask 2 Complete! Logs saved to logs/rag_logs.csv and logs/monitoring.db")


if __name__ == "__main__":
    run_simulation()
