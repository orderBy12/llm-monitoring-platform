"""
config.py — Central Configuration
------------------------------------
Single source of truth for all settings across the project.
Every other file imports from here instead of having hardcoded values.

To change any setting (model, threshold, path), change it HERE only.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  Azure OpenAI
# ══════════════════════════════════════════════════════════════════════════════

AZURE_API_KEY            = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT           = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION        = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
CHAT_DEPLOYMENT          = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBEDDING_DEPLOYMENT     = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# ══════════════════════════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data
DATA_DIR         = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR    = os.path.join(DATA_DIR, "documents")
PROMPTS_FILE     = os.path.join(DATA_DIR, "prompts.csv")
REF_QUERY_FILE   = os.path.join(DATA_DIR, "reference_queries.json")
# Vector store
VECTORSTORE_DIR  = os.path.join(BASE_DIR, "vectorstore", "faiss_index")
# Logs
LOGS_DIR         = os.path.join(BASE_DIR, "logs")
DB_FILE          = os.path.join(LOGS_DIR, "monitoring.db")
CSV_LOG_FILE     = os.path.join(LOGS_DIR, "rag_logs.csv")
EDGE_CASES_LOG   = os.path.join(LOGS_DIR, "edge_cases_log.csv")
BASELINE_FILE    = os.path.join(LOGS_DIR, "drift_baseline.json")
# Reports
REPORTS_DIR      = os.path.join(LOGS_DIR, "reports")

# ══════════════════════════════════════════════════════════════════════════════
#  RAG Pipeline Settings
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE          = 800     # characters per document chunk
CHUNK_OVERLAP       = 100     # overlap between adjacent chunks
TOP_K               = 3       # number of chunks to retrieve per query
LLM_TEMPERATURE     = 0.3     # 0 = deterministic, 1 = creative
LLM_MAX_TOKENS      = 800     # max response length
EMBED_DELAY         = 0.1     # seconds between embedding API calls
QUERY_DELAY         = 1.0     # seconds between simulation queries

# ── Prompt Version Registry ────────────────────────────────────────────────────
PROMPT_VERSIONS = {
    "v1": (
        "You are a helpful AI assistant. Use the provided context to answer "
        "the user's question. If the answer is not in the context, say so."
    ),
    "v2": (
        "You are an expert AI assistant. Be concise and ensure your answer is "
        "fully grounded in the provided evidence. Cite specific details from "
        "the context. If the answer is not in the context, explicitly state: "
        "'This information is not available in the provided documents.'"
    ),
}
DEFAULT_PROMPT_VERSION = "v1"

# ── Pricing (USD per 1,000 tokens) ─────────────────────────────────────────────
PRICING = {
    "gpt-4o":       {"input_per_1k": 0.005,  "output_per_1k": 0.015},
    "gpt-35-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    "default":      {"input_per_1k": 0.005,  "output_per_1k": 0.015},
}


# ── Alert Thresholds ───────────────────────────────────────────────────────────
LATENCY_SLOW_THRESHOLD     = 10.0    # seconds — flag if e2e > this
TOKEN_HIGH_THRESHOLD       = 2000    # tokens  — flag if single request > this
DAILY_COST_ALERT_USD       = 0.10    # USD     — alert if daily cost > this
# RAG evaluation thresholds
RAG_MIN_CONTEXT_RELEVANCE  = 0.5
RAG_MIN_FAITHFULNESS       = 0.5
RAG_MIN_GROUNDEDNESS       = 0.5
RAG_HALLUCINATION_THRESHOLD= 0.4     # faithfulness below this = hallucination
RAG_LOW_QUALITY_THRESHOLD  = 0.5

# Drift thresholds
DRIFT_WARNING_THRESHOLD    = 0.10
DRIFT_ANOMALY_THRESHOLD    = 0.20

# Safety keywords (basic filter)
SAFETY_KEYWORDS = ["hack", "weapon", "illegal", "kill", "bomb", "bypass", "jailbreak"]

# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation Settings
# ══════════════════════════════════════════════════════════════════════════════

EVAL_LIMIT           = 50     # max requests to evaluate in Task 4 (saves cost)
EVAL_LLM_TEMPERATURE = 0.0    # deterministic for scoring

# ══════════════════════════════════════════════════════════════════════════════
#  Azure OpenAI Pricing (per 1000 tokens, USD)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PRICING = {
    "gpt-4o": {
        "input_per_1k":  0.005,
        "output_per_1k": 0.015,
    },
    "gpt-35-turbo": {
        "input_per_1k":  0.0005,
        "output_per_1k": 0.0015,
    },
    "default": {
        "input_per_1k":  0.005,
        "output_per_1k": 0.015,
    },
}


def calculate_cost(input_tokens: int, output_tokens: int,
                   model: str = None) -> float:
    """Calculate USD cost for one request."""
    pricing = MODEL_PRICING.get(model or CHAT_DEPLOYMENT, MODEL_PRICING["default"])
    return round(
        (input_tokens  / 1000) * pricing["input_per_1k"] +
        (output_tokens / 1000) * pricing["output_per_1k"],
        6
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Validation: warn if required env vars are missing
# ══════════════════════════════════════════════════════════════════════════════

def validate_env() -> bool:
    """Check all required environment variables are set."""
    required = {
        "AZURE_OPENAI_API_KEY":           AZURE_API_KEY,
        "AZURE_OPENAI_ENDPOINT":          AZURE_ENDPOINT,
        "AZURE_OPENAI_CHAT_DEPLOYMENT":   CHAT_DEPLOYMENT,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(" Missing environment variables:")
        for m in missing:
            print(f"   {m}")
        print("\n   Copy .env.example to .env and fill in your values.")
        return False
    return True


if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"  Chat model       : {CHAT_DEPLOYMENT}")
    print(f"  Embedding model  : {EMBEDDING_DEPLOYMENT}")
    print(f"  DB file          : {DB_FILE}")
    print(f"  Vector store     : {VECTORSTORE_DIR}")
    print(f"  Latency alert    : > {LATENCY_SLOW_THRESHOLD}s")
    print(f"  Drift alert      : > {DRIFT_ANOMALY_THRESHOLD}")
    valid = validate_env()
    print(f"\n  Env valid        : {' Yes' if valid else ' No'}")
