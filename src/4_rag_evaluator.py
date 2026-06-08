"""
--------------------------------
Computes 3 core quality metrics for every RAG response using
an LLM-as-judge approach (same methodology as RAGAS):

  1. CONTEXT RELEVANCE  — Is the retrieved context relevant to the question?
  2. FAITHFULNESS       — Does the answer stay within the retrieved context?
                          (hallucination detector)
  3. GROUNDEDNESS       — Is the answer supported by specific evidence
                          from the context?

How it works:
  For each logged request, we send the (question, context, answer) to
  Azure OpenAI and ask it to score each metric 0.0–1.0 with reasoning.
  Scores are saved back to SQLite in a new 'rag_evaluations' table.

Why LLM-as-judge instead of the RAGAS library?
  RAGAS library has frequent breaking changes and Azure OpenAI setup is
  complex. This implementation uses the exact same logic but is stable,
  transparent, and works directly with your existing Azure deployment.
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_FILE     = "logs/monitoring.db"
REPORTS_DIR = "logs/reports"

# ── Azure OpenAI ───────────────────────────────────────────────────────────────
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_VERSION   = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ── Score thresholds (below these = poor quality) ─────────────────────────────
THRESHOLDS = {
    "context_relevance": 0.5,
    "faithfulness":      0.5,
    "groundedness":      0.5,
}

# ── How many requests to evaluate (set to None for all) ───────────────────────
EVAL_LIMIT = None   # Evaluating every request costs tokens; 50 is a good sample


# ══════════════════════════════════════════════════════════════════════════════
#  LLM Judge Setup
# ══════════════════════════════════════════════════════════════════════════════

def get_llm():
    return AzureChatOpenAI(
        azure_deployment = CHAT_DEPLOYMENT,
        azure_endpoint   = AZURE_ENDPOINT,
        api_key          = AZURE_API_KEY,
        api_version      = AZURE_VERSION,
        temperature      = 0.0,    # 0 = deterministic, important for scoring
        max_tokens       = 300,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation Prompts (one per metric)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI evaluator. You evaluate RAG system outputs.
Always respond with valid JSON only. No explanation outside the JSON block."""


def prompt_context_relevance(question: str, context: str) -> str:
    return f"""Evaluate how relevant the retrieved context is to answering the question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Scoring guide:
1.0 = Context directly and completely addresses the question
0.7 = Context mostly relevant, minor gaps
0.5 = Context partially relevant
0.3 = Context loosely related but mostly unhelpful
0.0 = Context is completely irrelevant to the question

Respond with this exact JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>"
}}"""


def prompt_faithfulness(question: str, context: str, answer: str) -> str:
    return f"""Evaluate whether the answer is faithful to the retrieved context.
A faithful answer only uses information present in the context.
An unfaithful answer introduces facts not found in the context (hallucination).

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Scoring guide:
1.0 = Every claim in the answer is directly supported by the context
0.7 = Most claims supported, minor unsupported details
0.5 = About half the claims are supported
0.3 = Most claims are not in the context
0.0 = Answer is completely fabricated / contradicts the context

Respond with this exact JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>",
  "hallucination_detected": <true or false>
}}"""


def prompt_groundedness(question: str, context: str, answer: str) -> str:
    return f"""Evaluate whether the answer is grounded in specific evidence from the context.
A grounded answer references specific facts, quotes, or details from the context.
An ungrounded answer is vague or generic even if technically correct.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Scoring guide:
1.0 = Answer clearly references specific evidence from context
0.7 = Answer uses context but references could be more specific
0.5 = Answer somewhat references context details
0.3 = Answer is generic, barely uses context evidence
0.0 = Answer ignores the context entirely

Respond with this exact JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>"
}}"""


# ══════════════════════════════════════════════════════════════════════════════
#  Core Evaluation Function
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_one(llm, question: str, context: str, answer: str) -> dict:
    """
    Run all 3 evaluations for a single RAG request.
    Returns a dict with scores and reasoning for each metric.
    """

    def call_judge(user_prompt: str, metric_name: str) -> dict:
        """Call the LLM judge and parse its JSON response."""
        try:
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            raw = response.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError:
            # If LLM returns malformed JSON, return a safe default
            return {"score": 0.0,
                    "reasoning": f"Parse error for {metric_name}",
                    "parse_error": True}
        except Exception as e:
            return {"score": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "api_error": True}

    # ── Run the 3 judges ───────────────────────────────────────────────────────
    cr_result = call_judge(
        prompt_context_relevance(question, context),
        "context_relevance"
    )
    time.sleep(0.3)   # Small delay to avoid rate limiting

    faith_result = call_judge(
        prompt_faithfulness(question, context, answer),
        "faithfulness"
    )
    time.sleep(0.3)

    ground_result = call_judge(
        prompt_groundedness(question, context, answer),
        "groundedness"
    )

    # ── Extract scores safely ──────────────────────────────────────────────────
    cr_score    = float(cr_result.get("score",    0.0))
    faith_score = float(faith_result.get("score", 0.0))
    ground_score= float(ground_result.get("score",0.0))

    # ── Quality flags (below threshold = poor quality) ─────────────────────────
    flags = []
    if cr_score    < THRESHOLDS["context_relevance"]: flags.append("low_context_relevance")
    if faith_score < THRESHOLDS["faithfulness"]:      flags.append("possible_hallucination")
    if ground_score< THRESHOLDS["groundedness"]:      flags.append("low_groundedness")

    return {
        "context_relevance_score":    round(cr_score,     3),
        "context_relevance_reason":   cr_result.get("reasoning", ""),
        "faithfulness_score":         round(faith_score,  3),
        "faithfulness_reason":        faith_result.get("reasoning", ""),
        "hallucination_detected":     faith_result.get("hallucination_detected", False),
        "groundedness_score":         round(ground_score, 3),
        "groundedness_reason":        ground_result.get("reasoning", ""),
        "overall_score":              round((cr_score + faith_score + ground_score) / 3, 3),
        "quality_flags":              json.dumps(flags),
        "evaluated_at":               datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SQLite: Create evaluations table + store results
# ══════════════════════════════════════════════════════════════════════════════

def setup_eval_table():
    """Create the rag_evaluations table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_evaluations (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id                TEXT NOT NULL,
            user_query                TEXT,
            context_relevance_score   REAL,
            context_relevance_reason  TEXT,
            faithfulness_score        REAL,
            faithfulness_reason       TEXT,
            hallucination_detected    INTEGER DEFAULT 0,
            groundedness_score        REAL,
            groundedness_reason       TEXT,
            overall_score             REAL,
            quality_flags             TEXT,
            evaluated_at              TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_eval_request
        ON rag_evaluations(request_id)
    """)
    conn.commit()
    conn.close()


def save_eval(request_id: str, user_query: str, scores: dict):
    """Save one evaluation result to SQLite."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO rag_evaluations (
            request_id, user_query,
            context_relevance_score, context_relevance_reason,
            faithfulness_score, faithfulness_reason,
            hallucination_detected,
            groundedness_score, groundedness_reason,
            overall_score, quality_flags, evaluated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        request_id,
        user_query,
        scores["context_relevance_score"],
        scores["context_relevance_reason"],
        scores["faithfulness_score"],
        scores["faithfulness_reason"],
        int(scores.get("hallucination_detected", False)),
        scores["groundedness_score"],
        scores["groundedness_reason"],
        scores["overall_score"],
        scores["quality_flags"],
        scores["evaluated_at"],
    ))
    conn.commit()
    conn.close()


def get_eval_summary() -> dict:
    """Aggregate statistics across all evaluations."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT
            COUNT(*)                                    AS total_evaluated,
            ROUND(AVG(context_relevance_score), 3)      AS avg_context_relevance,
            ROUND(AVG(faithfulness_score),      3)      AS avg_faithfulness,
            ROUND(AVG(groundedness_score),      3)      AS avg_groundedness,
            ROUND(AVG(overall_score),           3)      AS avg_overall,
            SUM(hallucination_detected)                 AS hallucination_count,
            SUM(CASE WHEN overall_score >= 0.7
                THEN 1 ELSE 0 END)                      AS high_quality_count,
            SUM(CASE WHEN overall_score < 0.5
                THEN 1 ELSE 0 END)                      AS low_quality_count
        FROM rag_evaluations
    """).fetchone()
    conn.close()
    return dict(row) if row else {}


# ══════════════════════════════════════════════════════════════════════════════
#  Main Evaluation Loop
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    print("=" * 60)
    print("  TASK 4: RAG Evaluation Metrics")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────────
    setup_eval_table()
    llm = get_llm()
    print(" LLM judge ready\n")

    # ── Load logs to evaluate ──────────────────────────────────────────────────
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    limit_clause = f"LIMIT {EVAL_LIMIT}" if EVAL_LIMIT else ""
    rows = conn.execute(f"""
        SELECT request_id, user_query, retrieved_context, model_response
        FROM llm_logs
        WHERE error_code = ''
          AND model_response NOT LIKE 'ERROR%'
        ORDER BY timestamp DESC
        {limit_clause}
    """).fetchall()
    conn.close()

    if not rows:
        print(" No logs found. Run simulate_queries.py first.")
        return

    total = len(rows)
    print(f"Evaluating {total} requests...\n")
    print("-" * 60)

    # ── Evaluation loop ────────────────────────────────────────────────────────
    for i, row in enumerate(rows, 1):
        request_id = row["request_id"]
        query      = row["user_query"]
        answer     = row["model_response"]

        # Parse context from JSON string
        try:
            chunks  = json.loads(row["retrieved_context"] or "[]")
            context = "\n\n".join([c.get("content", "") for c in chunks])
        except Exception:
            context = str(row["retrieved_context"])

        # Skip if already evaluated
        check_conn = sqlite3.connect(DB_FILE)
        already = check_conn.execute(
            "SELECT 1 FROM rag_evaluations WHERE request_id = ?", (request_id,)
        ).fetchone()
        check_conn.close()

        if already:
            print(f"[{i:3d}/{total}] ⏭ Already evaluated — skipping")
            continue

        print(f"[{i:3d}/{total}] Evaluating: {query[:55]}...")

        scores = evaluate_one(llm, query, context, answer)
        save_eval(request_id, query, scores)

        # Print inline result
        flags = json.loads(scores["quality_flags"])
        flag_str = f"  {', '.join(flags)}" if flags else " Clean"
        print(f"         CR={scores['context_relevance_score']:.2f} | "
              f"FA={scores['faithfulness_score']:.2f} | "
              f"GR={scores['groundedness_score']:.2f} | "
              f"Overall={scores['overall_score']:.2f} | {flag_str}")

        time.sleep(0.5)   # Rate limit buffer

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = get_eval_summary()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "task4_rag_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═" * 60)
    print("  TASK 4 — RAG EVALUATION SUMMARY")
    print("═" * 60)
    print(f"  Requests evaluated       : {summary.get('total_evaluated', 0)}")
    print(f"  Avg Context Relevance    : {summary.get('avg_context_relevance', 0):.3f}")
    print(f"  Avg Faithfulness         : {summary.get('avg_faithfulness', 0):.3f}")
    print(f"  Avg Groundedness         : {summary.get('avg_groundedness', 0):.3f}")
    print(f"  Avg Overall Score        : {summary.get('avg_overall', 0):.3f}")
    print(f"  Hallucinations detected  : {summary.get('hallucination_count', 0)}")
    print(f"  High quality (≥0.7)      : {summary.get('high_quality_count', 0)}")
    print(f"  Low quality  (<0.5)      : {summary.get('low_quality_count', 0)}")
    print(f"\n  Report saved to: {report_path}")
    print("═" * 60)
    print("\nTask 4 Complete!")


if __name__ == "__main__":
    run_evaluation()
