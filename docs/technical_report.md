# Technical Report: LLM Monitoring Platform

**Project:** LLM Monitoring Platform for Azure OpenAI RAG Pipelines
**Date:** June 2026
**Stack:** Python 3.11 · Azure OpenAI · LangChain · FAISS · SQLite · Streamlit

---

## 1. System Architecture

### 1.1 Overview

The platform is a file-based, single-node observability system built around an Azure OpenAI RAG pipeline. It is fully self-contained — no external databases, message queues, or cloud infrastructure are required beyond the Azure OpenAI API itself. All state lives in a local SQLite database (`logs/monitoring.db`), a FAISS vector index, and a set of JSON reports.

The system is divided into four logical layers:

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| **Ingestion** | Build and maintain the vector knowledge base | `a_setup.py`, FAISS index |
| **Core Pipeline** | Execute RAG queries and log every request | `b_rag_pipeline.py`, `c_logger_middleware.py`, `d_simulate_queries.py` |
| **Analysis** | Compute derived metrics offline from the log store | `f`, `g`, `h`, `i`, `e` scripts |
| **Presentation** | Expose all metrics through an interactive UI | `dashboard.py` (Streamlit) |

### 1.2 Data Flow

```
documents/ ──► a_setup.py ──► AzureEmbed ──► faiss_index/
                                                    │
prompts.csv ──► d_simulate ──► b_rag_pipeline ◄─────┘
                                    │   └──► AzureChat
                                    ▼
                             c_logger_middleware
                               │           │
                          rag_logs.csv   monitoring.db
                                              │
                    ┌─────────────────────────┤
                    │         │         │     │
                  f_token  g_rag    h_drift i_edge
                    │         │         │     │
                    └─────────┴────► e_eval ──┴──► reports/
                                         │
                                    dashboard.py
```

### 1.3 Central Configuration

All runtime parameters — Azure credentials, file paths, model names, pricing, and alert thresholds — are declared in `src/config.py` and imported by every other module. This prevents configuration drift across scripts and means any threshold or path change takes effect everywhere with a single edit.

Key settings:

| Parameter | Value |
|-----------|-------|
| `CHUNK_SIZE` | 800 characters |
| `CHUNK_OVERLAP` | 100 characters |
| `TOP_K` | 3 retrieved chunks per query |
| `LLM_TEMPERATURE` | 0.3 |
| `LATENCY_SLOW_THRESHOLD` | 10.0 s |
| `DRIFT_WARNING_THRESHOLD` | 0.10 |
| `DRIFT_ANOMALY_THRESHOLD` | 0.20 |
| `EVAL_LIMIT` | 50 requests per evaluation run |

### 1.4 Storage Design

Storage is split by access pattern to keep the system simple and portable:

- **SQLite (`monitoring.db`)** — three tables (`llm_logs`, `rag_evaluations`, `drift_logs`) indexed on `timestamp` and `prompt_version` for fast dashboard queries.
- **FAISS (`vectorstore/faiss_index/`)** — local vector similarity search using 1 536-dimension vectors from `text-embedding-ada-002`.
- **CSV (`logs/rag_logs.csv`, `edge_cases_log.csv`)** — human-readable mirror of `llm_logs` for quick inspection without a SQL client.
- **JSON (`logs/reports/*.json`)** — pre-aggregated reports consumed by the dashboard, avoiding expensive GROUP BY queries on every page load.

---

## 2. Logging Components

### 2.1 LLMLogger Middleware

The `LLMLogger` class in `c_logger_middleware.py` is the single point through which all RAG results pass before being persisted. It acts as a structured middleware: the RAG pipeline returns a result dictionary, and `LLMLogger.log()` normalises, validates, and writes every field to both CSV and SQLite.

**17 fields captured per request:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | TEXT | UUID4 — globally unique per request |
| `timestamp` | TEXT | ISO-8601 datetime at log time |
| `user_query` | TEXT | Raw user prompt |
| `retrieved_context` | TEXT | JSON array of top-K retrieved chunks |
| `retrieved_sources` | TEXT | JSON array of source filenames |
| `model_name` | TEXT | Azure deployment name used |
| `prompt_version` | TEXT | `v1` or `v2` |
| `model_response` | TEXT | Full LLM-generated response |
| `input_tokens` | INTEGER | Tokens in the combined prompt |
| `output_tokens` | INTEGER | Tokens in the model response |
| `total_tokens` | INTEGER | `input_tokens + output_tokens` |
| `retrieval_latency` | REAL | Seconds for FAISS vector search |
| `llm_latency` | REAL | Seconds for Azure OpenAI API call |
| `end_to_end_latency` | REAL | Total wall-clock time for the request |
| `top_k` | INTEGER | Number of chunks retrieved |
| `safety_flag` | INTEGER | 1 if query matched a safety keyword |
| `error_code` | TEXT | Non-empty string if the request failed |

### 2.2 Dual-Write Strategy

Every record is written to CSV first (flushed immediately), then to SQLite. This ensures that even if the database is locked or corrupted, the CSV provides a complete audit trail. The SQLite connection is opened once per logger instance with `check_same_thread=False` to allow reuse across the simulation loop without reconnecting on every request.

### 2.3 Safety Pre-Filtering

Before the RAG query is dispatched, the raw prompt is checked against a keyword list defined in `config.py`:

```
SAFETY_KEYWORDS = ["hack", "weapon", "illegal", "kill", "bomb", "bypass", "jailbreak"]
```

A case-insensitive substring match sets `safety_flag = 1`. The request still proceeds — it is not blocked — giving the platform visibility into adversarial traffic patterns without silently dropping data. In the simulation run of 591 requests, 27 (4.57%) were flagged, all originating from the edge case test suite and all attributed to prompt version v1.

### 2.4 Database Indexes

Three indexes are created at startup to maintain query performance as the log table grows:

- `idx_timestamp` on `llm_logs(timestamp)` — powers time-range filters in the Logs Viewer tab.
- `idx_prompt_version` on `llm_logs(prompt_version)` — powers v1 vs v2 comparison queries.
- `idx_eval_request` on `rag_evaluations(request_id)` — joins evaluation scores back to the originating request without a full table scan.

---

## 3. Metric Definitions

### 3.1 Token and Cost Metrics

Token counts are extracted directly from the Azure OpenAI API response object and stored verbatim. Cost is estimated post-hoc using the pricing table in `config.py`.

| Metric | Formula |
|--------|---------|
| Input tokens | Tokens in system prompt + retrieved context + user query |
| Output tokens | Tokens in the model response |
| Total tokens | `input_tokens + output_tokens` |
| Request cost (USD) | `(input / 1 000 × $0.005) + (output / 1 000 × $0.015)` |
| High-token flag | `total_tokens > 2 000` |

**Observed results across 591 requests:**

| Metric | Value |
|--------|-------|
| Total tokens consumed | 314 615 |
| Average tokens per request | 532.3 |
| Token standard deviation | 125.3 |
| Maximum tokens (single request) | 1 343 |
| Total estimated cost | $2.13 |
| Average cost per request | $0.0036 |
| High-token requests | 0 (0.0%) |
| Prompt v1 average tokens | 526.8 |
| Prompt v2 average tokens | 542.6 |

Prompt v2 consumes approximately 15.8 additional tokens per request because its system instruction is more detailed, but this difference is negligible relative to the context window and cost.

### 3.2 Latency Metrics

Latency is measured at three granularities using `time.time()` around each subsystem boundary.

| Component | Measured Interval |
|-----------|------------------|
| `retrieval_latency` | FAISS `similarity_search()` call |
| `llm_latency` | Azure OpenAI `invoke()` call |
| `end_to_end_latency` | Full `rag_query()` function wall time |

**Observed results:**

| Metric | E2E | LLM API | Retrieval |
|--------|-----|---------|-----------|
| Average | 2.230 s | 1.776 s | 0.454 s |
| Median (p50) | 1.912 s | 1.506 s | 0.390 s |
| p95 | 4.236 s | 3.370 s | 1.121 s |
| p99 | 6.973 s | 6.630 s | 1.807 s |
| Maximum | 12.098 s | 11.734 s | 2.767 s |

The LLM API accounts for 79.6% of end-to-end latency; FAISS retrieval accounts for 20.4%. Three requests (0.5%) exceeded the 10-second slow query threshold.

### 3.3 RAG Quality Metrics (LLM-as-Judge)

Quality evaluation uses the same Azure OpenAI chat model as a judge. For each request, three separate prompts are sent, each requesting a structured JSON response with a score (0.0–1.0) and a one-sentence reasoning string. This approach replicates the RAGAS evaluation methodology while remaining stable and fully transparent.

| Metric | What it measures | Alert threshold |
|--------|-----------------|-----------------|
| **Context Relevance** | Do the retrieved chunks contain information the query needs? | < 0.5 |
| **Faithfulness** | Is every claim traceable to the retrieved context? (hallucination proxy) | < 0.4 |
| **Groundedness** | Does the response cite specific evidence rather than speaking generically? | < 0.5 |
| **Overall Score** | Arithmetic mean of the three metrics | — |

A faithfulness score below 0.4 triggers a `possible_hallucination` quality flag. An overall score below 0.5 marks the request as low quality. Both flags are written to the `quality_flags` column as a JSON array.

**Observed results (179 requests evaluated):**

| Metric | Score |
|--------|-------|
| Avg Context Relevance | 0.557 |
| Avg Faithfulness | 0.899 |
| Avg Groundedness | 0.746 |
| Avg Overall Score | 0.734 (± 0.275) |
| Hallucinations detected | 16 (8.9%) |
| Low quality requests | 35 (19.6%) |

The high faithfulness score (0.899) indicates the model rarely introduces claims absent from the retrieved context. The lower context relevance (0.557) suggests the retriever occasionally returns tangentially related chunks — most evident in adversarial edge cases, where the five worst-performing requests all scored 0.0 across all three dimensions because the system prompt injection attempts produced responses unrelated to any retrieved evidence.

### 3.4 Embedding Drift Metrics

Drift is measured by re-embedding a fixed set of 20 reference queries and comparing each new vector to its saved baseline using cosine distance:

```
drift_score = 1 − cosine_similarity(current_vector, baseline_vector)
```

A score of 0 means the embedding is identical to the baseline. A score near 1 indicates the model produces fundamentally different representations for the same text, which may signal a model version change or embedding model update.

| Status | Threshold |
|--------|-----------|
| HEALTHY | drift\_score < 0.10 |
| WARNING | 0.10 ≤ drift\_score < 0.20 |
| ANOMALY | drift\_score ≥ 0.20 |

**Observed results (3 measurement runs, 60 total measurements):**

| Metric | Value |
|--------|-------|
| Average drift score | 0.336 |
| Maximum drift score | 1.051 |
| Healthy queries | 40 (66.7%) |
| Warning queries | 0 |
| Anomaly queries | 20 (33.3%) |
| Overall status | ANOMALY |

A maximum drift score exceeding 1.0 is only possible when the current and baseline vectors are nearly orthogonal, which is consistent with an Azure deployment model version change between baseline creation and subsequent measurements rather than genuine semantic drift in user queries.

### 3.5 System Health Score

The final health score is a weighted composite across five dimensions:

| Dimension | Weight | Observed Score |
|-----------|--------|----------------|
| Reliability (error rate) | 30% | 100.0 |
| Latency (p95 vs 5 s baseline) | 25% | 100.0 |
| RAG Quality (avg overall score) | 20% | 73.4 |
| Drift Stability | 15% | 0.0 |
| Token Efficiency (high-token requests) | 10% | 100.0 |

**Composite score: 79.7 / 100 — Grade B (Good)**

The drift component (0.0) is the sole significant drag on the composite score. All other dimensions are at or near full marks, reflecting a reliable, fast, and efficient system. Correcting the baseline drift issue would push the score to approximately 92 (Grade A).

---

## 4. Dashboard Design

### 4.1 Technology Choices

The dashboard is built with **Streamlit** for its zero-configuration Python server and native data binding, and **Plotly** for interactive, zoomable charts. A dark theme is applied via injected CSS to reduce eye strain during operational monitoring sessions. All data is loaded with `@st.cache_data(ttl=30)`, which re-queries SQLite every 30 seconds. A manual "Refresh Data" button in the sidebar triggers an immediate cache clear for on-demand updates.

### 4.2 Tab Structure

The dashboard has six tabs, each addressing a distinct monitoring concern:

**Tab 1 — Overview**
Five KPI cards (total requests, total tokens, average latency, estimated cost, error rate). A daily request volume bar chart and a v1/v2 donut chart sit below. System-level alert banners appear at the bottom.

**Tab 2 — Token & Cost**
Stacked bar chart of daily input vs output tokens. Scatter plot of cost per request over time coloured by prompt version. A distribution histogram overlays v1 and v2 token counts. A summary table shows mean and total cost per version.

**Tab 3 — Latency**
Line chart of E2E latency over time with a dashed 10-second alert threshold. Box plots comparing retrieval, LLM API, and E2E distributions side-by-side. A per-version box plot. Slow requests are listed in a table directly below the charts.

**Tab 4 — RAG Evaluation**
Four KPI cards for context relevance, faithfulness, groundedness, and hallucination count. A bar chart shows average scores against the 0.5 minimum threshold line. A scatter plot of faithfulness vs context relevance coloured by overall score surfaces correlation patterns. Low-quality requests are listed with all three scores.

**Tab 5 — Drift Tracking**
Average drift per monitoring run plotted as a line with dashed warning (0.10) and anomaly (0.20) thresholds. A horizontal bar chart shows per-query drift for the most recent run, colour-coded HEALTHY/WARNING/ANOMALY. Anomalous queries are listed in a table.

**Tab 6 — Logs Viewer**
Filterable table supporting prompt version, model name, error/safety/success status, and keyword search. Pagination at 20 rows per page. A row inspector renders the full response text for any selected row index.

### 4.3 Alert Banners

Alerts are surfaced inline in each tab using styled HTML boxes:

- **Red box** (`#450A0A` background) — errors, slow queries, drift anomalies.
- **Amber box** (`#451A03` background) — safety flags, low-quality RAG scores.
- **Green success** — shown when no issues exist for the relevant dimension.

---

## 5. Alerting Mechanism

The platform implements threshold-based alerting. Alerts do not send external notifications; they surface visually in the dashboard and are recorded in the final health report JSON. All thresholds are centralised in `config.py` — changing a value takes effect on the next dashboard load without any code change.

### 5.1 Alert Thresholds

| Dimension | Metric | Threshold | Config key |
|-----------|--------|-----------|------------|
| Latency | `end_to_end_latency` | > 10.0 s | `LATENCY_SLOW_THRESHOLD` |
| Token usage | `total_tokens` | > 2 000 | `TOKEN_HIGH_THRESHOLD` |
| Daily cost | Aggregated daily cost | > $0.10 | `DAILY_COST_ALERT_USD` |
| Context relevance | `context_relevance_score` | < 0.5 | `RAG_MIN_CONTEXT_RELEVANCE` |
| Faithfulness | `faithfulness_score` | < 0.4 | `RAG_HALLUCINATION_THRESHOLD` |
| Groundedness | `groundedness_score` | < 0.5 | `RAG_MIN_GROUNDEDNESS` |
| RAG overall | `overall_score` | < 0.5 | `RAG_LOW_QUALITY_THRESHOLD` |
| Drift (warning) | `drift_score` | ≥ 0.10 | `DRIFT_WARNING_THRESHOLD` |
| Drift (anomaly) | `drift_score` | ≥ 0.20 | `DRIFT_ANOMALY_THRESHOLD` |
| Safety | `safety_flag` | = 1 | `SAFETY_KEYWORDS` list |

### 5.2 Alert Evaluation Points

Alerts are evaluated at two points:

1. **At ingest time** — `safety_flag` is evaluated by the RAG pipeline before the response is returned. Token and latency alerts are flagged during `f_token_latency_tracker.py` and written to the tracking report.

2. **At dashboard render time** — Streamlit re-evaluates thresholds against live SQLite data on each tab load. This means newly logged requests are reflected immediately on the next page visit without re-running any analysis script.

### 5.3 Quality Flag Mechanism

The RAG evaluator writes a `quality_flags` JSON array to each `rag_evaluations` row. Possible values are `low_context_relevance`, `possible_hallucination`, and `low_groundedness`. The dashboard reads these flags to populate the warning tables in Tab 4, giving operators a queryable record of which specific requests were problematic and the dimension that failed.

---

## 6. Limitations and Future Improvements

### 6.1 Current Limitations

**Embedding drift — baseline dependency.**
Drift scores above 1.0 were observed (maximum: 1.051). A cosine distance score above 1.0 is only theoretically possible when vectors are in opposing directions, which indicates the Azure embedding deployment was updated between baseline creation and subsequent measurement runs. The current implementation provides no mechanism to detect or compensate for a model version change at the provider level. This caused the drift stability dimension to score 0/100 in the health report, pulling the composite score from a potential 92 down to 79.7.

**No real-time pipeline.**
All analysis scripts run manually in sequence after simulation. There is no streaming or continuous ingestion. A request logged during a live session will not appear in tracking reports until the relevant script is re-run.

**LLM judge cost and rate limits.**
The RAG evaluator sends three Azure OpenAI calls per evaluated request. Full evaluation of 591 requests would require 1 773 additional API calls, adding approximately $0.90 to the run cost. The `EVAL_LIMIT=50` default mitigates this but evaluates only a sample, which may not be representative if edge case requests cluster at certain timestamps.

**Safety filtering is keyword-only.**
The current safety check is a case-insensitive substring match against a fixed list. It will miss semantically harmful prompts that avoid the flagged words, and cannot detect prompt injection attempts that use encoding tricks or unusual phrasing. All 27 observed safety flags came from the explicitly adversarial edge case suite — zero came from the standard 105-prompt corpus.

**Single-user, single-process.**
SQLite with `check_same_thread=False` is sufficient for a single simulation process but will produce locking errors under concurrent writes from multiple processes or users. The platform is not suitable for production multi-user deployments in its current storage configuration.

**No automated test suite.**
There are no unit or integration tests. Correctness is validated only by running the full simulation pipeline and comparing outputs manually against the dashboard.

### 6.2 Future Improvements

**Versioned drift baselines.**
Store baseline embedding snapshots with a model version identifier. When a measurement run detects drift scores above 0.5 across the majority of reference queries, automatically classify the event as a model version change rather than semantic drift and prompt the operator to reset the baseline under the new version tag.

**Streaming log ingestion.**
Replace the batch simulation model with an event-driven ingestion path (e.g. a lightweight SQLite WAL-based watcher or a Redis Stream consumer). This would allow the dashboard to reflect live traffic without manual re-runs between sessions.

**External alerting integration.**
Add a post-run notification step using a Slack webhook or SMTP email when the composite health score drops below a configurable threshold, or when daily cost exceeds the budget limit. This would make the platform usable in an on-call rotation.

**Risk-based evaluation sampling.**
Rather than a fixed `EVAL_LIMIT`, prioritise evaluation of requests with high token counts, low cosine retrieval similarity, or active safety flags — the requests statistically most likely to expose quality problems — before falling back to random sampling.

**Prompt version registry.**
Replace the hardcoded `v1`/`v2` strings with a database table storing version ID, creation timestamp, system prompt text, and a description. This would enable reproducible A/B experiments and a complete audit trail of prompt changes over time.

**Production backend migration.**
For a production deployment, replacing SQLite with PostgreSQL and FAISS with a managed vector database (Azure AI Search, Weaviate, or Pinecone) would unlock concurrent access, automatic backups, horizontal scaling, and built-in access controls — with minimal changes required to the existing monitoring and evaluation logic.

---

## Appendix: Key Configuration Reference

```python
# src/config.py — selected parameters

CHUNK_SIZE              = 800      # characters per document chunk
CHUNK_OVERLAP           = 100      # overlap between adjacent chunks
TOP_K                   = 3        # retrieved chunks per query
LLM_TEMPERATURE         = 0.3
LLM_MAX_TOKENS          = 800

LATENCY_SLOW_THRESHOLD  = 10.0     # seconds — flag if E2E exceeds this
TOKEN_HIGH_THRESHOLD    = 2000     # tokens  — flag if single request exceeds this
DAILY_COST_ALERT_USD    = 0.10     # USD     — alert if daily cost exceeds this

RAG_MIN_CONTEXT_RELEVANCE   = 0.5
RAG_HALLUCINATION_THRESHOLD = 0.4  # faithfulness below this = hallucination
RAG_MIN_GROUNDEDNESS        = 0.5
RAG_LOW_QUALITY_THRESHOLD   = 0.5

DRIFT_WARNING_THRESHOLD = 0.10
DRIFT_ANOMALY_THRESHOLD = 0.20

EVAL_LIMIT              = 50       # max requests evaluated per run (cost control)
```
