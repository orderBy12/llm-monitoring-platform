# LLM Monitoring Platform

A production-oriented monitoring system for Azure OpenAI-based Retrieval-Augmented Generation (RAG) pipelines. The platform covers the full observability lifecycle: document indexing, query simulation, token and cost tracking, LLM-as-judge quality evaluation, embedding drift detection, edge case testing, and an interactive Streamlit dashboard.

**Authors:** Aadesh Shrivastava and Shaily Pandey

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Installation and Setup](#installation-and-setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Architecture](#architecture)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The platform is organized as lettered scripts under `src/`. All shared settings (paths, model names, thresholds, pricing) live in a single `config.py` — every other module imports from there.

| Script | What it does |
|--------|--------------|
| `config.py` | Central configuration — single source of truth for all settings |
| `a_setup.py` | Index documents into a FAISS vector store |
| `b_rag_pipeline.py` | RAG query engine (retrieval + Azure OpenAI generation) |
| `c_logger_middleware.py` | `LLMLogger` — persists request metadata to CSV and SQLite |
| `d_simulate_queries.py` | Fire 105 prompts through the pipeline (alternates v1/v2) |
| `f_token_latency_tracker.py` | Aggregate token usage, estimate costs, compute latency percentiles |
| `g_rag_evaluator.py` | Score each request on context relevance, faithfulness, and groundedness |
| `h_drift_detector.py` | Measure embedding drift against a fixed baseline; flag anomalies |
| `i_edge_cases.py` | Test 25 adversarial and boundary prompts across 9 categories (Task 7 Part A) |
| `e_evaluation.py` | Composite health score (0–100, grade A–D) from all monitoring dimensions (Task 7 Part B) |

> **Run order:** `a` → `b` (smoke test) → `d` → `f` → `g` → `h` → `i` → `e`. Note that `e_evaluation.py` is the final report step despite its letter position; `c_logger_middleware.py` is a shared module imported by other scripts and not run directly.

---

## Tech Stack

### Languages and Frameworks

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| LLM provider | Azure OpenAI (`openai`, `langchain-openai`) |
| RAG framework | LangChain (`langchain`, `langchain-community`) |
| Vector store | FAISS (`faiss-cpu`) |
| Dashboard | Streamlit + Plotly |
| Monitoring storage | SQLite (built-in) + CSV |

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >=1.30.0 | Azure OpenAI API client |
| `langchain` | >=0.1.0 | Document loading, text splitting, chain orchestration |
| `langchain-community` | >=0.0.1 | FAISS vector store integration |
| `langchain-openai` | >=0.0.1 | Azure OpenAI embeddings and chat wrappers |
| `faiss-cpu` | >=1.7.4 | Local vector similarity search |
| `streamlit` | >=1.32.0 | Web dashboard |
| `plotly` | >=5.18.0 | Interactive charts |
| `pandas` | >=2.0.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical operations |
| `scikit-learn` | >=1.3.0 | Cosine similarity for drift detection |
| `tiktoken` | ==0.8.0 | Token counting (locked version) |
| `python-dotenv` | >=1.0.0 | `.env` file loading |
| `pypdf` | >=4.0.0 | PDF document ingestion |
| `tqdm` | >=4.66.0 | Progress bars |

Full dependency list: [requirements.txt](requirements.txt)

---

## Folder Structure

```text
llm-monitoring-platform/
├── dashboard.py                   # Streamlit monitoring dashboard (6 tabs)
├── requirements.txt               # Python dependencies
├── .env-sample                    # Environment variable template
│
├── src/
│   ├── config.py                  # Central config — paths, model names, thresholds, pricing
│   ├── a_setup.py                 # Build FAISS vector index from documents
│   ├── b_rag_pipeline.py          # RAG query engine (retrieval + generation)
│   ├── c_logger_middleware.py     # LLMLogger: writes to CSV and SQLite
│   ├── d_simulate_queries.py      # Fire 105 prompts through the pipeline
│   ├── e_evaluation.py            # Final system health report (Task 7 Part B)
│   ├── f_token_latency_tracker.py # Token usage, cost, and latency reports
│   ├── g_rag_evaluator.py         # LLM-as-judge quality scoring
│   ├── h_drift_detector.py        # Embedding drift baseline and measurement
│   ├── i_edge_cases.py            # Adversarial and boundary prompt testing (Task 7 Part A)
│   └── sqlite.ipynb               # Notebook for ad-hoc SQLite exploration
│
├── data/
│   ├── documents/                 # Source .txt files indexed into FAISS
│   │   ├── ai_agents.txt
│   │   ├── ai_in_software_development.txt
│   │   ├── ai_regulation_ethics.txt
│   │   ├── ai_safety_and_ethics.txt
│   │   ├── large_language_models.txt
│   │   ├── llmops_and_infrastructure.txt
│   │   ├── multimodal_ai.txt
│   │   ├── rag_and_vector_databases.txt
│   │   └── retrieval_augmented_generation.txt
│   ├── prompts.csv                # 105 simulation prompts
│   └── reference_queries.json    # 20 fixed queries for drift baseline
│
├── logs/
│   ├── rag_logs.csv               # Per-request log (CSV mirror of SQLite)
│   ├── edge_cases_log.csv         # Edge case test outcomes
│   ├── drift_baseline.json        # Saved embedding vectors for drift baseline
│   ├── monitoring.db              # SQLite database (git-ignored, generated locally)
│   └── reports/
│       ├── task3_tracking_report.json
│       ├── task4_rag_evaluation.json
│       ├── task5_drift_report.json
│       ├── task7_final_evaluation.json
│       └── task7_summary.txt
│
└── vectorstore/
    └── faiss_index/               # Generated FAISS index (git-ignored, .gitkeep tracked)
```

---

## Installation and Setup

### Prerequisites

- Python 3.11 or later
- An Azure OpenAI resource with a chat deployment (e.g., `gpt-4o`) and an embedding deployment (e.g., `text-embedding-ada-002`)
- Git

### Steps

**1. Clone the repository**

```bash
git clone <repository-url>
cd llm-monitoring-platform
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env-sample .env
# Open .env and fill in your Azure OpenAI credentials
```

**5. Build the FAISS vector index**

```bash
python src/a_setup.py
```

This reads all `.txt` files under `data/documents/`, splits them into 800-character chunks with 100-character overlap, embeds them using Azure OpenAI, and saves the index to `vectorstore/faiss_index/`.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure endpoint URL (e.g., `https://my-resource.openai.azure.com/`) |
| `AZURE_OPENAI_API_VERSION` | No | `2024-02-01` | API version string |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | `gpt-4o` | Deployment name for the chat model |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | `text-embedding-ada-002` | Deployment name for embeddings |

All values are read through `src/config.py`, which also exposes `validate_env()` to check for missing variables before a run. See [.env-sample](.env-sample) for the full template.

---

## Usage

Run the scripts in the order below. Each step depends on outputs from the previous ones. All scripts must be invoked from the repo root with `src/` on the Python path so that `import config` resolves correctly:

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"

# macOS / Linux
export PYTHONPATH=src
```

### Step 1 — Index documents and verify retrieval

```bash
python src/a_setup.py        # builds vectorstore/faiss_index/
python src/b_rag_pipeline.py # smoke-tests the RAG pipeline
```

`a_setup.py` reads all `.txt` files under `data/documents/`, splits them into chunks (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` in `config.py`), embeds them via Azure OpenAI, and saves the FAISS index.

### Step 2 — Simulate queries and generate monitoring data

```bash
python src/d_simulate_queries.py
```

Fires all 105 prompts from `data/prompts.csv` through the RAG pipeline, alternating between prompt v1 and v2 (defined in `config.py`). Writes results to `logs/rag_logs.csv` and `logs/monitoring.db`. Uses `c_logger_middleware.py` as an imported module.

### Step 3 — Token and latency tracking

```bash
python src/f_token_latency_tracker.py
```

Reads `logs/monitoring.db` and writes `logs/reports/task3_tracking_report.json` with:
- Total and daily token usage (input / output / total)
- Cost estimates using pricing from `config.py`
- Latency percentiles (avg, min, max, p50, p95, p99)
- Side-by-side prompt v1 vs v2 comparison

### Step 4 — RAG quality evaluation

```bash
python src/g_rag_evaluator.py
```

Uses an Azure OpenAI LLM judge to score each logged request on three dimensions (0.0–1.0 each):

| Metric | Description |
|--------|-------------|
| Context Relevance | How well the retrieved chunks match the query |
| Faithfulness | Whether the response is grounded in the retrieved context (hallucination proxy) |
| Groundedness | How fully the answer is supported by the retrieved evidence |

Evaluates up to `EVAL_LIMIT=50` requests (configurable in `config.py`). Writes scores to the `rag_evaluations` table in SQLite and to `logs/reports/task4_rag_evaluation.json`.

### Step 5 — Embedding drift detection

```bash
# First time only — save baseline embeddings
python src/h_drift_detector.py --mode baseline

# On subsequent runs — measure drift against the baseline
python src/h_drift_detector.py --mode measure
```

Re-embeds the 20 fixed queries from `data/reference_queries.json` and computes cosine similarity against the saved baseline. Drift thresholds (set in `config.py`):

| Drift Score | Status |
|-------------|--------|
| < 0.10 | HEALTHY |
| 0.10 – 0.20 | WARNING |
| > 0.20 | ANOMALY |

### Step 6 — Edge case testing (Task 7 Part A)

```bash
python src/i_edge_cases.py
```

Runs 25 adversarial and boundary prompts across 9 categories (Safety, Off-Topic, Vague, Very Long, Empty-Like, Adversarial, Unanswerable, Repetitive, Mixed-Language). Results are written to `logs/edge_cases_log.csv`.

### Step 7 — Final system health report (Task 7 Part B)

```bash
python src/e_evaluation.py
```

Aggregates all monitoring tables into a composite health score (0–100) graded A–D across five weighted dimensions:

| Dimension | Weight | How it's scored |
|-----------|--------|----------------|
| Reliability | 30% | Error rate → `max(0, 100 - error_rate × 10)` |
| Latency | 25% | p95 vs 5 s baseline |
| RAG Quality | 20% | Average overall RAG score × 100 |
| Drift Stability | 15% | `(1 - avg_drift / 0.20) × 100` |
| Token Efficiency | 10% | Penalises high-token requests |

Outputs `logs/reports/task7_final_evaluation.json` and `logs/reports/task7_summary.txt`.

---

## Dashboard

```bash
streamlit run dashboard.py
```

Opens at [http://localhost:8501](http://localhost:8501). The dashboard has six tabs:

| Tab | Content |
|-----|---------|
| Overview | KPI cards, health score summary |
| Token & Cost | Daily token usage charts, cost trend |
| Latency | Response time distributions, slow query alerts |
| RAG Evaluation | Score trends, hallucination counts |
| Drift Tracking | Cosine similarity over time, anomaly markers |
| Logs Viewer | Paginated, filterable request table |

The dashboard reads from `logs/monitoring.db`. Run at least Step 2 before launching it.

---

## Architecture

```
data/documents/  ──►  a_setup.py  ──►  vectorstore/faiss_index/
                                                   │
data/prompts.csv ──►  d_simulate  ──►  c_logger  ──►  logs/monitoring.db
                           │                                  │
                     b_rag_pipeline                           │
                     (Azure OpenAI)                           │
                                                              ▼
                                         f_token_latency_tracker  ──►  reports/
                                         g_rag_evaluator          ──►  reports/
                                         h_drift_detector         ──►  reports/
                                         i_edge_cases             ──►  logs/
                                         e_evaluation             ──►  reports/
                                                              │
                                         dashboard.py  ◄───────┘
                                         (Streamlit)
```

**Design choices:**

- **`config.py` as the single source of truth.** All paths, model names, thresholds, and pricing are defined once and imported everywhere. To change a setting, edit `config.py` only.
- **File-based storage** (FAISS, CSV, SQLite, JSON) keeps the setup self-contained with no external infrastructure dependencies.
- **Alphabetical script names** (`a_`–`i_`) provide clear ordering in the file system while keeping filenames valid Python identifiers that import cleanly.
- **LLM-as-judge evaluation** makes the scoring logic fully visible in the repository instead of relying on a black-box library, but adds extra Azure OpenAI calls and cost.
- **Synchronous execution** is simple and easy to debug locally, but not designed for high-throughput serving.

---

## Known Limitations

- **`e_evaluation.py` runs last, not fifth.** The final report script is alphabetically at position `e` but must be run after all other analysis scripts. See the [Run order note](#project-overview) above.
- **`logs/monitoring.db` is not committed.** The SQLite database is git-ignored and must be generated locally by running `d_simulate_queries.py`. The dashboard and all report scripts depend on it.
- **FAISS index not committed.** `vectorstore/faiss_index/` tracks only `.gitkeep`. Run `a_setup.py` before issuing RAG queries.
- **`PYTHONPATH` must include `src/`.** All scripts import `config` as a top-level module. Set `PYTHONPATH=src` before running any script from the repo root.
- **No automated test suite.** There are no pytest or unittest tests. Quality assurance relies on the manual task workflow and edge case scripts.
- **No formal Python version constraint.** The project was developed with Python 3.11. No `pyproject.toml` or `setup.py` declares a minimum version.
- **Cost per evaluation run.** `g_rag_evaluator.py` calls Azure OpenAI once per logged request. The `EVAL_LIMIT` setting in `config.py` caps evaluation at 50 requests by default.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run the full script sequence (`a` → `d` → `f` → `g` → `h` → `i` → `e`) against your changes to confirm the pipeline works end-to-end.
3. If you change a path, model name, threshold, or pricing value, update `src/config.py` — do not hardcode values in individual scripts.
4. If you add a new monitoring metric, persist it to `logs/monitoring.db` via `LLMLogger` and expose it in `dashboard.py`.
5. Update this README if you change environment variables, generated artifacts, or the script sequence.
6. Keep changes aligned with the file-based, single-user design. Proposals to move to a managed backend belong in a discussion first.
7. Open a pull request with a clear description of what changed and why.

---

## License

No license file has been added to this repository. Add one before distributing or open-sourcing the project.
