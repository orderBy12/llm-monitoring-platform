# LLM Monitoring Platform

A production-oriented monitoring system for Azure OpenAI-based Retrieval-Augmented Generation (RAG) pipelines. The platform covers the full observability lifecycle: document indexing, query simulation, token and cost tracking, LLM-as-judge quality evaluation, embedding drift detection, edge case testing, and an interactive Streamlit dashboard.

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

The platform is organized as a sequence of numbered task scripts that build on one another:

| Task | Scripts | What it does |
|------|---------|--------------|
| 1 — Build & Query | `1.1_setup.py`, `1.2_rag_pipeline.py` | Index documents into a FAISS vector store; run RAG queries against Azure OpenAI |
| 2 — Log & Simulate | `2.1_logger_middleware.py`, `2.2_simulate_queries.py` | Persist request metadata to CSV and SQLite; fire 105 prompts through the pipeline |
| 3 — Token & Latency | `3_token_latency_tracker.py` | Aggregate token usage, estimate costs, compute latency percentiles, compare prompt versions |
| 4 — RAG Quality | `4_rag_evaluator.py` | Score each request on context relevance, faithfulness, and groundedness using an LLM judge |
| 5 — Drift Detection | `5_drift_detector.py` | Measure embedding drift against a fixed baseline; flag anomalies |
| 6 — Edge Cases | `6_edge_cases.py` | Test 25 deliberate adversarial and boundary prompts across 9 categories |
| 7 — Final Report | `7_evaluation.py` | Produce a composite health score (0–100, grade A–D) from all monitoring dimensions |
| — Dashboard | `dashboard.py` | Interactive Streamlit UI with six tabs over all collected data |

---

## Tech Stack

### Languages and Frameworks

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| LLM provider | Azure OpenAI (`openai`, `langchain-openai`) |
| RAG framework | LangChain (`langchain`, `langchain-community`) |
| Vector store | FAISS (`faiss-cpu`) |
| Alternative vector store | ChromaDB |
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
| `ragas` | >=0.1.7 | RAG evaluation framework |
| `tiktoken` | ==0.8.0 | Token counting (locked version) |
| `python-dotenv` | >=1.0.0 | `.env` file loading |
| `pypdf` | >=4.0.0 | PDF document ingestion |
| `tqdm` | >=4.66.0 | Progress bars |

Full dependency list: [requirements.txt](requirements.txt)

---

## Folder Structure

```text
llm-monitoring-platform/
├── dashboard.py                  # Streamlit monitoring dashboard (6 tabs)
├── requirements.txt              # Python dependencies
├── .env-sample                   # Environment variable template
│
├── src/
│   ├── 1.1_setup.py              # Build FAISS vector index from documents
│   ├── 1.2_rag_pipeline.py       # RAG query engine (retrieval + generation)
│   ├── 2.1_logger_middleware.py  # LLMLogger: writes to CSV and SQLite
│   ├── 2.2_simulate_queries.py   # Fire 105 prompts through the pipeline
│   ├── 3_token_latency_tracker.py# Token usage, cost, and latency reports
│   ├── 4_rag_evaluator.py        # LLM-as-judge quality scoring
│   ├── 5_drift_detector.py       # Embedding drift baseline and measurement
│   ├── 6_edge_cases.py           # Adversarial and boundary prompt testing
│   ├── 7_evaluation.py           # Final system health report
│   └── sqlite.ipynb              # Notebook for ad-hoc SQLite exploration
│
├── data/
│   ├── documents/                # Source .txt files indexed into FAISS
│   │   ├── ai_agents.txt
│   │   ├── ai_in_software_development.txt
│   │   ├── ai_regulation_ethics.txt
│   │   ├── ai_safety_and_ethics.txt
│   │   ├── large_language_models.txt
│   │   ├── llmops_and_infrastructure.txt
│   │   ├── multimodal_ai.txt
│   │   ├── rag_and_vector_databases.txt
│   │   └── retrieval_augmented_generation.txt
│   ├── prompts.csv               # 105 simulation prompts
│   └── reference_queries.json    # 20 fixed queries for drift baseline
│
├── logs/
│   ├── rag_logs.csv              # Per-request log (CSV mirror of SQLite)
│   ├── edge_cases_log.csv        # Edge case test outcomes
│   ├── drift_baseline.json       # Saved embedding vectors for drift baseline
│   ├── monitoring.db             # SQLite database (git-ignored, generated locally)
│   └── reports/
│       ├── task3_tracking_report.json
│       ├── task4_rag_evaluation.json
│       ├── task5_drift_report.json
│       ├── task7_final_evaluation.json
│       └── task7_summary.txt
│
└── vectorstore/
    └── faiss_index/              # Generated FAISS index (git-ignored, .gitkeep tracked)
```

---

## Installation and Setup

### Prerequisites

- Python 3.11 or later
- An Azure OpenAI resource with a chat deployment (e.g., `gpt-4o`) and an embedding deployment (e.g., `text-embedding-3-small`)
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
python src/1.1_setup.py
```

This reads all `.txt` files under `data/documents/`, splits them into 800-character chunks with 100-character overlap, embeds them using Azure OpenAI, and saves the index to `vectorstore/faiss_index/`.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure endpoint URL (e.g., `https://my-resource.openai.azure.com/`) |
| `AZURE_OPENAI_API_VERSION` | No | `2024-02-01` | API version string |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | `gpt-4o-mini` | Deployment name for the chat model |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | `text-embedding-ada-002` | Deployment name for embeddings |

See [.env-sample](.env-sample) for the full template.

---

## Usage

Run the tasks in order. Each step depends on outputs from the previous ones.

### Step 1 — Index documents and verify retrieval

```bash
python src/1.1_setup.py        # builds vectorstore/faiss_index/
python src/1.2_rag_pipeline.py # smoke-tests the RAG pipeline
```

### Step 2 — Simulate queries and generate monitoring data

```bash
python src/2.2_simulate_queries.py
```

Fires all 105 prompts from `data/prompts.csv` through the RAG pipeline, alternating between prompt v1 and v2. Writes results to `logs/rag_logs.csv` and `logs/monitoring.db`.

> **Note:** `2.2_simulate_queries.py` imports `rag_pipeline` and `logger_middleware` by module name. If you encounter `ModuleNotFoundError`, add `src/` to your `PYTHONPATH` and create module aliases:
> ```bash
> # From the repo root
> cp src/1.2_rag_pipeline.py src/rag_pipeline.py
> cp src/2.1_logger_middleware.py src/logger_middleware.py
> PYTHONPATH=src python src/2.2_simulate_queries.py
> ```

### Step 3 — Token and latency tracking

```bash
python src/3_token_latency_tracker.py
```

Reads `logs/monitoring.db` and writes `logs/reports/task3_tracking_report.json` with:
- Total and daily token usage (input / output / total)
- Cost estimates using gpt-4o pricing ($0.005 / 1K input tokens, $0.015 / 1K output tokens)
- Latency percentiles (avg, min, max, p95)
- Side-by-side prompt v1 vs v2 comparison

### Step 4 — RAG quality evaluation

```bash
python src/4_rag_evaluator.py
```

Uses an Azure OpenAI LLM judge to score each logged request on three dimensions (0.0–1.0 each):

| Metric | Description |
|--------|-------------|
| Context Relevance | How well the retrieved chunks match the query |
| Faithfulness | Whether the response is grounded in the retrieved context (hallucination proxy) |
| Groundedness | How fully the answer is supported by the retrieved evidence |

Writes scores to the `rag_evaluations` table in SQLite and to `logs/reports/task4_rag_evaluation.json`.

### Step 5 — Embedding drift detection

```bash
# First time only — save baseline embeddings
python src/5_drift_detector.py --mode baseline

# On subsequent runs — measure drift against the baseline
python src/5_drift_detector.py --mode measure
```

Re-embeds the 20 fixed queries from `data/reference_queries.json` and computes cosine similarity against the saved baseline. Drift thresholds:

| Drift Score | Status |
|-------------|--------|
| < 0.10 | HEALTHY |
| 0.10 – 0.20 | WARNING |
| > 0.20 | ANOMALY |

### Step 6 — Edge case testing

```bash
python src/6_edge_cases.py
```

Runs 25 adversarial and boundary prompts across 9 categories (Safety, Off-Topic, Vague, Very Long, Empty-Like, Adversarial, Unanswerable, Repetitive, Mixed-Language). Results are written to `logs/edge_cases_log.csv`.

> **Note:** Same import path caveat as Step 2 applies here.

### Step 7 — Final system health report

```bash
python src/7_evaluation.py
```

Aggregates all monitoring tables into a composite health score (0–100) graded A–D across five dimensions: reliability, latency, RAG quality, drift, and token efficiency. Outputs `logs/reports/task7_final_evaluation.json` and `logs/reports/task7_summary.txt`.

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
data/documents/  ──►  1.1_setup.py  ──►  vectorstore/faiss_index/
                                                      │
data/prompts.csv ──►  2.2_simulate  ──►  2.1_logger  ──►  logs/monitoring.db
                              │                                    │
                    1.2_rag_pipeline                               │
                    (Azure OpenAI)                                 │
                                                                   ▼
                                              3_token_latency_tracker  ──►  reports/
                                              4_rag_evaluator          ──►  reports/
                                              5_drift_detector         ──►  reports/
                                              7_evaluation             ──►  reports/
                                                                   │
                                              dashboard.py  ◄───────┘
                                              (Streamlit)
```

**Design choices:**

- **File-based storage** (FAISS, CSV, SQLite, JSON) keeps the setup self-contained with no external infrastructure dependencies.
- **Numbered task scripts** make the learning sequence explicit, at the cost of a non-standard module layout.
- **LLM-as-judge evaluation** makes the scoring logic fully visible in the repository instead of relying on a black-box library, but adds extra Azure OpenAI calls and cost.
- **Synchronous execution** is simple and easy to debug locally, but not designed for high-throughput serving.

---

## Known Limitations

- **Import path mismatch.** Scripts `2.2_simulate_queries.py` and `6_edge_cases.py` import `rag_pipeline` and `logger_middleware`, but the actual files are named `1.2_rag_pipeline.py` and `2.1_logger_middleware.py`. See the workaround in [Step 2](#step-2--simulate-queries-and-generate-monitoring-data).
- **`logs/monitoring.db` is not committed.** The SQLite database is git-ignored and must be generated locally by running `2.2_simulate_queries.py`. The dashboard and all report scripts depend on it.
- **FAISS index not committed.** `vectorstore/faiss_index/` tracks only `.gitkeep`. Run `1.1_setup.py` before issuing RAG queries.
- **No automated test suite.** There are no pytest or unittest tests. Quality assurance relies on the manual task workflow and edge case scripts.
- **No formal Python version constraint.** The project was developed with Python 3.11.9. No `pyproject.toml` or `setup.py` declares a minimum version.
- **Cost per evaluation run.** The LLM-as-judge in `4_rag_evaluator.py` calls Azure OpenAI once per logged request. Evaluate selectively or on a sample to manage cost.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run the full task sequence (`1.1` → `2.2` → `3` → `4` → `5` → `7`) against your changes to confirm the pipeline still works end-to-end.
3. If you add a new monitoring metric, persist it to `logs/monitoring.db` via `LLMLogger` and expose it in `dashboard.py`.
4. Update this README if you change environment variables, generated artifacts, or the task sequence.
5. Keep changes aligned with the file-based, single-user design. Proposals to move to a managed backend belong in a discussion first.
6. Open a pull request with a clear description of what changed and why.

---

## License

No license file has been added to this repository. Add one before distributing or open-sourcing the project.
