# System Architecture

---

## 1. End-to-End Data Flow

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        Docs["documents/\n9 .txt files"]
        Prompts["prompts.csv\n105 queries"]
        RefQ["reference_queries.json\n20 fixed queries"]
    end

    subgraph AZURE["Azure OpenAI"]
        Embed["Embedding Model"]
        Chat["Chat Model"]
    end

    subgraph CORE["Core Pipeline"]
        Setup["a_setup.py\nIndex Builder"]
        RAG["b_rag_pipeline.py\nRAG Engine"]
        Logger["c_logger_middleware.py\nRequest Logger"]
        Sim["d_simulate_queries.py\nQuery Simulator"]
    end

    subgraph STORE["Storage"]
        FAISS[("faiss_index/")]
        DB[("monitoring.db\nSQLite")]
        Baseline["drift_baseline.json"]
    end

    subgraph ANALYSIS["Analysis"]
        Tracker["f — Token & Latency"]
        Evaluator["g — RAG Evaluator"]
        Drift["h — Drift Detector"]
        Edges["i — Edge Cases"]
        Eval["e — Health Report"]
    end

    subgraph OUT["Output"]
        Reports["logs/reports/\n4 JSON reports"]
        Dash["dashboard.py\nlocalhost:8501"]
    end

    Docs --> Setup --> Embed --> FAISS
    Prompts --> Sim --> RAG
    RAG <--> Embed
    RAG <--> FAISS
    RAG <--> Chat
    RAG --> Logger --> DB

    RefQ --> Drift
    Drift <--> Embed
    Drift --- Baseline
    Drift --> DB

    Edges --> RAG

    DB --> Tracker & Evaluator & Drift & Eval
    Evaluator <--> Chat
    Evaluator --> DB

    Tracker & Evaluator & Drift & Eval --> Reports
    DB --> Dash
    Reports --> Dash
```

---

## 2. Script Execution Order

```mermaid
flowchart LR
    A["a_setup.py\nBuild index"] -->
    B["b_rag_pipeline.py\nSmoke test"] -->
    D["d_simulate_queries.py\n105 queries"] -->
    F["f_token_latency_tracker.py"] -->
    G["g_rag_evaluator.py"] -->
    H["h_drift_detector.py"] -->
    I["i_edge_cases.py"] -->
    E["e_evaluation.py\nFinal report"]

    style E fill:#2d6a4f,color:#fff
```

> `c_logger_middleware.py` and `config.py` are shared modules — not run directly.

---

## 3. SQLite Schema

Three tables are written by different pipeline stages and joined for the final report.

```mermaid
erDiagram
    llm_logs {
        INTEGER id PK
        TEXT    request_id
        TEXT    timestamp
        TEXT    user_query
        TEXT    model_response
        INTEGER input_tokens
        INTEGER output_tokens
        REAL    retrieval_latency
        REAL    llm_latency
        REAL    end_to_end_latency
        TEXT    prompt_version
        INTEGER safety_flag
        TEXT    error_code
    }

    rag_evaluations {
        INTEGER id PK
        TEXT    request_id FK
        REAL    context_relevance_score
        REAL    faithfulness_score
        REAL    groundedness_score
        INTEGER hallucination_detected
        REAL    overall_score
        TEXT    evaluated_at
    }

    drift_logs {
        INTEGER id PK
        TEXT    measured_at
        TEXT    query
        REAL    drift_score
        REAL    cosine_similarity
        TEXT    status
    }

    llm_logs ||--o{ rag_evaluations : "request_id"
```

---

## 4. Component Responsibilities

| Script | Reads | Writes | Azure OpenAI |
|--------|-------|--------|--------------|
| `a_setup.py` | `data/documents/` | `faiss_index/` | Embeddings |
| `b_rag_pipeline.py` | FAISS index | — | Embeddings + Chat |
| `c_logger_middleware.py` | RAG result | `llm_logs`, `rag_logs.csv` | — |
| `d_simulate_queries.py` | `prompts.csv` | via Logger | Indirect via RAG |
| `e_token_latency_tracker.py` | `llm_logs` | `task3_*.json` | — |
| `f_rag_evaluator.py` | `llm_logs` | `rag_evaluations`, `task4_*.json` | Chat (judge) |
| `g_drift_detector.py` | `reference_queries.json` | `drift_logs`, `drift_baseline.json`, `task5_*.json` | Embeddings |
| `h_edge_cases.py` | hardcoded prompts | `edge_cases_log.csv` via Logger | Indirect via RAG |
| `i_evaluation.py` | all three SQLite tables | `task7_*.json`, `task7_summary.txt` | — |
| `dashboard.py` | `monitoring.db`, all JSON reports | — | — |
| `config.py` | `.env` | — | — |
