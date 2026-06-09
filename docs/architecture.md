# System Architecture

This document describes the architecture of the LLM Monitoring Platform using Mermaid diagrams.

---

## 1. End-to-End Data Flow

The platform is organized as a sequential pipeline. Raw documents flow into a FAISS vector index; queries flow through the RAG engine and are logged to SQLite; monitoring scripts read from SQLite and write reports; the Streamlit dashboard consumes everything.

```mermaid
flowchart TD
    %% ── External Services ──────────────────────────────────────────────
    subgraph AZURE["Azure OpenAI  (external)"]
        EmbedAPI["Embedding Model\ntext-embedding-3-small / ada-002"]
        ChatAPI["Chat Model\ngpt-4o / gpt-4o-mini"]
    end

    %% ── Data Sources ────────────────────────────────────────────────────
    subgraph DATA["Data Sources"]
        Docs["data/documents/\n9 topic .txt files"]
        Prompts["data/prompts.csv\n105 simulation queries"]
        RefQ["data/reference_queries.json\n20 fixed drift queries"]
    end

    %% ── Core Pipeline ───────────────────────────────────────────────────
    subgraph PIPELINE["Core Pipeline  (src/)"]
        Setup["1.1_setup.py\nDocument Indexer\nLoad → Split → Embed → Store"]
        RAG["1.2_rag_pipeline.py\nRAG Query Engine\nEmbed query → Retrieve → Generate"]
        Logger["2.1_logger_middleware.py\nLLMLogger\n17 fields per request"]
        Sim["2.2_simulate_queries.py\nQuery Simulator\n105 prompts · prompt v1/v2"]
    end

    %% ── Storage ─────────────────────────────────────────────────────────
    subgraph STORAGE["Storage"]
        FAISS["vectorstore/faiss_index/\nFAISS Vector Index"]
        CSVLOG["logs/rag_logs.csv\nRequest Log"]
        EDGECSV["logs/edge_cases_log.csv\nEdge Case Results"]
        DRIFTBASE["logs/drift_baseline.json\nBaseline Embedding Vectors"]

        subgraph DB["logs/monitoring.db  (SQLite)"]
            LLMLogs["llm_logs\nrequest_id · query · response\ninput/output tokens · latency\nsafety_flag · error_code · prompt_version"]
            RAGEvals["rag_evaluations\ncontext_relevance · faithfulness\ngroundedness · hallucination_detected\noverall_score · quality_flags"]
            DriftLogs["drift_logs\nquery · drift_score\ncosine_similarity · status\nmeasured_at · overall_status"]
        end
    end

    %% ── Monitoring & Evaluation ─────────────────────────────────────────
    subgraph MONITORING["Monitoring & Evaluation  (src/)"]
        Tracker["3_token_latency_tracker.py\nToken & Cost Analysis\nLatency Percentiles · v1 vs v2"]
        Evaluator["4_rag_evaluator.py\nLLM-as-Judge Scorer\nRelevance · Faithfulness · Groundedness"]
        DriftDet["5_drift_detector.py\nEmbedding Drift Detector\nbaseline mode / measure mode"]
        EdgeCases["6_edge_cases.py\nAdversarial Tester\n25 prompts · 9 categories"]
        FinalEval["7_evaluation.py\nSystem Health Report\nComposite score · Grade A–D"]
    end

    %% ── Reports ─────────────────────────────────────────────────────────
    subgraph REPORTS["logs/reports/"]
        R3["task3_tracking_report.json\nToken totals · daily usage\ncost estimate · latency p95"]
        R4["task4_rag_evaluation.json\nPer-request RAG scores\nhallucination counts"]
        R5["task5_drift_report.json\nPer-query drift scores\nanomaly alerts"]
        R7["task7_final_evaluation.json\ntask7_summary.txt\nHealth score · grade · dimension breakdown"]
    end

    %% ── Dashboard ───────────────────────────────────────────────────────
    subgraph DASH["Dashboard"]
        Streamlit["dashboard.py\nStreamlit App · localhost:8501"]
        subgraph TABS["6 Tabs"]
            direction LR
            T1["Overview\nKPIs + Health"]
            T2["Token & Cost\nDaily Trends"]
            T3["Latency\nDistributions"]
            T4["RAG Evaluation\nScore Trends"]
            T5["Drift Tracking\nAnomaly Markers"]
            T6["Logs Viewer\nPaginated Table"]
        end
    end

    %% ── Indexing Flow ───────────────────────────────────────────────────
    Docs -->|"load & chunk"| Setup
    Setup -->|"embed 800-char chunks"| EmbedAPI
    EmbedAPI -->|"1536-dim vectors"| FAISS

    %% ── Query & Logging Flow ────────────────────────────────────────────
    Prompts --> Sim
    Sim -->|"one query at a time"| RAG
    RAG -->|"embed query"| EmbedAPI
    EmbedAPI -->|"query vector"| FAISS
    FAISS -->|"top-k=3 chunks"| RAG
    RAG -->|"query + context"| ChatAPI
    ChatAPI -->|"response + token counts"| RAG
    RAG -->|"structured result"| Logger
    Logger --> CSVLOG
    Logger --> LLMLogs

    %% ── Edge Case Flow ──────────────────────────────────────────────────
    EdgeCases -->|"25 adversarial prompts"| RAG
    EdgeCases --> Logger
    Logger --> EDGECSV

    %% ── Drift Detection Flow ────────────────────────────────────────────
    RefQ -->|"20 fixed queries"| DriftDet
    DriftDet -->|"re-embed queries"| EmbedAPI
    DriftDet -->|"save/load baseline"| DRIFTBASE
    DriftDet -->|"cosine similarity"| DriftLogs

    %% ── Monitoring Reads from SQLite ────────────────────────────────────
    LLMLogs --> Tracker
    LLMLogs --> Evaluator
    LLMLogs --> FinalEval
    DriftLogs --> FinalEval
    RAGEvals --> FinalEval

    %% ── LLM-as-Judge Calls ──────────────────────────────────────────────
    Evaluator -->|"judge prompt per request"| ChatAPI
    ChatAPI -->|"JSON scores"| Evaluator
    Evaluator --> RAGEvals

    %% ── Report Generation ───────────────────────────────────────────────
    Tracker --> R3
    RAGEvals --> R4
    DriftLogs --> R5
    FinalEval --> R7

    %% ── Dashboard Reads ─────────────────────────────────────────────────
    DB -->|"read all tables"| Streamlit
    R3 --> Streamlit
    R4 --> Streamlit
    R5 --> Streamlit
    R7 --> Streamlit
    Streamlit --> TABS
```

---

## 2. SQLite Database Schema

Three tables are written by different pipeline stages and read by monitoring scripts and the dashboard.

```mermaid
erDiagram
    llm_logs {
        INTEGER id PK
        TEXT    request_id
        TEXT    timestamp
        TEXT    user_query
        TEXT    retrieved_context
        TEXT    retrieved_sources
        TEXT    model_name
        TEXT    prompt_version
        TEXT    model_response
        INTEGER input_tokens
        INTEGER output_tokens
        INTEGER total_tokens
        REAL    retrieval_latency
        REAL    llm_latency
        REAL    end_to_end_latency
        INTEGER top_k
        INTEGER safety_flag
        TEXT    error_code
    }

    rag_evaluations {
        INTEGER id PK
        TEXT    request_id FK
        TEXT    user_query
        REAL    context_relevance_score
        TEXT    context_relevance_reason
        REAL    faithfulness_score
        TEXT    faithfulness_reason
        INTEGER hallucination_detected
        REAL    groundedness_score
        TEXT    groundedness_reason
        REAL    overall_score
        TEXT    quality_flags
        TEXT    evaluated_at
    }

    drift_logs {
        INTEGER id PK
        TEXT    measured_at
        TEXT    baseline_created
        TEXT    query
        REAL    drift_score
        REAL    cosine_similarity
        TEXT    status
        REAL    avg_drift_run
        TEXT    overall_status
    }

    llm_logs ||--o{ rag_evaluations : "request_id"
```

---

## 3. Task Execution Sequence

Scripts must be run in order; each step produces artifacts consumed by later steps.

```mermaid
sequenceDiagram
    actor User
    participant S1  as 1.1_setup.py
    participant S12 as 1.2_rag_pipeline.py
    participant S22 as 2.2_simulate_queries.py
    participant LOG as 2.1_logger_middleware.py
    participant S3  as 3_token_latency_tracker.py
    participant S4  as 4_rag_evaluator.py
    participant S5  as 5_drift_detector.py
    participant S6  as 6_edge_cases.py
    participant S7  as 7_evaluation.py
    participant DB  as monitoring.db
    participant AZ  as Azure OpenAI
    participant UI  as dashboard.py

    User->>S1: python src/1.1_setup.py
    S1->>AZ: embed document chunks
    AZ-->>S1: vectors
    S1-->>User: vectorstore/faiss_index/ written

    User->>S12: python src/1.2_rag_pipeline.py
    S12->>AZ: embed query + generate response
    AZ-->>S12: response
    S12-->>User: smoke test output

    User->>S22: python src/2.2_simulate_queries.py
    loop 105 prompts
        S22->>S12: run RAG query
        S12->>AZ: embed + generate
        AZ-->>S12: response
        S12->>LOG: structured result
        LOG->>DB: INSERT into llm_logs
        LOG-->>S22: logged
    end
    S22-->>User: logs/rag_logs.csv + monitoring.db

    User->>S5: python src/5_drift_detector.py --mode baseline
    S5->>AZ: embed 20 reference queries
    AZ-->>S5: baseline vectors
    S5-->>User: logs/drift_baseline.json written

    User->>S3: python src/3_token_latency_tracker.py
    S3->>DB: SELECT from llm_logs
    DB-->>S3: raw records
    S3-->>User: task3_tracking_report.json

    User->>S4: python src/4_rag_evaluator.py
    loop per logged request
        S4->>DB: SELECT from llm_logs
        S4->>AZ: judge prompt (relevance/faithfulness/groundedness)
        AZ-->>S4: JSON scores
        S4->>DB: INSERT into rag_evaluations
    end
    S4-->>User: task4_rag_evaluation.json

    User->>S5: python src/5_drift_detector.py --mode measure
    S5->>AZ: re-embed 20 reference queries
    AZ-->>S5: current vectors
    S5->>DB: INSERT into drift_logs
    S5-->>User: task5_drift_report.json

    User->>S6: python src/6_edge_cases.py
    loop 25 adversarial prompts
        S6->>S12: run RAG query
        S12->>AZ: embed + generate
        AZ-->>S12: response
        S12->>LOG: structured result
        LOG->>DB: INSERT into llm_logs
    end
    S6-->>User: logs/edge_cases_log.csv

    User->>S7: python src/7_evaluation.py
    S7->>DB: SELECT from all tables
    DB-->>S7: aggregated data
    S7-->>User: task7_final_evaluation.json + task7_summary.txt

    User->>UI: streamlit run dashboard.py
    UI->>DB: read llm_logs, rag_evaluations, drift_logs
    UI->>UI: load reports JSON
    UI-->>User: interactive dashboard on localhost:8501
```

---

## 4. Component Responsibilities

| Component | Reads | Writes | Azure OpenAI calls |
|-----------|-------|--------|--------------------|
| `1.1_setup.py` | `data/documents/` | `vectorstore/faiss_index/` | Embeddings |
| `1.2_rag_pipeline.py` | FAISS index | — | Embeddings + Chat |
| `2.1_logger_middleware.py` | RAG result dict | `llm_logs`, `rag_logs.csv` | None |
| `2.2_simulate_queries.py` | `data/prompts.csv` | via Logger | Indirect via RAG |
| `3_token_latency_tracker.py` | `llm_logs` | `task3_*.json` | None |
| `4_rag_evaluator.py` | `llm_logs` | `rag_evaluations`, `task4_*.json` | Chat (judge) |
| `5_drift_detector.py` | `reference_queries.json`, baseline | `drift_logs`, `task5_*.json`, baseline | Embeddings |
| `6_edge_cases.py` | hardcoded prompts | `edge_cases_log.csv` via Logger | Indirect via RAG |
| `7_evaluation.py` | all three SQLite tables | `task7_*.json`, `task7_summary.txt` | None |
| `dashboard.py` | `monitoring.db`, all JSON reports | — | None |
