import subprocess
import sys
import os

print("=" * 65)
print("  ENTERPRISE LLM MONITORING & OBSERVABILITY PLATFORM")
print("  Running all tasks in sequence...")
print("=" * 65)

# ── Task 1: Build FAISS index from documents ───────────────────────────────────
print("\n[TASK 1] Dataset Creation & FAISS Index Setup")
print("-" * 65)
subprocess.run([sys.executable, "src/a_setup.py"], check=True)

# ── Task 2: Run simulation — fires 105 prompts through RAG + logs everything ───
print("\n[TASK 2] Logging Middleware — Simulating 105 Queries")
print("-" * 65)
subprocess.run([sys.executable, "src/d_simulate_queries.py"], check=True)

# ── Task 3: Token, latency & prompt version analysis ──────────────────────────
print("\n[TASK 3] Token, Latency & Prompt Version Tracking")
print("-" * 65)
subprocess.run([sys.executable, "src/e_token_latency_tracker.py"], check=True)

# ── Task 4: RAG evaluation metrics (LLM-as-judge) ─────────────────────────────
print("\n[TASK 4] RAG Evaluation Metrics")
print("-" * 65)
subprocess.run([sys.executable, "src/f_rag_evaluator.py"], check=True)

# ── Task 5: Embedding drift — create baseline then measure ────────────────────
print("\n[TASK 5] Embedding Drift Detection — Creating Baseline")
print("-" * 65)
subprocess.run([sys.executable, "src/g_drift_detector.py", "--mode", "baseline"], check=True)

print("\n[TASK 5] Embedding Drift Detection — Measuring Drift")
print("-" * 65)
subprocess.run([sys.executable, "src/g_drift_detector.py", "--mode", "measure"], check=True)

# ── Task 7A: Edge case testing ─────────────────────────────────────────────────
print("\n[TASK 7A] Edge Case Testing")
print("-" * 65)
subprocess.run([sys.executable, "src/h_edge_cases.py"], check=True)

# ── Task 7B: Final system evaluation report ────────────────────────────────────
print("\n[TASK 7B] System Evaluation & Final Report")
print("-" * 65)
subprocess.run([sys.executable, "src/i_evaluation.py"], check=True)

# ── Task 6: Launch Streamlit dashboard (last — keeps terminal open) ────────────
print("\n[TASK 6] Launching Monitoring Dashboard")
print("-" * 65)
print("  Dashboard starting at http://localhost:8501")
print("  Press Ctrl+C to stop the dashboard\n")
subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
