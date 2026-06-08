"""
Task 6: LLM Monitoring Dashboard
----------------------------------
Streamlit-based visual analytics dashboard.
Reads from logs/monitoring.db and visualizes:

  Tab 1 — Overview        : KPI cards + system health summary
  Tab 2 — Token & Cost    : Usage charts, daily breakdown, cost estimation
  Tab 3 — Latency         : Response time charts, box plots, slow query alerts
  Tab 4 — RAG Evaluation  : Faithfulness, relevance, groundedness scores
  Tab 5 — Drift Tracking  : Cosine similarity graphs, anomaly markers
  Tab 6 — Logs Viewer     : Paginated, filterable raw logs table

Run with:
    streamlit run dashboard.py
"""

import os
import json
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config — must be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title  = "LLM Monitoring Platform",
    page_icon   = "🔭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_FILE = "logs/monitoring.db"

# ── Color palette ──────────────────────────────────────────────────────────────
COLOR_PRIMARY  = "#4F8EF7"
COLOR_SUCCESS  = "#22C55E"
COLOR_WARNING  = "#F59E0B"
COLOR_DANGER   = "#EF4444"
COLOR_NEUTRAL  = "#94A3B8"
CHART_COLORS   = [COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER]


# ══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F172A; color: #E2E8F0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4F8EF7;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 0.82rem;
        margin-top: 6px;
    }

    /* Status badges */
    .badge-healthy  { background:#14532D; color:#4ADE80;
                      padding:3px 10px; border-radius:20px; font-size:0.75rem; }
    .badge-warning  { background:#78350F; color:#FCD34D;
                      padding:3px 10px; border-radius:20px; font-size:0.75rem; }
    .badge-anomaly  { background:#7F1D1D; color:#FCA5A5;
                      padding:3px 10px; border-radius:20px; font-size:0.75rem; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #CBD5E1;
        border-left: 3px solid #4F8EF7;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8EF7 !important;
        color: white !important;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly { border-radius: 10px; }

    /* Alert box */
    .alert-box {
        background: #450A0A;
        border: 1px solid #EF4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #FCA5A5;
        font-size: 0.85rem;
    }
    .warning-box {
        background: #451A03;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #FCD34D;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "#1E293B",
    plot_bgcolor  = "#1E293B",
    font          = dict(color="#CBD5E1", size=12),
    margin        = dict(l=20, r=20, t=40, b=20),
    xaxis         = dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis         = dict(gridcolor="#334155", zerolinecolor="#334155"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading (cached for performance)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)   # Refresh data every 30 seconds
def load_logs() -> pd.DataFrame:
    if not os.path.isfile(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql("SELECT * FROM llm_logs ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"]      = df["timestamp"].dt.date.astype(str)
    return df


@st.cache_data(ttl=30)
def load_evaluations() -> pd.DataFrame:
    if not os.path.isfile(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM rag_evaluations ORDER BY evaluated_at DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_drift() -> pd.DataFrame:
    if not os.path.isfile(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM drift_logs ORDER BY measured_at DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def kpi(label: str, value, delta: str = ""):
    delta_html = f'<div class="kpi-delta" style="color:#94A3B8">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔭 LLM Monitor")
    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Database**")
    db_exists = os.path.isfile(DB_FILE)
    if db_exists:
        size_kb = round(os.path.getsize(DB_FILE) / 1024, 1)
        st.success(f"Connected  ({size_kb} KB)")
    else:
        st.error("No DB found")
        st.info("Run simulate_queries.py first")

    st.markdown("---")
    st.markdown("**Last Updated**")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='color:#E2E8F0; font-size:1.8rem; font-weight:700; margin-bottom:4px'>
  🔭 Enterprise LLM Monitoring Platform
</h1>
<p style='color:#64748B; font-size:0.9rem; margin-bottom:20px'>
  Real-time observability for LLM & RAG systems
</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════════

df_logs  = load_logs()
df_evals = load_evaluations()
df_drift = load_drift()

no_data = df_logs.empty


# ══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🪙 Token & Cost",
    "⏱️ Latency",
    "🎯 RAG Evaluation",
    "📡 Drift Tracking",
    "📋 Logs Viewer",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if no_data:
        st.warning("No log data found. Run `simulate_queries.py` first.")
    else:
        success_df = df_logs[df_logs["error_code"] == ""]
        error_df   = df_logs[df_logs["error_code"] != ""]
        safety_df  = df_logs[df_logs["safety_flag"] == 1]

        total_tokens  = int(success_df["total_tokens"].sum())
        avg_latency   = round(success_df["end_to_end_latency"].mean(), 3)
        total_cost    = round(total_tokens * 0.000002, 4)
        error_rate    = round(len(error_df) / len(df_logs) * 100, 1)

        # ── KPI Row ────────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi("Total Requests",   f"{len(df_logs):,}")
        with c2: kpi("Total Tokens",     f"{total_tokens:,}")
        with c3: kpi("Avg Latency",      f"{avg_latency}s")
        with c4: kpi("Est. Cost",        f"${total_cost:.4f}")
        with c5: kpi("Error Rate",       f"{error_rate}%")

        st.markdown("---")
        col_l, col_r = st.columns(2)

        # ── Requests over time ─────────────────────────────────────────────────
        with col_l:
            st.markdown('<div class="section-title">Requests Over Time</div>',
                        unsafe_allow_html=True)
            daily = df_logs.groupby("date").size().reset_index(name="count")
            fig = px.bar(daily, x="date", y="count",
                         color_discrete_sequence=[COLOR_PRIMARY])
            fig.update_layout(**PLOTLY_LAYOUT, title="Daily Request Volume")
            st.plotly_chart(fig, use_container_width=True)

        # ── Prompt version pie ─────────────────────────────────────────────────
        with col_r:
            st.markdown('<div class="section-title">Prompt Version Split</div>',
                        unsafe_allow_html=True)
            ver_counts = df_logs["prompt_version"].value_counts().reset_index()
            ver_counts.columns = ["version", "count"]
            fig = px.pie(ver_counts, names="version", values="count",
                         color_discrete_sequence=CHART_COLORS, hole=0.4)
            fig.update_layout(**PLOTLY_LAYOUT, title="v1 vs v2 Usage")
            st.plotly_chart(fig, use_container_width=True)

        # ── System alerts ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title">System Alerts</div>',
                    unsafe_allow_html=True)
        if len(error_df):
            st.markdown(f'<div class="alert-box">🚨 {len(error_df)} request(s) returned errors</div>',
                        unsafe_allow_html=True)
        if len(safety_df):
            st.markdown(f'<div class="warning-box">⚠️ {len(safety_df)} request(s) triggered safety flags</div>',
                        unsafe_allow_html=True)
        if not len(error_df) and not len(safety_df):
            st.success("✅ No active alerts — system is healthy")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TOKEN & COST
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if no_data:
        st.warning("No data yet.")
    else:
        sdf = df_logs[df_logs["error_code"] == ""].copy()

        # ── KPIs ───────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Total Input Tokens",  f"{int(sdf['input_tokens'].sum()):,}")
        with c2: kpi("Total Output Tokens", f"{int(sdf['output_tokens'].sum()):,}")
        with c3: kpi("Avg Tokens / Request",f"{sdf['total_tokens'].mean():.0f}")
        with c4: kpi("Total Cost (USD)",    f"${sdf['total_tokens'].sum() * 0.000002:.4f}")

        col_l, col_r = st.columns(2)

        # ── Daily token usage ──────────────────────────────────────────────────
        with col_l:
            st.markdown('<div class="section-title">Daily Token Usage</div>',
                        unsafe_allow_html=True)
            daily_tok = sdf.groupby("date")[["input_tokens", "output_tokens"]].sum().reset_index()
            fig = go.Figure()
            fig.add_bar(x=daily_tok["date"], y=daily_tok["input_tokens"],
                        name="Input",  marker_color=COLOR_PRIMARY)
            fig.add_bar(x=daily_tok["date"], y=daily_tok["output_tokens"],
                        name="Output", marker_color=COLOR_SUCCESS)
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Input vs Output Tokens per Day",
                              barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

        # ── Cost per request over time ─────────────────────────────────────────
        with col_r:
            st.markdown('<div class="section-title">Cost Per Request</div>',
                        unsafe_allow_html=True)
            sdf["cost"] = sdf["total_tokens"] * 0.000002
            fig = px.scatter(sdf, x="timestamp", y="cost",
                             color="prompt_version",
                             color_discrete_sequence=CHART_COLORS,
                             opacity=0.7)
            fig.update_layout(**PLOTLY_LAYOUT, title="Cost per Request Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # ── Token distribution histogram ───────────────────────────────────────
        st.markdown('<div class="section-title">Token Distribution</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(sdf, x="total_tokens", nbins=30,
                           color="prompt_version",
                           color_discrete_sequence=CHART_COLORS,
                           barmode="overlay", opacity=0.75)
        fig.update_layout(**PLOTLY_LAYOUT, title="Token Count Distribution by Prompt Version")
        st.plotly_chart(fig, use_container_width=True)

        # ── Version cost comparison ────────────────────────────────────────────
        st.markdown('<div class="section-title">Prompt Version Cost Comparison</div>',
                    unsafe_allow_html=True)
        ver_cost = sdf.groupby("prompt_version")["cost"].agg(["mean","sum","count"]).reset_index()
        ver_cost.columns = ["Version", "Avg Cost", "Total Cost", "Requests"]
        ver_cost["Avg Cost"]   = ver_cost["Avg Cost"].map("${:.6f}".format)
        ver_cost["Total Cost"] = ver_cost["Total Cost"].map("${:.4f}".format)
        st.dataframe(ver_cost, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LATENCY
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if no_data:
        st.warning("No data yet.")
    else:
        sdf = df_logs[df_logs["error_code"] == ""].copy()
        SLOW_THRESHOLD = 10.0

        # ── KPIs ───────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Avg E2E Latency",     f"{sdf['end_to_end_latency'].mean():.3f}s")
        with c2: kpi("P95 E2E Latency",     f"{sdf['end_to_end_latency'].quantile(0.95):.3f}s")
        with c3: kpi("Avg LLM Latency",     f"{sdf['llm_latency'].mean():.3f}s")
        with c4: kpi("Avg Retrieval Latency",f"{sdf['retrieval_latency'].mean():.4f}s")

        col_l, col_r = st.columns(2)

        # ── Latency over time line chart ───────────────────────────────────────
        with col_l:
            st.markdown('<div class="section-title">End-to-End Latency Over Time</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_scatter(x=sdf["timestamp"], y=sdf["end_to_end_latency"],
                            mode="lines", name="E2E Latency",
                            line=dict(color=COLOR_PRIMARY, width=1.5))
            # Slow threshold line
            fig.add_hline(y=SLOW_THRESHOLD, line_dash="dot",
                          line_color=COLOR_DANGER,
                          annotation_text=f"Alert threshold ({SLOW_THRESHOLD}s)")
            fig.update_layout(**PLOTLY_LAYOUT, title="E2E Latency Timeline")
            st.plotly_chart(fig, use_container_width=True)

        # ── Box plots ─────────────────────────────────────────────────────────
        with col_r:
            st.markdown('<div class="section-title">Latency Distribution (Box Plots)</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            for col, name, color in [
                ("retrieval_latency", "Retrieval", COLOR_SUCCESS),
                ("llm_latency",       "LLM API",   COLOR_PRIMARY),
                ("end_to_end_latency","End-to-End", COLOR_WARNING),
            ]:
                fig.add_box(y=sdf[col], name=name,
                            marker_color=color, boxmean=True)
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Latency Distribution Comparison",
                              yaxis_title="Seconds")
            st.plotly_chart(fig, use_container_width=True)

        # ── Latency by prompt version ──────────────────────────────────────────
        st.markdown('<div class="section-title">Latency by Prompt Version</div>',
                    unsafe_allow_html=True)
        fig = px.box(sdf, x="prompt_version", y="end_to_end_latency",
                     color="prompt_version",
                     color_discrete_sequence=CHART_COLORS)
        fig.update_layout(**PLOTLY_LAYOUT, title="E2E Latency: v1 vs v2")
        st.plotly_chart(fig, use_container_width=True)

        # ── Slow query alert table ─────────────────────────────────────────────
        slow = sdf[sdf["end_to_end_latency"] > SLOW_THRESHOLD]
        if not slow.empty:
            st.markdown(f'<div class="alert-box">🚨 {len(slow)} slow request(s) above {SLOW_THRESHOLD}s threshold</div>',
                        unsafe_allow_html=True)
            st.dataframe(slow[["timestamp","user_query","end_to_end_latency","model_name"]],
                         use_container_width=True)
        else:
            st.success(f"✅ No requests exceeded the {SLOW_THRESHOLD}s threshold")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — RAG EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    if df_evals.empty:
        st.warning("No evaluation data. Run `rag_evaluator.py` first.")
    else:
        # ── KPIs ───────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Avg Context Relevance", f"{df_evals['context_relevance_score'].mean():.3f}")
        with c2: kpi("Avg Faithfulness",      f"{df_evals['faithfulness_score'].mean():.3f}")
        with c3: kpi("Avg Groundedness",      f"{df_evals['groundedness_score'].mean():.3f}")
        with c4: kpi("Hallucinations Found",  f"{int(df_evals['hallucination_detected'].sum())}")

        col_l, col_r = st.columns(2)

        # ── Bar chart: average scores ──────────────────────────────────────────
        with col_l:
            st.markdown('<div class="section-title">Average RAG Scores</div>',
                        unsafe_allow_html=True)
            metrics = pd.DataFrame({
                "Metric": ["Context Relevance", "Faithfulness", "Groundedness"],
                "Score":  [
                    df_evals["context_relevance_score"].mean(),
                    df_evals["faithfulness_score"].mean(),
                    df_evals["groundedness_score"].mean(),
                ]
            })
            fig = px.bar(metrics, x="Metric", y="Score",
                         color="Metric",
                         color_discrete_sequence=CHART_COLORS,
                         range_y=[0, 1])
            fig.add_hline(y=0.5, line_dash="dot", line_color=COLOR_DANGER,
                          annotation_text="Minimum threshold (0.5)")
            fig.update_layout(**PLOTLY_LAYOUT, title="Average Evaluation Scores",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Scatter: faithfulness vs relevance ─────────────────────────────────
        with col_r:
            st.markdown('<div class="section-title">Faithfulness vs Relevance</div>',
                        unsafe_allow_html=True)
            fig = px.scatter(df_evals,
                             x="context_relevance_score",
                             y="faithfulness_score",
                             color="overall_score",
                             color_continuous_scale="Blues",
                             opacity=0.75,
                             hover_data=["user_query"])
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Faithfulness vs Context Relevance",
                              xaxis_title="Context Relevance",
                              yaxis_title="Faithfulness")
            st.plotly_chart(fig, use_container_width=True)

        # ── Score distribution histograms ──────────────────────────────────────
        st.markdown('<div class="section-title">Score Distributions</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        for col, name, color in [
            ("context_relevance_score", "Context Relevance", COLOR_PRIMARY),
            ("faithfulness_score",      "Faithfulness",      COLOR_SUCCESS),
            ("groundedness_score",      "Groundedness",      COLOR_WARNING),
        ]:
            fig.add_histogram(x=df_evals[col], name=name,
                              marker_color=color, opacity=0.7, nbinsx=20)
        fig.update_layout(**PLOTLY_LAYOUT,
                          title="Distribution of Evaluation Scores",
                          barmode="overlay", xaxis_title="Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # ── Low quality requests ───────────────────────────────────────────────
        low_q = df_evals[df_evals["overall_score"] < 0.5]
        if not low_q.empty:
            st.markdown(f'<div class="warning-box">⚠️ {len(low_q)} request(s) scored below 0.5 overall</div>',
                        unsafe_allow_html=True)
            st.dataframe(low_q[["user_query","context_relevance_score",
                                 "faithfulness_score","groundedness_score","overall_score"]],
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — DRIFT TRACKING
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    if df_drift.empty:
        st.warning("No drift data. Run `drift_detector.py --mode baseline` then `--mode measure`.")
    else:
        # Summary by run
        run_summary = df_drift.groupby("measured_at").agg(
            avg_drift    = ("drift_score", "mean"),
            max_drift    = ("drift_score", "max"),
            anomaly_count= ("status",      lambda x: (x == "ANOMALY").sum()),
            overall      = ("overall_status", "first"),
        ).reset_index().sort_values("measured_at")

        latest = run_summary.iloc[-1]

        # ── KPIs ───────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Monitoring Runs",   f"{len(run_summary)}")
        with c2: kpi("Latest Avg Drift",  f"{latest['avg_drift']:.4f}")
        with c3: kpi("Latest Max Drift",  f"{latest['max_drift']:.4f}")
        with c4: kpi("Anomalies (latest)",f"{int(latest['anomaly_count'])}")

        col_l, col_r = st.columns(2)

        # ── Drift over time ────────────────────────────────────────────────────
        with col_l:
            st.markdown('<div class="section-title">Average Drift Score Over Time</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_scatter(x=run_summary["measured_at"],
                            y=run_summary["avg_drift"],
                            mode="lines+markers",
                            name="Avg Drift",
                            line=dict(color=COLOR_PRIMARY, width=2),
                            marker=dict(size=6))
            fig.add_hline(y=0.10, line_dash="dot", line_color=COLOR_WARNING,
                          annotation_text="Warning (0.10)")
            fig.add_hline(y=0.20, line_dash="dot", line_color=COLOR_DANGER,
                          annotation_text="Anomaly (0.20)")
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Drift Score Timeline",
                              yaxis_title="Drift Score")
            st.plotly_chart(fig, use_container_width=True)

        # ── Per-query drift bar chart ──────────────────────────────────────────
        with col_r:
            st.markdown('<div class="section-title">Per-Query Drift (Latest Run)</div>',
                        unsafe_allow_html=True)
            latest_run = df_drift[df_drift["measured_at"] == latest["measured_at"]].copy()
            latest_run["query_short"] = latest_run["query"].str[:40]
            color_map = {"HEALTHY": COLOR_SUCCESS,
                         "WARNING": COLOR_WARNING,
                         "ANOMALY": COLOR_DANGER}
            latest_run["color"] = latest_run["status"].map(color_map)

            fig = px.bar(latest_run.sort_values("drift_score", ascending=True),
                         x="drift_score", y="query_short",
                         orientation="h",
                         color="status",
                         color_discrete_map=color_map)
            fig.add_vline(x=0.10, line_dash="dot", line_color=COLOR_WARNING)
            fig.add_vline(x=0.20, line_dash="dot", line_color=COLOR_DANGER)
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Drift Score per Reference Query",
                              height=500,
                              xaxis_title="Drift Score",
                              yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        # ── Anomaly table ──────────────────────────────────────────────────────
        anomalies = df_drift[df_drift["status"] == "ANOMALY"]
        if not anomalies.empty:
            st.markdown(f'<div class="alert-box">🚨 {len(anomalies)} anomalous embeddings detected across all runs</div>',
                        unsafe_allow_html=True)
            st.dataframe(anomalies[["measured_at","query","drift_score",
                                     "cosine_similarity","status"]],
                         use_container_width=True)
        else:
            st.success("✅ No drift anomalies detected")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — LOGS VIEWER
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    if no_data:
        st.warning("No logs yet.")
    else:
        st.markdown('<div class="section-title">Filter Logs</div>',
                    unsafe_allow_html=True)

        # ── Filters ────────────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            versions    = ["All"] + sorted(df_logs["prompt_version"].unique().tolist())
            sel_version = st.selectbox("Prompt Version", versions)

        with fc2:
            models    = ["All"] + sorted(df_logs["model_name"].unique().tolist())
            sel_model = st.selectbox("Model", models)

        with fc3:
            sel_errors = st.selectbox("Show", ["All", "Errors only", "Safety flags only", "Successful only"])

        with fc4:
            search = st.text_input("Search query text", placeholder="keyword...")

        # ── Apply filters ──────────────────────────────────────────────────────
        filtered = df_logs.copy()
        if sel_version != "All":
            filtered = filtered[filtered["prompt_version"] == sel_version]
        if sel_model != "All":
            filtered = filtered[filtered["model_name"] == sel_model]
        if sel_errors == "Errors only":
            filtered = filtered[filtered["error_code"] != ""]
        elif sel_errors == "Safety flags only":
            filtered = filtered[filtered["safety_flag"] == 1]
        elif sel_errors == "Successful only":
            filtered = filtered[filtered["error_code"] == ""]
        if search:
            filtered = filtered[filtered["user_query"].str.contains(
                search, case=False, na=False)]

        st.caption(f"Showing {len(filtered):,} of {len(df_logs):,} total logs")

        # ── Paginated table ────────────────────────────────────────────────────
        PAGE_SIZE = 20
        total_pages = max(1, (len(filtered) - 1) // PAGE_SIZE + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages,
                               value=1, step=1)
        start = (page - 1) * PAGE_SIZE
        end   = start + PAGE_SIZE

        display_cols = [
            "timestamp", "user_query", "model_name", "prompt_version",
            "total_tokens", "end_to_end_latency", "safety_flag", "error_code"
        ]
        page_df = filtered[display_cols].iloc[start:end].copy()
        page_df["user_query"] = page_df["user_query"].str[:80]

        st.dataframe(page_df, use_container_width=True, height=420)

        # ── Expanded view for selected row ─────────────────────────────────────
        st.markdown('<div class="section-title">Full Response Viewer</div>',
                    unsafe_allow_html=True)
        row_index = st.number_input(
            "Enter row index to inspect (from table above)",
            min_value=int(filtered.index.min()) if not filtered.empty else 0,
            max_value=int(filtered.index.max()) if not filtered.empty else 0,
            value=int(filtered.index[0]) if not filtered.empty else 0,
        )
        if row_index in filtered.index:
            row = filtered.loc[row_index]
            st.markdown(f"**Query:** {row['user_query']}")
            st.markdown(f"**Model:** `{row['model_name']}` | "
                        f"**Version:** `{row['prompt_version']}` | "
                        f"**Tokens:** `{row['total_tokens']}` | "
                        f"**Latency:** `{row['end_to_end_latency']}s`")
            st.markdown("**Response:**")
            st.info(str(row.get("model_response", ""))[:1000])
