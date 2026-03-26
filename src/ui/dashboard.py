"""
Streamlit Dashboard - Interactive Battery Health Monitoring.

Usage: streamlit run src/ui/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.features.extractor import FeatureExtractor
from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel

st.set_page_config(page_title="Battery Health Monitor", layout="wide", page_icon="🔋")

SAFETY_COLORS = {
    SafetyLevel.GREEN: "#2ecc71",
    SafetyLevel.YELLOW: "#f39c12",
    SafetyLevel.RED: "#e74c3c",
}


@st.cache_data
def load_data():
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
    validator = DataValidator()
    df, report = validator.validate(df)
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)
    return df, report


def main():
    st.title("🔋 Battery Health Management System")
    st.markdown("Uncertainty-aware prognostics with safety decision support")

    try:
        df, report = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Sidebar
    st.sidebar.header("Configuration")
    batteries = df["battery_id"].unique().tolist()
    selected_bat = st.sidebar.selectbox("Battery", batteries)
    rul_critical = st.sidebar.slider("RUL Critical Threshold", 5, 30, 10)
    rul_warning = st.sidebar.slider("RUL Warning Threshold", 20, 60, 30)

    bat_df = df[df["battery_id"] == selected_bat].sort_values("cycle")

    # ── Overview Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cycles", len(bat_df))
    col2.metric("Initial Capacity", f"{bat_df['capacity'].iloc[0]:.3f} Ah")
    col3.metric("Current Capacity", f"{bat_df['capacity'].iloc[-1]:.3f} Ah")
    fade_pct = (1 - bat_df['capacity'].iloc[-1] / bat_df['capacity'].iloc[0]) * 100
    col4.metric("Capacity Fade", f"{fade_pct:.1f}%")

    # ── Degradation Curve ──
    st.subheader("Capacity Degradation")
    fig = go.Figure()
    for bat in batteries:
        sub = df[df["battery_id"] == bat].sort_values("cycle")
        opacity = 1.0 if bat == selected_bat else 0.3
        width = 3 if bat == selected_bat else 1
        fig.add_trace(go.Scatter(
            x=sub["cycle"], y=sub["capacity"], name=bat,
            opacity=opacity, line=dict(width=width),
        ))
    fig.update_layout(xaxis_title="Cycle", yaxis_title="Capacity (Ah)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── Safety Status ──
    st.subheader("Safety Decision")
    engine = SafetyDecisionEngine(rul_critical=rul_critical, rul_warning=rul_warning)

    if "rul" in bat_df.columns:
        last_rul = bat_df["rul"].iloc[-1]
        decision = engine.decide(
            rul_mean=float(last_rul),
            rul_lower=float(last_rul * 0.8),
            rul_upper=float(last_rul * 1.2),
            epistemic_std=float(last_rul * 0.1),
        )
        color = SAFETY_COLORS[decision.level]
        st.markdown(
            f'<div style="background-color:{color};padding:20px;border-radius:10px;color:white">'
            f'<h2>Status: {decision.level.value}</h2>'
            f'<p>RUL Estimate: {decision.rul_estimate:.0f} cycles '
            f'[{decision.confidence_lower:.0f}, {decision.confidence_upper:.0f}]</p>'
            f'<p>Action: {decision.action}</p>'
            f'<p>Reason: {decision.reason}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Feature Trends ──
    st.subheader("Feature Trends")
    numeric_cols = [c for c in bat_df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul") and bat_df[c].notna().sum() > 5]
    selected_features = st.multiselect("Select features", numeric_cols, default=numeric_cols[:3])

    if selected_features:
        fig2 = make_subplots(rows=len(selected_features), cols=1, shared_xaxes=True,
                             subplot_titles=selected_features)
        for i, feat in enumerate(selected_features, 1):
            fig2.add_trace(go.Scatter(x=bat_df["cycle"], y=bat_df[feat], name=feat), row=i, col=1)
        fig2.update_layout(height=250 * len(selected_features), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Data Quality ──
    with st.expander("Data Quality Report"):
        st.write(f"Pass Rate: {report.pass_rate:.1%}")
        st.write(f"Flagged Rows: {report.flagged_rows}")
        st.write(f"Total Rows: {report.total_rows}")
        if report.issues:
            st.dataframe(pd.DataFrame(report.issues))


if __name__ == "__main__":
    main()
