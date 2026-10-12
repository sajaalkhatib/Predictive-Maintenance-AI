"""
Predictive Maintenance AI Dashboard
Power BI-Style Interactive Dashboard built with Streamlit + Plotly
NASA C-MAPSS FD004 — Turbofan Engine RUL Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Dark Aviation Theme ────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1E293B; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #94A3B8 !important; }
    [data-testid="stSidebarNav"] { display: none; }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .kpi-value { font-size: 2.4rem; font-weight: 800; margin: 4px 0; }
    .kpi-label { font-size: 0.78rem; color: #64748B; letter-spacing: 0.1em; text-transform: uppercase; }
    .kpi-delta { font-size: 0.82rem; margin-top: 4px; }

    .kpi-blue  .kpi-value { color: #38BDF8; }
    .kpi-red   .kpi-value { color: #F87171; }
    .kpi-amber .kpi-value { color: #FBBF24; }
    .kpi-green .kpi-value { color: #34D399; }

    /* Section headers */
    .section-header {
        color: #94A3B8;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Alert banners */
    .alert-critical {
        background: linear-gradient(90deg, #7F1D1D, #450A0A);
        border: 1px solid #EF4444;
        border-radius: 8px; padding: 10px 16px; margin: 4px 0;
    }
    .alert-warning {
        background: linear-gradient(90deg, #78350F, #451A03);
        border: 1px solid #F59E0B;
        border-radius: 8px; padding: 10px 16px; margin: 4px 0;
    }

    /* Metric overrides */
    [data-testid="stMetric"] { background: #1E293B; border-radius: 8px; padding: 12px; border: 1px solid #334155; }
    [data-testid="stMetricValue"] { color: #E2E8F0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #1E293B; border-radius: 8px; gap: 4px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #64748B; background: transparent; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background-color: #334155 !important; color: #E2E8F0 !important; }

    /* Dataframe */
    .dataframe { font-size: 0.8rem; }
    [data-testid="stDataFrameResizable"] { border: 1px solid #334155; border-radius: 8px; }

    /* Divider */
    hr { border-color: #334155; margin: 12px 0; }

    /* Plotly charts background */
    .js-plotly-plot { border-radius: 10px; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Layout Template ───────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="#1E293B",
    plot_bgcolor="#0F172A",
    font=dict(color="#94A3B8", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ─── Data Loading & Model ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    columns = (
        ["unit_number", "time_in_cycles"]
        + ["setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    try:
        df = pd.read_csv("train_FD004.txt", sep=r"\s+", header=None, names=columns)
    except FileNotFoundError:
        return None, None, None, None

    # RUL calculation with cap
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["Actual_RUL"] = (max_cycles - df["time_in_cycles"]).clip(upper=125)

    feature_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("setting_")]
    X = df[feature_cols]
    y = df["Actual_RUL"]
    groups = df["unit_number"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    results = df.iloc[test_idx].copy()
    results["Predicted_RUL"] = np.round(model.predict(X_test), 1)
    results["Actual_RUL"] = y.iloc[test_idx].values
    results["Error"] = np.round(results["Predicted_RUL"] - results["Actual_RUL"], 1)
    results["Abs_Error"] = results["Error"].abs()
    results["Risk"] = pd.cut(
        results["Predicted_RUL"],
        bins=[-1, 30, 75, 125],
        labels=["🔴 Critical", "🟡 Warning", "🟢 Healthy"]
    )
    results["Maintenance"] = results["Predicted_RUL"].apply(
        lambda v: "🚨 Urgent" if v <= 30 else ("⚠️ Schedule" if v <= 75 else "✅ Normal")
    )

    # Feature importance
    fi = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    mae = mean_absolute_error(results["Actual_RUL"], results["Predicted_RUL"])

    return results, fi, df, round(mae, 2)


# ─── Demo data when train_FD004.txt is not available ─────────────────────────
@st.cache_data(show_spinner=False)
def make_demo_data():
    rng = np.random.default_rng(42)
    n_engines = 80
    records = []
    for eid in range(1, n_engines + 1):
        lifespan = rng.integers(100, 350)
        for cycle in range(1, lifespan + 1):
            actual_rul = min(125, lifespan - cycle)
            noise = rng.normal(0, 5)
            predicted_rul = float(np.clip(actual_rul + noise, 0, 125))
            base = 30 + rng.uniform(-2, 2)
            s11 = base + cycle * 0.02 + rng.normal(0, 0.3)
            s13 = 100 - cycle * 0.05 + rng.normal(0, 0.5)
            s15 = 14.62 + cycle * 0.003 + rng.normal(0, 0.1)
            s9  = 9000 - cycle * 1.2 + rng.normal(0, 10)
            s14 = 8138 - cycle * 0.9 + rng.normal(0, 8)
            records.append({
                "unit_number": eid,
                "time_in_cycles": cycle,
                "Actual_RUL": actual_rul,
                "Predicted_RUL": round(predicted_rul, 1),
                "Error": round(predicted_rul - actual_rul, 1),
                "Abs_Error": round(abs(predicted_rul - actual_rul), 1),
                "sensor_9": round(s9, 1),
                "sensor_11": round(s11, 1),
                "sensor_13": round(s13, 1),
                "sensor_14": round(s14, 1),
                "sensor_15": round(s15, 2),
                "setting_1": round(rng.choice([0, 0.1, 0.25, 0.42, 0.55, 0.84]), 2),
                "setting_2": round(rng.choice([0, 0.25, 0.62, 0.84]), 2),
            })

    results = pd.DataFrame(records)
    results["Risk"] = pd.cut(
        results["Predicted_RUL"],
        bins=[-1, 30, 75, 125],
        labels=["🔴 Critical", "🟡 Warning", "🟢 Healthy"]
    )
    results["Maintenance"] = results["Predicted_RUL"].apply(
        lambda v: "🚨 Urgent" if v <= 30 else ("⚠️ Schedule" if v <= 75 else "✅ Normal")
    )

    fi = pd.DataFrame({
        "Feature": [f"sensor_{i}" for i in [13, 11, 15, 9, 14, 4, 7, 12, 3, 8]],
        "Importance": [0.21, 0.17, 0.14, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03]
    })
    mae = round(results["Abs_Error"].mean(), 2)
    return results, fi, results, mae


# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading model & predictions..."):
    results, fi, raw_df, mae = load_and_train()
    if results is None:
        results, fi, raw_df, mae = make_demo_data()
        demo_mode = True
    else:
        demo_mode = False

# Latest reading per engine
latest = (
    results.sort_values("time_in_cycles")
    .groupby("unit_number")
    .last()
    .reset_index()
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Predictive Maintenance")
    st.markdown("<p style='color:#64748B;font-size:0.75rem;margin-top:-10px;'>NASA C-MAPSS FD004 · Random Forest</p>", unsafe_allow_html=True)

    if demo_mode:
        st.warning("⚠️ **Demo Mode**\nPlace `train_FD004.txt` in the project folder to use real data.", icon="📂")

    st.divider()
    st.markdown("<div class='section-header'>Fleet Filter</div>", unsafe_allow_html=True)
    risk_filter = st.multiselect(
        "Risk Category",
        options=["🔴 Critical", "🟡 Warning", "🟢 Healthy"],
        default=["🔴 Critical", "🟡 Warning", "🟢 Healthy"],
    )

    st.divider()
    st.markdown("<div class='section-header'>Engine Inspector</div>", unsafe_allow_html=True)
    engine_ids = sorted(latest["unit_number"].unique())
    selected_engine = st.selectbox("Select Engine ID", engine_ids)

    st.divider()
    st.markdown("<div class='section-header'>Sensor Overlay</div>", unsafe_allow_html=True)
    sensor_opts = [c for c in results.columns if c.startswith("sensor_")]
    selected_sensors = st.multiselect(
        "Sensors to Plot",
        sensor_opts,
        default=["sensor_11", "sensor_13", "sensor_15"] if "sensor_11" in sensor_opts else sensor_opts[:3],
    )

    st.divider()
    st.caption("Built with Streamlit + Plotly · Antigravity AI")

# ─── Apply Fleet Filter ───────────────────────────────────────────────────────
filtered_latest = latest[latest["Risk"].isin(risk_filter)] if risk_filter else latest

# ─── Top Header ───────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown(
        "<h1 style='color:#E2E8F0;margin:0;font-size:1.7rem;'>✈️ Engine Health Monitoring Dashboard</h1>"
        "<p style='color:#64748B;margin:0;font-size:0.85rem;'>Real-Time Remaining Useful Life (RUL) Prediction · Turbofan Engine Fleet</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Fleet Overview", "🔍 Engine Deep-Dive", "🤖 AI Model Insights"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ════════════════════════════════════════════════════════════════
with tab1:
    # ── KPI Row
    total_eng   = latest["unit_number"].nunique()
    critical_n  = (latest["Risk"] == "🔴 Critical").sum()
    warning_n   = (latest["Risk"] == "🟡 Warning").sum()
    healthy_n   = (latest["Risk"] == "🟢 Healthy").sum()
    avg_rul     = round(latest["Predicted_RUL"].mean(), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='kpi-card kpi-blue'>
            <div class='kpi-label'>Total Engines</div>
            <div class='kpi-value'>{total_eng}</div>
            <div class='kpi-delta' style='color:#38BDF8;'>Active Fleet</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card kpi-red'>
            <div class='kpi-label'>⚠️ Critical</div>
            <div class='kpi-value'>{critical_n}</div>
            <div class='kpi-delta' style='color:#F87171;'>RUL ≤ 30 cycles</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card kpi-amber'>
            <div class='kpi-label'>🔔 Warning</div>
            <div class='kpi-value'>{warning_n}</div>
            <div class='kpi-delta' style='color:#FBBF24;'>RUL 31–75 cycles</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card kpi-green'>
            <div class='kpi-label'>✅ Healthy</div>
            <div class='kpi-value'>{healthy_n}</div>
            <div class='kpi-delta' style='color:#34D399;'>RUL > 75 cycles</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='kpi-card kpi-blue'>
            <div class='kpi-label'>Avg Fleet RUL</div>
            <div class='kpi-value'>{avg_rul}</div>
            <div class='kpi-delta' style='color:#38BDF8;'>Cycles remaining</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts Row
    row1_l, row1_r = st.columns([1, 2])

    with row1_l:
        st.markdown("<div class='section-header'>Fleet Risk Distribution</div>", unsafe_allow_html=True)
        risk_counts = latest["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map = {"🔴 Critical": "#EF4444", "🟡 Warning": "#F59E0B", "🟢 Healthy": "#10B981"}
        fig_donut = go.Figure(go.Pie(
            labels=risk_counts["Risk"],
            values=risk_counts["Count"],
            hole=0.60,
            marker=dict(colors=[color_map.get(r, "#64748B") for r in risk_counts["Risk"]],
                        line=dict(color="#0F172A", width=3)),
            textfont=dict(size=13, color="#E2E8F0"),
            hovertemplate="<b>%{label}</b><br>Engines: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total_eng}</b><br><span style='font-size:11px'>Engines</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#E2E8F0"), align="center"
        )
        fig_donut.update_layout(**PLOTLY_DARK, height=280, showlegend=True,
                                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1, font=dict(size=11)))
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_r:
        st.markdown("<div class='section-header'>Predicted RUL Distribution — All Engines</div>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_vline(x=30, line_dash="dot", line_color="#EF4444", annotation_text="Critical", annotation_font_color="#EF4444")
        fig_hist.add_vline(x=75, line_dash="dot", line_color="#F59E0B", annotation_text="Warning", annotation_font_color="#F59E0B")
        fig_hist.add_trace(go.Histogram(
            x=latest["Predicted_RUL"], nbinsx=40,
            marker=dict(color="#38BDF8", line=dict(color="#0F172A", width=1)),
            hovertemplate="RUL: %{x}<br>Engines: %{y}<extra></extra>",
            name="Engine Count",
        ))
        fig_hist.update_layout(**PLOTLY_DARK, height=280,
                               xaxis_title="Predicted RUL (cycles)", yaxis_title="Engine Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Alerts
    critical_list = latest[latest["Risk"] == "🔴 Critical"].sort_values("Predicted_RUL")
    if not critical_list.empty:
        st.markdown("<div class='section-header'>🚨 Urgent Maintenance Alerts</div>", unsafe_allow_html=True)
        for _, row in critical_list.head(5).iterrows():
            st.markdown(
                f"<div class='alert-critical'>🚨 <b>Engine #{int(row['unit_number'])}</b> — "
                f"Predicted RUL: <b>{row['Predicted_RUL']} cycles</b> · "
                f"Operating: {int(row['time_in_cycles'])} cycles · Requires immediate inspection.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Fleet Table
    st.markdown("<div class='section-header'>Full Fleet Status Table</div>", unsafe_allow_html=True)
    display_cols = ["unit_number", "time_in_cycles", "Actual_RUL", "Predicted_RUL", "Abs_Error", "Risk", "Maintenance"]
    available = [c for c in display_cols if c in filtered_latest.columns]
    st.dataframe(
        filtered_latest[available].rename(columns={
            "unit_number": "Engine ID", "time_in_cycles": "Cycles", "Actual_RUL": "Actual RUL",
            "Predicted_RUL": "Predicted RUL", "Abs_Error": "Abs Error", "Risk": "Risk Level",
            "Maintenance": "Action"
        }).sort_values("Predicted RUL").reset_index(drop=True),
        use_container_width=True, height=340,
        column_config={
            "Predicted RUL": st.column_config.ProgressColumn("Predicted RUL", min_value=0, max_value=125),
            "Actual RUL": st.column_config.NumberColumn("Actual RUL"),
        }
    )

# ════════════════════════════════════════════════════════════════
# TAB 2 — Engine Deep-Dive
# ════════════════════════════════════════════════════════════════
with tab2:
    engine_data = results[results["unit_number"] == selected_engine].sort_values("time_in_cycles")

    if engine_data.empty:
        st.warning(f"No data found for Engine #{selected_engine}")
    else:
        current   = engine_data.iloc[-1]
        pred_rul  = current["Predicted_RUL"]
        act_rul   = current.get("Actual_RUL", pred_rul)
        risk_cat  = str(current.get("Risk", "Unknown"))
        max_cycle = int(engine_data["time_in_cycles"].max())

        # ── Top KPIs
        d1, d2, d3, d4 = st.columns(4)
        risk_color = "#EF4444" if "Critical" in risk_cat else ("#F59E0B" if "Warning" in risk_cat else "#10B981")
        with d1:
            st.markdown(f"""<div class='kpi-card' style='border-color:{risk_color};'>
                <div class='kpi-label'>Engine ID</div>
                <div class='kpi-value' style='color:{risk_color};'>#{int(current['unit_number'])}</div>
                <div class='kpi-delta' style='color:{risk_color};'>{risk_cat}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class='kpi-card kpi-blue'>
                <div class='kpi-label'>Operating Cycles</div>
                <div class='kpi-value'>{max_cycle}</div>
                <div class='kpi-delta' style='color:#38BDF8;'>Total Run Time</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class='kpi-card' style='border-color:{risk_color};'>
                <div class='kpi-label'>Predicted RUL</div>
                <div class='kpi-value' style='color:{risk_color};'>{pred_rul}</div>
                <div class='kpi-delta' style='color:#64748B;'>Actual: {act_rul}</div>
            </div>""", unsafe_allow_html=True)
        with d4:
            err = abs(pred_rul - act_rul)
            st.markdown(f"""<div class='kpi-card kpi-amber'>
                <div class='kpi-label'>Prediction Error</div>
                <div class='kpi-value'>±{round(err,1)}</div>
                <div class='kpi-delta' style='color:#FBBF24;'>Abs. Error (cycles)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Gauge + RUL Trend
        g1, g2 = st.columns([1, 2])

        with g1:
            st.markdown("<div class='section-header'>Health Gauge</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_rul,
                delta={"reference": 75, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
                number={"font": {"color": risk_color, "size": 42}},
                gauge=dict(
                    axis=dict(range=[0, 125], tickwidth=1, tickcolor="#334155", tickfont=dict(color="#64748B")),
                    bar=dict(color=risk_color, thickness=0.25),
                    bgcolor="#0F172A",
                    borderwidth=2, bordercolor="#334155",
                    steps=[
                        dict(range=[0, 30], color="#7F1D1D"),
                        dict(range=[30, 75], color="#78350F"),
                        dict(range=[75, 125], color="#14532D"),
                    ],
                    threshold=dict(line=dict(color="#FFFFFF", width=3), thickness=0.75, value=30),
                ),
                title=dict(text="Predicted RUL (cycles)", font=dict(color="#94A3B8", size=13)),
            ))
            fig_gauge.update_layout(**PLOTLY_DARK, height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            st.markdown("<div class='section-header'>RUL Degradation Over Time</div>", unsafe_allow_html=True)
            fig_rul = go.Figure()
            fig_rul.add_hrect(y0=0, y1=30, fillcolor="#7F1D1D", opacity=0.15, layer="below", line_width=0)
            fig_rul.add_hrect(y0=30, y1=75, fillcolor="#78350F", opacity=0.15, layer="below", line_width=0)
            fig_rul.add_trace(go.Scatter(
                x=engine_data["time_in_cycles"], y=engine_data["Actual_RUL"],
                mode="lines", name="Actual RUL",
                line=dict(color="#64748B", width=1.5, dash="dot"),
            ))
            fig_rul.add_trace(go.Scatter(
                x=engine_data["time_in_cycles"], y=engine_data["Predicted_RUL"],
                mode="lines", name="Predicted RUL",
                line=dict(color=risk_color, width=2.5),
                fill="tozeroy", fillcolor=f"rgba(56,189,248,0.08)",
            ))
            fig_rul.update_layout(**PLOTLY_DARK, height=260,
                                  xaxis_title="Operating Cycle", yaxis_title="RUL (cycles)",
                                  legend=dict(orientation="h", x=0, y=-0.2, font=dict(size=11)))
            st.plotly_chart(fig_rul, use_container_width=True)

        # ── Sensor Trends
        if selected_sensors:
            st.markdown("<div class='section-header'>Sensor Trend Analysis</div>", unsafe_allow_html=True)
            avail_sensors = [s for s in selected_sensors if s in engine_data.columns]
            if avail_sensors:
                fig_sensors = make_subplots(
                    rows=len(avail_sensors), cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=[f"<b>{s}</b>" for s in avail_sensors],
                )
                colors = ["#38BDF8", "#34D399", "#FBBF24", "#F87171", "#A78BFA"]
                for i, sensor in enumerate(avail_sensors, 1):
                    fig_sensors.add_trace(
                        go.Scatter(
                            x=engine_data["time_in_cycles"],
                            y=engine_data[sensor],
                            mode="lines", name=sensor,
                            line=dict(color=colors[(i - 1) % len(colors)], width=1.8),
                        ),
                        row=i, col=1,
                    )
                    fig_sensors.update_yaxes(gridcolor="#1E293B", row=i, col=1,
                                            title_font=dict(color="#64748B"), tickfont=dict(color="#64748B"))
                fig_sensors.update_xaxes(title_text="Cycle", gridcolor="#1E293B",
                                         tickfont=dict(color="#64748B"), row=len(avail_sensors), col=1)
                fig_sensors.update_layout(
                    paper_bgcolor="#1E293B", plot_bgcolor="#0F172A",
                    font=dict(color="#94A3B8"),
                    height=200 * len(avail_sensors),
                    showlegend=False,
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                for ann in fig_sensors.layout.annotations:
                    ann.font.color = "#94A3B8"
                    ann.font.size = 11
                st.plotly_chart(fig_sensors, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — AI Model Insights
# ════════════════════════════════════════════════════════════════
with tab3:
    m1, m2 = st.columns(2)

    with m1:
        # Actual vs Predicted scatter
        st.markdown("<div class='section-header'>Actual vs Predicted RUL</div>", unsafe_allow_html=True)
        sample = results.sample(min(2000, len(results)), random_state=42) if len(results) > 2000 else results
        color_map_num = sample["Risk"].map({"🔴 Critical": "#EF4444", "🟡 Warning": "#F59E0B", "🟢 Healthy": "#10B981"}).fillna("#64748B")

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=[0, 125], y=[0, 125], mode="lines",
            line=dict(color="#64748B", dash="dash", width=1.5),
            name="Perfect Prediction", showlegend=True,
        ))
        for risk_label, color in [("🔴 Critical", "#EF4444"), ("🟡 Warning", "#F59E0B"), ("🟢 Healthy", "#10B981")]:
            sub = sample[sample["Risk"] == risk_label]
            if not sub.empty:
                fig_scatter.add_trace(go.Scatter(
                    x=sub["Actual_RUL"], y=sub["Predicted_RUL"],
                    mode="markers", name=risk_label,
                    marker=dict(color=color, size=4, opacity=0.55),
                    hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>",
                ))
        fig_scatter.update_layout(
            **PLOTLY_DARK, height=360,
            xaxis_title="Actual RUL (cycles)", yaxis_title="Predicted RUL (cycles)",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=11)),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Model metrics
        st.markdown("<div class='section-header'>Model Performance Metrics</div>", unsafe_allow_html=True)
        rmse = round(np.sqrt((results["Error"] ** 2).mean()), 2)
        r2   = round(1 - (results["Error"] ** 2).sum() / ((results["Actual_RUL"] - results["Actual_RUL"].mean()) ** 2).sum(), 3)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("MAE", f"{mae} cycles", help="Mean Absolute Error")
        with mc2:
            st.metric("RMSE", f"{rmse} cycles", help="Root Mean Squared Error")
        with mc3:
            st.metric("R² Score", f"{r2}", help="Coefficient of Determination")

    with m2:
        # Feature Importance
        st.markdown("<div class='section-header'>Top Feature Importance (AI Decision Factors)</div>", unsafe_allow_html=True)
        top_fi = fi.head(12).sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            x=top_fi["Importance"],
            y=top_fi["Feature"],
            orientation="h",
            marker=dict(
                color=top_fi["Importance"],
                colorscale=[[0, "#1E3A5F"], [0.5, "#38BDF8"], [1, "#7DD3FC"]],
                line=dict(color="#0F172A", width=1),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(**PLOTLY_DARK, height=360,
                             xaxis_title="Feature Importance Score", yaxis_title="",
                             yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_fi, use_container_width=True)

        # Error distribution by life stage
        st.markdown("<div class='section-header'>MAE by Engine Life Stage</div>", unsafe_allow_html=True)
        results["Life_Stage"] = pd.cut(
            results["Actual_RUL"],
            bins=[-1, 30, 75, 125],
            labels=["Critical (0-30)", "Middle (31-75)", "Healthy (76-125)"]
        )
        stage_mae = results.groupby("Life_Stage")["Abs_Error"].mean().reset_index()
        stage_mae.columns = ["Stage", "MAE"]

        fig_stage = go.Figure(go.Bar(
            x=stage_mae["Stage"],
            y=stage_mae["MAE"],
            marker=dict(color=["#EF4444", "#F59E0B", "#10B981"],
                        line=dict(color="#0F172A", width=1)),
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f} cycles<extra></extra>",
            text=stage_mae["MAE"].round(1),
            textposition="outside", textfont=dict(color="#E2E8F0"),
        ))
        fig_stage.update_layout(**PLOTLY_DARK, height=200,
                                yaxis_title="MAE (cycles)", xaxis_title="")
        st.plotly_chart(fig_stage, use_container_width=True)

    # Error Histogram full-width
    st.markdown("<div class='section-header'>Prediction Error Distribution</div>", unsafe_allow_html=True)
    fig_err = go.Figure()
    fig_err.add_vline(x=0, line_dash="dash", line_color="#64748B", annotation_text="Zero Error")
    fig_err.add_trace(go.Histogram(
        x=results["Error"], nbinsx=60,
        marker=dict(color="#38BDF8", opacity=0.75, line=dict(color="#0F172A", width=0.5)),
        name="Prediction Error",
        hovertemplate="Error: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig_err.update_layout(
        **PLOTLY_DARK, height=200,
        xaxis_title="Prediction Error (Predicted − Actual RUL, cycles)",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.75rem;'>"
    "✈️ Predictive Maintenance AI · NASA C-MAPSS FD004 · Random Forest Regressor · "
    "Built with Streamlit & Plotly · © 2025</p>",
    unsafe_allow_html=True,
)
