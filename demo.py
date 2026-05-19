"""
IIoT Privacy-Preserving Fog Aggregation — Presentation Demo
Run: streamlit run demo.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
RESULTS = HERE / "results"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IIoT Fog Demo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ────────────────────────────────────────────────────────────
C_PROPOSED = "#2196F3"
C_BASELINE = ["#9E9E9E", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
C_GOOD = "#4CAF50"
C_WARN = "#FF5722"
C_BG = "#0E1117"
C_CARD = "#1E2329"

CUSTOM_CSS = """
<style>
  [data-testid="stSidebar"] { background: #13171f; }
  .metric-card {
    background: #1E2329;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 6px 0;
    border-left: 4px solid #2196F3;
  }
  .metric-label { font-size: 0.78rem; color: #9aa5b4; text-transform: uppercase; letter-spacing: .06em; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #e8ecf0; }
  .metric-sub   { font-size: 0.78rem; color: #4CAF50; }
  .takeaway {
    background: #162032;
    border-left: 4px solid #2196F3;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.95rem;
    color: #cdd6e0;
  }
  .warn-box {
    background: #1f1508;
    border-left: 4px solid #FF5722;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #e0b090;
  }
  .section-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 4px; }
  .arch-box {
    background: #1E2329;
    border: 1px solid #2a3140;
    border-radius: 10px;
    padding: 20px;
    font-family: monospace;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #b0bec5;
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "", warn: bool = False):
    border = C_WARN if warn else C_PROPOSED
    color = C_WARN if warn else C_GOOD
    st.markdown(
        f"""<div class="metric-card" style="border-left-color:{border}">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub" style="color:{color}">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def takeaway(text: str):
    st.markdown(f'<div class="takeaway">💡 {text}</div>', unsafe_allow_html=True)


def warn_box(text: str):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def load(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if path.exists():
        return pd.read_csv(path)
    st.warning(f"CSV not found: {name} — run the experiments first.")
    return pd.DataFrame()


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0bec5", size=12),
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor="#2a3140", linecolor="#2a3140"),
    yaxis=dict(gridcolor="#2a3140", linecolor="#2a3140"),
)


def apply_layout(fig, **kw):
    merged = {**PLOTLY_LAYOUT, **kw}
    # deep-merge nested dicts (e.g. xaxis/yaxis) so caller values win
    for key in ("xaxis", "yaxis", "yaxis2", "legend", "polar"):
        if key in PLOTLY_LAYOUT and key in kw and isinstance(kw[key], dict):
            merged[key] = {**PLOTLY_LAYOUT[key], **kw[key]}
    fig.update_layout(**merged)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 IIoT Fog Demo")
    st.markdown("Privacy-Preserving Aggregation with Paillier HE + KMM")
    st.divider()
    section = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🎬  Real Simulation",
            "⚡  Live Pipeline",
            "📦  E1 · Storage",
            "⏱️  E2 · Latency",
            "✅  E3 · Correctness",
            "🔑  E4 · KMM Overhead",
            "⚖️  E5 · Task Scheduling",
            "🛡️  E6 · Fault Detection",
            "🔄  E7 · Pipeline Latency",
            "💥  E8 · Blast Radius",
            "📊  Summary",
        ],
    )
    st.divider()
    st.caption("Run all experiments first:\n`python run_all_p1.py`\n`python run_all_p2.py`\n`python run_all_p3.py`")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section.startswith("🏠"):
    st.title("Privacy-Preserving IIoT Fog Aggregation")
    st.markdown("**Evaluating Paillier Homomorphic Encryption + Key Management Module (KMM) in a Fog-Computing IIoT Deployment**")
    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### System Architecture")
        st.markdown(
            """<div class="arch-box">
  ┌──────────────────────────────────────────────────────────┐
  │                    IIoT Sensor Layer                     │
  │  [S1]──AES-GCM──▶  [S2]──AES-GCM──▶  ... [Sₙ]          │
  └───────────────────────┬──────────────────────────────────┘
                          │  (AES ciphertexts)
  ┌───────────────────────▼──────────────────────────────────┐
  │                  Fog Layer  (TEE / SGX)                  │
  │  AES-decrypt → scale → Paillier-encrypt → HE-accumulate  │
  │                                                          │
  │  [F1] ──KMM──▶ [F2]  ◀──KMM──  [F3]                    │
  │         Key Management Module  (provision / revoke)      │
  └───────────────────────┬──────────────────────────────────┘
                          │  (1 Paillier ciphertext)
  ┌───────────────────────▼──────────────────────────────────┐
  │                     Cloud / Gateway                      │
  │     Paillier-decrypt  →  result  →  store (AES-GCM)      │
  └──────────────────────────────────────────────────────────┘</div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Key Properties")
        props = {
            "Encryption": "Paillier 2048-bit HE\n+ AES-GCM per sensor",
            "Privacy": "Fog never sees\nplaintext sensor values",
            "Aggregation": "Homomorphic addition\n(n ciphertexts → 1)",
            "Key Mgmt": "KMM delegation\n+ revocation",
            "Scheduling": "Capacity-score\nload balancer",
            "Fault": "ACK + KMM\nfault detection",
        }
        for k, v in props.items():
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{k}</div>
                    <div style="font-size:0.9rem;color:#e8ecf0">{v.replace(chr(10),"<br>")}</div>
                   </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Experiment Overview")
    exps = [
        ("E1", "Storage", "100×–1000× ciphertext reduction", C_PROPOSED),
        ("E2", "Latency", "444.128ms calibrated 2048-bit path", C_GOOD),
        ("E3a/b", "Correctness", "100% accuracy across all k delegations", C_GOOD),
        ("E4", "KMM Overhead", "1.474ms host aggregate reference", C_GOOD),
        ("E5", "Task Scheduling", "78.77% deadline satisfaction (E5 S6)", C_PROPOSED),
        ("E6", "Fault Detection", "6.5% data loss vs 75% (gossip)", C_GOOD),
        ("E7", "Pipeline", "≈399ms end-to-end (analytical hardware model)", C_PROPOSED),
        ("E8", "Blast Radius", "60–80% exposure reduction", C_GOOD),
    ]
    cols = st.columns(4)
    for i, (eid, name, finding, color) in enumerate(exps):
        with cols[i % 4]:
            st.markdown(
                f"""<div class="metric-card" style="border-left-color:{color}">
                    <div class="metric-label">{eid} · {name}</div>
                    <div style="font-size:0.85rem;color:#e0e6ed;margin-top:4px">{finding}</div>
                   </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# REAL SIMULATION — multi-window time-series with live crypto
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("🎬"):
    import math
    import copy

    st.title("🎬 Real-Time IIoT Simulation — All Sensors + Load Balancing")
    st.markdown(
        "100 sensors across 5 heterogeneous fog nodes, streaming every 500 ms window. "
        "Watch CapacityScore re-route tasks when F2 overloads and F5 fails — live."
    )

    sys.path.insert(0, str(HERE))
    try:
        from config import (
            BASE_WORKLOAD, FOG_NODES, TAU, SCALE,
            build_schedule, full_cap_score,
        )
        from crypto_sim import aes_key, aes_encrypt, generate_paillier_keypair, sgx_enclave_process
        cfg_ok = True
    except Exception as e:
        st.error(f"Cannot import modules: {e}")
        cfg_ok = False

    if not cfg_ok:
        st.stop()

    # ── fog node colours ───────────────────────────────────────────────────
    FOG_COLORS = {
        "F1": "#2196F3", "F2": "#FF9800",
        "F3": "#9E9E9E", "F4": "#4CAF50", "F5": "#9C27B0",
    }
    NODE_ORDER = ["F1", "F2", "F3", "F4", "F5"]
    SENSORS_PER_NODE = 20   # 20 × 5 = 100 sensors total

    # sensor signal per fog zone
    ZONE_CFG = {
        "F1": dict(base=88.0,  amp=5.0,  noise=1.2, unit="°C",    label="Industrial Temp"),
        "F2": dict(base=10.5,  amp=2.5,  noise=0.6, unit="A",     label="Motor Current"),
        "F3": dict(base=3.5,   amp=1.5,  noise=0.8, unit="m/s²",  label="Vibration"),
        "F4": dict(base=5.2,   amp=1.0,  noise=0.3, unit="bar",   label="Pressure"),
        "F5": dict(base=72.0,  amp=4.0,  noise=1.0, unit="°C",    label="Ambient Temp"),
    }

    # inline service-time (mirrors experiments_p1._service_time)
    def _service_time(node_id: str, task_type: str, wt: dict) -> float:
        node = FOG_NODES[node_id]
        base = 28.0 if task_type == "LS" else 90.0
        return (base * (4.0 / node["cpu_cap"])
                + wt[node_id]["latency"] * 60.0
                + wt[node_id]["queue"] * (70.0 if task_type == "LS" else 120.0))

    # inline capacity picker
    def _pick_capacity(candidates: list, task_type: str, wt: dict) -> str | None:
        if not candidates:
            return None
        return max(candidates, key=lambda nid: full_cap_score(nid, wt, task_type))

    # ── controls ───────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        lb_method = st.selectbox(
            "Load-balancing method",
            ["capacity (proposed)", "threshold", "round_robin", "random"],
            key="lb_method",
        )
    with ctrl2:
        speed = st.selectbox("Animation speed", ["Fast (0.2s)", "Normal (0.4s)", "Slow (0.8s)"], key="lb_speed")
    with ctrl3:
        show_heatmap = st.checkbox("Show sensor heatmap", value=True, key="lb_heat")

    delay = {"Fast (0.2s)": 0.2, "Normal (0.4s)": 0.4, "Slow (0.8s)": 0.8}[speed]
    method_key = lb_method.split()[0]   # "capacity" | "threshold" | "round_robin" | "random"

    col_btn, _ = st.columns([2, 8])
    with col_btn:
        run_sim = st.button("▶  Run 20-Window Simulation", type="primary", use_container_width=True)

    st.divider()

    if not run_sim:
        # idle: show static architecture
        st.markdown("#### Fog network topology")
        cols = st.columns(5)
        for col, nid in zip(cols, NODE_ORDER):
            nd = FOG_NODES[nid]
            with col:
                st.markdown(
                    f"""<div class="metric-card" style="border-left-color:{FOG_COLORS[nid]}">
                        <div class="metric-label">{nid} · {nd['class']}</div>
                        <div style="font-size:0.82rem;color:#cdd6e0">
                          CPU: {nd['cpu_cap']} cores<br>
                          RAM: {nd['ram_gb']} GB<br>
                          BW: {nd['bw_mbps']} Mbps<br>
                          Sensors: {nd['sensors']}<br>
                          Base load: {BASE_WORKLOAD[nid]['workload']:.0%}
                        </div></div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("#### E5 scenario script (20 windows)")
        schedule_display = pd.DataFrame(build_schedule())
        schedule_display.columns = ["Window", "F2 Load", "F5 Alive", "Event"]
        st.dataframe(schedule_display, use_container_width=True, hide_index=True)
        st.info("Select a method above and press **▶ Run 20-Window Simulation** to begin.")
        st.stop()

    # ════════════════════════════════════════════════════════════════════════
    # RUN SIMULATION
    # ════════════════════════════════════════════════════════════════════════
    schedule = build_schedule()
    rng = random.Random(42)
    rr_idx = 0   # round-robin counter

    # ── layout placeholders ────────────────────────────────────────────────
    event_ph   = st.empty()                      # event banner
    metric_ph  = st.empty()                      # 5 live metrics
    st.divider()

    if show_heatmap:
        heat_ph = st.empty()                     # sensor heatmap (full width)
        st.divider()

    row1l, row1r = st.columns([3, 2])
    score_ph   = row1l.empty()                   # capacity score bar chart
    status_ph  = row1r.empty()                   # fog node live status

    row2l, row2r = st.columns([3, 2])
    assign_ph  = row2l.empty()                   # task assignment stacked bar
    workload_ph = row2r.empty()                  # workload evolution lines

    metrics_ph = st.empty()                      # completion / deadline / redeleg lines

    # ── history buffers ────────────────────────────────────────────────────
    sensor_matrix: dict[str, list[list[float]]] = {nid: [] for nid in NODE_ORDER}
    workload_hist  = {nid: [] for nid in NODE_ORDER}
    score_hist_ls  = {nid: [] for nid in NODE_ORDER}
    assign_hist    = {nid: [] for nid in NODE_ORDER}
    completion_h, deadline_h, redeleg_h = [], [], []
    events_seen = []

    # cumulative task counters
    total_tasks = completed = deadline_met = delegated = redelegated = 0

    current = {nid: dict(v) for nid, v in BASE_WORKLOAD.items()}

    for win_data in schedule:
        w      = win_data["window"] - 1   # 0-indexed
        event  = win_data["event"]
        events_seen.append(event)

        # ── update workloads per schedule ──────────────────────────────────
        for nid in ["F1", "F3", "F4", "F5"]:
            current[nid]["workload"] = max(
                current[nid]["workload"] * 0.85 + BASE_WORKLOAD[nid]["workload"] * 0.15,
                BASE_WORKLOAD[nid]["workload"],
            )
            current[nid]["queue"] = max(
                current[nid]["queue"] * 0.85 + BASE_WORKLOAD[nid]["queue"] * 0.15,
                BASE_WORKLOAD[nid]["queue"],
            )
        current["F2"]["workload"] = float(win_data["f2_load"])
        current["F2"]["queue"]    = float(win_data["f2_load"]) * 0.8
        if not win_data["f5_alive"]:
            current["F5"]["workload"] = 1.0
            current["F5"]["queue"]    = 1.0

        overloaded = {nid for nid, v in current.items() if v["workload"] >= TAU}
        wt_snap = copy.deepcopy(current)

        # ── generate sensor readings (20 per fog node) ─────────────────────
        for nid in NODE_ORDER:
            zcfg = ZONE_CFG[nid]
            node_readings = []
            for sid in range(SENSORS_PER_NODE):
                phase = sid * 0.45 + NODE_ORDER.index(nid) * 1.2
                t_val = w * 0.4
                val = (zcfg["base"] + zcfg["amp"] * math.sin(t_val + phase)
                       + rng.gauss(0, zcfg["noise"]))
                node_readings.append(round(val, 3))
            sensor_matrix[nid].append(node_readings)

        # ── capacity scores ────────────────────────────────────────────────
        for nid in NODE_ORDER:
            score_hist_ls[nid].append(full_cap_score(nid, wt_snap, "LS"))

        # ── task assignment (100 tasks per window) ─────────────────────────
        window_assign = {nid: 0 for nid in NODE_ORDER}
        E5_TASKS = 100
        E5_TO_RATIO = 0.70
        for tidx in range(E5_TASKS):
            task_type = "TO" if rng.random() < E5_TO_RATIO else "LS"
            source = "F2" if current["F2"]["workload"] >= TAU else f"F{1 + (tidx % 5)}"
            is_delegated = source in overloaded

            if not is_delegated:
                target = source
            else:
                delegated += 1
                candidates = [nid for nid in NODE_ORDER
                              if nid not in overloaded and wt_snap[nid]["workload"] < TAU]
                if method_key == "capacity":
                    target = _pick_capacity(candidates, task_type, wt_snap)
                elif method_key == "threshold":
                    target = min(candidates, key=lambda n: wt_snap[n]["workload"]) if candidates else None
                elif method_key == "round_robin":
                    if candidates:
                        target = candidates[rr_idx % len(candidates)]
                        rr_idx += 1
                    else:
                        target = None
                else:  # random
                    target = rng.choice(candidates) if candidates else None

            total_tasks += 1
            if target is None:
                if is_delegated:
                    redelegated += 1
                continue

            window_assign[target] += 1
            latency = _service_time(target, task_type, current)
            deadline_limit = 100.0 if task_type == "LS" else 1000.0
            if is_delegated and target == "F3":
                redelegated += 1
            else:
                completed += 1
                deadline_met += int(latency <= deadline_limit)

            if is_delegated:
                current[target]["workload"] = min(0.99, current[target]["workload"] + 0.20 / E5_TASKS)
                current[target]["queue"]    = min(0.99, current[target]["queue"]    + 0.12 / E5_TASKS)

        for nid in NODE_ORDER:
            assign_hist[nid].append(window_assign[nid])
        for nid in NODE_ORDER:
            workload_hist[nid].append(current[nid]["workload"])

        # cumulative metrics
        completion_h.append(completed / total_tasks * 100)
        deadline_h.append(deadline_met / total_tasks * 100)
        redeleg_h.append(redelegated / max(1, delegated) * 100)

        w_labels = [f"W{i+1}" for i in range(w + 1)]

        # ── event banner ───────────────────────────────────────────────────
        evt_color = C_WARN if ("overload" in event.lower() or "fail" in event.lower()) else (
            C_GOOD if "recovery" in event.lower() else C_PROPOSED)
        with event_ph.container():
            st.markdown(
                f'<div style="background:#1E2329;border-left:4px solid {evt_color};'
                f'border-radius:6px;padding:10px 16px;font-size:1.0rem;color:#e8ecf0">'
                f'<b>Window {w+1}/20</b> &nbsp;·&nbsp; {event} &nbsp;·&nbsp; '
                f'Overloaded: {", ".join(sorted(overloaded)) or "none"}</div>',
                unsafe_allow_html=True,
            )

        # ── live metric strip ──────────────────────────────────────────────
        with metric_ph.container():
            mc = st.columns(5)
            mc[0].metric("Completion", f"{completion_h[-1]:.1f}%",
                          f"{completion_h[-1]-completion_h[-2]:.1f}%" if w > 0 else "")
            mc[1].metric("Deadline Met", f"{deadline_h[-1]:.1f}%",
                          f"{deadline_h[-1]-deadline_h[-2]:.1f}%" if w > 0 else "")
            mc[2].metric("Re-delegation", f"{redeleg_h[-1]:.1f}%",
                          f"{redeleg_h[-1]-redeleg_h[-2]:.1f}%" if w > 0 else "")
            mc[3].metric("Delegated Tasks", str(delegated), f"of {total_tasks} total")
            mc[4].metric("Method", lb_method.split()[0].upper())

        # ── sensor heatmap ─────────────────────────────────────────────────
        if show_heatmap:
            # rows = 100 sensors (F1:0-19, F2:20-39, ...), cols = windows so far
            z_rows, y_labels = [], []
            for nid in NODE_ORDER:
                zcfg = ZONE_CFG[nid]
                for sid in range(SENSORS_PER_NODE):
                    row = [sensor_matrix[nid][wi][sid] for wi in range(w + 1)]
                    # pad to full width for uniform matrix
                    z_rows.append(row)
                    y_labels.append(f"{nid}-S{sid+1:02d}")

            fig_heat = go.Figure(go.Heatmap(
                z=z_rows,
                x=w_labels,
                y=y_labels,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(thickness=12, len=0.8, title="Reading"),
                hovertemplate="Sensor %{y}<br>Window %{x}<br>Value: %{z:.3f}<extra></extra>",
            ))
            # fog-group separator lines
            for i, nid in enumerate(NODE_ORDER):
                boundary = (i + 1) * SENSORS_PER_NODE - 0.5
                fig_heat.add_hline(y=boundary, line_color="#2a3140", line_width=2)
                fig_heat.add_annotation(
                    x=0, y=(i * SENSORS_PER_NODE + SENSORS_PER_NODE / 2 - 0.5),
                    text=f"<b>{nid}</b>",
                    showarrow=False, xanchor="right", xref="paper",
                    font=dict(color=FOG_COLORS[nid], size=11),
                )
            apply_layout(
                fig_heat,
                title="All 100 Sensor Readings — Heatmap (5 fog zones × 20 sensors × windows)",
                xaxis_title="Window",
                yaxis=dict(showticklabels=False, gridcolor="#1E2329"),
                height=420,
                margin=dict(l=80, r=60, t=50, b=40),
            )
            heat_ph.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{w}")

        # ── capacity score bar chart ───────────────────────────────────────
        current_scores_ls = [score_hist_ls[nid][-1] for nid in NODE_ORDER]
        current_scores_to = [full_cap_score(nid, wt_snap, "TO") for nid in NODE_ORDER]
        winner_ls = NODE_ORDER[current_scores_ls.index(max(current_scores_ls))]
        winner_to = NODE_ORDER[current_scores_to.index(max(current_scores_to))]

        fig_score = go.Figure()
        fig_score.add_trace(go.Bar(
            name="LS task score",
            x=NODE_ORDER,
            y=current_scores_ls,
            marker_color=[
                FOG_COLORS[nid] if nid == winner_ls else "#37474F"
                for nid in NODE_ORDER
            ],
            text=[f"{s:.3f}" + (" ★" if nid == winner_ls else "")
                  for nid, s in zip(NODE_ORDER, current_scores_ls)],
            textposition="outside",
        ))
        fig_score.add_trace(go.Bar(
            name="TO task score",
            x=NODE_ORDER,
            y=current_scores_to,
            marker_color=[
                FOG_COLORS[nid] if nid == winner_to else "#37474F"
                for nid in NODE_ORDER
            ],
            opacity=0.55,
        ))
        # draw overload threshold indicator per node
        for nid in NODE_ORDER:
            if current[nid]["workload"] >= TAU:
                fig_score.add_annotation(
                    x=nid, y=max(current_scores_ls) * 1.08,
                    text="OVERLOADED", showarrow=False,
                    font=dict(color=C_WARN, size=10),
                )
        apply_layout(
            fig_score,
            barmode="group",
            title=f"Capacity Scores — W{w+1}  (★ = winner)",
            yaxis=dict(range=[0, 1.1], gridcolor="#2a3140"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=280,
        )
        score_ph.plotly_chart(fig_score, use_container_width=True, key=f"score_{w}")

        # ── fog node live status (5 mini gauges) ───────────────────────────
        fig_status = go.Figure()
        bar_colors = []
        for nid in NODE_ORDER:
            wl = current[nid]["workload"]
            color = C_WARN if wl >= TAU else (C_GOOD if wl < 0.5 else "#FF9800")
            bar_colors.append(color)
        fig_status.add_trace(go.Bar(
            x=NODE_ORDER,
            y=[current[nid]["workload"] for nid in NODE_ORDER],
            name="Workload",
            marker_color=bar_colors,
            text=[f"{current[nid]['workload']:.0%}" for nid in NODE_ORDER],
            textposition="outside",
        ))
        fig_status.add_trace(go.Bar(
            x=NODE_ORDER,
            y=[current[nid]["queue"] for nid in NODE_ORDER],
            name="Queue depth",
            marker_color=[FOG_COLORS[nid] for nid in NODE_ORDER],
            opacity=0.45,
        ))
        fig_status.add_hline(y=TAU, line_dash="dash", line_color=C_WARN, line_width=1.5,
                              annotation_text=f"τ={TAU}", annotation_font_color=C_WARN)
        apply_layout(
            fig_status,
            barmode="group",
            title="Fog Node Workload & Queue",
            yaxis=dict(range=[0, 1.15], gridcolor="#2a3140"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=280,
        )
        status_ph.plotly_chart(fig_status, use_container_width=True, key=f"status_{w}")

        # ── task assignment accumulation (stacked bar per window) ──────────
        fig_assign = go.Figure()
        for nid in NODE_ORDER:
            fig_assign.add_trace(go.Bar(
                name=nid,
                x=w_labels,
                y=assign_hist[nid],
                marker_color=FOG_COLORS[nid],
            ))
        # event annotations
        for wi, ev in enumerate(events_seen):
            if "overload" in ev.lower():
                fig_assign.add_vline(x=wi, line_color=C_WARN, line_dash="dot", line_width=1)
            elif "fail" in ev.lower():
                fig_assign.add_vline(x=wi, line_color="#F44336", line_dash="dot", line_width=2)
            elif "recovery" in ev.lower():
                fig_assign.add_vline(x=wi, line_color=C_GOOD, line_dash="dot", line_width=1)
        apply_layout(
            fig_assign,
            barmode="stack",
            title=f"Task Distribution per Window ({lb_method.split()[0]})",
            xaxis_title="Window", yaxis_title="Tasks assigned",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
            height=280,
        )
        assign_ph.plotly_chart(fig_assign, use_container_width=True, key=f"assign_{w}")

        # ── workload evolution lines ───────────────────────────────────────
        fig_wl = go.Figure()
        for nid in NODE_ORDER:
            fig_wl.add_trace(go.Scatter(
                x=w_labels, y=workload_hist[nid],
                mode="lines+markers", name=nid,
                line=dict(color=FOG_COLORS[nid], width=2),
                marker=dict(size=5),
            ))
        fig_wl.add_hline(y=TAU, line_dash="dash", line_color=C_WARN, line_width=1.5,
                          annotation_text=f"overload τ={TAU}", annotation_font_color=C_WARN)
        apply_layout(
            fig_wl,
            title="Fog Node Workload over Time",
            xaxis_title="Window", yaxis_title="Workload (0–1)",
            yaxis=dict(range=[0, 1.1], gridcolor="#2a3140"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=280,
        )
        workload_ph.plotly_chart(fig_wl, use_container_width=True, key=f"wl_{w}")

        # ── cumulative metrics lines ───────────────────────────────────────
        fig_met = go.Figure()
        for series, name, color, dash in [
            (completion_h, "Completion %", C_PROPOSED, "solid"),
            (deadline_h,   "Deadline met %", C_GOOD, "dash"),
            (redeleg_h,    "Re-delegation %", C_WARN, "dot"),
        ]:
            fig_met.add_trace(go.Scatter(
                x=w_labels, y=series, mode="lines+markers", name=name,
                line=dict(color=color, dash=dash, width=2.5),
                marker=dict(size=6),
            ))
        apply_layout(
            fig_met,
            title="Cumulative E5 Metrics",
            xaxis_title="Window", yaxis_title="%",
            yaxis=dict(range=[0, 110], gridcolor="#2a3140"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=220,
        )
        metrics_ph.plotly_chart(fig_met, use_container_width=True, key=f"met_{w}")

        time.sleep(delay)

    # ── final summary ──────────────────────────────────────────────────────
    st.divider()
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1: metric_card("Task Completion", f"{completion_h[-1]:.1f}%", "all 20 windows")
    with fc2: metric_card("Deadline Satisfaction", f"{deadline_h[-1]:.1f}%", f"{lb_method.split()[0]}")
    with fc3: metric_card("Re-delegation Rate", f"{redeleg_h[-1]:.1f}%", "overload + failure windows")
    with fc4: metric_card("Total Tasks", f"{total_tasks}", f"{delegated} delegated")
    takeaway(
        f"CapacityScore routes overloaded tasks from F2 to F4 (best LS score: low latency=0.10, "
        f"moderate load) — not to F3 (weak, high latency). When F5 fails, it scores 0 and is "
        f"automatically excluded. Final deadline satisfaction: {deadline_h[-1]:.1f}%."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PIPELINE DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("⚡"):
    st.title("⚡ Live Pipeline Demo")
    st.markdown("Run a real AES-GCM → Paillier HE aggregation pipeline in your browser.")

    sys.path.insert(0, str(HERE))
    try:
        from crypto_sim import (
            aes_key, aes_encrypt, aes_decrypt,
            generate_paillier_keypair, sgx_enclave_process,
            paillier_backend_name,
        )
        from config import SCALE
        crypto_ok = True
    except Exception as e:
        st.error(f"Could not import crypto modules: {e}")
        crypto_ok = False

    if crypto_ok:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_sensors = st.slider("Number of sensors", 2, 20, 5)
        with col2:
            key_bits = st.selectbox("Paillier key bits", [256, 512, 1024], index=0,
                                    help="256-bit is fast for demo; production uses 2048-bit")
        with col3:
            seed_val = st.number_input("Random seed", value=42, step=1)

        if st.button("▶  Run Pipeline", type="primary", use_container_width=True):
            rng = random.Random(int(seed_val))
            log = st.container()
            progress = st.progress(0)

            with st.spinner("Generating Paillier keypair…"):
                t0 = time.perf_counter()
                pub_key, priv_key = generate_paillier_keypair(int(key_bits), rng)
                keygen_ms = (time.perf_counter() - t0) * 1000
            log.success(f"Keypair generated ({key_bits}-bit) in {keygen_ms:.1f} ms  [{paillier_backend_name(pub_key)} backend]")
            progress.progress(15)

            # Sensor values
            sensor_values = [round(rng.uniform(10.0, 99.9), 3) for _ in range(n_sensors)]
            true_sum = sum(sensor_values)
            log.info(f"Sensor readings: {sensor_values}")
            progress.progress(25)

            # AES encrypt
            k_fog = aes_key(rng)
            aes_cts = []
            for v in sensor_values:
                nonce, ct = aes_encrypt(k_fog, v, rng)
                aes_cts.append((nonce, ct))
            log.info(f"AES-GCM encrypted {n_sensors} readings → {n_sensors} ciphertexts ({len(aes_cts[0][0])+len(aes_cts[0][1])} bytes each)")
            progress.progress(40)

            # SGX enclave: decrypt AES, encrypt Paillier, accumulate
            t1 = time.perf_counter()
            encrypted_sum = None
            for nonce, ct in aes_cts:
                enc_val = sgx_enclave_process(k_fog, nonce, ct, pub_key, rng)
                encrypted_sum = enc_val if encrypted_sum is None else encrypted_sum + enc_val
            enclave_ms = (time.perf_counter() - t1) * 1000
            log.info(f"SGX enclave: AES-decrypt → scale → Paillier-encrypt × {n_sensors}, HE-sum  ({enclave_ms:.1f} ms)")
            progress.progress(75)

            # Cloud: Paillier decrypt
            t2 = time.perf_counter()
            result_scaled = priv_key.decrypt(encrypted_sum)
            decrypt_ms = (time.perf_counter() - t2) * 1000
            result = result_scaled / SCALE
            progress.progress(95)

            error = abs(result - true_sum)
            log.success(f"Decrypted sum = {result:.3f}  (true = {true_sum:.3f},  error = {error:.2e})")
            progress.progress(100)

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Plaintext Sum", f"{true_sum:.3f}", "ground truth")
            with c2: metric_card("Decrypted Sum", f"{result:.3f}", "from ciphertext")
            with c3: metric_card("Keygen Time", f"{keygen_ms:.0f} ms", f"{key_bits}-bit")
            with c4: metric_card("Enclave Time", f"{enclave_ms:.1f} ms", f"{n_sensors} sensors", warn=enclave_ms > 500)

            st.divider()
            takeaway(
                f"The fog node aggregated {n_sensors} sensor readings homomorphically "
                f"without ever seeing plaintext values. The result error is {error:.2e} — "
                "negligible due to integer scaling."
            )
            if key_bits < 2048:
                warn_box(
                    f"Demo uses {key_bits}-bit keys for speed. E2 evaluates 2048-bit Paillier with "
                    "a calibrated host-reference benchmark. The benchmark is OP-TEE QEMU timing only, "
                    "not physical hardware timing."
                )


# ══════════════════════════════════════════════════════════════════════════════
# E1 — STORAGE
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("📦"):
    st.title("E1 · Storage Overhead")
    df = load("e1_storage.csv")
    if df.empty:
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1: metric_card("Max Reduction", "1000×", "ciphertexts at n=1000")
    with col2: metric_card("Proposed Ciphertexts", "1", "always exactly 1 per window")
    with col3: metric_card("Proposed Paillier Size", "512 B", "fixed regardless of n")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Bytes per aggregation window")
        fig = go.Figure()
        for col, name, color, dash in [
            ("plaintext_bytes", "Plaintext", "#9E9E9E", "dot"),
            ("aes_bytes", "AES-GCM (baseline)", "#FF9800", "dash"),
            ("paillier_nobatch_bytes", "Paillier no-batch", "#F44336", "dashdot"),
            ("ours_paillier_bytes", "Proposed (Paillier)", C_PROPOSED, "solid"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["n"], y=df[col], mode="lines+markers",
                                         name=name, line=dict(color=color, dash=dash, width=2)))
        apply_layout(fig, xaxis_title="Sensors (n)", yaxis_title="Bytes", yaxis_type="log",
                     legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Ciphertext count reduction")
        fig2 = go.Figure(go.Bar(
            x=df["n"].astype(str),
            y=df["ciphertext_count_reduction"],
            marker_color=C_PROPOSED,
            text=df["ciphertext_count_reduction"].astype(int).astype(str) + "×",
            textposition="outside",
        ))
        apply_layout(fig2, xaxis_title="Sensors (n)", yaxis_title="Reduction ratio",
                     showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    takeaway(
        "The proposed system always transmits exactly 1 Paillier ciphertext (512 bytes) to the cloud, "
        "regardless of how many sensors feed the fog node. At n=1000 sensors this is a 1000× reduction "
        "versus individual encryption."
    )


# ══════════════════════════════════════════════════════════════════════════════
# E2 — LATENCY
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("⏱️"):
    st.title("E2 · Aggregation Latency")
    df = load("e2_latency.csv")
    if df.empty:
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1: metric_card("Window Budget", "500 ms", "real-time constraint")
    with col2: metric_card("Plaintext (n=500)", f"{df['plaintext_ms_median'].iloc[-1]:.3f} ms", "✓ within budget")
    with col3: metric_card("Proposed", f"{df['ours_ms_median'].iloc[0]:.3f} ms",
                            "calibrated 2048-bit path")

    st.divider()
    fig = go.Figure()
    palette = {"plaintext_ms_median": ("#9E9E9E", "Plaintext"),
               "aes_ms_median": ("#FF9800", "AES-GCM"),
               "ours_ms_median": (C_PROPOSED, "Proposed (2048-bit Paillier)")}
    for col, (color, name) in palette.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["n"], y=df[col], mode="lines+markers",
                                     name=name, line=dict(color=color, width=2.5)))

    fig.add_hline(y=500, line_dash="dash", line_color=C_WARN, line_width=2,
                  annotation_text="500 ms deadline", annotation_position="top left",
                  annotation_font_color=C_WARN)
    apply_layout(fig, xaxis_title="Sensors (n)", yaxis_title="Latency (ms)",
                 legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    warn_box(
        "E2 uses the calibrated 2048-bit host-reference benchmark: fog TA 227.537ms, "
        "host aggregate/KMM 1.474ms, and storage TA 215.117ms, totaling 444.128ms."
    )
    takeaway(
        "The calibrated proposed path is within the 500ms window. Caveat: this is OP-TEE QEMU timing only, "
        "not physical hardware timing, and TA-side 2048-bit Paillier is not enabled in this prototype."
    )


# ══════════════════════════════════════════════════════════════════════════════
# E3 — CORRECTNESS
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("✅"):
    st.title("E3 · KMM Correctness")
    df3a = load("e3a_correctness.csv")
    df3b = load("e3b_multisource_correctness.csv")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("E3a Accuracy", "100%", "all k-delegation values")
    with c2: metric_card("E3b Accuracy", "100%", "multi-source delegation")
    with c3: metric_card("Max Quant. Error", f"{df3a['max_quantization_error'].max():.2e}" if not df3a.empty else "—", "near machine-epsilon")

    if not df3a.empty:
        st.divider()
        st.markdown("##### E3a — Single-source correctness vs. delegation depth (k)")
        fig = go.Figure(go.Bar(
            x=df3a["k_delegated"].astype(str),
            y=df3a["accuracy_scaled_pct"],
            marker_color=C_PROPOSED,
            text=df3a["accuracy_scaled_pct"].astype(str) + "%",
            textposition="inside",
        ))
        apply_layout(fig, xaxis_title="Delegations (k)", yaxis_title="Accuracy (%)",
                     yaxis=dict(range=[99, 100.1], gridcolor="#2a3140"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if not df3b.empty:
        st.markdown("##### E3b — Multi-source summary")
        st.dataframe(df3b[["sources", "backup", "accuracy_scaled_pct", "provisioned_edges"]].style
                     .format({"accuracy_scaled_pct": "{:.1f}%"}), use_container_width=True)

    takeaway(
        "The KMM slot-protocol achieves 100% arithmetic correctness across all tested delegation depths "
        "and multi-source topologies. Quantization error is < 1e-9 — due to integer scaling, not cryptographic noise."
    )


# ══════════════════════════════════════════════════════════════════════════════
# E4 — KMM COMBINE OVERHEAD
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("🔑"):
    st.title("E4 · KMM Combine Overhead")
    df = load("e4_kmm_combine.csv")
    if df.empty:
        st.stop()

    col1, col2 = st.columns(2)
    with col1: metric_card("Max Overhead (k=100)", f"{df['combine_latency_ms_median'].max():.3f} ms", "well under 500ms budget")
    with col2: metric_card("All Within Budget", "100%", "within_500ms = True for all k")

    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["k_fog_aggregates"], y=df["combine_latency_ms_median"],
                             mode="lines+markers", name="Combine latency (median)",
                             line=dict(color=C_PROPOSED, width=2.5),
                             error_y=dict(type="data", array=df["combine_latency_ms_std"],
                                         visible=True, color="#555")))
    fig.add_hline(y=500, line_dash="dash", line_color=C_WARN, line_width=2,
                  annotation_text="500 ms budget")
    apply_layout(fig, xaxis_title="Fog aggregates (k)", yaxis_title="Combine latency (ms)",
                 legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    takeaway(
        "The calibrated host aggregate/KMM combine reference is 1.474ms while ciphertext input size "
        "scales linearly at 512 bytes per fog aggregate."
    )


# ══════════════════════════════════════════════════════════════════════════════
# E5 — TASK SCHEDULING
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("⚖️"):
    st.title("E5 · Task Scheduling — E5 Full-CapScore")
    df = load("e5_results.csv")
    task_df = load("e5_task_records.csv")
    if df.empty:
        st.stop()

    method_colors = {"S1": "#d62728", "S2": "#ff7f0e", "S3": "#bcbd22",
                     "S4": "#1f77b4", "S5": "#2ca02c", "S6": C_PROPOSED}
    method_labels = {
        "S1": "S1 Random",
        "S2": "S2 Round-Robin",
        "S3": "S3 Threshold",
        "S4": "S4 CapScore-Fixed",
        "S5": "S5 CapScore-Adaptive",
        "S6": "S6 Full-CapScore",
    }

    c1, c2, c3 = st.columns(3)
    cap = df[df["strategy"] == "S6"].iloc[0]
    with c1: metric_card("Deadline Satisfaction", f"{cap['deadline_rate_mean'] * 100:.2f}%", "S6 Full-CapScore")
    with c2: metric_card("Task Completion", f"{cap['completion_rate_mean'] * 100:.0f}%", "S6 Full-CapScore")
    with c3: metric_card("Re-delegation Rate", f"{cap['redelegation_rate_mean'] * 100:.1f}%", "E5 simulator")

    st.divider()
    metrics = [
        ("deadline_rate_mean", "deadline_rate_ci95", "Deadline Satisfaction (%)"),
        ("completion_rate_mean", "completion_rate_ci95", "Task Completion (%)"),
        ("delegation_rate_mean", "delegation_rate_ci95", "Delegation Rate (%)"),
        ("sigma_bw_mean_mean", "sigma_bw_mean_ci95", "Bandwidth Std Dev"),
    ]
    fig = go.Figure()
    for method, row in df.set_index("strategy").iterrows():
        fig.add_trace(go.Bar(
            name=method_labels.get(method, method),
            x=[m[2] for m in metrics],
            y=[row[m[0]] * 100 if "rate" in m[0] else row[m[0]] for m in metrics],
            error_y=dict(type="data", array=[row[m[1]] * 100 if "rate" in m[1] else row[m[1]] for m in metrics], visible=True),
            marker_color=method_colors.get(method, "#999"),
        ))
    apply_layout(fig, barmode="group", yaxis_title="Metric value", legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    if not task_df.empty:
        hp = task_df[task_df["task_type"] == "HP"]
        hp_rates = hp.groupby("strategy").agg(
            hp_miss=("deadline_met", lambda values: 1.0 - values.astype(int).mean()),
            hp_f3=("assigned_node", lambda values: (values == "F3").mean()),
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="HP deadline miss",
            x=[method_labels.get(s, s) for s in hp_rates["strategy"]],
            y=hp_rates["hp_miss"] * 100,
            marker_color=[method_colors.get(s, "#999") for s in hp_rates["strategy"]],
        ))
        fig2.add_trace(go.Bar(
            name="HP assigned to F3",
            x=[method_labels.get(s, s) for s in hp_rates["strategy"]],
            y=hp_rates["hp_f3"] * 100,
            marker_color=[method_colors.get(s, "#999") for s in hp_rates["strategy"]],
            opacity=0.55,
        ))
        apply_layout(fig2, barmode="group", yaxis_title="%", legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

    takeaway(
        "E5 uses six strategies across 20 seeds. S6 Full-CapScore achieves 78.77% "
        "deadline satisfaction, 100% completion, and the lowest bandwidth-utilization "
        "standard deviation by routing HP tasks away from the slowest node."
    )


# ══════════════════════════════════════════════════════════════════════════════
# E6 — FAULT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("🛡️"):
    st.title("E6 · Fault Detection & Recovery")
    df = load("e6_fault_detection.csv")
    if df.empty:
        st.stop()

    # aggregate across seeds
    agg = df.groupby(["method", "scenario"])["data_loss_rate"].mean().reset_index()
    agg["data_loss_pct"] = agg["data_loss_rate"] * 100

    scenario_map = {"fail_0ms": "Fail at 0ms", "mid_window_250ms": "Fail at 250ms", "late_window_450ms": "Fail at 450ms"}
    method_map = {"b1_gossip": "Gossip", "b2_replication": "Replication",
                  "checkpoint": "Checkpoint", "b4_multilayer": "Multilayer",
                  "b5_fog_clustering": "Fog-Clustering", "proposed_ack_kmm": "Proposed ACK+KMM"}
    method_colors_e6 = {
        "b1_gossip": "#F44336", "b2_replication": "#FF9800", "checkpoint": "#9C27B0",
        "b4_multilayer": "#00BCD4", "b5_fog_clustering": "#9E9E9E", "proposed_ack_kmm": C_PROPOSED,
    }

    c1, c2, c3 = st.columns(3)
    prop = agg[agg["method"] == "proposed_ack_kmm"]
    worst_prop = prop["data_loss_pct"].max()
    gossip_worst = agg[agg["method"] == "b1_gossip"]["data_loss_pct"].max()
    with c1: metric_card("Proposed Max Data Loss", f"{worst_prop:.1f}%", "at latest failure time")
    with c2: metric_card("Gossip Max Data Loss", f"{gossip_worst:.0f}%", "at earliest failure", warn=True)
    with c3: metric_card("ACK Window", "100 ms", "28-byte key per ACK")

    st.divider()
    fig = go.Figure()
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("scenario")
        fig.add_trace(go.Bar(
            name=method_map.get(method, method),
            x=sub["scenario"].map(scenario_map),
            y=sub["data_loss_pct"],
            marker_color=method_colors_e6.get(method, "#999"),
        ))
    apply_layout(fig, barmode="group", xaxis_title="Failure scenario",
                 yaxis_title="Data loss (%)", legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    takeaway(
        "The proposed ACK+KMM method limits data loss to ≈6.5% even at the latest failure time (450ms), "
        "compared to 75% for gossip-based detection. The 28-byte per-ACK key overhead is minimal."
    )
    warn_box("E6 uses an analytical fault model — not live distributed fault injection. Results represent expected behaviour under modelled timing assumptions.")


# ══════════════════════════════════════════════════════════════════════════════
# E7 — PIPELINE LATENCY
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("🔄"):
    st.title("E7 · End-to-End Pipeline Latency")
    df = load("e7_pipeline_latency_summary.csv")
    if df.empty:
        st.stop()

    method_labels = {
        "cloud_only": "Cloud-Only",
        "fog_plaintext": "Fog Plaintext",
        "paillier_fog_convert": "Paillier Fog-Convert",
        "ours": "Proposed",
    }
    method_colors_e7 = {
        "cloud_only": "#9E9E9E", "fog_plaintext": "#FF9800",
        "paillier_fog_convert": "#00BCD4", "ours": C_PROPOSED,
    }

    c1, c2, c3 = st.columns(3)
    ours = df[df["method"] == "ours"].iloc[0]
    with c1: metric_card("Proposed Median", f"{ours['total_ms_median']:.0f} ms", "analytical hardware model")
    with c2: metric_card("Within-Budget Windows", f"{ours['within_500ms_windows']}/20", "8 delegation windows exceed 500ms", warn=True)
    with c3: metric_card("Storage per Window", f"{ours['storage_bytes_per_window_median']:.0f} B", "1 Paillier ciphertext")

    st.divider()
    fig = go.Figure(go.Bar(
        x=[method_labels.get(r["method"], r["method"]) for _, r in df.iterrows()],
        y=df["total_ms_median"],
        marker_color=[method_colors_e7.get(m, "#999") for m in df["method"]],
        error_y=dict(type="data", array=df["total_ms_std"], visible=True, color="#555"),
        text=df["total_ms_median"].round(0).astype(int).astype(str) + " ms",
        textposition="outside",
    ))
    fig.add_hline(y=500, line_dash="dash", line_color=C_WARN, line_width=2,
                  annotation_text="500 ms budget")
    apply_layout(fig, xaxis_title="Method", yaxis_title="Total latency (ms)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    takeaway(
        "The proposed system achieves a median of ~399ms on the analytical hardware model — within the 500ms budget "
        "for normal windows. Delegation + recovery windows (8/20) exceed the budget due to KMM provisioning overhead (50ms)."
    )
    warn_box(
        "E7 uses analytical hardware-target latency constants (AES hardware accelerator, SGX enclave, "
        "Paillier co-processor). E2 is a separate calibrated 2048-bit host-reference benchmark."
    )


# ══════════════════════════════════════════════════════════════════════════════
# E8 — BLAST RADIUS
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("💥"):
    st.title("E8 · Blast Radius Analysis")
    df = load("e8_blast_radius.csv")
    if df.empty:
        st.stop()

    scenario_map = {
        "one_fog_compromised": "One Fog\nCompromised",
        "backup_during_delegation": "Backup During\nDelegation",
        "kmm_compromised": "KMM\nCompromised",
        "host_os_reads_enclave": "Host OS Reads\nEnclave (SGX)",
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Exposure: global-key scheme vs. proposed fog-scoped")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Global Key (baseline)",
                             x=df["scenario"].map(scenario_map),
                             y=df["global_key_exposed"],
                             marker_color="#F44336"))
        fig.add_trace(go.Bar(name="Proposed Fog-Scoped",
                             x=df["scenario"].map(scenario_map),
                             y=df["fog_scoped_exposed"],
                             marker_color=C_PROPOSED))
        apply_layout(fig, barmode="group", xaxis_title="Compromise scenario",
                     yaxis_title="Exposed sensor count",
                     legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Exposure reduction (%)")
        fig2 = go.Figure(go.Bar(
            x=df["scenario"].map(scenario_map),
            y=df["exposure_reduction_pct"],
            marker_color=[C_GOOD if v > 0 else "#9E9E9E" for v in df["exposure_reduction_pct"]],
            text=df["exposure_reduction_pct"].astype(str) + "%",
            textposition="outside",
        ))
        apply_layout(fig2, xaxis_title="Scenario", yaxis_title="Reduction (%)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Scenario details")
    display_cols = ["scenario", "total_sensors", "global_key_exposed_pct", "fog_scoped_exposed_pct",
                    "exposure_reduction_pct", "interpretation"]
    cols_present = [c for c in display_cols if c in df.columns]
    st.dataframe(df[cols_present], use_container_width=True)

    takeaway(
        "Fog-scoped key isolation limits a single fog compromise to 20% exposure (one group of 20 sensors) "
        "vs. 100% with a global key. KMM compromise remains a trust-anchor vulnerability — acknowledged as "
        "a single point of failure for both schemes."
    )
    warn_box("E8 is an analytical exposure accounting model, not a cryptographic attack simulation.")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("📊"):
    st.title("📊 Summary & Key Findings")

    findings = [
        ("E1 Storage", "100×–1000× reduction", "1 ciphertext regardless of n sensors", C_GOOD, False),
        ("E2 Latency", "444.128ms", "Calibrated 2048-bit host-reference path", C_GOOD, False),
        ("E3 Correctness", "100% accuracy", "Across all k-delegations & multi-source", C_GOOD, False),
        ("E4 KMM Combine", "1.474ms overhead", "Calibrated host aggregate reference", C_GOOD, False),
        ("E5 Scheduling", "78.77% deadline satisfaction", "E5 S6 improves routing quality", C_PROPOSED, False),
        ("E6 Fault Detection", "6.5% data loss", "vs. 75% gossip (at 450ms failure)", C_GOOD, False),
        ("E7 Pipeline", "~399ms median", "Analytical hardware model; 8/20 windows over budget", C_PROPOSED, False),
        ("E8 Blast Radius", "60–80% reduction", "KMM compromise still full exposure", C_GOOD, False),
    ]

    cols = st.columns(2)
    for i, (exp, headline, detail, color, is_warn) in enumerate(findings):
        with cols[i % 2]:
            border = C_WARN if is_warn else color
            sub_color = C_WARN if is_warn else C_GOOD
            st.markdown(
                f"""<div class="metric-card" style="border-left-color:{border}">
                    <div class="metric-label">{exp}</div>
                    <div class="metric-value" style="font-size:1.4rem;color:#e8ecf0">{headline}</div>
                    <div class="metric-sub" style="color:{sub_color}">{detail}</div>
                   </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Honest Limitations")
    limitations = [
        "E2 calibration is OP-TEE QEMU host-reference timing only; it is not physical hardware timing or TA-side 2048-bit Paillier timing.",
        "SGX enclave is simulated; formal security relies on TEE assumption, not hardware enforcement.",
        "E6 fault detection and E7 pipeline latency use analytical models, not live distributed experiments.",
        "KMM is a single trust anchor — KMM compromise exposes all groups in both schemes (E8).",
        "E5 is a deterministic task-level simulator; it is not a live distributed scheduler benchmark.",
    ]
    for lim in limitations:
        warn_box(lim)

    st.divider()
    st.markdown("#### Radar: Normalised System Properties")

    categories = ["Storage\nEfficiency", "Correctness", "Fault\nTolerance",
                  "Scheduling\nEfficiency", "Security\nIsolation"]
    proposed_scores = [1.0, 1.0, 0.87, 0.79, 0.80]
    baseline_scores = [0.10, 1.0, 0.25, 0.78, 0.20]

    fig = go.Figure()
    for name, scores, color in [
        ("Proposed ACK+KMM", proposed_scores, C_PROPOSED),
        ("Naive baseline", baseline_scores, "#F44336"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line_color=color,
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
        ))

    apply_layout(fig,
                 polar=dict(
                     bgcolor="#1E2329",
                     radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a3140", color="#9aa5b4"),
                     angularaxis=dict(gridcolor="#2a3140", color="#b0bec5"),
                 ),
                 showlegend=True,
                 legend=dict(bgcolor="rgba(0,0,0,0)"),
                 height=420)
    st.plotly_chart(fig, use_container_width=True)
