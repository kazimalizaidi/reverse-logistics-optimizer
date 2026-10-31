"""
Returns Control Tower  -  Streamlit dashboard.

Run:  python run_pipeline.py   (once, produces outputs/)
      streamlit run app.py

Reads only the CSVs in outputs/. Focus: processing backlog, facility SLA breaches and
recaptured inventory value margins for the selected plan.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rl_network.config import DISPOSITIONS, FACILITIES, HUBS

OUT = Path(__file__).parent / "outputs"
OUTLINE = json.load(open(Path(__file__).parent / "data" / "south_asia_outline.json"))["countries"]

# ---------------------------------------------------------------- palette
INK, SURF, LINE = "#070B0A", "#0E1514", "#1B2926"
TEXT, MUTED, DIM = "#E8EFE9", "#7F918B", "#4B5C57"
LIME, MINT, AMBER, CORAL, SAGE = "#C6F25D", "#4FE3C1", "#FFB02E", "#FF5C47", "#8C9A94"
DISP_COLOR = {"restock": MINT, "refurbish": LIME, "scrap": SAGE}

SCEN = {
    "A_greedy_nearest":  ("Nearest-facility baseline", "A"),
    "B_milp_prior_only": ("MILP, history-only forecast", "B"),
    "C_milp_classifier": ("MILP, classifier-driven", "C"),
    "D_milp_oracle":     ("MILP, perfect information", "D"),
}

LABEL_POS = {"Karachi": "bottom center", "Lahore": "bottom right", "Islamabad": "top right", "Faisalabad": "middle left",
             "Multan": "bottom right", "Peshawar": "top left", "Hyderabad": "middle right", "Quetta": "bottom center",
             "Sialkot": "middle right"}
SHORT = {"A_greedy_nearest": "Greedy baseline", "B_milp_prior_only": "No classifier",
         "C_milp_classifier": "Classifier MILP", "D_milp_oracle": "Oracle MILP"}

st.set_page_config(page_title="Returns Control Tower", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- styling
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
:root {{
  --ink:{INK}; --surf:{SURF}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED}; --dim:{DIM};
  --lime:{LIME}; --mint:{MINT}; --amber:{AMBER}; --coral:{CORAL}; --sage:{SAGE};
  --display:'Space Grotesk','Segoe UI',system-ui,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace;
  --body:'Inter','Segoe UI',system-ui,sans-serif;
}}
html, body, .stApp {{ font-family: var(--body); color: var(--text); }}
.stApp {{
  background:
    radial-gradient(1100px 520px at 88% -8%, rgba(198,242,93,.075), transparent 60%),
    radial-gradient(900px 480px at -8% 6%, rgba(79,227,193,.07), transparent 55%),
    radial-gradient(rgba(232,239,233,.035) 1px, transparent 1px) 0 0/26px 26px,
    {INK};
}}
/* keep the header (it holds the button that re-opens a collapsed sidebar) but make it invisible chrome */
header[data-testid="stHeader"] {{ background: transparent !important; height: 3rem; }}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stMainMenuButton"] {{ display:none !important; }}
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] button {{ color: {LIME} !important; }}
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] {{ display:flex !important; visibility:visible !important;
  background: rgba(198,242,93,.10); border:1px solid rgba(198,242,93,.35); border-radius:12px; z-index:999; }}
[data-testid="stExpandSidebarButton"] svg, [data-testid="collapsedControl"] svg {{ fill: {LIME} !important; color: {LIME} !important; }}
.block-container {{ padding: 2.8rem 2.2rem 3rem !important; max-width: 1560px; }}
h1,h2,h3 {{ font-family: var(--display); }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: #060A09; border-right: 1px solid #14201D; min-width: 286px !important; max-width: 286px !important; }}
section[data-testid="stSidebar"] > div {{ padding-top: 1.1rem; }}
section[data-testid="stSidebar"] label p {{ font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }}
[data-baseweb="select"] > div {{ background: {SURF} !important; border: 1px solid #1E2D2A !important; border-radius: 12px !important; min-height: 44px; }}
[data-baseweb="select"] * {{ color: var(--text) !important; font-family: var(--body); }}
[data-baseweb="popover"] ul {{ background: #0B1211 !important; border: 1px solid #1E2D2A; border-radius: 12px; }}

/* bordered containers = cards */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: linear-gradient(180deg, rgba(19,31,29,.78), rgba(10,16,15,.92));
  border: 1px solid {LINE} !important; border-radius: 20px !important; padding: 8px 12px 4px;
  box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 24px 48px -28px rgba(0,0,0,.8);
}}

/* html cards */
.card {{ position:relative; overflow:hidden; background: linear-gradient(180deg, rgba(19,31,29,.78), rgba(10,16,15,.92));
  border:1px solid {LINE}; border-radius:20px; padding:18px 20px; box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 24px 48px -28px rgba(0,0,0,.8); }}
.card .glow {{ position:absolute; right:-40px; top:-60px; width:180px; height:180px; border-radius:50%; filter: blur(46px); opacity:.30; pointer-events:none; }}
.eyebrow {{ font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }}
.crumb {{ font-family: var(--mono); font-size: 11.5px; color: var(--muted); letter-spacing:.04em; }}
.crumb b {{ color: var(--text); font-weight:500; }}
.h1 {{ font-family: var(--display); font-weight:600; font-size: 40px; letter-spacing:-.02em; line-height:1.05; margin: 14px 0 6px; }}
.h1 em {{ font-style: normal; color: var(--lime); }}
.sub {{ color: var(--muted); font-size: 14px; }}
.sub b {{ color: var(--text); font-weight:500; }}
.chip {{ display:inline-flex; align-items:center; gap:8px; padding:7px 12px; border:1px solid #20322e; border-radius:999px; background: rgba(198,242,93,.06); font-family: var(--mono); font-size:11.5px; color: var(--text); }}
.dot {{ width:7px; height:7px; border-radius:50%; background: var(--lime); box-shadow: 0 0 0 0 rgba(198,242,93,.6); animation: pulse 2s infinite; }}
.dot.coral {{ background: var(--coral); box-shadow: 0 0 0 0 rgba(255,92,71,.6); animation: pulsec 1.6s infinite; }}
@keyframes pulse {{ 70% {{ box-shadow: 0 0 0 8px rgba(198,242,93,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(198,242,93,0); }} }}
@keyframes pulsec {{ 70% {{ box-shadow: 0 0 0 8px rgba(255,92,71,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(255,92,71,0); }} }}

/* KPI */
.card.kpi {{ height: 236px; }}
.kpi-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; height:92px; }}
.kpi-val {{ font-family: var(--mono); font-weight:500; font-size: 38px; letter-spacing:-.03em; margin: 8px 0 4px; line-height:1; }}
.kpi-val small {{ font-size:15px; color: var(--muted); letter-spacing:0; margin-left:4px; }}
.kpi-sub {{ color: var(--muted); font-size: 12.5px; min-height: 18px; }}
.delta {{ display:inline-block; margin-top:10px; font-family: var(--mono); font-size: 11px; padding: 3px 8px; border-radius: 999px; }}
.delta.good {{ color: var(--lime); background: rgba(198,242,93,.10); }}
.delta.bad {{ color: var(--coral); background: rgba(255,92,71,.12); }}
.delta.flat {{ color: var(--muted); background: rgba(127,145,139,.12); }}
.spark {{ display:flex; align-items:flex-end; gap:4px; height:46px; }}
.spark i {{ display:block; width:7px; border-radius:3px; min-height:3px; }}

/* panel headers */
.ph {{ display:flex; justify-content:space-between; align-items:baseline; margin: 6px 4px 2px; }}
.pt {{ font-family: var(--mono); font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color: var(--muted); }}
.ps {{ font-size:12px; color: var(--dim); }}
.big {{ font-family: var(--mono); font-size: 30px; letter-spacing:-.02em; }}

/* utilisation list */
.ul-row {{ margin: 13px 0; }}
.ul-head {{ display:flex; justify-content:space-between; font-size:13px; }}
.ul-head span:last-child {{ font-family: var(--mono); color: var(--muted); }}
.bar {{ height:6px; border-radius:6px; background:#142120; margin-top:7px; overflow:hidden; position:relative; }}
.bar i {{ position:absolute; inset:0 auto 0 0; border-radius:6px; }}

/* watch card */
.watch {{ position:relative; border-radius:18px; padding:16px 18px; margin-top:6px; overflow:hidden;
  border:1px solid rgba(198,242,93,.30);
  background: radial-gradient(260px 120px at 50% 0%, rgba(198,242,93,.22), rgba(198,242,93,.04) 60%, rgba(10,16,15,.9));
  box-shadow: 0 0 40px -12px rgba(198,242,93,.45); }}
.watch.coral {{ border-color: rgba(255,92,71,.38); background: radial-gradient(260px 120px at 50% 0%, rgba(255,92,71,.26), rgba(255,92,71,.05) 60%, rgba(10,16,15,.9)); box-shadow: 0 0 40px -12px rgba(255,92,71,.5); }}
.watch .wt {{ display:flex; justify-content:center; align-items:center; gap:8px; font-family: var(--mono); font-size:10.5px; letter-spacing:.16em; color: var(--text); }}
.watch p {{ margin:10px 0 0; font-size:12.8px; line-height:1.55; color:#cfdad4; text-align:center; }}
.watch p b {{ color:#fff; font-weight:600; }}

/* detail grid */
.dgrid {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px 12px; margin-top:8px; }}
.dgrid .k {{ font-family: var(--mono); font-size:9.5px; letter-spacing:.13em; text-transform:uppercase; color: var(--muted); }}
.dgrid .v {{ font-family: var(--mono); font-size:15px; margin-top:4px; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color: var(--muted); margin-top:4px; }}
.legend span::before {{ content:''; display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; background: var(--c); }}

/* tables */
table.t, table.t thead, table.t tbody, table.t tr {{ border:0 !important; background:transparent !important; }}
table.t {{ width:100%; border-collapse: collapse; font-size:12.5px; }}
.tw {{ overflow-x:auto; padding: 0 4px 6px; }}
table.t th, table.t td {{ border-left:0 !important; border-right:0 !important; border-top:0 !important; white-space:nowrap; }}
table.t th {{ border-bottom:1px solid {LINE} !important; text-align:left; font-family: var(--mono); font-weight:400; font-size:9.5px; letter-spacing:.1em; text-transform:uppercase; color: var(--muted); padding: 9px 8px; border-bottom:1px solid {LINE}; }}
table.t td {{ padding: 11px 8px; border-bottom:1px solid #121d1b !important; }}
table.t tr:last-child td {{ border-bottom:none; }}
table.t td.m {{ font-family: var(--mono); }}
table.t tr.sel td {{ background: rgba(198,242,93,.06); }}
table.t tr.sel td:first-child {{ box-shadow: inset 3px 0 0 var(--lime); }}
.pill {{ display:inline-flex; align-items:center; gap:6px; font-family: var(--mono); font-size:11px; padding:3px 9px; border-radius:999px; }}
.pill::before {{ content:''; width:6px; height:6px; border-radius:50%; background: currentColor; }}
.pill.ok {{ color: var(--mint); background: rgba(79,227,193,.10); }}
.pill.late {{ color: var(--coral); background: rgba(255,92,71,.12); }}
.pill.warn {{ color: var(--amber); background: rgba(255,176,46,.12); }}
.pill.off {{ color: var(--muted); background: rgba(127,145,139,.12); }}
.tag {{ font-family: var(--mono); font-size:11px; padding:2px 8px; border-radius:6px; border:1px solid var(--c); color: var(--c); }}

/* brand */
.brand {{ display:flex; align-items:center; gap:12px; padding: 4px 4px 18px; }}
.brand b {{ font-family: var(--display); font-size:15.5px; font-weight:600; display:block; letter-spacing:-.01em; }}
.brand span {{ font-size:11.5px; color: var(--muted); }}
.side-sec {{ font-family: var(--mono); font-size:10px; letter-spacing:.16em; color: var(--dim); margin: 22px 4px 8px; text-transform:uppercase; }}
.side-card {{ border:1px solid #16241f; border-radius:14px; padding:12px 14px; background: rgba(14,21,20,.7); margin-bottom:8px; }}
.side-card .n {{ font-family: var(--mono); font-size:20px; }}
.side-card .l {{ font-size:11.5px; color: var(--muted); }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def H(s: str):
    """Render an HTML block. Lines are stripped (indentation would trigger markdown code
    blocks) and '$' is escaped upstream so Streamlit does not read it as LaTeX."""
    st.markdown(re.sub(r"\n\s+", "\n", s.strip()), unsafe_allow_html=True)


def usd(x, signed=False):
    s = f"{abs(x):,.0f}"
    sign = ("+" if x > 0 else "-" if x < 0 else "") if signed else ("-" if x < 0 else "")
    return f"{sign}&#36;{s}"


def num(x, d=0):
    return f"{x:,.{d}f}"


def spark(values, color, hi=None, labels=None):
    mx = max(max(values), 1e-9)
    bars = []
    for i, v in enumerate(values):
        h = max(3, 46 * v / mx)
        c = color if (hi is None or i == hi) else "rgba(127,145,139,.35)"
        op = 1 if v > 0 else .25
        t = f' title="{labels[i]}: {v:,.0f}"' if labels else ""
        bars.append(f'<i style="height:{h:.0f}px;background:{c};opacity:{op}"{t}></i>')
    return f'<div class="spark">{"".join(bars)}</div>'


def ring(frac, color):
    c = 2 * np.pi * 34
    return (f'<svg width="92" height="92" viewBox="0 0 92 92"><circle cx="46" cy="46" r="34" fill="none" '
            f'stroke="#16241f" stroke-width="9"/><circle cx="46" cy="46" r="34" fill="none" stroke="{color}" '
            f'stroke-width="9" stroke-linecap="round" stroke-dasharray="{frac * c:.1f} {c:.1f}" '
            f'transform="rotate(-90 46 46)"/><text x="46" y="51" text-anchor="middle" fill="{TEXT}" '
            f'font-family="JetBrains Mono, monospace" font-size="13.5">{frac:.1%}</text></svg>')


def delta(cur, base, lower_is_better, fmt, base_label):
    d = cur - base
    if abs(d) < 0.5:
        return f'<span class="delta flat">no change vs {base_label}</span>'
    good = (d < 0) == lower_is_better
    arrow = "▲" if d > 0 else "▼"
    return f'<span class="delta {"good" if good else "bad"}">{arrow} {fmt(abs(d))} vs {base_label}</span>'


def plot_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=4, r=4, t=8, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=MUTED, size=12),
                      hoverlabel=dict(bgcolor="#0B1211", bordercolor="#24352f",
                                      font=dict(family="JetBrains Mono, monospace", color=TEXT, size=12)))
    return fig


def panel_head(title, sub=""):
    H(f'<div class="ph"><span class="pt">{title}</span><span class="ps">{sub}</span></div>')


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    need = ["scenario_summary.csv", "facility_kpis.csv", "routing_plan.csv"]
    if not all((OUT / n).exists() for n in need):
        return None
    return tuple(pd.read_csv(OUT / n) for n in need)


data = load()
if data is None:
    st.error("No pipeline outputs found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()
summary, fac_all, routes_all = data
summary = summary.set_index("scenario")

CITY = {f["name"]: f["city"] for f in FACILITIES}
XY = {h[0]: (h[1], h[2]) for h in HUBS}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    H("""
    <div class="brand">
      <svg width="38" height="38" viewBox="0 0 38 38"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#C6F25D"/><stop offset="1" stop-color="#4FE3C1"/></linearGradient></defs>
      <circle cx="19" cy="19" r="14" fill="none" stroke="url(#g)" stroke-width="4" stroke-dasharray="64 24" stroke-linecap="round"/>
      <circle cx="19" cy="19" r="5" fill="url(#g)"/></svg>
      <div><b>Returns Control Tower</b><span>Circular network optimizer</span></div>
    </div>""")
    scen_keys = [k for k in SCEN if k in summary.index]
    scen = st.selectbox("Plan under review", scen_keys, index=scen_keys.index("C_milp_classifier")
                        if "C_milp_classifier" in scen_keys else 0, format_func=lambda k: f"{SCEN[k][1]} · {SHORT[k]}")
    base_opts = [k for k in scen_keys if k != scen]
    base_default = "A_greedy_nearest" if "A_greedy_nearest" in base_opts else base_opts[0]
    base = st.selectbox("Compare against", base_opts, index=base_opts.index(base_default),
                        format_func=lambda k: f"{SCEN[k][1]} · {SHORT[k]}")

S, B = summary.loc[scen], summary.loc[base]
F = fac_all[fac_all.scenario == scen].reset_index(drop=True)
R = routes_all[routes_all.scenario == scen].copy()
F["city"] = F.facility.map(CITY)
F["load"] = F.units_assigned / F.capacity
base_label = f"plan {SCEN[base][1]}"

with st.sidebar:
    H(f"""
    <div class="side-sec">Network</div>
    <div class="side-card"><div class="n">{num(S.units_returned)}</div><div class="l">returns in the planning week</div></div>
    <div class="side-card"><div class="n">{int(S.facilities_open)} / {len(F)} sites open</div><div class="l">{len(HUBS)} collection hubs feed them</div></div>
    <div class="side-sec">Legend</div>
    <div class="legend">
      <span style="--c:{MINT}">Restock</span><span style="--c:{LIME}">Refurbish</span><span style="--c:{SAGE}">Scrap</span>
      <span style="--c:{AMBER}">Backlog</span><span style="--c:{CORAL}">SLA breach</span>
    </div>
    <div class="side-sec">Reading guide</div>
    <div class="l" style="font-size:12px;color:{MUTED};line-height:1.55">KPIs replay the actual week through the selected plan.
    Lanes on the map show the plan itself. Backlog is load above weekly capacity.</div>""")

# ---------------------------------------------------------------- header
attention = int(((F.backlog_units > 0.5) | (F.sla_breach_units > 0.5)).sum())
status_dot = "coral" if attention else ""
c1, c2 = st.columns([3, 1.1])
with c1:
    H(f"""
    <div class="crumb">Operations / <b>Returns control tower</b></div>
    <div class="h1">Network health, <em>planning week</em></div>
    <div class="sub"><b>{num(S.units_returned)}</b> returns across <b>{len(HUBS)}</b> hubs and <b>{len(F)}</b> facilities.
    {"<b>" + str(attention) + "</b> facilit" + ("y needs" if attention == 1 else "ies need") + " attention." if attention else "No facility needs attention."}</div>""")
with c2:
    H(f"""<div style="display:flex;justify-content:flex-end;padding-top:6px"><span class="chip"><span class="dot {status_dot}"></span>
    {SCEN[scen][1]} · {SCEN[scen][0]}</span></div>""")
st.write("")

# ---------------------------------------------------------------- KPI row
labels = list(F.city)
worst_days = F.backlog_days.max()
breach_site = F.loc[F.sla_breach_units.idxmax()] if S.sla_breach_units > 0 else None
margin = S.net_contribution / S.recovered_value
base_margin = B.net_contribution / B.recovered_value
net_labels = [SCEN[k][1] for k in scen_keys]
net_vals = [summary.loc[k, "net_contribution"] for k in scen_keys]

k1, k2, k3, k4 = st.columns(4)
with k1:
    H(f"""<div class="card kpi"><div class="glow" style="background:{AMBER}"></div>
    <div class="kpi-top"><span class="eyebrow">Processing backlog</span>{spark(list(F.backlog_units), AMBER, labels=labels)}</div>
    <div class="kpi-val">{num(S.backlog_units)}<small>units</small></div>
    <div class="kpi-sub">{f"Worst site clears in {worst_days:.1f} days of full-rate work" if S.backlog_units > 0.5 else "Every site clears its load within the week"}</div>
    {delta(S.backlog_units, B.backlog_units, True, lambda d: num(d) + " units", base_label)}</div>""")
with k2:
    H(f"""<div class="card kpi"><div class="glow" style="background:{CORAL}"></div>
    <div class="kpi-top"><span class="eyebrow">Facility SLA breaches</span>{spark(list(F.sla_breach_units), CORAL, labels=labels)}</div>
    <div class="kpi-val">{num(S.sla_breach_units)}<small>units · {S.sla_breach_rate:.1%}</small></div>
    <div class="kpi-sub">{f"Concentrated at {breach_site.facility}" if breach_site is not None else "No unit arrives after its SLA"}</div>
    {delta(S.sla_breach_units, B.sla_breach_units, True, lambda d: num(d) + " units", base_label)}</div>""")
with k3:
    H(f"""<div class="card kpi"><div class="glow" style="background:{LIME}"></div>
    <div class="kpi-top"><span class="eyebrow">Recaptured value</span>{ring(S.value_recovery_rate, LIME)}</div>
    <div class="kpi-val">{usd(S.recovered_value)}</div>
    <div class="kpi-sub">{S.value_recovery_rate:.1%} of {usd(S.original_value)} original value</div>
    {delta(S.recovered_value, B.recovered_value, False, lambda d: usd(d), base_label)}</div>""")
with k4:
    H(f"""<div class="card kpi"><div class="glow" style="background:{MINT}"></div>
    <div class="kpi-top"><span class="eyebrow">Recapture margin</span>{spark(net_vals, MINT, hi=scen_keys.index(scen), labels=net_labels)}</div>
    <div class="kpi-val">{margin:.1%}</div>
    <div class="kpi-sub">{usd(S.net_contribution)} net after {usd(S.cost_to_serve)} cost to serve</div>
    {delta(margin * 100, base_margin * 100, False, lambda d: f"{d:.1f} pts", base_label)}</div>""")
st.write("")

# ---------------------------------------------------------------- load vs capacity + utilisation
left, right = st.columns([2.15, 1])
with left:
    with st.container(border=True):
        panel_head("Facility load vs capacity", "units this week · bars = processed, amber = backlog")
        fig = go.Figure()
        for d in DISPOSITIONS:
            fig.add_bar(x=F.city, y=F[f"units_{d}"], name=d.capitalize(), marker_color=DISP_COLOR[d],
                        hovertemplate="%{x}<br>" + d + ": %{y:,.0f} units<extra></extra>")
        fig.add_bar(x=F.city, y=F.backlog_units, name="Backlog", marker_color=AMBER,
                    hovertemplate="%{x}<br>backlog: %{y:,.0f} units<extra></extra>")
        fig.add_trace(go.Scatter(x=F.city, y=F.capacity, mode="markers", name="Weekly capacity",
                                 marker=dict(symbol="line-ew", size=64, line=dict(width=3, color=TEXT)),
                                 hovertemplate="%{x}<br>capacity: %{y:,.0f} units<extra></extra>"))
        for _, r in F[F.open == 0].iterrows():
            fig.add_annotation(x=r.city, y=r.capacity / 2, text="CLOSED<br>THIS WEEK", showarrow=False,
                               font=dict(family="JetBrains Mono, monospace", size=11, color=MUTED))
        plot_layout(fig, 372).update_layout(
            barmode="stack", bargap=.42, legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=TEXT, size=12.5)),
            yaxis=dict(gridcolor="#13201e", zeroline=False, tickformat=","))
        try:
            fig.update_traces(marker_cornerradius=6, selector=dict(type="bar"))
        except Exception:
            pass
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")

with right:
    rows = []
    for _, r in F.iterrows():
        if not r.open:
            rows.append(f'<div class="ul-row"><div class="ul-head"><span style="color:{DIM}">{r.facility}</span>'
                        f'<span>closed</span></div><div class="bar"></div></div>')
            continue
        col = CORAL if r.load > 1.0001 else AMBER if r.load > .9 else MINT
        rows.append(f'<div class="ul-row"><div class="ul-head"><span>{r.facility}</span><span>{r.load:.0%}</span></div>'
                    f'<div class="bar"><i style="width:{min(r.load, 1) * 100:.1f}%;background:linear-gradient(90deg,{col}66,{col})"></i></div></div>')
    # watch item: backlog first, otherwise the largest SLA breach, otherwise all-clear
    if S.backlog_units > 0.5:
        w = F.loc[F.backlog_units.idxmax()]
        cls, title = "coral", "WATCH ITEM · BACKLOG"
        msg = (f"<b>{w.facility}</b> is assigned <b>{w.load:.0%}</b> of its weekly capacity. "
               f"<b>{num(w.backlog_units)}</b> units cannot be processed this week, about <b>{w.backlog_days:.1f} days</b> of full-rate work.")
    elif S.sla_breach_units > 0.5:
        late = R[(R.late == True) & (R.facility != "DISPOSAL")].sort_values("planned_units", ascending=False)
        cls, title = "coral", "WATCH ITEM · SLA"
        if len(late):
            l = late.iloc[0]
            msg = (f"<b>{l.hub} → {l.facility}</b> carries <b>{num(l.planned_units)}</b> {l.disposition} units at "
                   f"<b>{l.transit_days:.1f} days</b> against a <b>{l.sla_days:.0f}-day</b> SLA. "
                   f"{num(S.sla_breach_units)} units breach SLA in total.")
        else:
            msg = f"<b>{num(S.sla_breach_units)}</b> units breach their SLA this week."
    else:
        cls, title = "", "ALL CLEAR"
        msg = "No facility is over capacity and no unit arrives after its SLA in this plan."
    with st.container(border=True):
        panel_head("Facility load", "assigned ÷ weekly capacity")
        H("".join(rows))
        H(f'<div class="watch {cls}"><div class="wt"><span class="dot {"coral" if cls else ""}"></span>{title}</div><p>{msg}</p></div>')
        st.write("")

# ---------------------------------------------------------------- map + facility detail
mapc, detc = st.columns([2.15, 1])
with mapc:
    with st.container(border=True):
        panel_head("Return lanes", "curves = planned flows · coral = arrives after SLA")
        fig = go.Figure()
        for cname, rings in OUTLINE.items():
            home = cname == "Pakistan"
            for ring_ in rings:
                xs, ys = zip(*ring_)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", hoverinfo="skip", showlegend=False,
                                         fillcolor="#122220" if home else "#0B1312",
                                         line=dict(width=1.4 if home else 0.8, color="#2B4640" if home else "#1A2A27")))
        lanes = R[R.facility != "DISPOSAL"].copy()
        mx = max(lanes.planned_units.max(), 1) if len(lanes) else 1
        for _, r in lanes.iterrows():
            city = CITY[r.facility]
            if city == r.hub:
                continue
            (la1, lo1), (la2, lo2) = XY[r.hub], XY[city]
            t = np.linspace(0, 1, 28)
            mla, mlo = (la1 + la2) / 2, (lo1 + lo2) / 2
            nx, ny = -(la2 - la1), (lo2 - lo1)                       # perpendicular offset -> arc
            nrm = max(np.hypot(nx, ny), 1e-9)
            bulge = 0.16 * np.hypot(la2 - la1, lo2 - lo1)
            cla, clo = mla + nx / nrm * bulge, mlo + ny / nrm * bulge
            la = (1 - t) ** 2 * la1 + 2 * (1 - t) * t * cla + t ** 2 * la2
            lo = (1 - t) ** 2 * lo1 + 2 * (1 - t) * t * clo + t ** 2 * lo2
            col = CORAL if r.late else MINT
            fig.add_trace(go.Scatter(x=lo, y=la, mode="lines", showlegend=False,
                                     line=dict(width=1.0 + 3.6 * r.planned_units / mx, color=col),
                                     opacity=.85 if r.late else .55, hoverinfo="text",
                                     text=f"{r.hub} → {r.facility}<br>{r.disposition}: {r.planned_units:,.0f} units<br>"
                                          f"transit {r.transit_days:.1f} d vs SLA {r.sla_days:.0f} d"))
        hub_units = lanes.groupby("hub").planned_units.sum().reindex([h[0] for h in HUBS]).fillna(0)
        fig.add_trace(go.Scatter(x=[h[2] for h in HUBS], y=[h[1] for h in HUBS], mode="markers+text",
                                 text=[h[0] for h in HUBS], textposition=[LABEL_POS[h[0]] for h in HUBS], showlegend=False,
                                 textfont=dict(color=MUTED, size=11, family="Inter, sans-serif"),
                                 marker=dict(size=9 + 14 * hub_units.values / max(hub_units.max(), 1), color="rgba(0,0,0,0)",
                                             line=dict(width=1.6, color=TEXT)),
                                 hovertext=[f"{h}: {u:,.0f} planned units out" for h, u in hub_units.items()], hoverinfo="text"))
        scol = [DIM if not r.open else CORAL if (r.backlog_units > .5 or r.sla_breach_units > .5) else LIME for _, r in F.iterrows()]
        fig.add_trace(go.Scatter(x=[XY[c][1] for c in F.city], y=[XY[c][0] for c in F.city], mode="markers",
                                 showlegend=False, marker=dict(size=13, symbol="diamond", color=scol, line=dict(width=2, color=INK)),
                                 hovertext=[f"{r.facility}<br>{'closed' if not r.open else f'load {r.load:.0%}'}" for _, r in F.iterrows()],
                                 hoverinfo="text"))
        plot_layout(fig, 440).update_layout(
            xaxis=dict(range=[60.2, 77.8], visible=False, fixedrange=True),
            yaxis=dict(range=[23.2, 36.6], visible=False, scaleanchor="x", scaleratio=1.1547, fixedrange=True),
            plot_bgcolor="#080D0C", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
        H(f'<div class="legend" style="margin:6px 4px 8px"><span style="--c:{LIME}">Facility, healthy</span>'
          f'<span style="--c:{CORAL}">Facility, backlog or SLA breach</span><span style="--c:{DIM}">Closed</span>'
          f'<span style="--c:{MINT}">On-time lane</span><span style="--c:{TEXT}">Hub, sized by volume</span></div>')

with detc:
    with st.container(border=True):
        panel_head("Facility detail")
        pick = st.selectbox("Facility", list(F.facility), index=int(F.backlog_units.idxmax()) if S.backlog_units > .5
                            else int(F.sla_breach_units.idxmax()) if S.sla_breach_units > .5 else 0,
                            label_visibility="collapsed")
        r = F[F.facility == pick].iloc[0]
        vals = [r[f"units_{d}"] for d in DISPOSITIONS]
        donut = go.Figure(go.Pie(labels=[d.capitalize() for d in DISPOSITIONS], values=vals, hole=.74, sort=False,
                                 marker=dict(colors=[DISP_COLOR[d] for d in DISPOSITIONS], line=dict(color=INK, width=3)),
                                 textinfo="none", hovertemplate="%{label}: %{value:,.0f} units<extra></extra>"))
        donut.add_annotation(text=f"<b>{sum(vals):,.0f}</b><br><span style='font-size:11px'>units processed</span>",
                             showarrow=False, font=dict(family="JetBrains Mono, monospace", size=22, color=TEXT))
        plot_layout(donut, 215).update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(donut, config={"displayModeBar": False}, width="stretch")
        if not r.open:
            stat = '<span class="pill off">Closed this week</span>'
        elif r.backlog_units > .5:
            stat = '<span class="pill warn">Backlogged</span>'
        elif r.sla_breach_units > .5:
            stat = '<span class="pill late">SLA breaches</span>'
        else:
            stat = '<span class="pill ok">Healthy</span>'
        mg = f"{r.recapture_margin:.1%}" if pd.notna(r.recapture_margin) else "n/a"
        H(f"""<div class="dgrid">
        <div><div class="k">Status</div><div class="v">{stat}</div></div>
        <div><div class="k">Weekly capacity</div><div class="v">{num(r.capacity)}</div></div>
        <div><div class="k">Assigned</div><div class="v">{num(r.units_assigned)}</div></div>
        <div><div class="k">Backlog</div><div class="v" style="color:{AMBER if r.backlog_units > .5 else TEXT}">{num(r.backlog_units)} u · {r.backlog_days:.1f} d</div></div>
        <div><div class="k">SLA breaches</div><div class="v" style="color:{CORAL if r.sla_breach_units > .5 else TEXT}">{num(r.sla_breach_units)} u</div></div>
        <div><div class="k">Recaptured value</div><div class="v">{usd(r.recovered_value)}</div></div>
        <div><div class="k">Cost to serve</div><div class="v">{usd(r.cost_to_serve)}</div></div>
        <div><div class="k">Recapture margin</div><div class="v" style="color:{LIME}">{mg}</div></div></div><div style="height:8px"></div>""")

# ---------------------------------------------------------------- lanes table + policy table
lt, pt = st.columns([1.45, 1])
with lt:
    with st.container(border=True):
        panel_head("Lanes in motion", "late lanes first, then by volume")
        ln = R.copy()
        ln["late"] = ln["late"].astype(bool)
        ln = ln.sort_values(["late", "planned_units"], ascending=[False, False]).head(8)
        body = []
        for _, r in ln.iterrows():
            if r.facility == "DISPOSAL":
                route, status = f"{r.hub} → Disposal", '<span class="pill late">Disposal</span>'
                dist = "n/a"
            else:
                route = f"{r.hub} → {r.facility}"
                over = r.transit_days - r.sla_days
                status = (f'<span class="pill late">+{over:.1f} d</span>' if r.late else '<span class="pill ok">On time</span>')
                dist = f"{r.distance_km:,.0f} km · {r.transit_days:.1f} / {r.sla_days:.0f} d"
            body.append(f'<tr><td>{route}</td><td><span class="tag" style="--c:{DISP_COLOR[r.disposition]}">{r.disposition}</span></td>'
                        f'<td class="m">{num(r.planned_units)}</td><td class="m">{dist}</td><td>{status}</td></tr>')
        H(f'<div class="tw"><table class="t"><thead><tr><th>Lane</th><th>Disposition</th><th>Units</th><th>Distance · transit / SLA</th><th>Late by</th></tr></thead>'
          f'<tbody>{"".join(body)}</tbody></table></div>')
        st.write("")

with pt:
    with st.container(border=True):
        panel_head("Plan comparison", "same realized week")
        body = []
        for k in scen_keys:
            s = summary.loc[k]
            body.append(f'<tr class="{"sel" if k == scen else ""}"><td><b>{SCEN[k][1]}</b> <span style="color:{MUTED}">{SHORT[k]}</span></td>'
                        f'<td class="m">{usd(s.net_contribution)}</td><td class="m">{s.value_recovery_rate:.1%}</td>'
                        f'<td class="m">{num(s.backlog_units)}</td><td class="m">{num(s.sla_breach_units)}</td></tr>')
        H(f'<div class="tw"><table class="t"><thead><tr><th>Plan</th><th>Net / wk</th><th>Recapt.</th><th>Backlog</th><th>SLA</th></tr></thead>'
          f'<tbody>{"".join(body)}</tbody></table></div>')
        st.write("")
