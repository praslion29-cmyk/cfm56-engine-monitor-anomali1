# cfm56-engine-monitor-anomali1
"""
CFM56-7B Engine Health Monitoring System
Web App — Streamlit
Run with: streamlit run cfm56_health_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CFM56-7B Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #080c14;
    color: #c9d1d9;
}

/* ── Header ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 50%, #091020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,120,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 2px;
    margin: 0;
    text-shadow: 0 0 20px rgba(88,166,255,0.4);
}
.hero-sub {
    font-size: 0.95rem;
    color: #8b949e;
    margin-top: 6px;
    letter-spacing: 1px;
}
.hero-badge {
    display: inline-block;
    background: rgba(88,166,255,0.12);
    border: 1px solid #1e3a5f;
    color: #58a6ff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-top: 12px;
    letter-spacing: 1px;
}

/* ── Upload zone ── */
.upload-wrapper {
    background: #0d1117;
    border: 2px dashed #1e3a5f;
    border-radius: 14px;
    padding: 40px 20px;
    text-align: center;
    transition: border-color 0.3s;
}
.upload-icon { font-size: 3rem; }
.upload-label {
    font-family: 'Share Tech Mono', monospace;
    color: #58a6ff;
    font-size: 1rem;
    margin-top: 8px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: #0d1117;
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    border: 1px solid #21262d;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.metric-card.green::after  { background: #2ea043; }
.metric-card.yellow::after { background: #d29922; }
.metric-card.red::after    { background: #da3633; }
.metric-card.blue::after   { background: #58a6ff; }

.metric-num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-num.green  { color: #3fb950; }
.metric-num.yellow { color: #e3b341; }
.metric-num.red    { color: #f85149; }
.metric-num.blue   { color: #58a6ff; }
.metric-label {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Verdict banner ── */
.verdict-healthy {
    background: linear-gradient(135deg, #0d2119, #0a1f14);
    border: 1.5px solid #2ea043;
    border-radius: 14px;
    padding: 24px 30px;
    margin: 20px 0;
}
.verdict-warning {
    background: linear-gradient(135deg, #1f1a0d, #1a160a);
    border: 1.5px solid #d29922;
    border-radius: 14px;
    padding: 24px 30px;
    margin: 20px 0;
}
.verdict-critical {
    background: linear-gradient(135deg, #1f0d0d, #1a0a0a);
    border: 1.5px solid #da3633;
    border-radius: 14px;
    padding: 24px 30px;
    margin: 20px 0;
}
.verdict-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.verdict-desc { font-size: 0.95rem; color: #8b949e; line-height: 1.6; }

/* ── Section headers ── */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 28px 0 16px;
}

/* ── Parameter table ── */
.param-row {
    display: flex;
    align-items: center;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    gap: 12px;
}
.param-name { font-weight: 600; font-size: 0.95rem; flex: 2; }
.param-value { font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; flex: 1; text-align: right; color: #c9d1d9; }
.param-range { font-size: 0.78rem; color: #8b949e; flex: 1.5; text-align: center; }
.status-pill {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 1px;
    white-space: nowrap;
}
.pill-normal   { background: rgba(46,160,67,0.15);  color: #3fb950; border: 1px solid #2ea043; }
.pill-warning  { background: rgba(210,153,34,0.15); color: #e3b341; border: 1px solid #d29922; }
.pill-critical { background: rgba(218,54,51,0.15);  color: #f85149; border: 1px solid #da3633; }

/* ── Recommendation cards ── */
.rec-card {
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    border-left: 4px solid;
}
.rec-critical { background: #160b0b; border-color: #da3633; }
.rec-warning  { background: #14110a; border-color: #d29922; }
.rec-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.rec-critical .rec-title { color: #f85149; }
.rec-warning  .rec-title { color: #e3b341; }
.rec-body { font-size: 0.88rem; color: #8b949e; line-height: 1.7; }
.rec-action { color: #58a6ff; font-weight: 600; }

/* ── Data table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Streamlit overrides ── */
.stFileUploader > div { background: #0d1117 !important; border-radius: 14px !important; }
.stButton > button {
    background: linear-gradient(135deg, #1158cc, #0d4baf);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    padding: 10px 28px;
    font-size: 0.9rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a6de0, #1158cc);
    box-shadow: 0 0 20px rgba(88,166,255,0.3);
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CFM56-7B OPERATING LIMITS
# ─────────────────────────────────────────────────────────────
LIMITS = {
    "Engine rpm": {
        "min": 400, "max": 1200,
        "warn_low": 500, "warn_high": 1100,
        "unit": "rpm", "desc": "Engine / Fan Speed",
        "icon": "⚙️"
    },
    "Lub oil pressure": {
        "min": 2.0, "max": 5.5,
        "warn_low": 2.5, "warn_high": 5.0,
        "unit": "bar", "desc": "Lubricating Oil Pressure",
        "icon": "🛢️"
    },
    "Fuel pressure": {
        "min": 3.0, "max": 18.0,
        "warn_low": 4.0, "warn_high": 17.0,
        "unit": "psi", "desc": "Fuel Pressure",
        "icon": "⛽"
    },
    "Coolant pressure": {
        "min": 1.0, "max": 4.0,
        "warn_low": 1.2, "warn_high": 3.8,
        "unit": "bar", "desc": "Coolant Pressure",
        "icon": "💧"
    },
    "lub oil temp": {
        "min": 70.0, "max": 88.0,
        "warn_low": 72.0, "warn_high": 86.0,
        "unit": "°C", "desc": "Lubricating Oil Temperature",
        "icon": "🌡️"
    },
    "Coolant temp": {
        "min": 65.0, "max": 90.0,
        "warn_low": 68.0, "warn_high": 87.0,
        "unit": "°C", "desc": "Coolant Temperature",
        "icon": "🌡️"
    },
}

SENSOR_COLS = list(LIMITS.keys())


# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def classify(col, value):
    lim = LIMITS[col]
    if value < lim["min"] or value > lim["max"]:
        return "CRITICAL"
    elif value < lim["warn_low"] or value > lim["warn_high"]:
        return "WARNING"
    return "NORMAL"


def analyze(df):
    df = df.copy()
    for col in SENSOR_COLS:
        if col in df.columns:
            df[f"{col}_status"] = df[col].apply(lambda v: classify(col, v))

    def row_status(row):
        statuses = [row.get(f"{c}_status", "NORMAL") for c in SENSOR_COLS if c in df.columns]
        if "CRITICAL" in statuses: return "CRITICAL"
        if "WARNING"  in statuses: return "WARNING"
        return "NORMAL"

    df["Overall_Status"] = df.apply(row_status, axis=1)
    df["Is_Anomaly"]     = df["Overall_Status"] != "NORMAL"
    return df


def build_recommendations(df):
    recs = []
    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        lim    = LIMITS[col]
        sc     = df[f"{col}_status"]
        vals   = df[col]
        n_crit = (sc == "CRITICAL").sum()
        n_warn = (sc == "WARNING").sum()
        mean_v = vals.mean()
        max_v  = vals.max()
        min_v  = vals.min()

        if col == "Engine rpm":
            if n_crit > 0:
                if (vals > lim["max"]).any():
                    recs.append(("CRITICAL", lim["icon"], lim["desc"],
                        f"RPM terdeteksi melebihi batas atas ({max_v:.0f} rpm; batas maks {lim['max']} rpm). "
                        "Indikasi <b>over-speed condition</b>.",
                        "Periksa Fuel Control Unit (FCU), FADEC fault codes, dan N1/N2 limiter. "
                        "Lakukan engine run-up test. Rujuk AMM TASK 73-20-00."))
                if (vals < lim["min"]).any():
                    recs.append(("CRITICAL", lim["icon"], lim["desc"],
                        f"RPM terdeteksi di bawah batas minimum ({min_v:.0f} rpm; batas min {lim['min']} rpm). "
                        "Indikasi <b>under-speed / stall condition</b>.",
                        "Periksa FADEC, igniter plug, dan bleed air system. "
                        "Cek compressor stall indicator. Rujuk AMM TASK 72-30-00."))
            elif n_warn > 0:
                recs.append(("WARNING", lim["icon"], lim["desc"],
                    f"RPM mendekati batas operasi (min={min_v:.0f}, maks={max_v:.0f} rpm).",
                    "Monitor FADEC fault codes dan VSV/VBV schedule. "
                    "Jadwalkan inspeksi pada next A-Check."))

        if col == "Lub oil pressure":
            if (sc == "CRITICAL").any():
                if (vals < lim["min"]).any():
                    recs.append(("CRITICAL", lim["icon"], lim["desc"],
                        f"Tekanan oli <b>RENDAH KRITIS</b> ({min_v:.2f} bar; minimum {lim['min']} bar). "
                        "Risiko kerusakan bearing, gear, dan seal.",
                        "Cek level oli segera. Inspeksi oil pump, filter oli, dan oil chip detector. "
                        "<b>AOG (Aircraft on Ground)</b> jika tekanan &lt; 1.5 bar. Rujuk AMM TASK 79-00-00."))
                if (vals > lim["max"]).any():
                    recs.append(("CRITICAL", lim["icon"], lim["desc"],
                        f"Tekanan oli <b>TINGGI KRITIS</b> ({max_v:.2f} bar; maksimum {lim['max']} bar). "
                        "Indikasi oil pressure relief valve stuck atau sumbatan return line.",
                        "Periksa dan test oil pressure relief valve. "
                        "Cek sumbatan pada oil return line dan scavenge system. Rujuk AMM TASK 79-20-00."))
            elif n_warn > 0:
                recs.append(("WARNING", lim["icon"], lim["desc"],
                    f"Tekanan oli mendekati batas (avg={mean_v:.2f} bar).",
                    "Ganti filter oli. Cek kondisi seal dan O-ring. Monitor trend setiap 10 flight hours."))

        if col == "Fuel pressure":
            if n_crit > 0:
                recs.append(("CRITICAL", lim["icon"], lim["desc"],
                    f"Tekanan bahan bakar di luar batas (maks={max_v:.2f} psi; batas {lim['min']}–{lim['max']} psi). "
                    "Risiko <b>hot section damage</b> atau <b>lean blowout</b>.",
                    "Inspeksi high-pressure fuel pump dan fuel flow divider. "
                    "Cek fuel filter dan manifold untuk kebocoran. Periksa fuel nozzle atomisasi. Rujuk AMM TASK 73-10-00."))
            elif n_warn > 0:
                recs.append(("WARNING", lim["icon"], lim["desc"],
                    f"Tekanan bahan bakar mendekati batas (avg={mean_v:.2f} psi).",
                    "Ganti fuel filter. Inspeksi fuel pump impeller dan inlet strainer."))

        if col == "Coolant pressure":
            if n_crit > 0:
                recs.append(("CRITICAL", lim["icon"], lim["desc"],
                    f"Tekanan coolant di luar batas (maks={max_v:.2f} bar; batas maks {lim['max']} bar). "
                    "Risiko <b>overheating komponen aksesori</b>.",
                    "Periksa coolant pump, expansion tank, dan pressure cap. "
                    "Cek kebocoran pada hose dan heat exchanger. Rujuk AMM TASK 75-10-00."))
            elif n_warn > 0:
                recs.append(("WARNING", lim["icon"], lim["desc"],
                    f"Tekanan coolant mendekati batas (avg={mean_v:.2f} bar).",
                    "Cek level coolant, bleed udara dari sistem, dan inspeksi kondisi hose."))

        if col == "lub oil temp":
            if n_crit > 0:
                if (vals > lim["max"]).any():
                    recs.append(("CRITICAL", lim["icon"], lim["desc"],
                        f"Suhu oli <b>TERLALU TINGGI</b> ({max_v:.1f}°C; batas {lim['max']}°C). "
                        "Risiko <b>oil coking</b> dan kegagalan bearing.",
                        "Inspeksi air-oil cooler dan fuel-oil heat exchanger. "
                        "Cek airflow di nacelle. Ganti oli jika TBO terlampaui. Rujuk AMM TASK 79-20-00."))
            elif n_warn > 0:
                recs.append(("WARNING", lim["icon"], lim["desc"],
                    f"Suhu oli mendekati batas (maks={max_v:.1f}°C).",
                    "Bersihkan oil cooler. Periksa regulasi bleed air dan kondisi heat exchanger."))

        if col == "Coolant temp":
            if n_crit > 0:
                recs.append(("CRITICAL", lim["icon"], lim["desc"],
                    f"Suhu coolant di luar batas (maks={max_v:.1f}°C; batas {lim['max']}°C). "
                    "Risiko <b>thermal damage</b> pada komponen aksesori.",
                    "Inspeksi thermostat, coolant pump, dan radiator. "
                    "Drain dan flush sistem jika coolant terkontaminasi. Rujuk AMM TASK 75-20-00."))
            elif n_warn > 0:
                recs.append(("WARNING", lim["icon"], lim["desc"],
                    f"Suhu coolant mendekati batas (maks={max_v:.1f}°C).",
                    "Periksa thermostat operation dan level coolant. Monitor di setiap 25 flight hours."))

    return sorted(recs, key=lambda x: 0 if x[0] == "CRITICAL" else 1)


def make_chart(df):
    """
    5-panel dashboard:
      1. Time-series line chart per sensor (nilai aktual + batas)
      2. Anomaly heatmap (record × parameter)
      3. Bar chart rata-rata vs batas
      4. Pie distribusi status
      5. Scatter anomaly per record
    """
    COLOR_MAP = {
        "NORMAL":   "#3fb950",
        "WARNING":  "#e3b341",
        "CRITICAL": "#f85149",
    }
    BG   = "#0d1117"
    PANEL = "#161b22"
    GRID  = "#21262d"
    TXT   = "#c9d1d9"
    SUB   = "#8b949e"

    cols_present = [c for c in SENSOR_COLS if c in df.columns]
    idx          = df.index.tolist()
    total        = len(df)

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── CHART 1: Line chart semua sensor ──────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL)

    line_colors = ["#58a6ff", "#3fb950", "#e3b341", "#f0883e", "#bc8cff", "#f85149"]
    for i, col in enumerate(cols_present):
        lim    = LIMITS[col]
        vals   = df[col].values
        color  = line_colors[i % len(line_colors)]

        # Normalise 0-1 untuk multi-axis overlay
        vmin, vmax = lim["min"] * 0.9, lim["max"] * 1.1
        norm_vals  = (vals - vmin) / (vmax - vmin)
        norm_max   = (lim["max"] - vmin) / (vmax - vmin)
        norm_min   = (lim["min"] - vmin) / (vmax - vmin)

        ax1.plot(idx, norm_vals, color=color, linewidth=1.6,
                 label=f"{col} ({lim['unit']})", alpha=0.9, zorder=3)

        # Mark anomalies
        for j, (v, n) in enumerate(zip(vals, norm_vals)):
            st_col = df[f"{col}_status"].iloc[j]
            if st_col == "CRITICAL":
                ax1.scatter(j, n, color=COLOR_MAP["CRITICAL"], s=70, zorder=5, marker="X")
            elif st_col == "WARNING":
                ax1.scatter(j, n, color=COLOR_MAP["WARNING"],  s=50, zorder=4, marker="D")

        # Limit lines (normalised)
        ax1.axhline(norm_max, color=color, linewidth=0.7, linestyle="--", alpha=0.4)
        ax1.axhline(norm_min, color=color, linewidth=0.7, linestyle=":",  alpha=0.3)

    ax1.set_xlim(-0.5, total - 0.5)
    ax1.set_xlabel("Record Index", color=SUB, fontsize=9)
    ax1.set_ylabel("Nilai Ternormalisasi (0–1 per sensor)", color=SUB, fontsize=9)
    ax1.set_title(
        "CHART 1 — Time-Series Semua Parameter Sensor\n"
        "(X = Critical · ◆ = Warning · garis putus = batas maks/min ternormalisasi)",
        color=TXT, fontsize=11, fontweight="bold", pad=10
    )
    ax1.tick_params(colors=SUB, labelsize=8)
    ax1.spines[:].set_color(GRID)
    ax1.grid(color=GRID, linestyle="--", alpha=0.5, zorder=0)
    ax1.legend(loc="upper right", facecolor=PANEL, labelcolor=TXT,
               edgecolor=GRID, fontsize=7.5, ncol=3)

    # ── CHART 2: Anomaly Heatmap ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(PANEL)

    status_num = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    heat_data  = np.array([
        [status_num[df[f"{c}_status"].iloc[r]] for c in cols_present]
        for r in range(total)
    ]).T  # shape: (n_params, n_records)

    cmap = plt.matplotlib.colors.ListedColormap(
        [COLOR_MAP["NORMAL"], COLOR_MAP["WARNING"], COLOR_MAP["CRITICAL"]]
    )
    im = ax2.imshow(heat_data, aspect="auto", cmap=cmap,
                    vmin=0, vmax=2, interpolation="nearest")

    ax2.set_yticks(range(len(cols_present)))
    ax2.set_yticklabels(
        [f"{LIMITS[c]['icon']} {c}" for c in cols_present],
        color=TXT, fontsize=8.5
    )
    ax2.set_xticks(range(total))
    ax2.set_xticklabels([str(i) for i in idx], color=SUB, fontsize=7)
    ax2.set_xlabel("Record Index", color=SUB, fontsize=9)
    ax2.set_title(
        "CHART 2 — Anomaly Heatmap (Hijau = Normal · Kuning = Warning · Merah = Critical)",
        color=TXT, fontsize=11, fontweight="bold", pad=10
    )
    ax2.tick_params(colors=SUB)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    # Annotate critical cells
    for row_i, col in enumerate(cols_present):
        for rec_i in range(total):
            st_val = df[f"{col}_status"].iloc[rec_i]
            if st_val != "NORMAL":
                label = "C" if st_val == "CRITICAL" else "W"
                ax2.text(rec_i, row_i, label, ha="center", va="center",
                         fontsize=6.5, fontweight="bold",
                         color="white" if st_val == "CRITICAL" else "#1a1400")

    # ── CHART 3: Bar rata-rata vs batas ──────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(PANEL)

    x      = np.arange(len(cols_present))
    means  = [df[c].mean() for c in cols_present]
    maxes  = [LIMITS[c]["max"] for c in cols_present]
    mins   = [LIMITS[c]["min"] for c in cols_present]
    bcols  = []
    for c in cols_present:
        sc = df[f"{c}_status"]
        if (sc == "CRITICAL").any():  bcols.append(COLOR_MAP["CRITICAL"])
        elif (sc == "WARNING").any(): bcols.append(COLOR_MAP["WARNING"])
        else:                         bcols.append(COLOR_MAP["NORMAL"])

    # Normalise per-parameter to 0–100% of operating range
    pct = [(m - mn) / (mx - mn) * 100
           for m, mn, mx in zip(means, mins, maxes)]

    bars = ax3.bar(x, pct, width=0.55, color=bcols, alpha=0.85, zorder=3)
    ax3.axhline(100, color=COLOR_MAP["CRITICAL"], linewidth=1.2,
                linestyle="--", alpha=0.7, label="Batas Maks (100%)")
    ax3.axhline(0,   color=COLOR_MAP["WARNING"],  linewidth=1.2,
                linestyle=":", alpha=0.5, label="Batas Min (0%)")

    for i, (p, m, col) in enumerate(zip(pct, means, cols_present)):
        ax3.text(i, p + 2, f"{m:.1f}\n{LIMITS[col]['unit']}",
                 ha="center", fontsize=7, color=TXT, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [f"{c[:12]}\n({LIMITS[c]['unit']})" for c in cols_present],
        color=SUB, fontsize=7.5
    )
    ax3.set_ylabel("% dalam rentang operasi", color=SUB, fontsize=9)
    ax3.set_ylim(-10, 130)
    ax3.set_title("CHART 3 — Rata-rata Sensor\nvs Rentang Operasi Normal",
                  color=TXT, fontsize=11, fontweight="bold", pad=10)
    ax3.tick_params(colors=SUB, labelsize=8)
    ax3.spines[:].set_color(GRID)
    ax3.grid(axis="y", color=GRID, linestyle="--", alpha=0.4, zorder=0)
    ax3.legend(facecolor=PANEL, labelcolor=TXT, edgecolor=GRID, fontsize=7.5)

    # ── CHART 4: Pie + Scatter side-by-side ──────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(PANEL)

    n_norm = (df["Overall_Status"] == "NORMAL").sum()
    n_warn = (df["Overall_Status"] == "WARNING").sum()
    n_crit = (df["Overall_Status"] == "CRITICAL").sum()

    sizes  = [s for s in [n_norm, n_warn, n_crit] if s > 0]
    labels = [f"{l}\n{s} rec" for l, s in
              zip(["Normal", "Warning", "Critical"], [n_norm, n_warn, n_crit]) if s > 0]
    colors = [c for c, s in
              zip([COLOR_MAP["NORMAL"], COLOR_MAP["WARNING"], COLOR_MAP["CRITICAL"]],
                  [n_norm, n_warn, n_crit]) if s > 0]

    if sizes:
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140, pctdistance=0.72,
            wedgeprops=dict(edgecolor=BG, linewidth=2.5),
            explode=[0.04] * len(sizes)
        )
        for t in texts:     t.set_color(SUB);  t.set_fontsize(8.5)
        for t in autotexts: t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(9)

    ax4.set_title("CHART 4 — Distribusi Status\nSemua Record",
                  color=TXT, fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        "CFM56-7B Engine Health Monitoring — Anomaly Detection Dashboard",
        color="#58a6ff", fontsize=15, fontweight="bold",
        fontfamily="monospace", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


# ─────────────────────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">✈ CFM56-7B ENGINE HEALTH MONITOR</div>
    <div class="hero-sub">Aircraft Engine Anomaly Detection & Diagnostic System</div>
    <div class="hero-badge">CFM International · Boeing 737 Next Generation · v2.0</div>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ──
st.markdown('<div class="section-header">// DATA INPUT</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload file CSV data sensor engine (format: Engine rpm, Lub oil pressure, Fuel pressure, Coolant pressure, lub oil temp, Coolant temp)",
    type=["csv"],
    label_visibility="visible"
)

if uploaded is None:
    st.markdown("""
    <div style="background:#0d1117; border:2px dashed #1e3a5f; border-radius:14px; padding:40px 20px; text-align:center; margin-top:20px;">
        <div style="font-size:3rem;">📂</div>
        <div style="font-family:'Share Tech Mono',monospace; color:#58a6ff; font-size:1rem; margin-top:8px;">
            Upload file CSV untuk memulai analisis
        </div>
        <div style="color:#8b949e; font-size:0.85rem; margin-top:8px;">
            Pastikan CSV berisi kolom: Engine rpm · Lub oil pressure · Fuel pressure · Coolant pressure · lub oil temp · Coolant temp
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Load & Validate ──
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

    missing = [c for c in SENSOR_COLS if c not in df_raw.columns]
    if missing:
        st.warning(f"⚠️ Kolom berikut tidak ditemukan di CSV: **{', '.join(missing)}**. Analisis dilakukan pada kolom yang tersedia.")

    # ── Preview & Confirmation ──
    st.markdown('<div class="section-header">// PREVIEW DATA</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid #1e3a5f; border-radius:12px;
                padding:18px 24px; margin-bottom:20px;">
        <div style="font-family:'Share Tech Mono',monospace; color:#58a6ff; font-size:0.85rem; letter-spacing:1px;">
            📄 FILE TERDETEKSI
        </div>
        <div style="margin-top:10px; color:#c9d1d9; font-size:0.92rem;">
            <b>{uploaded.name}</b> &nbsp;·&nbsp;
            <span style="color:#8b949e;">{len(df_raw)} baris &nbsp;·&nbsp; {len(df_raw.columns)} kolom</span>
        </div>
        <div style="margin-top:6px; color:#8b949e; font-size:0.8rem;">
            Kolom: {', '.join(df_raw.columns.tolist())}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_raw.head(5), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confirmation Button ──
    col_l, col_c, col_r = st.columns([2, 1.5, 2])
    with col_c:
        run_analysis = st.button("❓  Lanjutkan Analisis?", use_container_width=True)

    if not run_analysis:
        st.markdown("""
        <div style="text-align:center; color:#8b949e; font-size:0.85rem;
                    font-family:'Share Tech Mono',monospace; margin-top:12px; letter-spacing:1px;">
            Klik tombol di atas untuk memulai deteksi anomali engine CFM56-7B
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Run Analysis ──
    with st.spinner("🔍 Menganalisis data sensor engine..."):
        df = analyze(df_raw)

    available_cols = [c for c in SENSOR_COLS if c in df.columns]

    total    = len(df)
    n_normal = (df["Overall_Status"] == "NORMAL").sum()
    n_warn   = (df["Overall_Status"] == "WARNING").sum()
    n_crit   = (df["Overall_Status"] == "CRITICAL").sum()
    n_anom   = df["Is_Anomaly"].sum()

    crit_rate  = n_crit / total * 100
    anom_rate  = n_anom / total * 100
    health_score = max(0, 100 - (crit_rate * 3) - (n_warn / total * 100))

    # ── Metric Cards ──
    st.markdown('<div class="section-header">// RINGKASAN ANOMALI</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="metric-num blue">{total}</div>
            <div class="metric-label">Total Record</div>
        </div>
        <div class="metric-card green">
            <div class="metric-num green">{n_normal}</div>
            <div class="metric-label">Normal</div>
        </div>
        <div class="metric-card yellow">
            <div class="metric-num yellow">{n_warn}</div>
            <div class="metric-label">Warning</div>
        </div>
        <div class="metric-card red">
            <div class="metric-num red">{n_crit}</div>
            <div class="metric-label">Critical</div>
        </div>
        <div class="metric-card {'green' if health_score >= 80 else 'yellow' if health_score >= 60 else 'red'}">
            <div class="metric-num {'green' if health_score >= 80 else 'yellow' if health_score >= 60 else 'red'}">{health_score:.0f}</div>
            <div class="metric-label">Health Score / 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Verdict ──
    st.markdown('<div class="section-header">// KONDISI ENGINE</div>', unsafe_allow_html=True)
    if health_score >= 80 and n_crit == 0:
        st.markdown(f"""
        <div class="verdict-healthy">
            <div class="verdict-title" style="color:#3fb950;">✅ ENGINE SEHAT</div>
            <div class="verdict-desc">
                Semua parameter berada dalam batas operasi normal CFM56-7B.
                Health Score: <b style="color:#3fb950;">{health_score:.0f}/100</b>.
                Lanjutkan jadwal perawatan rutin sesuai CMM CFM56-7B.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif health_score >= 60 and n_crit == 0:
        st.markdown(f"""
        <div class="verdict-warning">
            <div class="verdict-title" style="color:#e3b341;">⚠️ ENGINE PERLU PERHATIAN</div>
            <div class="verdict-desc">
                Beberapa parameter mendekati batas operasi. Health Score: <b style="color:#e3b341;">{health_score:.0f}/100</b>.
                Tingkatkan frekuensi monitoring. Jadwalkan inspeksi pada next scheduled check (A-Check).
                Anomaly rate: <b>{anom_rate:.1f}%</b> dari total data.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-critical">
            <div class="verdict-title" style="color:#f85149;">🚨 ENGINE TIDAK SEHAT — PERLU INSPEKSI SEGERA</div>
            <div class="verdict-desc">
                Terdeteksi <b style="color:#f85149;">{n_crit} record KRITIS</b> dan {n_warn} record warning.
                Health Score: <b style="color:#f85149;">{health_score:.0f}/100</b>.
                Anomaly rate: <b>{anom_rate:.1f}%</b>.<br><br>
                ⛔ <b>Engine harus diinspeksi sebelum dioperasikan kembali.</b>
                Ground the aircraft jika diperlukan.
                Referensi: <i>CFM56-7B Aircraft Maintenance Manual (AMM), TASK 72-00-00-200-xxx</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Parameter Health Table ──
    st.markdown('<div class="section-header">// STATUS PARAMETER SENSOR</div>', unsafe_allow_html=True)
    for col in available_cols:
        lim    = LIMITS[col]
        sc     = df[f"{col}_status"]
        mean_v = df[col].mean()
        min_v  = df[col].min()
        max_v  = df[col].max()
        n_crit_p = (sc == "CRITICAL").sum()
        n_warn_p = (sc == "WARNING").sum()

        if n_crit_p > 0:
            pill_class = "pill-critical"
            pill_text  = f"🚨 CRITICAL ({n_crit_p} record)"
        elif n_warn_p > 0:
            pill_class = "pill-warning"
            pill_text  = f"⚠️ WARNING ({n_warn_p} record)"
        else:
            pill_class = "pill-normal"
            pill_text  = "✅ NORMAL"

        st.markdown(f"""
        <div class="param-row">
            <div style="font-size:1.4rem;">{lim['icon']}</div>
            <div class="param-name">{lim['desc']}<br>
                <span style="font-size:0.75rem;color:#8b949e;">{col}</span>
            </div>
            <div class="param-range">
                Batas: {lim['min']} – {lim['max']} {lim['unit']}<br>
                <span style="font-size:0.75rem;">Min: {min_v:.2f} · Avg: {mean_v:.2f} · Maks: {max_v:.2f}</span>
            </div>
            <div class="param-value">{mean_v:.2f} {lim['unit']}</div>
            <div><span class="status-pill {pill_class}">{pill_text}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ──
    recs = build_recommendations(df)
    if recs:
        st.markdown('<div class="section-header">// REKOMENDASI TINDAKAN</div>', unsafe_allow_html=True)
        for sev, icon, param_desc, problem, action in recs:
            card_class = "rec-critical" if sev == "CRITICAL" else "rec-warning"
            label      = "🚨 CRITICAL" if sev == "CRITICAL" else "⚠️ WARNING"
            st.markdown(f"""
            <div class="rec-card {card_class}">
                <div class="rec-title">{icon} {label} — {param_desc}</div>
                <div class="rec-body">
                    <b>Masalah:</b> {problem}<br>
                    <b class="rec-action">▶ Tindakan:</b> {action}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Tidak ada rekomendasi tindakan — semua parameter dalam batas normal.")

    # ── Charts ──
    st.markdown('<div class="section-header">// VISUALISASI DATA</div>', unsafe_allow_html=True)
    fig = make_chart(df)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    st.image(buf, use_container_width=True)
    plt.close(fig)

    # ── Raw Data Table ──
    with st.expander("📋 Lihat Data Lengkap dengan Status Anomali"):
        display_df = df[available_cols + ["Overall_Status", "Is_Anomaly"]].copy()
        def color_status(val):
            if val == "CRITICAL": return "background-color:#1f0d0d; color:#f85149"
            if val == "WARNING":  return "background-color:#14110a; color:#e3b341"
            return "background-color:#0d2119; color:#3fb950"
        styled = display_df.style.applymap(color_status, subset=["Overall_Status"])
        st.dataframe(styled, use_container_width=True)

    # ── Download ──
    st.markdown('<div class="section-header">// EXPORT</div>', unsafe_allow_html=True)
    export_df  = df.copy()
    csv_bytes  = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️  Download Hasil Analisis (CSV)",
        data      = csv_bytes,
        file_name = "cfm56_7b_analysis_result.csv",
        mime      = "text/csv",
    )

    st.markdown("""
    <div style="text-align:center; color:#30363d; font-size:0.78rem;
                font-family:'Share Tech Mono',monospace; margin-top:40px; padding-top:20px;
                border-top:1px solid #21262d; letter-spacing:1px;">
        CFM56-7B ENGINE HEALTH MONITOR · Referensi: CFM56-7B AMM / CMM · For Maintenance Use Only
    </div>
    """, unsafe_allow_html=True)
