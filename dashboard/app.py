import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL THEME
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --border:    #1e2d45;
    --accent1:   #38bdf8;
    --accent2:   #f472b6;
    --accent3:   #a78bfa;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: var(--bg) !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--accent1); }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--accent1) !important; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
h1 { font-size: 2.6rem !important; font-weight: 800 !important; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; color: var(--accent1) !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.pill {
    display: inline-block; padding: 2px 12px; border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; margin-left: 8px; vertical-align: middle;
}
.pill-good     { background:#166534; color:#4ade80; }
.pill-sat      { background:#14532d; color:#86efac; }
.pill-moderate { background:#713f12; color:#facc15; }
.pill-poor     { background:#7c2d12; color:#fb923c; }
.pill-vpoor    { background:#7f1d1d; color:#f87171; }
.pill-severe   { background:#450a0a; color:#fca5a5; }

.hero {
    background: linear-gradient(135deg, #0f1f3d 0%, #0b0f1a 60%, #1a0b2e 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #fff; line-height: 1.1; margin: 0; }
.hero-sub   { color: var(--muted); font-size: 0.85rem; margin-top: 0.4rem; }
.hero-tag   { font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent1); margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#94a3b8"),
    title_font=dict(family="Syne, sans-serif", color="#e2e8f0", size=15),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    colorway=["#38bdf8","#f472b6","#a78bfa","#4ade80","#facc15","#fb923c","#f87171"],
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45"),
    margin=dict(t=50, b=40, l=40, r=20),
)

def dark(fig):
    fig.update_layout(**DARK_LAYOUT)
    return fig

AQI_COLORS = {
    "Good":         "#4ade80",
    "Satisfactory": "#86efac",
    "Moderate":     "#facc15",
    "Poor":         "#fb923c",
    "Very Poor":    "#f87171",
    "Severe":       "#fca5a5",
}
BUCKET_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

# Search for city_day.csv starting from app.py's directory, then walking up
def _find_csv(filename):
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):  # look up to 4 levels up
        candidate = os.path.join(current, filename)
        if os.path.exists(candidate):
            return candidate
        current = os.path.dirname(current)
    raise FileNotFoundError(
        f"Could not find '{filename}' in or above: {os.path.dirname(os.path.abspath(__file__))}\n"
        f"Please place city_day.csv in the same folder as app.py."
    )

@st.cache_data
def load_data():
    path = _find_csv("city_day.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

df = load_data()

POLLUTANTS = [c for c in ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"] if c in df.columns]
MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌫️ AQI Intelligence")
    st.markdown("<hr style='border-color:#1e2d45'>", unsafe_allow_html=True)

    cities = sorted(df["City"].dropna().unique())
    default_city = "Delhi" if "Delhi" in cities else cities[0]
    selected_city = st.selectbox("📍 City", cities, index=cities.index(default_city))

    years = sorted(df["Year"].dropna().unique().astype(int))
    selected_years = st.multiselect("📅 Year(s)", years, default=years)

    pollutant = st.selectbox("🧪 Pollutant Focus", POLLUTANTS, index=0)

    st.markdown("<hr style='border-color:#1e2d45'>", unsafe_allow_html=True)
    show_raw = st.toggle("Show raw data table", value=False)
    st.markdown("<p style='color:#475569;font-size:0.7rem;margin-top:1rem;'>Dataset: 26 cities · 2015–2020<br>Source: India CPCB via Kaggle</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTERED DATA
# ─────────────────────────────────────────────

city_df = df[(df["City"] == selected_city) & (df["Year"].isin(selected_years))].copy()

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────

avg_aqi  = city_df["AQI"].mean()
avg_pm25 = city_df["PM2.5"].mean() if "PM2.5" in city_df.columns else float("nan")

def aqi_label(v):
    if pd.isna(v): return "Unknown", "moderate"
    if v <= 50:    return "Good", "good"
    if v <= 100:   return "Satisfactory", "sat"
    if v <= 200:   return "Moderate", "moderate"
    if v <= 300:   return "Poor", "poor"
    if v <= 400:   return "Very Poor", "vpoor"
    return "Severe", "severe"

label, cls = aqi_label(avg_aqi)

st.markdown(f"""
<div class="hero">
  <div class="hero-tag">Air Quality Intelligence Dashboard</div>
  <div class="hero-title">{selected_city}</div>
  <div class="hero-sub">
    Average AQI: <strong style="color:#e2e8f0">{"—" if pd.isna(avg_aqi) else f"{avg_aqi:.0f}"}</strong>
    <span class="pill pill-{cls}">{label}</span>
    &nbsp;·&nbsp; {len(city_df):,} daily records across {len(selected_years)} year(s)
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg AQI",       f"{avg_aqi:.0f}"                                        if not pd.isna(avg_aqi) else "—")
k2.metric("Avg PM2.5",     f"{avg_pm25:.1f} µg/m³"                                 if not pd.isna(avg_pm25) else "—")
k3.metric("Avg PM10",      f"{city_df['PM10'].mean():.1f} µg/m³"                   if "PM10" in city_df and city_df['PM10'].notna().any() else "—")
k4.metric("Peak AQI",      f"{city_df['AQI'].max():.0f}"                           if city_df['AQI'].notna().any() else "—")
k5.metric("Good Air Days", f"{(city_df['AQI_Bucket']=='Good').sum()}")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 1 — AQI TREND (full width) + DONUT BELOW
# ─────────────────────────────────────────────

st.markdown("## AQI Trend & Category Breakdown")

trend = city_df[["Date","AQI"]].dropna().sort_values("Date")
trend["AQI_30d"] = trend["AQI"].rolling(30, min_periods=1).mean()

fig_trend = dark(go.Figure())
fig_trend.add_trace(go.Scatter(
    x=trend["Date"], y=trend["AQI"],
    mode="lines", name="Daily AQI",
    line=dict(color="#1e3a5f", width=1),
    fill="tozeroy", fillcolor="rgba(56,189,248,0.06)"
))
fig_trend.add_trace(go.Scatter(
    x=trend["Date"], y=trend["AQI_30d"],
    mode="lines", name="30-day avg",
    line=dict(color="#38bdf8", width=2.5)
))
for lvl, yval, color in [("Good",50,"#4ade80"),("Satisfactory",100,"#86efac"),("Moderate",200,"#facc15")]:
    fig_trend.add_hline(y=yval, line_dash="dot", line_color=color, line_width=1,
                        annotation=dict(text=lvl, font_color=color, font_size=10))
fig_trend.update_layout(
    title=f"Daily AQI — {selected_city}", xaxis_title=None, yaxis_title="AQI",
    height=380, showlegend=True, legend=dict(orientation="h", y=1.08, x=0)
)
st.plotly_chart(fig_trend, use_container_width=True)

bucket_counts = city_df["AQI_Bucket"].value_counts()
bucket_counts = bucket_counts.reindex([b for b in BUCKET_ORDER if b in bucket_counts.index])
mode_bucket = city_df["AQI_Bucket"].mode()
center_text = mode_bucket.iloc[0] if not mode_bucket.empty else "—"

fig_donut = dark(go.Figure(go.Pie(
    labels=bucket_counts.index, values=bucket_counts.values,
    hole=0.55,
    marker_colors=[AQI_COLORS.get(b,"#94a3b8") for b in bucket_counts.index],
    textinfo="label+percent", textfont_size=12,
    hovertemplate="%{label}: %{value} days<extra></extra>"
)))
fig_donut.update_layout(title="AQI Categories", height=340,
    showlegend=False,
    annotations=[dict(text=f"<b>{center_text}</b>", x=0.5, y=0.5,
                      font_size=15, font_color="#e2e8f0",
                      font_family="Syne, sans-serif", showarrow=False)]
)
st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 2 — SEASONAL HEATMAP + BOX
# ─────────────────────────────────────────────

st.markdown("## Seasonal Patterns")
col_heat, col_box = st.columns([2, 1])

with col_heat:
    pivot = city_df.pivot_table(values="AQI", index="Year", columns="Month", aggfunc="mean")
    pivot.columns = [MONTH_LABELS[m] for m in pivot.columns]

    fig_heat = dark(go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=pivot.index.astype(str),
        colorscale=[[0.0,"#166534"],[0.2,"#facc15"],[0.5,"#fb923c"],[0.8,"#f87171"],[1.0,"#7f1d1d"]],
        text=np.round(pivot.values, 0), texttemplate="%{text}",
        hoverongaps=False, showscale=True,
        colorbar=dict(tickfont=dict(size=10), len=0.8)
    )))
    fig_heat.update_layout(title="Monthly Avg AQI Heatmap (Year × Month)",
        height=300, xaxis_title=None, yaxis_title=None
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_box:
    city_df["MonthLabel"] = city_df["Month"].map(MONTH_LABELS)
    fig_box = px.box(
        city_df.dropna(subset=["AQI"]),
        x="MonthLabel", y="AQI",
        category_orders={"MonthLabel": list(MONTH_LABELS.values())},
        color_discrete_sequence=["#38bdf8"], title="AQI by Month",
    )
    fig_box = dark(fig_box)
    fig_box.update_layout(height=300, xaxis_title=None)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 3 — POLLUTANT DEEP DIVE
# ─────────────────────────────────────────────

st.markdown(f"## Pollutant Focus — {pollutant}")

col_hist, col_violin = st.columns(2)
with col_hist:
    fig_hist = px.histogram(
        city_df.dropna(subset=[pollutant]),
        x=pollutant, nbins=60,
        color_discrete_sequence=["#a78bfa"], title=f"{pollutant} Distribution",
        marginal="rug", opacity=0.85
    )
    fig_hist = dark(fig_hist)
    fig_hist.update_layout(height=320, bargap=0.04)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_violin:
    fig_viol = px.violin(
        city_df.dropna(subset=[pollutant, "AQI_Bucket"]),
        x="AQI_Bucket", y=pollutant,
        color="AQI_Bucket", color_discrete_map=AQI_COLORS,
        category_orders={"AQI_Bucket": BUCKET_ORDER},
        box=True, points=False, title=f"{pollutant} by AQI Category",
    )
    fig_viol = dark(fig_viol)
    fig_viol.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_viol, use_container_width=True)

# Pollutant time series with rolling average
poll_series = city_df[["Date", pollutant]].dropna().sort_values("Date")
poll_series["roll7"] = poll_series[pollutant].rolling(7, min_periods=1).mean()
fig_poll_ts = dark(go.Figure())
fig_poll_ts.add_trace(go.Scatter(
    x=poll_series["Date"], y=poll_series[pollutant],
    mode="lines", name="Daily", line=dict(color="#312e81", width=1),
    fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"
))
fig_poll_ts.add_trace(go.Scatter(
    x=poll_series["Date"], y=poll_series["roll7"],
    mode="lines", name="7-day avg", line=dict(color="#a78bfa", width=2.5)
))
fig_poll_ts.update_layout(
    title=f"{pollutant} Time Series with 7-day Rolling Average",
    height=300, xaxis_title=None, legend=dict(orientation="h", y=1.08)
)
st.plotly_chart(fig_poll_ts, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 4 — CORRELATION + RADAR
# ─────────────────────────────────────────────

st.markdown("## Pollutant Correlations")
col_corr, col_radar = st.columns([2, 1])

CORR_COLS = [c for c in ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","AQI"] if c in city_df.columns]

with col_corr:
    corr = city_df[CORR_COLS].corr()
    fig_corr = dark(go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0.0,"#7f1d1d"],[0.5,"#0b0f1a"],[1.0,"#1e3a5f"]],
        zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True, hoverongaps=False,
    )))
    fig_corr.update_layout(title="Correlation Matrix", height=380,
        xaxis=dict(tickfont_size=10), yaxis=dict(tickfont_size=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col_radar:
    radar_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3","NH3"] if c in city_df.columns]
    means = city_df[radar_cols].mean()
    means_norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
    r_vals = list(means_norm) + [means_norm.iloc[0]]
    theta_vals = list(means_norm.index) + [means_norm.index[0]]

    fig_radar = dark(go.Figure(go.Scatterpolar(
        r=r_vals, theta=theta_vals,
        fill="toself", fillcolor="rgba(244,114,182,0.18)",
        line=dict(color="#f472b6", width=2), name="Normalised mean"
    )))
    fig_radar.update_layout(title="Pollutant Fingerprint",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color="#1e2d45", gridcolor="#1e2d45"),
            angularaxis=dict(color="#64748b", gridcolor="#1e2d45"),
        ),
        height=380, showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 5 — CITY COMPARISON
# ─────────────────────────────────────────────

st.markdown("## City Comparison")

city_summary = (
    df[df["Year"].isin(selected_years)]
    .groupby("City")
    .agg(Avg_AQI=("AQI","mean"), Avg_PM25=("PM2.5","mean"),
         Max_AQI=("AQI","max"), Good_Days=("AQI_Bucket", lambda x: (x=="Good").sum()))
    .reset_index()
    .sort_values("Avg_AQI", ascending=True)
)

col_bar, col_scatter = st.columns(2)

with col_bar:
    bar_colors = ["#38bdf8" if c == selected_city else "#1e3a5f" for c in city_summary["City"]]
    fig_bar = dark(go.Figure(go.Bar(
        x=city_summary["Avg_AQI"], y=city_summary["City"],
        orientation="h", marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>Avg AQI: %{x:.0f}<extra></extra>"
    )))
    fig_bar.update_layout(title="Average AQI — All Cities",
        height=440, xaxis_title="Avg AQI", yaxis_title=None,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_scatter:
    fig_sc = px.scatter(
        city_summary, x="Avg_AQI", y="Avg_PM25",
        size="Max_AQI", color="Good_Days", text="City",
        color_continuous_scale=["#f87171","#facc15","#4ade80"],
        title="AQI vs PM2.5 (bubble = peak AQI, colour = good-air days)",
        size_max=35,
    )
    fig_sc = dark(fig_sc)
    fig_sc.update_traces(textposition="top center", textfont_size=9)
    fig_sc.update_layout(height=440, coloraxis_colorbar=dict(len=0.6, thickness=12))
    st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 6 — YEAR-OVER-YEAR + MISSING VALUES
# ─────────────────────────────────────────────

st.markdown("## Year-over-Year & Data Quality")
col_yoy, col_miss = st.columns(2)

with col_yoy:
    yoy = city_df.groupby("Year")["AQI"].mean().reset_index()
    fig_yoy = dark(go.Figure(go.Bar(
        x=yoy["Year"], y=yoy["AQI"],
        marker=dict(
            color=yoy["AQI"],
            colorscale=[[0,"#4ade80"],[0.4,"#facc15"],[0.7,"#fb923c"],[1,"#f87171"]],
            line=dict(width=0)
        ),
        hovertemplate="Year %{x}: AQI %{y:.0f}<extra></extra>"
    )))
    fig_yoy.update_layout(title=f"Yearly Average AQI — {selected_city}",
        height=320, xaxis=dict(tickmode="array", tickvals=yoy["Year"]), yaxis_title="Avg AQI",
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

with col_miss:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig_miss = dark(go.Figure(go.Bar(
        x=missing.index, y=missing.values,
        marker=dict(color=missing.values, colorscale=[[0,"#1e3a5f"],[1,"#f87171"]]),
        hovertemplate="%{x}: %{y} missing<extra></extra>"
    )))
    fig_miss.update_layout(title="Missing Values by Column (full dataset)",
        height=320, xaxis_title=None, yaxis_title="Count"
    )
    st.plotly_chart(fig_miss, use_container_width=True)


st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 7 — AQI CALENDAR HEATMAP
# ─────────────────────────────────────────────

st.markdown("## AQI Calendar Heatmap")

cal_year = st.selectbox("Select Year for Calendar", sorted(city_df["Year"].dropna().unique().astype(int), reverse=True), key="cal_year")
cal_df = city_df[city_df["Year"] == cal_year][["Date","AQI"]].dropna().copy()
cal_df["Week"]    = cal_df["Date"].dt.isocalendar().week.astype(int)
cal_df["Weekday"] = cal_df["Date"].dt.weekday  # 0=Mon, 6=Sun
cal_df["Label"]   = cal_df["Date"].dt.strftime("%b %d") + "<br>AQI: " + cal_df["AQI"].round(0).astype(int).astype(str)

fig_cal = dark(go.Figure(go.Heatmap(
    x=cal_df["Week"],
    y=cal_df["Weekday"],
    z=cal_df["AQI"],
    text=cal_df["Label"],
    hovertemplate="%{text}<extra></extra>",
    colorscale=[[0,"#166534"],[0.25,"#facc15"],[0.55,"#fb923c"],[0.75,"#f87171"],[1,"#7f1d1d"]],
    showscale=True,
    colorbar=dict(title="AQI", len=0.8, thickness=12),
    xgap=3, ygap=3,
)))
fig_cal.update_layout(
    title=f"Daily AQI Calendar — {selected_city} {cal_year}",
    height=280,
    xaxis=dict(title="Week of Year", tickmode="linear", dtick=4),
    yaxis=dict(
        title=None,
        tickmode="array",
        tickvals=list(range(7)),
        ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        autorange="reversed",
    ),
)
st.plotly_chart(fig_cal, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 8 — TOP 10 WORST & BEST DAYS
# ─────────────────────────────────────────────

st.markdown("## Extreme Days")
col_worst, col_best = st.columns(2)

with col_worst:
    worst = city_df[["Date","AQI","AQI_Bucket"]].dropna(subset=["AQI"]).nlargest(10, "AQI").copy()
    worst["DateStr"] = worst["Date"].dt.strftime("%d %b %Y")
    fig_worst = dark(go.Figure(go.Bar(
        x=worst["AQI"],
        y=worst["DateStr"],
        orientation="h",
        marker=dict(color=worst["AQI"], colorscale=[[0,"#fb923c"],[1,"#7f1d1d"]], line=dict(width=0)),
        text=worst["AQI_Bucket"],
        textposition="inside",
        insidetextfont=dict(color="#fff", size=10),
        hovertemplate="<b>%{y}</b><br>AQI: %{x}<extra></extra>",
    )))
    fig_worst.update_layout(
        title="🔴 10 Worst Air Quality Days",
        height=340, xaxis_title="AQI",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_worst, use_container_width=True)

with col_best:
    best = city_df[["Date","AQI","AQI_Bucket"]].dropna(subset=["AQI"]).nsmallest(10, "AQI").copy()
    best["DateStr"] = best["Date"].dt.strftime("%d %b %Y")
    fig_best = dark(go.Figure(go.Bar(
        x=best["AQI"],
        y=best["DateStr"],
        orientation="h",
        marker=dict(color=best["AQI"], colorscale=[[0,"#166534"],[1,"#86efac"]], line=dict(width=0)),
        text=best["AQI_Bucket"],
        textposition="inside",
        insidetextfont=dict(color="#000", size=10),
        hovertemplate="<b>%{y}</b><br>AQI: %{x}<extra></extra>",
    )))
    fig_best.update_layout(
        title="🟢 10 Best Air Quality Days",
        height=340, xaxis_title="AQI",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_best, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 9 — MULTI-POLLUTANT AREA STACK + SUNBURST
# ─────────────────────────────────────────────

st.markdown("## Multi-Pollutant Breakdown")
col_area, col_sun = st.columns([2, 1])

with col_area:
    area_cols = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in city_df.columns]
    area_df = city_df[["Date"] + area_cols].dropna().sort_values("Date")
    # Resample to weekly mean for clarity
    area_df = area_df.set_index("Date").resample("W").mean().reset_index()
    AREA_COLORS = ["#38bdf8","#f472b6","#a78bfa","#4ade80","#facc15","#fb923c"]
    fig_area = dark(go.Figure())
    for col, clr in zip(area_cols, AREA_COLORS):
        fig_area.add_trace(go.Scatter(
            x=area_df["Date"], y=area_df[col],
            mode="lines", name=col,
            line=dict(color=clr, width=1.5),
            stackgroup="one",
            fillcolor=clr.replace("#", "rgba(") + ",0.6)" if False else clr,
        ))
    fig_area.update_layout(
        title="Weekly Pollutant Levels (Stacked Area)",
        height=360, xaxis_title=None, yaxis_title="µg/m³",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_area, use_container_width=True)

with col_sun:
    # Sunburst: Year → AQI Bucket → count
    sun_df = city_df.dropna(subset=["AQI_Bucket","Year"]).copy()
    sun_df["Year"] = sun_df["Year"].astype(str)
    sun_agg = sun_df.groupby(["Year","AQI_Bucket"]).size().reset_index(name="Days")
    fig_sun = px.sunburst(
        sun_agg, path=["Year","AQI_Bucket"], values="Days",
        color="AQI_Bucket",
        color_discrete_map=AQI_COLORS,
        title="Year → Category Breakdown",
    )
    fig_sun = dark(fig_sun)
    fig_sun.update_layout(height=360)
    fig_sun.update_traces(textfont_size=11)
    st.plotly_chart(fig_sun, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 10 — ROLLING STATS BAND CHART
# ─────────────────────────────────────────────

st.markdown("## AQI Rolling Statistics")

roll_df = city_df[["Date","AQI"]].dropna().sort_values("Date").set_index("Date")
roll_df["mean_30"]  = roll_df["AQI"].rolling(30).mean()
roll_df["upper_30"] = roll_df["AQI"].rolling(30).mean() + roll_df["AQI"].rolling(30).std()
roll_df["lower_30"] = roll_df["AQI"].rolling(30).mean() - roll_df["AQI"].rolling(30).std()
roll_df = roll_df.dropna().reset_index()

fig_band = dark(go.Figure())
fig_band.add_trace(go.Scatter(
    x=roll_df["Date"], y=roll_df["upper_30"],
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig_band.add_trace(go.Scatter(
    x=roll_df["Date"], y=roll_df["lower_30"],
    fill="tonexty", fillcolor="rgba(56,189,248,0.12)",
    line=dict(width=0), name="±1 std band", hoverinfo="skip",
))
fig_band.add_trace(go.Scatter(
    x=roll_df["Date"], y=roll_df["mean_30"],
    line=dict(color="#38bdf8", width=2), name="30-day mean",
))
fig_band.add_trace(go.Scatter(
    x=roll_df["Date"], y=roll_df["AQI"],
    mode="markers", marker=dict(size=2, color="#64748b", opacity=0.4),
    name="Daily AQI",
))
fig_band.update_layout(
    title=f"30-Day Rolling Mean ± Std Band — {selected_city}",
    height=340, xaxis_title=None, yaxis_title="AQI",
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig_band, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 11 — POLLUTANT RANKING TABLE
# ─────────────────────────────────────────────

st.markdown("## Pollutant Summary Statistics")

poll_stats = city_df[POLLUTANTS].agg(["mean","median","std","min","max"]).T.round(2)
poll_stats.columns = ["Mean","Median","Std Dev","Min","Max"]
poll_stats.index.name = "Pollutant"
poll_stats = poll_stats.reset_index()

fig_table = dark(go.Figure(go.Table(
    columnwidth=[80, 80, 80, 80, 80, 80],
    header=dict(
        values=["<b>Pollutant</b>","<b>Mean</b>","<b>Median</b>","<b>Std Dev</b>","<b>Min</b>","<b>Max</b>"],
        fill_color="#1e2d45",
        font=dict(color="#38bdf8", size=12, family="Syne, sans-serif"),
        align="center", height=36,
        line_color="#0b0f1a",
    ),
    cells=dict(
        values=[poll_stats[c] for c in poll_stats.columns],
        fill_color=[["#111827","#0f172a"] * (len(poll_stats)//2 + 1)],
        font=dict(color="#e2e8f0", size=11, family="DM Mono, monospace"),
        align="center", height=30,
        line_color="#1e2d45",
    )
)))
fig_table.update_layout(title="Pollutant Statistics — Selected City & Years", height=420, margin=dict(t=50,b=10,l=10,r=10))
st.plotly_chart(fig_table, use_container_width=True)

# ─────────────────────────────────────────────
# OPTIONAL RAW TABLE
# ─────────────────────────────────────────────

if show_raw:
    st.markdown("## Raw Data")
    st.dataframe(
        city_df.sort_values("Date", ascending=False).reset_index(drop=True),
        use_container_width=True, height=380,
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:#334155;font-size:0.72rem;letter-spacing:0.1em;'>
    AQI INTELLIGENCE &nbsp;·&nbsp; INDIA CPCB DATA &nbsp;·&nbsp; 2015–2020 &nbsp;·&nbsp; BUILT WITH STREAMLIT & PLOTLY
</div>
""", unsafe_allow_html=True)
