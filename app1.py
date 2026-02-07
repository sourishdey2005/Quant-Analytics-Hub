import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import *

# --------------------------------------------------
# PAGE CONFIG (WHITE THEME)
# --------------------------------------------------
st.set_page_config(
    page_title="NIFTY-50 Decision Intelligence",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: white; color: black; }
</style>
""", unsafe_allow_html=True)

st.title("📘 NIFTY-50 Multi-Factor Stock Peer Analysis")
st.caption("Quant-style decision intelligence | Market • Peer • Single Stock")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_all():
    df = load_data("data/nifty50_final_features.csv")
    df = engineer_features(df)
    df = add_peer_metrics(df)
    return df

df = load_all()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🔍 Controls")

symbols = sorted(df["Symbol"].unique())
selected = st.sidebar.multiselect("Select Stocks", symbols)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["Date"].min(), df["Date"].max()]
)

df = df[
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# --------------------------------------------------
# MODE
# --------------------------------------------------
if len(selected) == 0:
    mode = "MARKET"
elif len(selected) == 1:
    mode = "SINGLE"
else:
    mode = "PEER"

st.subheader(f"📊 MODE: {mode}")

# ==================================================
# MODULE 1 — PRICE & TREND
# ==================================================
st.header("📈 Price & Trend Intelligence")

if mode == "SINGLE":
    sdf = df[df["Symbol"] == selected[0]]

    st.plotly_chart(
        px.line(sdf, x="Date", y="Close", title="Close Price"),
        use_container_width=True
    )

    fig = go.Figure(go.Candlestick(
        x=sdf["Date"],
        open=sdf["Open"],
        high=sdf["High"],
        low=sdf["Low"],
        close=sdf["Close"]
    ))
    fig.update_layout(title="OHLC Candlestick")
    st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(
        px.line(sdf, x="Date", y="NormPrice",
                title="Normalized Price (Base 100)"),
        use_container_width=True
    )

    # Moving averages (SAFE)
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=sdf["Date"], y=sdf["Close"], name="Close"
    ))
    fig_ma.add_trace(go.Scatter(
        x=sdf["Date"], y=sdf["MA20"], name="MA20", line=dict(dash="dot")
    ))
    fig_ma.add_trace(go.Scatter(
        x=sdf["Date"], y=sdf["MA50"], name="MA50", line=dict(dash="dash")
    ))
    fig_ma.update_layout(title="Price with Moving Averages")
    st.plotly_chart(fig_ma, use_container_width=True)

# ==================================================
# MODULE 2 — PERFORMANCE
# ==================================================
st.header("📊 Performance Analysis")

if mode != "MARKET":
    perf = df[df["Symbol"].isin(selected)]

    st.plotly_chart(
        px.line(perf, x="Date", y="CumReturn", color="Symbol",
                title="Cumulative Returns"),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(perf, x="Return", color="Symbol",
                     title="Return Distribution"),
        use_container_width=True
    )

# ==================================================
# MODULE 3 — RISK
# ==================================================
st.header("⚠️ Risk & Volatility")

st.plotly_chart(
    px.scatter(df, x="Volatility", y="Return", color="Symbol",
               title="Volatility vs Return"),
    use_container_width=True
)

# ==================================================
# MODULE 4 — VWAP
# ==================================================
st.header("🏦 VWAP & Institutional Flow")

if mode == "SINGLE":
    st.plotly_chart(
        px.line(sdf, x="Date", y=["Close", "VWAP"],
                title="Price vs VWAP"),
        use_container_width=True
    )

# ==================================================
# MODULE 5 — PEER COMPARISON
# ==================================================
st.header("🆚 Peer Comparison")

latest = df.groupby("Symbol").last().reset_index()
latest["Score"] = latest.appl
