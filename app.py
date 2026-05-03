import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# -----------------------------
# REPORTLAB (PDF EXPORT)
# -----------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

st.set_page_config(page_title="Quant Research Terminal", layout="wide")

# -----------------------------
# UNIVERSE
# -----------------------------
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO",
    "JPM", "BAC", "V", "MA",
    "UNH", "JNJ", "PFE", "LLY",
    "AMZN", "COST", "HD", "MCD",
    "XOM", "CVX",
    "CAT", "BA"
]

# -----------------------------
# DATA
# -----------------------------
def load_data(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all")
    returns = prices.pct_change().dropna()

    return prices, returns

# -----------------------------
# METRICS (SAFE)
# -----------------------------
def metrics(r):
    r = pd.Series(np.ravel(r)).dropna().astype(float)

    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1

    return ann_ret, ann_vol, sharpe, float(dd.min()), equity

# -----------------------------
# STRATEGIES
# -----------------------------
def momentum_weights(prices, selected, lookback=90):
    recent = prices[selected].dropna().iloc[-lookback:]
    mom = (recent.iloc[-1] / recent.iloc[0]) - 1

    w = np.exp(mom)
    return (w / w.sum()).values, mom

def risk_parity_weights(returns, selected):
    vol = returns[selected].std()
    w = 1 / vol
    return (w / w.sum()).values

# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(mode, selected, weights, comparison_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Quant Research Investment Memo", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Strategy Mode: {mode}", styles["Heading2"]))
    story.append(Spacer(1, 12))

    # Weights
    story.append(Paragraph("Portfolio Weights", styles["Heading2"]))

    w_data = [["Asset", "Weight"]]
    for t, w in zip(selected, weights):
        w_data.append([t, f"{w:.4f}"])

    w_table = Table(w_data)
    w_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))

    story.append(w_table)
    story.append(Spacer(1, 12))

    # Comparison table
    story.append(Paragraph("Strategy Comparison", styles["Heading2"]))

    comp_data = [["Metric"] + list(comparison_df.columns)]
    for idx in comparison_df.index:
        comp_data.append([idx] + list(comparison_df.loc[idx].values))

    comp_table = Table(comp_data)
    comp_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))

    story.append(comp_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------------
# UI
# -----------------------------
st.title("📊 Quant Research Terminal")

st.sidebar.header("Controls")

selected = st.sidebar.multiselect(
    "Universe",
    UNIVERSE,
    default=["AAPL", "MSFT", "NVDA", "JPM", "AMZN"]
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

mode = st.sidebar.radio(
    "Strategy Mode",
    ["Manual", "Momentum", "Risk Parity"]
)

if len(selected) < 2:
    st.stop()

prices, returns = load_data(selected, start)

# -----------------------------
# STATE
# -----------------------------
if "weights" not in st.session_state:
    st.session_state.weights = np.ones(len(selected)) / len(selected)

weights = st.session_state.weights.copy()

# -----------------------------
# STRATEGY LOGIC
# -----------------------------
if mode == "Momentum":
    weights, _ = momentum_weights(prices, selected)

elif mode == "Risk Parity":
    weights = risk_parity_weights(returns, selected)

if mode == "Manual":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Weights")

    manual = []

    for i, t in enumerate(selected):
        manual.append(
            st.sidebar.slider(t, 0.0, 1.0, float(st.session_state.weights[i]), key=f"w_{t}")
        )

    manual = np.array(manual)
    weights = manual / manual.sum() if manual.sum() > 0 else np.ones(len(manual)) / len(manual)

st.session_state.weights = weights

# -----------------------------
# PORTFOLIO
# -----------------------------
port_ret = returns @ weights

spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)["Close"]
spy = spy.pct_change().dropna().squeeze()

port_ret, spy = port_ret.align(spy, join="inner")

# -----------------------------
# METRICS
# -----------------------------
p_r, p_v, p_s, p_dd, p_eq = metrics(port_ret)
s_r, s_v, s_s, s_dd, s_eq = metrics(spy)

# -----------------------------
# DASHBOARD
# -----------------------------
st.subheader("Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Return", f"{p_r:.2%}")
c2.metric("Volatility", f"{p_v:.2%}")
c3.metric("Sharpe", f"{p_s:.2f}")
c4.metric("Max DD", f"{p_dd:.2%}")

# -----------------------------
# EQUITY CURVE
# -----------------------------
st.subheader("Equity Curve")

st.line_chart(pd.DataFrame({
    "Portfolio": p_eq,
    "SPY": s_eq
}).dropna())

# -----------------------------
# STRATEGY COMPARISON
# -----------------------------
st.subheader("Strategy Comparison Engine")

base = np.ones(len(selected)) / len(selected)
manual_ret = returns @ base

mom_w, _ = momentum_weights(prices, selected)
mom_ret = returns @ mom_w

rp_w = risk_parity_weights(returns, selected)
rp_ret = returns @ rp_w

comparison = pd.DataFrame({
    "Manual": metrics(manual_ret)[:4],
    "Momentum": metrics(mom_ret)[:4],
    "Risk Parity": metrics(rp_ret)[:4],
    "SPY": metrics(spy)[:4]
}, index=["Return", "Volatility", "Sharpe", "Max Drawdown"]).T

st.dataframe(comparison.style.format({
    "Return": "{:.2%}",
    "Volatility": "{:.2%}",
    "Sharpe": "{:.2f}",
    "Max Drawdown": "{:.2%}"
}))

# -----------------------------
# 📄 PDF EXPORT
# -----------------------------
st.subheader("📄 Investment Memo Export")

if st.button("Generate PDF Research Report"):

    pdf = generate_pdf(
        mode,
        selected,
        weights,
        comparison
    )

    st.download_button(
        label="Download Investment Memo",
        data=pdf,
        file_name="quant_research_memo.pdf",
        mime="application/pdf"
    )

# -----------------------------
# DEBUG
# -----------------------------
st.subheader("Portfolio Breakdown")

st.write("Tickers:", selected)
st.write("Weights:", weights)
st.write("Correlation Matrix")
st.write(returns.corr())