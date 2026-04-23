# ============================================================
# app.py - Streamlit UI
# LSTM + FinBERT + RoBERTa Stock Predictor
# Features: OHLCV + RSI + MACD + BB_position (8 features)
# Predicts: Open + Close (dual output)
# ============================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = torch.device("cpu")

from calendar_dates import (
    get_next_trading_days, get_last_trading_day,
    assign_to_trading_day
)
from data_pipeline import (
    prepare_data, N_FEATURES, N_OUTPUTS, SEQ_LEN,
    OPEN_IDX, CLOSE_IDX, inverse_transform_col
)
from lstm_model import load_or_train, forecast_ohlcv
from sentiment2 import (
    load_nlp, finbert_sentiment, roberta_sentiment,
    fuse_sentiment, compute_total_bias, apply_sentiment_fusion
)
from evaluation import (
    evaluate_predictions, baseline_comparison,
    mcnemar_significance, ablation_summary
)
from sp500 import load_sp500

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fused Market Predictor",
    layout="centered"
)
st.title("Fused Market Predictor")
st.caption(
    "LSTM + FinBERT + RoBERTa  |  "
    "Features: OHLCV + RSI + MACD + Bollinger  |  "
    "Predicts: Open & Close"
)

# ============================================================
# SIDEBAR
# ============================================================
@st.cache_data
def load_sp500_cached():
    return load_sp500()

df_sp500        = load_sp500_cached()

if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = None

company = st.sidebar.selectbox(
    "S&P 500 Company", df_sp500["Security"]
)
symbol = df_sp500[
    df_sp500["Security"] == company
].iloc[0]["Symbol"]

if st.session_state.last_symbol != symbol:
    st.session_state.last_symbol = symbol
    st.session_state.news_fetched = False

st.sidebar.success(f"{company} ({symbol})")

days_to_predict = st.sidebar.number_input(
    label="Trading days to predict",
    min_value=5, max_value=90,
    value=30, step=1, format="%d"
)

model_path = f"models/{symbol}_lstm_ohlcv_indicators_v4.h5"

# ============================================================
# LOAD & PREPARE DATA
# ============================================================
@st.cache_resource
def get_data(symbol):
    return prepare_data(symbol)

with st.spinner(f"Loading data for {symbol}..."):
    ctx = get_data(symbol)

data     = ctx["df"]
if data.empty:
    st.error("No price data available.")
    st.stop()

st.sidebar.info(
    f"Data: {data.index[0].date()} to {data.index[-1].date()}\n"
    f"({len(data)} trading days)\n"
    f"Features: {N_FEATURES} (OHLCV+RSI+MACD+BB)"
)

# ============================================================
# LOAD / TRAIN MODEL
# ============================================================
@st.cache_resource
def get_model(_X_train, _y_train, _X_val, _y_val, path):
    return load_or_train(_X_train, _y_train, _X_val, _y_val, path)

with st.spinner("Loading model..."):
    model, was_loaded = get_model(
        ctx["X_train"], ctx["y_train"],
        ctx["X_val"], ctx["y_val"],
        model_path
    )

status = "Loaded saved model" if was_loaded else "Model trained"
st.success(f"✓ {status} | Input: {N_FEATURES} features | "
           f"Output: {N_OUTPUTS} (Open + Close)")

DECAY_RATE = 10.0
ARTICLE_SCALE = 0.10
TOTAL_SCALE = 0.20
st.caption(
    f"Fusion params: decay={DECAY_RATE}, "
    f"article_scale={ARTICLE_SCALE}, total_scale={TOTAL_SCALE}"
)

# ============================================================
# MODEL EVALUATION
# ============================================================
st.header("📊 Model Evaluation on Test Set")

y_pred_both = model.predict(ctx["X_test"], verbose=0)

# Evaluate Open prediction
m_open, df_open, yt_open, yp_open = evaluate_predictions(
    ctx["y_test"][:, 0], y_pred_both[:, 0],
    ctx["scaler"], OPEN_IDX, "Open"
)

# Evaluate Close prediction
m_close, df_close, yt_close, yp_close = evaluate_predictions(
    ctx["y_test"][:, 1], y_pred_both[:, 1],
    ctx["scaler"], CLOSE_IDX, "Close"
)

# Display metrics side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Open Price Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE",      f"{m_open['RMSE']:.6f}")
    c2.metric("MAE",       f"{m_open['MAE']:.6f}")
    c3.metric("Direction", f"{m_open['Directional Accuracy']:.2%}")
    c4.metric("F1",        f"{m_open['F1 Score']:.4f}")
    if m_open["Directional Accuracy"] > 0.60:
        st.success(f"✓ {m_open['Directional Accuracy']:.2%} — beats 60% target")
    elif m_open["Directional Accuracy"] > 0.52:
        st.warning(f"⚠ {m_open['Directional Accuracy']:.2%} — beats random")
    else:
        st.error(f"✗ {m_open['Directional Accuracy']:.2%} — at or below random")

with col2:
    st.subheader("Close Price Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE",      f"{m_close['RMSE']:.6f}")
    c2.metric("MAE",       f"{m_close['MAE']:.6f}")
    c3.metric("Direction", f"{m_close['Directional Accuracy']:.2%}")
    c4.metric("F1",        f"{m_close['F1 Score']:.4f}")
    if m_close["Directional Accuracy"] > 0.60:
        st.success(f"✓ {m_close['Directional Accuracy']:.2%} — beats 60% target")
    elif m_close["Directional Accuracy"] > 0.52:
        st.warning(f"⚠ {m_close['Directional Accuracy']:.2%} — beats random")
    else:
        st.error(f"✗ {m_close['Directional Accuracy']:.2%} — at or below random")

# Full metrics tables
with st.expander("Full Metrics Detail"):
    st.dataframe(df_open,  use_container_width=True)
    st.dataframe(df_close, use_container_width=True)

# ── Baseline Comparison ───────────────────────────────────────
st.subheader("📊 Baseline Comparison")
st.caption(
    "Honest comparison against naive baselines. "
    "A good model must beat both."
)

comp_open,  imp_open  = baseline_comparison(
    yt_open,  yp_open,  "Open"
)
comp_close, imp_close = baseline_comparison(
    yt_close, yp_close, "Close"
)

col1, col2 = st.columns(2)
with col1:
    st.write("**Open**")
    st.dataframe(comp_open,  use_container_width=True)
    delta = imp_open["mse_improvement_pct"]
    if delta > 10:
        st.success(f"✓ {imp_open['verdict']} (+{delta:.1f}% MSE)")
    elif delta > 0:
        st.warning(f"⚠ {imp_open['verdict']} (+{delta:.1f}% MSE)")
    else:
        st.error(f"✗ {imp_open['verdict']}")

with col2:
    st.write("**Close**")
    st.dataframe(comp_close, use_container_width=True)
    delta = imp_close["mse_improvement_pct"]
    if delta > 10:
        st.success(f"✓ {imp_close['verdict']} (+{delta:.1f}% MSE)")
    elif delta > 0:
        st.warning(f"⚠ {imp_close['verdict']} (+{delta:.1f}% MSE)")
    else:
        st.error(f"✗ {imp_close['verdict']}")

# ── Statistical Significance ──────────────────────────────────
st.subheader("📐 Statistical Significance (McNemar Test)")
st.caption(
    "Tests whether improvement over baseline is statistically "
    "significant or could be due to chance. "
    "p < 0.05 = significant for thesis."
)

sig_open  = mcnemar_significance(yt_open,  yp_open,  label="Open")
sig_close = mcnemar_significance(yt_close, yp_close, label="Close")

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Open** χ²={sig_open['chi2']}  "
             f"p={sig_open['p_value']}")
    if sig_open["significant"]:
        st.success(f"✓ {sig_open['conclusion']}")
    else:
        st.warning(f"⚠ {sig_open['conclusion']}")

with col2:
    st.write(f"**Close** χ²={sig_close['chi2']}  "
             f"p={sig_close['p_value']}")
    if sig_close["significant"]:
        st.success(f"✓ {sig_close['conclusion']}")
    else:
        st.warning(f"⚠ {sig_close['conclusion']}")

# ── Ablation Study ────────────────────────────────────────────
st.subheader("🔬 Ablation Study")
st.caption(
    "Shows contribution of each component. "
    "This table belongs directly in your thesis results chapter."
)
st.info(
    "Run the model with different feature sets to fill this table. "
    "Current run uses full feature set (OHLCV + RSI + MACD + BB)."
)

ablation_data = {
    "LSTM — OHLC only (baseline)":
        {"direction": 0.4959, "mae": 0.796519},
    "LSTM — OHLCV (+ Volume)":
        {"direction": 0.00,   "mae": 0.00},
    "LSTM — OHLCV + RSI":
        {"direction": 0.00,   "mae": 0.00},
    "LSTM — OHLCV + RSI + MACD":
        {"direction": 0.00,   "mae": 0.00},
    "LSTM — OHLCV + RSI + MACD + BB (full)":
        {"direction": m_close["Directional Accuracy"],
         "mae":       m_close["MAE"]},
}
st.dataframe(
    ablation_summary(ablation_data),
    use_container_width=True
)
st.caption(
    "Fill in the empty rows by running the model "
    "with each configuration separately."
)

# ── Prediction Visualization ──────────────────────────────────
st.subheader("📈 Prediction Quality — Test Set")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Open: actual vs predicted
axes[0, 0].plot(yt_open[:100],  label="Actual",    alpha=0.8)
axes[0, 0].plot(yp_open[:100],  label="Predicted", alpha=0.8)
axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.3)
axes[0, 0].set_title("Open Log Returns — Actual vs Predicted")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Close: actual vs predicted
axes[0, 1].plot(yt_close[:100], label="Actual",    alpha=0.8)
axes[0, 1].plot(yp_close[:100], label="Predicted", alpha=0.8)
axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
axes[0, 1].set_title("Close Log Returns — Actual vs Predicted")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Cumulative Open
axes[1, 0].plot(np.cumsum(yt_open),  label="Actual Cumulative",    lw=2)
axes[1, 0].plot(np.cumsum(yp_open),  label="Predicted Cumulative", lw=2)
axes[1, 0].set_title("Cumulative Open Returns")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cumulative Close
axes[1, 1].plot(np.cumsum(yt_close), label="Actual Cumulative",    lw=2)
axes[1, 1].plot(np.cumsum(yp_close), label="Predicted Cumulative", lw=2)
axes[1, 1].set_title("Cumulative Close Returns")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(
    f"{symbol} — LSTM+Indicators Prediction Quality",
    fontsize=13
)
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# FORECAST
# ============================================================
st.header("🔮 Future Price Forecast")

last_real_day = get_last_trading_day(data.index[-1])
future_dates  = get_next_trading_days(last_real_day, days_to_predict)

weekend_check = future_dates[future_dates.dayofweek >= 5]
if len(weekend_check) > 0:
    st.error(f"Weekend dates in forecast: {weekend_check}")
else:
    st.success(
        f"✓ All {days_to_predict} dates are valid trading days"
    )

forecast = forecast_ohlcv(
    model,
    ctx["scaled"][-SEQ_LEN:],
    days_to_predict,
    ctx["scaler"],
    ctx["raw_ohlcv"]
)

# Forecast table
st.subheader("LSTM Forecast (No Sentiment)")
forecast_df = pd.DataFrame({
    "Date":            future_dates.strftime("%Y-%m-%d (%A)"),
    "Predicted Open":  forecast["open_prices"],
    "Predicted Close": forecast["close_prices"],
    "Open Return":     [round(r, 6) for r in forecast["open_returns"]],
    "Close Return":    [round(r, 6) for r in forecast["close_returns"]],
    "Daily Range $":   [round(abs(o - c), 4)
                        for o, c in zip(forecast["open_prices"],
                                        forecast["close_prices"])]
})
st.dataframe(forecast_df, use_container_width=True)

# Forecast chart
fig_fc, ax = plt.subplots(figsize=(13, 5))
ax.plot(
    data.index[-90:], data["Close"].values[-90:],
    label="Historical Close", color="steelblue", lw=2
)
ax.plot(
    future_dates, forecast["open_prices"],
    label="Predicted Open", color="orange",
    marker="^", lw=2, linestyle="--"
)
ax.plot(
    future_dates, forecast["close_prices"],
    label="Predicted Close", color="green",
    marker="o", lw=2
)
ax.fill_between(
    future_dates,
    forecast["open_prices"],
    forecast["close_prices"],
    alpha=0.15, color="gray", label="Open-Close range"
)
ax.axvline(
    x=last_real_day, color="red",
    linestyle="--", alpha=0.5, label="Forecast start"
)
ax.set_title(
    f"{symbol} — LSTM Forecast: Open & Close "
    f"({days_to_predict} trading days)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_fc)

# ============================================================
# NLP MODELS
# ============================================================
@st.cache_resource
def get_nlp():
    return load_nlp()

fin_tok, fin_mod, roberta = get_nlp()

# ============================================================
# NEWS + SENTIMENT FUSION
# ============================================================
st.header(f"📰 News Sentiment for {symbol}")

if "news_fetched" not in st.session_state:
    st.session_state.news_fetched = False

if st.button("Fetch News & Apply Sentiment", type="primary"):
    st.session_state.news_fetched = True

if st.session_state.news_fetched:
    try:
        # Use Alpha Vantage instead of Finlight
        import requests
        api_key = os.environ.get("ALPHA_VANTAGE_KEY", "")
        
        url = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "apikey": api_key, "limit": 10}
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        articles = data.get("feed", [])[:5]  # Limit to 5 articles

        if not articles:
            st.warning("No news found.")
            st.session_state.news_fetched = False
            st.stop()

        # ── STEP 1: collect per-article data ──
        biases     = []
        pub_times = []
        sentiments = []

        for art in articles:
            title = art.get("title", "No title")
            summary = art.get("summary", "")
            st.markdown(f"### {title}")
            
            time_str = art.get("time_published", "")
            from datetime import datetime
            if time_str:
                try:
                    pub_date = datetime.strptime(time_str, "%Y%m%d%H%M%S")
                except:
                    pub_date = datetime.now()
            else:
                pub_date = datetime.now()
            
            st.caption(f"Time published: {pub_date}")
            
            text = f"{title} {summary}"
            fin_l, fin_s = finbert_sentiment(text, fin_tok, fin_mod)
            rob_l, rob_s = roberta_sentiment(text, roberta)
            bias = fuse_sentiment(fin_l, fin_s, rob_l, rob_s)
            
            biases.append(bias)
            pub_times.append(pub_date)
            
            sentiments.append({
                "Title": title[:60] + "...",
                "FinBERT": f"{fin_l} ({fin_s:.2f})",
                "RoBERTa": f"{rob_l} ({rob_s:.2f})",
                "Bias": f"{bias:+.4f}",
            })

            col1, col2 = st.columns(2)
            with col1:
                if fin_l == "Bullish":
                    st.success(f"FinBERT Bullish ({fin_s:.2f})")
                elif fin_l == "Bearish":
                    st.error(f"FinBERT Bearish ({fin_s:.2f})")
                else:
                    st.info(f"FinBERT Neutral ({fin_s:.2f})")
            with col2:
                rob_up = rob_l.upper()
                if "POS" in rob_up:
                    st.success(f"RoBERTa {rob_l} ({rob_s:.2f})")
                elif "NEG" in rob_up:
                    st.error(f"RoBERTa {rob_l} ({rob_s:.2f})")
                else:
                    st.info(f"RoBERTa {rob_l} ({rob_s:.2f})")

            if summary:
                st.write(summary)
            
            link = art.get("url", "")
            if link:
                st.markdown(f"[Read article]({link})")
            
            st.divider()

        # ── STEP 2: aggregate with recency weighting ──
        total_bias = compute_total_bias(
            article_biases=biases,
            publish_times=pub_times,
        )

        signal = (
            "Bullish" if total_bias > 0.02 else
            "Bearish" if total_bias < -0.02 else
            "Neutral"
        )

        st.subheader("Sentiment Summary")
        st.dataframe(pd.DataFrame(sentiments), use_container_width=True)
        st.info(
            f"Signal: **{signal}** | "
            f"Bias: {total_bias:+.4f} | "
            f"Articles: {len(articles)}"
        )

        # ── STEP 3: compute recent volatility ──
        recent_log_returns = np.diff(np.log(data["Close"].values[-25:]))

        # ── STEP 4: apply improved fusion ──
        fused = apply_sentiment_fusion(
            total_bias=total_bias,
            open_returns=forecast["open_returns"],
            close_returns=forecast["close_returns"],
            last_open=float(ctx["raw_ohlcv"][-1, OPEN_IDX]),
            last_close=float(ctx["raw_ohlcv"][-1, CLOSE_IDX]),
            days_to_predict=days_to_predict,
            recent_returns=recent_log_returns,
            forecast_decay=10.0,
        )

        fused_open = fused["fused_open"]
        fused_close = fused["fused_close"]

        # Show impact curve
        with st.expander("Sentiment impact curve (per-day bias applied)"):
            impact_df = pd.DataFrame({
                "Day": range(1, days_to_predict + 1),
                "Sentiment Bias Applied": fused["impact_curve"],
            })
            st.line_chart(impact_df.set_index("Day"))

        # Fused forecast chart
        st.subheader(
            "Fused Forecast — LSTM + FinBERT + RoBERTa"
        )
        fig_f, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Open comparison
        axes[0].plot(
            data.index[-60:], data["Open"].values[-60:],
            label="Historical", color="steelblue", lw=2
        )
        axes[0].plot(
            future_dates, forecast["open_prices"],
            label="LSTM Only", color="orange",
            marker="^", lw=2, linestyle="--", alpha=0.7
        )
        axes[0].plot(
            future_dates, fused_open,
            label="LSTM + Sentiment", color="red",
            marker="^", lw=2
        )
        axes[0].axvline(
            x=last_real_day, color="gray",
            linestyle="--", alpha=0.5
        )
        axes[0].set_title(f"{symbol} Open Forecast")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        plt.setp(
            axes[0].xaxis.get_majorticklabels(), rotation=45
        )

        # Close comparison
        axes[1].plot(
            data.index[-60:], data["Close"].values[-60:],
            label="Historical", color="steelblue", lw=2
        )
        axes[1].plot(
            future_dates, forecast["close_prices"],
            label="LSTM Only", color="green",
            marker="o", lw=2, linestyle="--", alpha=0.7
        )
        axes[1].plot(
            future_dates, fused_close,
            label="LSTM + Sentiment", color="darkgreen",
            marker="o", lw=2
        )
        axes[1].axvline(
            x=last_real_day, color="gray",
            linestyle="--", alpha=0.5
        )
        axes[1].set_title(f"{symbol} Close Forecast")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.setp(
            axes[1].xaxis.get_majorticklabels(), rotation=45
        )

        plt.suptitle(
            f"{symbol} — LSTM vs Sentiment Fusion "
            f"({days_to_predict} trading days)",
            fontsize=13
        )
        plt.tight_layout()
        st.pyplot(fig_f)

        # Comparison table
        st.dataframe(pd.DataFrame({
            "Date":         future_dates.strftime(
                                "%Y-%m-%d (%A)"
                            ),
            "LSTM Open":    forecast["open_prices"],
            "Fused Open":   fused_open,
            "Open Diff $":  [round(f - b, 4) for b, f in
                             zip(forecast["open_prices"],
                                 fused_open)],
            "LSTM Close":   forecast["close_prices"],
            "Fused Close":  fused_close,
            "Close Diff $": [round(f - b, 4) for b, f in
                             zip(forecast["close_prices"],
                                 fused_close)],
        }), use_container_width=True)

    except Exception as e:
        st.error("Failed to fetch or process news")
        st.code(str(e))

# ============================================================
# EXPORT REPORT
# ============================================================
st.sidebar.divider()
if st.sidebar.button("Download Evaluation Report"):
    report = f"""
STOCK PREDICTION EVALUATION REPORT
{symbol} - {company}
Features: OHLCV + RSI + MACD + BB_position (8 total)
Predicts: Open + Close (dual output)

OPEN PREDICTION
  RMSE:      {m_open['RMSE']:.8f}
  MAE:       {m_open['MAE']:.8f}
  R2:        {m_open['R2']:.4f}
  Direction: {m_open['Directional Accuracy']:.2%}
  F1:        {m_open['F1 Score']:.4f}

CLOSE PREDICTION
  RMSE:      {m_close['RMSE']:.8f}
  MAE:       {m_close['MAE']:.8f}
  R2:        {m_close['R2']:.4f}
  Direction: {m_close['Directional Accuracy']:.2%}
  F1:        {m_close['F1 Score']:.4f}

BASELINE COMPARISON (Close)
  {imp_close['verdict']}
  MSE improvement vs naive: {imp_close['mse_improvement_pct']:+.1f}%
  Direction improvement:    {imp_close['direction_improvement']:+.1f}%

STATISTICAL SIGNIFICANCE
  Open:  p={sig_open['p_value']} - {sig_open['conclusion']}
  Close: p={sig_close['p_value']} - {sig_close['conclusion']}

DATASET
  Training samples: {len(ctx['X_train'])}
  Test samples:     {len(ctx['X_test'])}
  Sequence length:  {SEQ_LEN} trading days
  Features:         {N_FEATURES}

DISCLAIMER: Research purposes only. Not financial advice.
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    st.sidebar.download_button(
        label="Download Report",
        data=report,
        file_name=f"{symbol}_thesis_evaluation.txt",
        mime="text/plain"
    )