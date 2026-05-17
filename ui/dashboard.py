# ui/dashboard.py
"""
Streamlit Dashboard — the visible face of the AI Crypto Trading Assistant.

Architecture:
- Pure UI layer — displays data, never runs business logic
- Calls FastAPI backend via HTTP requests
- Uses st.cache_data to avoid redundant API calls on reruns
- Uses st.session_state for chat history persistence

Run with:
    streamlit run ui/dashboard.py
(Keep FastAPI running separately: uvicorn src.api.main:app --port 8000)
"""

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "https://crypto-trading-assistant.onrender.com"

st.set_page_config(
    page_title="AI Crypto Trading Assistant",
    page_icon="📈",
    layout="wide",          # use full browser width
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# API helper functions
# ─────────────────────────────────────────────────────────────────────────────

def call_api(endpoint: str, method: str = "GET", body: dict = None) -> dict | None:
    """
    Call the FastAPI backend.
    Returns parsed JSON or None on failure.
    Displays error in the UI — never crashes the dashboard.
    """
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=body, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API server. "
            "Make sure FastAPI is running: "
            "`uvicorn src.api.main:app --port 8000`"
        )
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(ttl=30)      # refresh price every 30 seconds
def get_price(coin_id: str) -> dict | None:
    return call_api(f"/price/{coin_id}")


@st.cache_data(ttl=300)     # refresh prediction every 5 minutes
def get_prediction(coin_id: str) -> dict | None:
    return call_api(f"/prediction/{coin_id}")


@st.cache_data(ttl=600)     # refresh analysis every 10 minutes
def get_analysis(coin_id: str) -> dict | None:
    return call_api(f"/analysis/{coin_id}")


@st.cache_data(ttl=3600)    # refresh history once per hour
def get_price_history(coin_id: str, days: int = 90) -> list | None:
    """
    Fetch price history directly from CoinGecko via the API.
    WHY: we need historical data for the chart — the /price endpoint
    only returns current price. We call CoinGecko through our fetcher.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        from src.ingestion.fetcher import fetch_price_history
        return fetch_price_history(coin_id, days=days)
    except Exception as e:
        st.warning(f"Could not load price history: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    coin = st.selectbox(
        "Select Coin",
        options=["bitcoin", "ethereum", "solana"],
        format_func=lambda x: {
            "bitcoin":  "₿ Bitcoin (BTC)",
            "ethereum": "Ξ Ethereum (ETH)",
            "solana":   "◎ Solana (SOL)"
        }[x]
    )

    chart_days = st.slider(
        "Chart History (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )

    st.divider()

    # API health indicator
    health = call_api("/health")
    if health:
        model_status = "✅ Loaded" if health.get("model_loaded") else "⚠️ Not loaded"
        st.success(f"API: Online")
        st.info(f"ML Model: {model_status}")
    else:
        st.error("API: Offline")

    st.divider()
    st.caption("AI Crypto Trading Assistant v1.0")
    st.caption("Predictions are for educational purposes only.")


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 AI Crypto Trading Assistant")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab layout — keeps the dashboard organised
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🤖 AI Prediction", "💬 Chat"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Market Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"{coin.capitalize()} Market Overview")

    # Fetch current price
    price_data = get_price(coin)

    if price_data:
        # Metric cards — three columns
        col1, col2, col3 = st.columns(3)

        price      = price_data.get("price", 0)
        change_24h = price_data.get("change_24h", 0) or 0
        market_cap = price_data.get("market_cap", 0) or 0
        volume     = price_data.get("volume_24h", 0) or 0

        col1.metric(
            label="Current Price",
            value=f"${price:,.2f}",
            delta=f"{change_24h:.4f}%"
        )
        col2.metric(
            label="Market Cap",
            value=f"${market_cap/1e9:.2f}B"
        )
        col3.metric(
            label="24h Volume",
            value=f"${volume/1e9:.2f}B"
        )
    else:
        st.warning("Could not load price data.")

    st.divider()

    # Price chart
    st.subheader(f"Price Chart — Last {chart_days} Days")

    with st.spinner("Loading price history..."):
        history = get_price_history(coin, days=chart_days)

    if history:
        # Convert to DataFrame for plotting
        df_chart = pd.DataFrame(history, columns=["timestamp_ms", "price"])
        df_chart["date"] = pd.to_datetime(df_chart["timestamp_ms"], unit="ms")
        df_chart = df_chart.set_index("date")

        # Plot with matplotlib
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0E1117")  # dark background
        ax.set_facecolor("#0E1117")

        # Price line
        ax.plot(
            df_chart.index,
            df_chart["price"],
            color="#00D4FF",
            linewidth=1.5,
            label="Price"
        )

        # Moving averages
        df_chart["ma_7"]  = df_chart["price"].rolling(7).mean()
        df_chart["ma_30"] = df_chart["price"].rolling(30).mean()

        ax.plot(df_chart.index, df_chart["ma_7"],
                color="#FFD700", linewidth=1, linestyle="--",
                label="7-day MA", alpha=0.8)
        ax.plot(df_chart.index, df_chart["ma_30"],
                color="#FF6B6B", linewidth=1, linestyle="--",
                label="30-day MA", alpha=0.8)

        # Formatting
        ax.set_ylabel("Price (USD)", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)
        ax.legend(facecolor="#1E2130", labelcolor="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["top"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["right"].set_color("#333")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )

        st.pyplot(fig)
        plt.close()
    else:
        st.info("Price history not available.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: AI Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header(f"AI Prediction — {coin.capitalize()}")

    col_pred, col_analysis = st.columns([1, 2])

    with col_pred:
        st.subheader("ML Model Prediction")

        if st.button("🔮 Get Prediction", type="primary"):
            # Clear cache to force a fresh prediction
            get_prediction.clear()

        with st.spinner("Running ML pipeline..."):
            pred_data = get_prediction(coin)

        if pred_data:
            direction  = pred_data.get("direction", "UNKNOWN")
            confidence = pred_data.get("confidence", 0)
            p_up       = pred_data.get("probability_up", 0)
            p_down     = pred_data.get("probability_down", 0)

            # Direction indicator
            if direction == "UP":
                st.success(f"## ↑ {direction}")
            else:
                st.error(f"## ↓ {direction}")

            # Confidence meter
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)

            # Probability breakdown
            st.write("**Probability Breakdown:**")
            prob_df = pd.DataFrame({
                "Direction": ["↑ UP", "↓ DOWN"],
                "Probability": [p_up, p_down]
            })

            fig2, ax2 = plt.subplots(figsize=(4, 2))
            fig2.patch.set_facecolor("#0E1117")
            ax2.set_facecolor("#0E1117")
            colors = ["#00C853", "#FF3D00"]
            bars = ax2.barh(
                prob_df["Direction"],
                prob_df["Probability"],
                color=colors
            )
            ax2.set_xlim(0, 1)
            ax2.tick_params(colors="white")
            ax2.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x:.0%}")
            )
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_color("#333")
            ax2.spines["left"].set_color("#333")
            st.pyplot(fig2)
            plt.close()

        else:
            st.warning("Prediction unavailable.")

    with col_analysis:
        st.subheader("AI Market Analysis")

        if st.button("📝 Get Analysis", type="primary"):
            get_analysis.clear()

        with st.spinner("Generating AI analysis (this takes ~10 seconds)..."):
            analysis_data = get_analysis(coin)

        if analysis_data:
            direction  = analysis_data.get("direction", "")
            confidence = analysis_data.get("confidence", 0)
            analysis   = analysis_data.get("analysis", "")

            # Analysis text in a styled box
            st.info(analysis)

            # Summary metrics
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Predicted Direction", direction)
            mcol2.metric("Model Confidence", f"{confidence:.1%}")

        else:
            st.warning("Analysis unavailable.")

    # Disclaimer
    st.divider()
    st.caption(
        "⚠️ This prediction is generated by a machine learning model "
        "for educational purposes only. "
        "Cryptocurrency markets are highly volatile and unpredictable. "
        "This is not financial advice."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("💬 Ask the AI Analyst")
    st.caption(
        "Ask any question about the market. "
        "The AI uses current price data to ground its answers."
    )

    # Initialise chat history in session_state
    # WHY session_state: persists across Streamlit reruns
    # Without it, the chat history resets on every interaction
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input(
        f"Ask about {coin}... (e.g. 'Is {coin} showing bullish signals?')"
    )

    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_question)

        # Get AI answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_api(
                    "/chat",
                    method="POST",
                    body={"question": user_question, "coin": coin}
                )

            if result:
                answer = result.get("answer", "No answer received.")
                st.write(answer)
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error("Could not get an answer. Is the API running?")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()