# src/llm/analyst.py
"""
LLM Integration Layer — OpenAI GPT for market analysis and chat.

Design decisions:
- RAG pattern: real market data injected into every prompt
  → LLM explains provided data, never invents market conditions
- System prompt constrains: no data invention, no direct trade advice
- temperature=0.2: consistent factual responses, not creative hallucination
- Graceful degradation: LLM failure returns raw prediction, not a crash
- Completely isolated from API layer: swap providers without touching main.py
"""

import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file into environment variables
# WHY here: this module is the only one that needs the key
# load_dotenv() is idempotent — safe to call multiple times
load_dotenv()

logger = logging.getLogger(__name__)

# Initialise OpenAI client once at module level
# WHY once: creating a client is expensive — reuse across calls
# WHY not in function: avoids re-reading env var on every request
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt — the persona and rules for ALL LLM calls in this module
# WHY strict rules: financial context — hallucination is genuinely harmful
ANALYST_SYSTEM_PROMPT = """You are a cryptocurrency market analyst assistant.

Rules you must always follow:
1. Only use the data explicitly provided in the prompt. Never invent numbers.
2. Always acknowledge that crypto markets are highly unpredictable.
3. Never give direct buy or sell advice. Explain signals only.
4. If data is insufficient to draw a conclusion, say so clearly.
5. Keep responses to 3-5 sentences maximum.
6. End every analysis with one sentence acknowledging prediction uncertainty."""


def get_market_analysis(
    coin_id: str,
    price_data: dict,
    prediction: dict,
    features: dict
) -> str:
    """
    Generate a natural language explanation of the ML prediction.

    This is RAG in practice:
    - Retrieve: price_data, prediction, features are retrieved from
      your pipeline before this function is called
    - Augment: all retrieved data is injected into the prompt
    - Generate: GPT explains the data it was given

    The LLM never needs to "know" about crypto markets —
    it only needs to explain the numbers we provide.

    Args:
        coin_id:    e.g. "bitcoin"
        price_data: dict from fetch_current_price()
        prediction: dict from predict_direction()
        features:   dict of latest feature values from df_features.iloc[-1]

    Returns:
        Natural language analysis string
        Falls back to structured summary if LLM call fails
    """
    # Build the data context block
    # WHY format numbers explicitly: prevents LLM from misreading floats
    price = price_data.get("price", 0)
    change_24h = price_data.get("change_24h", 0) or 0
    ma_7 = features.get("ma_7", 0)
    ma_30 = features.get("ma_30", 0)
    ma_cross = features.get("ma_cross", 0)
    volatility = features.get("volatility_7d", 0)
    pct_1d = features.get("pct_1d", 0)
    price_pos = features.get("price_position", 0)
    direction = prediction.get("direction", "UNKNOWN")
    confidence = prediction.get("confidence", 0)
    p_up = prediction.get("probability_up", 0)

    # Crossover signal in plain English
    crossover_text = (
        "Bullish (MA7 above MA30)" if ma_cross == 1.0
        else "Bearish (MA7 below MA30)"
    )

    user_prompt = f"""Analyze the following {coin_id.upper()} market data and explain what the ML model's prediction means.

CURRENT MARKET DATA:
- Current price:    ${price:,.2f}
- 24h change:       {change_24h:+.2f}%
- 7-day MA:         ${ma_7:,.2f}
- 30-day MA:        ${ma_30:,.2f}
- MA crossover:     {crossover_text}
- 7d volatility:    {volatility:.2f}% (std of daily returns)
- Today's return:   {pct_1d:+.2f}%
- Price position:   {price_pos:.2f} (0=30d low, 1=30d high)

ML MODEL OUTPUT:
- Predicted direction: {direction}
- Model confidence:    {confidence:.1%}
- Probability UP:      {p_up:.1%}
- Probability DOWN:    {1-p_up:.1%}

Write a 3-5 sentence analysis explaining what these signals mean together.
End with one sentence about prediction uncertainty."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",     # cheap, fast, good enough for analysis
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,         # low = consistent factual responses
            max_tokens=400           # ~300 words maximum
        )

        analysis = response.choices[0].message.content
        logger.info(f"Analysis generated for {coin_id} ({len(analysis)} chars)")
        return analysis

    except Exception as e:
        logger.error(f"LLM analysis failed for {coin_id}: {e}")
        # Graceful degradation — return structured data as plain text
        # The API endpoint stays up even when OpenAI is down
        return (
            f"{coin_id.upper()} market analysis unavailable. "
            f"ML prediction: {direction} with {confidence:.1%} confidence "
            f"(P_up={p_up:.1%}). "
            f"Current price: ${price:,.2f} ({change_24h:+.2f}% 24h)."
        )


def answer_question(
    question: str,
    coin_id: str,
    market_context: Optional[dict]
) -> str:
    """
    Answer a user's natural language question about the market.

    WHY market_context injected here:
    The LLM has no live market data. Without context injection,
    it would invent current prices. We always provide real data
    so answers are grounded in actual market conditions.

    Args:
        question:       User's question string
        coin_id:        Coin being asked about
        market_context: dict from fetch_current_price() or None

    Returns:
        Answer string, or fallback message if LLM fails
    """
    # Build context string from market data
    if market_context:
        price      = market_context.get("price", 0)
        change_24h = market_context.get("change_24h", 0) or 0
        context_text = (
            f"Current {coin_id.upper()} data: "
            f"price=${price:,.2f}, "
            f"24h change={change_24h:+.2f}%"
        )
    else:
        context_text = f"No current {coin_id.upper()} market data available."

    user_prompt = f"""Market context: {context_text}

User question: {question}

Answer the question using the market context provided.
If the question cannot be answered with available data, say so clearly.
Do not recommend specific buy or sell actions."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.3,     # slightly higher for conversational responses
            max_tokens=500
        )

        answer = response.choices[0].message.content
        logger.info(f"Question answered for {coin_id} ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        return (
            "I'm unable to answer right now due to a technical issue. "
            "Please try again in a moment."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# Usage: python -m src.llm.analyst
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))

    from src.ingestion.fetcher import fetch_current_price, fetch_price_history
    from src.processing.cleaner import raw_history_to_dataframe
    from src.features.engineer import build_features
    from src.ml.trainer import load_pipeline
    from src.ml.predictor import predict_direction

    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing analyst.py ===\n")

    # Check API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_key_here":
        print("ERROR: OPENAI_API_KEY not set in .env file")
        print("Get your key from platform.openai.com")
        sys.exit(1)
    print(f"OpenAI key loaded: sk-...{api_key[-6:]}\n")

    # Get price data
    print("Fetching market data...")
    price_data = fetch_current_price("ethereum")
    if not price_data:
        print("Could not fetch price data")
        sys.exit(1)
    print(f"Price: ${price_data['price']:,.2f}")

    # Build features
    print("Building features...")
    raw      = fetch_price_history("ethereum", days=180)
    df_clean = raw_history_to_dataframe(raw, "ethereum")
    df_feat  = build_features(df_clean)

    # Load model and predict
    print("Loading model and predicting...")
    try:
        pipeline   = load_pipeline()
        prediction = predict_direction(df_feat, pipeline)
        print(f"Prediction: {prediction['direction']} "
              f"({prediction['confidence']:.1%} confidence)")
    except FileNotFoundError:
        print("No model.pkl found. Using dummy prediction for LLM test.")
        prediction = {
            "direction": "UP",
            "confidence": 0.6,
            "probability_up": 0.6,
            "probability_down": 0.4
        }

    # Get latest features as dict
    latest_features = df_feat.iloc[-1].to_dict()

    # Test 1: Market analysis
    print("\nTest 1: Generating market analysis...")
    print("-" * 50)
    analysis = get_market_analysis(
        coin_id="ethereum",
        price_data=price_data,
        prediction=prediction,
        features=latest_features
    )
    print(analysis)

    # Test 2: Question answering
    print("\nTest 2: Answering a user question...")
    print("-" * 50)
    answer = answer_question(
        question="Is ethereum showing any strong trend signals right now?",
        coin_id="ethereum",
        market_context=price_data
    )
    print(answer)

    print("\n=== All tests complete ===\n")