# src/api/main.py
"""
FastAPI Backend — exposes the AI pipeline as HTTP endpoints.

Four endpoints:
  GET  /price/{coin_id}       → live price from CoinGecko
  GET  /prediction/{coin_id}  → ML direction prediction
  GET  /analysis/{coin_id}    → prediction + LLM explanation
  POST /chat                  → natural language Q&A

Design decisions:
- Lifespan pattern: ML model loaded ONCE at startup, not per request
- API layer has ZERO business logic — only orchestrates pipeline calls
- HTTPException with correct status codes (503 for dependencies, 500 for bugs)
- CORS enabled for Streamlit dashboard on different port
- response_model on every route — validates output shape automatically
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.ingestion.fetcher import (
    fetch_current_price,
    fetch_price_history,
    SUPPORTED_COINS
)
from src.processing.cleaner import raw_history_to_dataframe
from src.features.engineer import build_features
from src.ml.predictor import predict_direction
from src.ml.trainer import load_pipeline
from src.llm.analyst import get_market_analysis, answer_question
from src.api.schemas import (
    PriceResponse,
    PredictionResponse,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline variable — loaded once at startup
# WHY global: the lifespan function sets it, route functions read it
# This avoids loading model.pkl on every single request (~200ms saved)
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup code before the server accepts requests.
    Runs shutdown code after the server stops.

    WHY lifespan pattern:
    Load expensive resources (ML model) once at startup.
    Every request then uses the already-loaded pipeline.
    Without this, model.pkl would be read from disk on every /prediction call.
    """
    global pipeline

    logger.info("Server starting — loading ML pipeline...")
    try:
        pipeline = load_pipeline()
        logger.info("ML pipeline loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "No model.pkl found. "
            "Run 'python -m src.ml.trainer' first. "
            "/prediction and /analysis endpoints will return 503."
        )

    yield  # server runs here, handling requests

    logger.info("Server shutting down")


# Create FastAPI app with lifespan and metadata
# Metadata appears in /docs UI
app = FastAPI(
    title="AI Crypto Trading Assistant",
    description=(
        "Live crypto prices, ML direction predictions, "
        "and LLM market analysis. "
        f"Supported coins: {', '.join(SUPPORTED_COINS)}"
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware — allows Streamlit dashboard (port 8501) to call
# this API (port 8000). Without this, browsers block cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # development: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: GET /price/{coin_id}
# Simplest endpoint — fetch and return current price
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "AI Crypto Trading Assistant API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/price/{coin}", "/prediction/{coin}", "/analysis/{coin}", "/chat"]
    }


@app.get(
    "/price/{coin_id}",
    response_model=PriceResponse,
    summary="Get current price for a coin",
    tags=["Market Data"]
)
async def get_price(coin_id: str):
    """
    Fetch current price, 24h change, market cap, and volume.

    - **coin_id**: CoinGecko coin ID (bitcoin, ethereum, solana)
    """
    # Validate coin is supported
    # WHY: prevents unnecessary API calls for unsupported coins
    if coin_id not in SUPPORTED_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported coin: {coin_id}. "
                   f"Supported: {SUPPORTED_COINS}"
        )

    data = fetch_current_price(coin_id)

    if data is None:
        # 503 = dependency (CoinGecko) is down, not our code's fault
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch price for {coin_id}. "
                   f"CoinGecko API may be unavailable."
        )

    return PriceResponse(**data)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: GET /prediction/{coin_id}
# Full ML pipeline: fetch → clean → engineer → predict
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/prediction/{coin_id}",
    response_model=PredictionResponse,
    summary="Get ML price direction prediction",
    tags=["Predictions"]
)
async def get_prediction(coin_id: str):
    """
    Run the full ML pipeline and return price direction prediction.

    Pipeline: CoinGecko (180 days) → clean → engineer features → Random Forest

    - **coin_id**: CoinGecko coin ID (bitcoin, ethereum, solana)
    """
    if coin_id not in SUPPORTED_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported coin: {coin_id}"
        )

    # Check model is loaded
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run 'python -m src.ml.trainer' first."
        )

    # Run the pipeline — each step can fail independently
    raw = fetch_price_history(coin_id, days=180)
    if raw is None:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch price history for {coin_id}"
        )

    df_clean = raw_history_to_dataframe(raw, coin_id)
    if df_clean is None:
        raise HTTPException(
            status_code=422,
            detail="Data cleaning failed — insufficient or malformed data"
        )

    df_features = build_features(df_clean)
    if df_features is None:
        raise HTTPException(
            status_code=422,
            detail="Feature engineering failed — not enough data rows"
        )

    result = predict_direction(df_features, pipeline)
    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed unexpectedly"
        )

    return PredictionResponse(coin_id=coin_id, **result)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: GET /analysis/{coin_id}
# Full pipeline + LLM explanation
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/analysis/{coin_id}",
    response_model=AnalysisResponse,
    summary="Get ML prediction with LLM explanation",
    tags=["Predictions"]
)
async def get_analysis(coin_id: str):
    """
    Run the full ML pipeline AND generate a natural language explanation.

    Uses RAG: real market data and ML prediction are injected into
    the GPT prompt. The LLM explains the data — never invents it.

    - **coin_id**: CoinGecko coin ID (bitcoin, ethereum, solana)
    """
    if coin_id not in SUPPORTED_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported coin: {coin_id}"
        )

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run trainer first."
        )

    # Fetch current price (for LLM context)
    price_data = fetch_current_price(coin_id)
    if price_data is None:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch price for {coin_id}"
        )

    # Run prediction pipeline
    raw         = fetch_price_history(coin_id, days=180)
    df_clean    = raw_history_to_dataframe(raw, coin_id)
    df_features = build_features(df_clean)

    if df_features is None:
        raise HTTPException(
            status_code=422,
            detail="Feature engineering failed"
        )

    prediction = predict_direction(df_features, pipeline)
    if prediction is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    # Get latest feature values for LLM context
    latest_features = df_features.iloc[-1].to_dict()

    # Generate LLM analysis
    # Note: get_market_analysis has its own try/except and returns
    # a fallback string on failure — this endpoint never 500s due to LLM
    analysis = get_market_analysis(
        coin_id=coin_id,
        price_data=price_data,
        prediction=prediction,
        features=latest_features
    )

    return AnalysisResponse(
        coin_id=coin_id,
        direction=prediction["direction"],
        confidence=prediction["confidence"],
        analysis=analysis
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: POST /chat
# Natural language Q&A with market context
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about the crypto market",
    tags=["Chat"]
)
async def chat(request: ChatRequest):
    """
    Answer a natural language question about crypto markets.

    Fetches current market data as context and passes it to GPT.
    The LLM answers using only the provided data — not invented knowledge.

    Request body:
    - **question**: Your question (3-500 characters)
    - **coin**: Which coin to use as context (default: bitcoin)
    """
    # Fetch market context for grounding the LLM
    # If this fails, answer_question handles None context gracefully
    market_context = fetch_current_price(request.coin)

    answer = answer_question(
        question=request.question,
        coin_id=request.coin,
        market_context=market_context
    )

    return ChatResponse(answer=answer, coin=request.coin)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    tags=["System"]
)
async def health():
    """
    Returns server status and whether the ML model is loaded.
    Use this to verify the server is running before testing other endpoints.
    """
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None
    )


