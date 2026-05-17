# CORRECT — both imported
from pydantic import BaseModel, Field
from typing import Optional


class PriceResponse(BaseModel):
    coin_id: str
    price: float
    change_24h: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    currency: str = "usd"


class PredictionResponse(BaseModel):
    coin_id:          str
    direction:        str    # "UP" or "DOWN"
    confidence:       float  # 0.0 to 1.0
    probability_up:   float
    probability_down: float


class AnalysisResponse(BaseModel):
    coin_id:    str
    direction:  str
    confidence: float
    analysis:   str


class ChatRequest(BaseModel):
    question: str = Field(
        ...,                    # ... means required — no default
        min_length=3,
        max_length=500,
        description="Your question about the crypto market"
    )
    coin: str = Field(
        default="bitcoin",
        description="Which coin to use as market context"
    )


class ChatResponse(BaseModel):
    """Response shape for POST /chat"""
    answer: str
    coin:   str


class HealthResponse(BaseModel):
    """Response shape for GET /health"""
    status:       str   # "ok" always
    model_loaded: bool

