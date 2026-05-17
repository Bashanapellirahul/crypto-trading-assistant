# src/ml/predictor.py
"""
ML Prediction Layer.

Loads the saved Pipeline and generates predictions for new data.
Called by the FastAPI endpoints on every request.

Design decisions:
- Never re-trains — loads model.pkl from disk (fast, milliseconds)
- Uses only the most recent row for live prediction
- Returns probability scores not just binary labels
- iloc[[-1]] not iloc[-1] → returns DataFrame not Series (sklearn needs 2D)
"""

import logging
import pandas as pd
from typing import Optional
from sklearn.pipeline import Pipeline

from src.features.engineer import FEATURE_COLS

logger = logging.getLogger(__name__)


def predict_direction(
    df_features: pd.DataFrame,
    pipeline: Pipeline
) -> Optional[dict]:
    """
    Generate a prediction for the most recent row of features.

    Uses only the last row because this is a LIVE prediction —
    we want to predict tomorrow based on what we know today.

    Args:
        df_features: Full features DataFrame from build_features()
        pipeline:    Fitted Pipeline from load_pipeline()

    Returns:
        dict with direction, confidence, and both probabilities
        or None if prediction fails

    Example return:
        {
            "direction":        "UP",
            "confidence":       0.67,
            "probability_up":   0.67,
            "probability_down": 0.33
        }
    """
    if df_features is None or df_features.empty:
        logger.error("Cannot predict: empty features DataFrame")
        return None

    try:
        # Take only the most recent row
        # WHY iloc[[-1]] with double brackets:
        # iloc[-1]  returns a Series  — shape (14,)   — sklearn REJECTS this
        # iloc[[-1]] returns DataFrame — shape (1, 14) — sklearn ACCEPTS this
        latest = df_features[FEATURE_COLS].iloc[[-1]]

        # predict() returns array: [0] = DOWN, [1] = UP
        prediction = pipeline.predict(latest)[0]

        # predict_proba() returns [[p_down, p_up]]
        proba  = pipeline.predict_proba(latest)[0]
        p_down = float(proba[0])
        p_up   = float(proba[1])

        direction  = "UP" if prediction == 1 else "DOWN"
        confidence = p_up if prediction == 1 else p_down

        result = {
            "direction":        direction,
            "confidence":       round(confidence, 4),
            "probability_up":   round(p_up, 4),
            "probability_down": round(p_down, 4),
        }

        logger.info(
            f"Prediction: {direction} "
            f"(confidence: {confidence:.1%})"
        )
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None