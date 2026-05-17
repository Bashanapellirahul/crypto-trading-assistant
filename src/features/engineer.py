import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ma_7", "ma_30", "ma_cross",
    "price_vs_ma7", "price_vs_ma30",
    "pct_1d", "pct_7d", "pct_30d",
    "volatility_7d", "volatility_30d",
    "price_position",
    "price_lag_1", "price_lag_3", "price_lag_7"
]


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["ma_7"] = df["price"].rolling(window=7, min_periods=7).mean()
    df["ma_30"] = df["price"].rolling(window=30, min_periods=30).mean()

    # Crossover signal: 1.0 = MA7 above MA30 (bullish), 0.0 = bearish
    # astype(float) converts bool True/False to 1.0/0.0
    # WHY float not int: scikit-learn StandardScaler expects float
    df["ma_cross"] = (df["ma_7"] > df["ma_30"]).astype(float)

    # How far is price from its moving averages, as a percentage
    # WHY percentage: scale-invariant across coins
    df["price_vs_ma7"] = (
        (df["price"] - df["ma_7"]) / df["ma_7"] * 100
    )
    df["price_vs_ma30"] = (
        (df["price"] - df["ma_30"]) / df["ma_30"] * 100
    )

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["pct_1d"] = df["price"].pct_change(1) * 100   # daily return
    df["pct_7d"] = df["price"].pct_change(7) * 100   # weekly return
    df["pct_30d"] = df["price"].pct_change(30) * 100  # monthly return

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["volatility_7d"] = df["pct_1d"].rolling(7).std()
    df["volatility_30d"] = df["pct_1d"].rolling(30).std()

    return df


def add_price_position(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    rolling_max = df["price"].rolling(30).max()
    rolling_min = df["price"].rolling(30).min()
    range_size = rolling_max - rolling_min

    df["price_position"] = np.where(
        range_size > 0,
        (df["price"] - rolling_min) / range_size,
        0.5  # flat market — place at midpoint
    )

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_3"] = df["price"].shift(3)
    df["price_lag_7"] = df["price"].shift(7)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["target"] = (
        df["price"].shift(-1) > df["price"]
    ).astype(int)

    # The last row has NaN because shift(-1) has nothing to look forward to
    # We cast to int first but need to re-handle that NaN
    # Replace the last row's target with NaN explicitly
    df.loc[df.index[-1], "target"] = np.nan

    return df


def build_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
   
    if df is None or df.empty:
        logger.error("Cannot engineer features: empty or None DataFrame")
        return None

    df = df.copy()  # never mutate the input

    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_volatility(df)    # depends on pct_1d from add_momentum
    df = add_price_position(df)
    df = add_lag_features(df)
    df = add_target(df)

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)

    logger.info(
        f"Feature engineering complete: "
        f"{rows_before} input rows → "
        f"{len(df)} usable rows "
        f"({rows_dropped} dropped for NaN)"
    )

    if len(df) < 30:
        logger.error(
            f"Only {len(df)} rows after dropna — not enough to train. "
            f"Fetch more history (increase days parameter)."
        )
        return None

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# Usage: python src/features/engineer.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))

    from src.ingestion.fetcher import fetch_price_history
    from src.processing.cleaner import raw_history_to_dataframe

    print("\n=== Testing engineer.py ===\n")

    # Fetch and clean
    print("Step 1: Fetching data...")
    raw = fetch_price_history("bitcoin", days=90)
    df_clean = raw_history_to_dataframe(raw, "bitcoin")
    print(f"  Clean DataFrame: {df_clean.shape}")

    # Engineer features
    print("\nStep 2: Engineering features...")
    df_features = build_features(df_clean)

    if df_features is not None:
        print(f"\nFeatures DataFrame:")
        print(f"  Shape:    {df_features.shape}")
        print(f"  Columns:  {list(df_features.columns)}")
        print(f"\nAll 14 features present: "
              f"{all(col in df_features.columns for col in FEATURE_COLS)}")

        print(f"\nNull values per column:")
        print(df_features.isnull().sum())

        print(f"\nSample row (most recent):")
        print(df_features.iloc[-1][FEATURE_COLS])

        print(f"\nTarget distribution:")
        counts = df_features["target"].value_counts()
        total = len(df_features)
        print(f"  UP   (1): {counts.get(1,0)} days "
              f"({counts.get(1,0)/total*100:.1f}%)")
        print(f"  DOWN (0): {counts.get(0,0)} days "
              f"({counts.get(0,0)/total*100:.1f}%)")

        print(f"\nFirst 3 rows:\n{df_features[FEATURE_COLS].head(3)}")
    else:
        print("Feature engineering FAILED")

    print("\n=== All tests complete ===\n")