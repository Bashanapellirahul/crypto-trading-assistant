import pandas as pd
import logging

logger = logging.getLogger(__name__)

def raw_history_to_dataframe(raw_prices: list, 
                             coin_id: str) -> pd.DataFrame:
    if not raw_prices:
        logger.info(f'Empty price list received for {coin_id}')
        return None
    
    try:
        df = pd.DataFrame(raw_prices, columns=["timestamp_ms", "price"])
        #Convert millisecond timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp_ms"],unit="ms")
        # Drop the raw timestamp — we have 'date' now
        df = df.drop(columns=["timestamp_ms"])
        # Step 4: Add coin identifier
        df["coin"] = coin_id
        # Step 5: Set date as the index
        df = df.set_index("date")
        df = df.sort_index()
        # Step 7: Ensure price is float64
        df["price"] = df["price"].astype(float)
        # Step 8: Handle missing values
        missing_count = df["price"].isnull().sum()
        if missing_count > 0:
            logger.warning(
                f"{coin_id}: {missing_count} missing prices — "
                f"applying forward fill"
            )
            # ffill: use last known price for the missing row
            df["price"] = df["price"].ffill()
        
        # Step 9: Drop any remaining nulls
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows still containing nulls")
        
        # Step 10: Remove duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]
        logger.info(
            f"{coin_id}: cleaned DataFrame ready — "
            f"{len(df)} rows, index: {df.index[0].date()} "
            f"to {df.index[-1].date()}"
        )

        return df
    except Exception as e:
        logger.error(f"Failed to clean {coin_id} data: {e}")
        return None
    

def validate_dataframe(df: pd.DataFrame, min_rows: int=60) -> bool:
    if df is None:
        logger.error("Validation failed: DataFrame is None")
        return False
    
    if df.empty:
        logger.error("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        logger.error(
            f"Validation failed: only {len(df)} rows, "
            f"need at least {min_rows}"
        )
        return False
    
    if "price" not in df.columns:
        logger.error("Validation failed: 'price' column missin")
        return False
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Validation failed: index is not DatetimeIndex")
        return False
    
    null_count = df["price"].isnull().sum()
    if null_count > 0:
        logger.error(f"Validation failed: {null_count} null prices")
        return False
    
    if (df["price"] <= 0).any():
        logger.error("Validation failed: non-positive prices detected")
        return False
    
    logger.info(
        f"Validation passed: {len(df)} rows, "
        f"price range ${df['price'].min():,.2f} "
        f"to ${df['price'].max():,.2f}"
    )

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run directly to verify
# Usage: python src/processing/cleaner.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    # Add project root to path so we can import fetcher
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))

    from src.ingestion.fetcher import fetch_price_history

    print("\n=== Testing cleaner.py ===\n")

    # Fetch raw data
    print("Fetching raw data from CoinGecko...")
    raw = fetch_price_history("bitcoin", days=90)

    if not raw:
        print("Could not fetch data. Check internet connection.")
        sys.exit(1)

    print(f"Raw data: {len(raw)} entries")
    print(f"Raw sample: {raw[0]}")

        # Clean it
    print("\nCleaning data...")
    df = raw_history_to_dataframe(raw, "bitcoin")

    if df is not None:
        print(f"\nCleaned DataFrame:")
        print(f"  Shape:      {df.shape}")
        print(f"  Index type: {type(df.index).__name__}")
        print(f"  Columns:    {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Price range: ${df['price'].min():,.2f} "
              f"to ${df['price'].max():,.2f}")
        print(f"  Null values: {df.isnull().sum().to_dict()}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nLast 3 rows:\n{df.tail(3)}")
    else:
        print("Cleaning FAILED")

    # Validate
    print("\nValidating...")
    is_valid = validate_dataframe(df)
    print(f"Valid: {is_valid}")

    print("\n=== All tests complete ===\n")