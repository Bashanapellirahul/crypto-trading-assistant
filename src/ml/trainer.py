import os
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.features.engineer import FEATURE_COLS

logger = logging.getLogger(__name__)

MODEL_PATH = "data/model.pkl"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,      
            max_depth=6,            
            min_samples_leaf=5,     
            class_weight="balanced",
            random_state=42,        
            n_jobs=-1              
        ))
    ])


def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx]   # oldest 80%
    test_df = df.iloc[split_idx:]   # newest 20%

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"].astype(int)

    logger.info(f"Train: {len(X_train)} rows (rows 0 to {split_idx-1})")
    logger.info(f"Test:  {len(X_test)} rows (rows {split_idx} to {len(df)-1})")

    return X_train, X_test, y_train, y_test


def train(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = build_pipeline()

    logger.info("Training Random Forest pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    return pipeline


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # P(UP)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print(f"  Test Accuracy:  {accuracy:.2%}")
    print(f"  ROC-AUC Score:  {roc_auc:.3f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["DOWN (0)", "UP (1)"]
    ))
    print("Confusion Matrix:")
    print(f"  True Negative  (DOWN→DOWN): {cm[0,0]}")
    print(f"  False Positive (DOWN→UP):   {cm[0,1]}  ← false buy signal")
    print(f"  False Negative (UP→DOWN):   {cm[1,0]}  ← missed opportunity")
    print(f"  True Positive  (UP→UP):     {cm[1,1]}")
    print("="*50)

    rf_model = pipeline.named_steps["model"]
    importances = pd.Series(
        rf_model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\nTop 5 features by importance:")
    for feat, imp in importances.head(5).items():
        bar = "█" * int(imp * 100)
        print(f"  {feat:<20} {imp:.3f}  {bar}")

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "feature_importances": importances.to_dict()
    }


def save_pipeline(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Pipeline saved to {path}")


def load_pipeline(path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path}. "
            f"Run trainer.py first: python src/ml/trainer.py"
        )
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Run training
# Usage: python src/ml/trainer.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))

    logging.basicConfig(level=logging.INFO)

    from src.ingestion.fetcher import (
        fetch_price_history, SUPPORTED_COINS, DEFAULT_HISTORY_DAYS
    )
    from src.processing.cleaner import (
        raw_history_to_dataframe, validate_dataframe
    )
    from src.features.engineer import build_features

    print("\n" + "="*50)
    print("  CRYPTO TRADING ASSISTANT — MODEL TRAINING")
    print("="*50)

    all_features = []

    for coin in SUPPORTED_COINS:
        print(f"\nFetching {coin}...")

        raw = fetch_price_history(coin, days=180)
        if not raw:
            print(f"  Skipping {coin} — fetch failed")
            continue

        df_clean = raw_history_to_dataframe(raw, coin)
        if not validate_dataframe(df_clean):
            print(f"  Skipping {coin} — validation failed")
            continue

        df_feat = build_features(df_clean)
        if df_feat is not None:
            all_features.append(df_feat)
            print(f"  {coin}: {len(df_feat)} usable rows")

        time.sleep(1.5)  # rate limit between coins

    if not all_features:
        print("No training data collected. Exiting.")
        sys.exit(1)

    df_combined = pd.concat(all_features).reset_index(drop=True)
    print(f"\nCombined training data: {len(df_combined)} rows "
          f"across {len(all_features)} coins")

    dist = df_combined["target"].value_counts(normalize=True)
    print(f"Target: {dist.get(1,0)*100:.1f}% UP, "
          f"{dist.get(0,0)*100:.1f}% DOWN")

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = chronological_split(df_combined)
    print(f"  Train: {len(X_train)} rows")
    print(f"  Test:  {len(X_test)} rows")

    print("\nTraining...")
    pipeline = train(X_train, y_train)

    print("\nEvaluating on test set...")
    metrics = evaluate(pipeline, X_test, y_test)

    save_pipeline(pipeline)
    print(f"\nModel saved to {MODEL_PATH}")
    print("\nTraining complete. You can now start the API server.")
    print("="*50 + "\n")