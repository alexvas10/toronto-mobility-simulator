"""XGBoost regressor forecasting ward-hour trip duration from time/weather/delay context.

Target is duration_avg (the "how bad is the commute" signal) rather than fare_avg --
duration ties more directly to the congestion story the dashboard tells. Evaluated
with a time-based split (last 2 months held out) rather than a random shuffle split,
since random shuffling would leak future information into training via adjacent hours.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import MODELS_DIR, PROCESSED_DIR

FACTS_PATH = PROCESSED_DIR / "hourly_ward_facts.parquet"
MODEL_PATH = MODELS_DIR / "duration_forecast_xgb.json"

FEATURE_COLUMNS = [
    "ward_num",
    "hour",
    "dayofweek",
    "is_weekend",
    "is_rush_hour",
    "temp_c",
    "precip_mm",
    "is_snowing",
    "is_raining",
    "active_ttc_delays_ward",
    "active_ttc_delays_citywide",
    "trips_total_last_week",
]
TARGET_COLUMN = "duration_avg"
TEST_HOLDOUT_DAYS = 60


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = pd.to_datetime(df["datetime"]).dt.dayofweek
    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])


def time_based_split(df: pd.DataFrame):
    cutoff = df["datetime"].max() - pd.Timedelta(days=TEST_HOLDOUT_DAYS)
    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]
    return train, test


def train_and_evaluate(df: pd.DataFrame) -> XGBRegressor:
    df = prepare(df)
    train, test = time_based_split(df)

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    preds = model.predict(test[FEATURE_COLUMNS])
    mae = mean_absolute_error(test[TARGET_COLUMN], preds)
    rmse = np.sqrt(mean_squared_error(test[TARGET_COLUMN], preds))
    print(f"held-out test ({TEST_HOLDOUT_DAYS} days, n={len(test)}): MAE={mae:.2f} min, RMSE={rmse:.2f} min")

    model.save_model(MODEL_PATH)
    return model


if __name__ == "__main__":
    facts = pd.read_parquet(FACTS_PATH)
    train_and_evaluate(facts)
