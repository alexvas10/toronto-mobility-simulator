"""Flag anomalous ward-hours with Isolation Forest.

Trips_total has strong, expected seasonality (rush hour, weekend, weather), so the
model is given hour/day-of-week/weather/delay context alongside the trip metrics --
the goal is to flag hours that are unusual *given that context*, not just any busy
rush hour. Isolation Forest is tree-based, so features don't need scaling.

Framing: flags are "associated with" unusual conditions, not proof of causation.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import PROCESSED_DIR

FACTS_PATH = PROCESSED_DIR / "hourly_ward_facts.parquet"
OUTPUT_PATH = PROCESSED_DIR / "hourly_ward_facts_with_anomalies.parquet"

FEATURE_COLUMNS = [
    "ward_num",
    "hour",
    "is_weekend",
    "is_rush_hour",
    "temp_c",
    "precip_mm",
    "is_snowing",
    "active_ttc_delays_ward",
    "active_ttc_delays_citywide",
    "trips_total",
    "fare_avg",
    "wait_avg",
    "duration_avg",
]
CONTAMINATION = 0.01


def fit_anomaly_model(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.dropna(subset=FEATURE_COLUMNS).copy()
    X = model_df[FEATURE_COLUMNS].astype(float)

    model = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    model.fit(X)

    model_df["anomaly_score"] = -model.score_samples(X)  # higher = more anomalous
    model_df["is_anomaly"] = model.predict(X) == -1

    df = df.merge(
        model_df[["ward_num", "datetime", "anomaly_score", "is_anomaly"]],
        on=["ward_num", "datetime"],
        how="left",
    )
    return df


def report_top_anomalies(df: pd.DataFrame, n: int = 15) -> None:
    cols = ["ward_num", "datetime", "trips_total", "duration_avg", "fare_avg",
            "temp_c", "is_snowing", "active_ttc_delays_ward", "active_ttc_delays_citywide", "anomaly_score"]
    top = df.sort_values("anomaly_score", ascending=False).head(n)
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    facts = pd.read_parquet(FACTS_PATH)
    flagged = fit_anomaly_model(facts)
    flagged.to_parquet(OUTPUT_PATH, index=False)
    print(f"flagged {flagged['is_anomaly'].sum()} of {len(flagged)} ward-hours as anomalous")
    report_top_anomalies(flagged)
