"""Join PTC trips, weather, and TTC delays into an hourly ward-level fact table.

Geography note: PTC data only has ward name/number (e.g. "10 - Spadina-Fort York"),
not lat/long, so this aggregates to pickup ward x hour rather than any point/hex grid.

Weather note: Environment Canada's hourly feed does not split rain vs snow amount at
the hourly grain (that split only exists in their *daily* summaries) -- it gives one
combined "Precip. Amount (mm)" plus a free-text "Weather" description. We derive
is_snowing/is_raining flags from that text rather than inventing a fake snowfall_mm.

TTC delay note: subway delays are matched to a ward via src.ttc_station_wards (GTFS
station coordinates + a fuzzy text match against the delay log's free-text Station
field, ~99% row match rate). Streetcar/bus/LRT delay logs use intersection-style free
text instead of station names, which isn't reliably geocodable here, so those (plus
the ~1% of unmatched subway rows) are folded into a single city-wide
active_ttc_delays_citywide count instead of a per-ward one.
"""
import glob

import dask.dataframe as dd
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, RUSH_HOUR_RANGES
from src.ttc_station_wards import build_match_name_list, build_station_ward_lookup, match_station_text

PTC_TRIPS_GLOB = str(RAW_DIR / "ptc_trips" / "trips_??????.csv")
WEATHER_GLOB = str(RAW_DIR / "weather" / "weather_*.csv")
DELAY_DIR = RAW_DIR / "ttc_delays"
OUTPUT_PATH = PROCESSED_DIR / "hourly_ward_facts.parquet"


def _is_rush_hour(hour: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=hour.index)
    for start, end in RUSH_HOUR_RANGES:
        mask |= (hour >= start) & (hour < end)
    return mask


def load_ptc_ward_hourly() -> pd.DataFrame:
    paths = sorted(glob.glob(PTC_TRIPS_GLOB))
    ddf = dd.read_csv(paths, dtype={"pickup_ward": "object", "dropoff_ward": "object"})
    # A handful of rows have placeholder wards like "Not included elsewhere" instead
    # of a real "<number> - <name>" ward -- drop those, they can't be mapped to a
    # ward boundary polygon anyway.
    keep = (ddf["pickup_municipality"] == "Toronto") & ddf["pickup_ward"].str.match(r"^\d+ - ")
    ddf = ddf[keep]
    ddf["pickup_hr"] = dd.to_datetime(ddf["pickup_hr"], utc=True)
    ddf["ward_num"] = ddf["pickup_ward"].str.split(" - ").str[0].astype("int64")
    ddf["weighted_fare"] = ddf["fare_avg"] * ddf["trips_total"]
    ddf["weighted_distance"] = ddf["distance_avg"] * ddf["trips_total"]
    ddf["weighted_duration"] = ddf["duration_avg"] * ddf["trips_total"]
    ddf["weighted_wait"] = ddf["waittime_avg"] * ddf["trips_total"]

    grouped = (
        ddf.groupby(["ward_num", "pickup_hr"])
        .agg(
            trips_total=("trips_total", "sum"),
            weighted_fare=("weighted_fare", "sum"),
            weighted_distance=("weighted_distance", "sum"),
            weighted_duration=("weighted_duration", "sum"),
            weighted_wait=("weighted_wait", "sum"),
        )
        .reset_index()
    )
    df = grouped.compute()

    for col in ("fare", "distance", "duration", "wait"):
        df[f"{col}_avg"] = df[f"weighted_{col}"] / df["trips_total"]
    df = df.drop(columns=[c for c in df.columns if c.startswith("weighted_")])

    df["datetime"] = pd.to_datetime(df["pickup_hr"], utc=True).dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df.drop(columns=["pickup_hr"])


def load_weather_hourly() -> pd.DataFrame:
    frames = []
    for path in glob.glob(WEATHER_GLOB):
        frames.append(pd.read_csv(path))
    weather = pd.concat(frames, ignore_index=True)
    weather["datetime"] = pd.to_datetime(weather["Date/Time (LST)"])
    weather["date"] = weather["datetime"].dt.date
    weather["hour"] = weather["datetime"].dt.hour
    weather["temp_c"] = weather["Temp (°C)"]
    weather["precip_mm"] = weather["Precip. Amount (mm)"].fillna(0)
    weather_text = weather["Weather"].fillna("")
    weather["is_snowing"] = weather_text.str.contains("Snow", case=False)
    weather["is_raining"] = weather_text.str.contains("Rain", case=False)
    return weather[["date", "hour", "temp_c", "precip_mm", "is_snowing", "is_raining"]]


def _read_delay_file(path: str, columns: list[str]) -> pd.DataFrame | None:
    reader = pd.read_csv if path.endswith(".csv") else pd.read_excel
    try:
        return reader(path, usecols=lambda c: c in columns)
    except ValueError:
        return None  # readme/code-lookup files without the expected columns


STATION_MATCHABLE_MODES = ("subway", "lrt")  # Line 5/6 LRT has clean station names too


def load_ttc_delay_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ward_counts, citywide_counts): subway + Line 5/6 LRT delays matched to
    a station get counted per ward-hour; everything else (streetcar/bus + unmatched
    rows) gets counted per city-wide hour."""
    station_lookup = build_station_ward_lookup()
    canonical_names = build_match_name_list(station_lookup)
    name_to_ward = dict(zip(station_lookup["base_name"], station_lookup["ward_num"]))

    ward_frames, citywide_frames = [], []
    for mode_dir in sorted(DELAY_DIR.iterdir()):
        mode = mode_dir.name
        for path in glob.glob(str(mode_dir / "*")):
            columns = ["Date", "Time", "Station"] if mode in STATION_MATCHABLE_MODES else ["Date", "Time"]
            df = _read_delay_file(path, columns)
            if df is None or "Date" not in df.columns or "Time" not in df.columns:
                continue
            df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
            df = df.dropna(subset=["datetime"])
            df["date"] = df["datetime"].dt.date
            df["hour"] = df["datetime"].dt.hour

            if mode in STATION_MATCHABLE_MODES and "Station" in df.columns:
                df["ward_num"] = df["Station"].apply(lambda t: name_to_ward.get(match_station_text(t, canonical_names)))
                ward_frames.append(df.dropna(subset=["ward_num"])[["ward_num", "date", "hour"]])
                citywide_frames.append(df[df["ward_num"].isna()][["date", "hour"]])
            else:
                citywide_frames.append(df[["date", "hour"]])

    ward_counts = (
        pd.concat(ward_frames, ignore_index=True)
        .astype({"ward_num": "int64"})
        .groupby(["ward_num", "date", "hour"])
        .size()
        .rename("active_ttc_delays_ward")
        .reset_index()
    )
    citywide_counts = (
        pd.concat(citywide_frames, ignore_index=True)
        .groupby(["date", "hour"])
        .size()
        .rename("active_ttc_delays_citywide")
        .reset_index()
    )
    return ward_counts, citywide_counts


def build() -> pd.DataFrame:
    trips = load_ptc_ward_hourly()
    weather = load_weather_hourly()
    ward_delays, citywide_delays = load_ttc_delay_counts()

    fact = trips.merge(weather, on=["date", "hour"], how="left")
    fact = fact.merge(ward_delays, on=["ward_num", "date", "hour"], how="left")
    fact = fact.merge(citywide_delays, on=["date", "hour"], how="left")
    fact["active_ttc_delays_ward"] = fact["active_ttc_delays_ward"].fillna(0).astype(int)
    fact["active_ttc_delays_citywide"] = fact["active_ttc_delays_citywide"].fillna(0).astype(int)

    fact["datetime"] = pd.to_datetime(fact["date"].astype(str)) + pd.to_timedelta(fact["hour"], unit="h")
    fact["is_weekend"] = fact["datetime"].dt.dayofweek >= 5
    fact["is_rush_hour"] = _is_rush_hour(fact["hour"])

    fact = fact.sort_values(["ward_num", "datetime"])
    fact["trips_total_last_week"] = fact.groupby("ward_num")["trips_total"].shift(7 * 24)

    fact.to_parquet(OUTPUT_PATH, index=False)
    print(f"wrote {len(fact)} rows to {OUTPUT_PATH}")
    return fact


if __name__ == "__main__":
    build()
