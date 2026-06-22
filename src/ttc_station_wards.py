"""Map TTC rail delay log "Station" text to a ward, via GTFS station coordinates.

Covers the subway (route_type 1) plus Line 5 Eglinton and Line 6 Finch West, the two
light-rail lines (GTFS files them under route_type 0 "streetcar" alongside the actual
streetcar network, but they have clean station names like a subway, unlike streetcars'
intersection-style stop names, so they're treated the same way here).

TTC delay logs only give a free-text Station field (e.g. "BATHURST STATION (ENTE",
truncated, with stray punctuation, "APPROACHING X", building names, etc.) -- there is
no ward column. GTFS gives us clean coordinates for each station platform, which we
collapse to one point per station and spatial-join into the ward boundary polygons.
We then match the messy delay-log text back to a canonical station name with a
word-boundary substring search (validated at a 99.2% row match rate on 2025+ subway
delay data and 95.8% on the Line 5/6 LRT delay data; the unmatched remainder are
genuinely not single-station events -- "SYSTEM WIDE", "TRACK LEVEL ACTIVITY", named
buildings/garages/yards, or a handful of stations not yet in this GTFS snapshot's
active trip patterns, like Mount Dennis -- added manually below).

Regular streetcar and bus delay logs use intersection-style free text instead of
station names, which isn't reliably geocodable without a real geocoding API, so this
module intentionally only covers subway + Line 5/6 LRT.
"""
import glob
import re

import geopandas as gpd
import pandas as pd

from src.config import RAW_DIR

GTFS_DIR = RAW_DIR / "gtfs"
WARDS_PATH = RAW_DIR / "city_wards_4326.geojson"

SUBWAY_ROUTE_TYPE = 1
LRT_ROUTE_IDS = (5, 6)  # Line 5 Eglinton, Line 6 Finch West

# Stations missing from this GTFS snapshot's active trip patterns at the time this
# was built -- filled in manually from public TTC station data.
MANUAL_STATIONS = [
    {"base_name": "DUNDAS", "stop_lat": 43.6562, "stop_lon": -79.3805},  # Line 1, downtown
    {"base_name": "MOUNT DENNIS", "stop_lat": 43.6886, "stop_lon": -79.4878},  # Line 5 west terminus
]

# Common abbreviations seen in the delay logs that don't literally contain the
# canonical GTFS station name.
ALIASES = {
    "VMC": "VAUGHAN METROPOLITAN CENTRE",
    "VAUGHAN MC": "VAUGHAN METROPOLITAN CENTRE",
    "SHEPPARD": "SHEPPARD-YONGE",
    "NORTH YORK CTR": "NORTH YORK CENTRE",
    "MT DENNIS": "MOUNT DENNIS",
}


def _canonical_rail_stations() -> pd.DataFrame:
    stops = pd.read_csv(GTFS_DIR / "stops.txt")
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")

    subway_route_ids = routes.loc[routes["route_type"] == SUBWAY_ROUTE_TYPE, "route_id"]
    rail_route_ids = pd.concat([subway_route_ids, pd.Series(LRT_ROUTE_IDS)])
    rail_trip_ids = trips.loc[trips["route_id"].isin(rail_route_ids), "trip_id"].unique()
    rail_stop_ids = stop_times.loc[stop_times["trip_id"].isin(rail_trip_ids), "stop_id"].unique()
    rail_stops = stops[stops["stop_id"].isin(rail_stop_ids)].copy()

    # Subway names look like "Finch Station - Southbound Platform"; LRT names look
    # like "Eglinton Station Eastbound Platform" or "... Station LRT Platform" --
    # splitting on "Station" handles both.
    rail_stops["base_name"] = (
        rail_stops["stop_name"].str.split(" Station").str[0]
        .str.upper().str.strip()
    )
    canon = rail_stops.groupby("base_name")[["stop_lat", "stop_lon"]].mean().reset_index()
    missing_manual = [s for s in MANUAL_STATIONS if s["base_name"] not in set(canon["base_name"])]
    return pd.concat([canon, pd.DataFrame(missing_manual)], ignore_index=True)


def build_station_ward_lookup() -> pd.DataFrame:
    """Returns one row per canonical rail station: base_name, ward_num."""
    stations = _canonical_rail_stations()
    points = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["stop_lon"], stations["stop_lat"]),
        crs="EPSG:4326",
    )
    wards = gpd.read_file(WARDS_PATH)[["AREA_SHORT_CODE", "geometry"]]
    wards["ward_num"] = wards["AREA_SHORT_CODE"].astype(int)
    joined = gpd.sjoin(points, wards[["ward_num", "geometry"]], how="left", predicate="within")
    return joined[["base_name", "ward_num"]]


def station_points_with_delay_counts() -> pd.DataFrame:
    """Station points (base_name, stop_lat, stop_lon) plus total historical delay
    count, for plotting on the map -- sized markers showing which stations have had
    the most delays historically. Covers subway + Line 5/6 LRT delay logs."""
    stations = _canonical_rail_stations()
    canonical_names = sorted(set(stations["base_name"]) | set(ALIASES.keys()), key=len, reverse=True)

    counts = {}
    for mode_dir in ("subway", "lrt"):
        for path in glob.glob(str(RAW_DIR / "ttc_delays" / mode_dir / "*")):
            reader = pd.read_csv if path.endswith(".csv") else pd.read_excel
            try:
                df = reader(path, usecols=lambda c: c == "Station")
            except ValueError:
                continue
            if "Station" not in df.columns:
                continue
            for text in df["Station"].dropna():
                name = match_station_text(text, canonical_names)
                if name:
                    counts[name] = counts.get(name, 0) + 1

    stations["delay_count"] = stations["base_name"].map(counts).fillna(0).astype(int)
    return stations


def match_station_text(text: str, canonical_names: list[str]) -> str | None:
    """Word-boundary substring match of a messy delay-log Station string against
    canonical station names (checked longest-first so e.g. "ST CLAIR WEST" wins over
    "ST CLAIR" when both would match)."""
    cleaned = re.sub(r"[.,]", "", str(text).upper())
    for name in canonical_names:
        if re.search(r"\b" + re.escape(name) + r"\b", cleaned):
            return ALIASES.get(name, name)
    return None


def build_match_name_list(lookup: pd.DataFrame) -> list[str]:
    names = set(lookup["base_name"]) | set(ALIASES.keys())
    return sorted(names, key=len, reverse=True)
