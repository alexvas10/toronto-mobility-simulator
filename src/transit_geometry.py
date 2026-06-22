"""Subway/streetcar line geometry and station points, for drawing on top of the
ward choropleth -- pulled from the same GTFS feed used for the station-ward lookup.
"""
import pandas as pd

from src.config import RAW_DIR

GTFS_DIR = RAW_DIR / "gtfs"

MODE_NAMES = {0: "streetcar", 1: "subway"}
MODE_COLORS = {"subway": "#1f1fb4", "streetcar": "#c0392b"}


def load_transit_lines() -> pd.DataFrame:
    """One row per (route, representative shape): mode, route_name, lat/lon point lists."""
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt")
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")

    routes = routes[routes["route_type"].isin(MODE_NAMES)].copy()
    routes["mode"] = routes["route_type"].map(MODE_NAMES)

    trip_shapes = trips[trips["route_id"].isin(routes["route_id"])][["route_id", "shape_id"]]
    # GTFS splits each route into many trip patterns/branches/directions; keep the
    # single longest shape per route so the map shows one clean line per route
    # instead of dozens of near-duplicate overlapping traces.
    shape_lengths = shapes.groupby("shape_id").size().rename("n_points")
    trip_shapes = trip_shapes.merge(shape_lengths, on="shape_id").drop_duplicates("shape_id")
    longest_shape_per_route = trip_shapes.sort_values("n_points").drop_duplicates("route_id", keep="last")

    rows = []
    for _, row in longest_shape_per_route.merge(routes, on="route_id").iterrows():
        shape_points = shapes[shapes["shape_id"] == row["shape_id"]].sort_values("shape_pt_sequence")
        rows.append(
            {
                "route_id": row["route_id"],
                "mode": row["mode"],
                "route_name": row.get("route_long_name") or row.get("route_short_name"),
                "lats": shape_points["shape_pt_lat"].tolist(),
                "lons": shape_points["shape_pt_lon"].tolist(),
            }
        )
    return pd.DataFrame(rows)
