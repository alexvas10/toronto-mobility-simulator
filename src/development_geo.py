"""Load and geolocate Toronto development data for the map overlays.

Development Applications carry projected X/Y coordinates and a ward number, so they map as
points (one dot per proposed/approved development). Active Building Permits have no
coordinates at all -- only a postal FSA -- so they are aggregated to a ward-level count
using an FSA -> dominant-ward lookup derived from the Development Applications data. That
lookup is coarse (a postal FSA can straddle ward boundaries), so the permit layer is an
approximate "where is construction concentrated", not an exact per-permit location.
"""
import pandas as pd
from pyproj import Transformer

from src.config import DEVELOPMENT_XY_CRS
from src.ingest_development import BUILDING_PERMITS_PATH, DEV_APPLICATIONS_PATH

# Toronto planning application type codes -> human-readable labels.
APPLICATION_TYPE_LABELS = {
    "OZ": "Official Plan / Rezoning",
    "SA": "Site Plan Approval",
    "CD": "Draft Plan of Condominium",
    "SB": "Draft Plan of Subdivision",
    "PL": "Part Lot Control",
}

# Statuses that represent a decision still in motion (proposed, under review, or approved
# but not yet closed out) -- the default view for "where new city decisions will be built".
ACTIVE_APPLICATION_STATUSES = {
    "Under Review", "Application Received", "Circulated", "NOAC Issued",
    "OMB Appeal", "Appeal Received", "Council Approved", "Draft Plan Approved",
    "Final Approval Completed", "Approved", "OMB Approved", "OMB Partially Approved",
    "Amend Drft Plan App",
}


def load_development_applications() -> pd.DataFrame:
    """One row per development application with WGS84 lat/lon, status, type, and ward."""
    df = pd.read_csv(DEV_APPLICATIONS_PATH, low_memory=False)
    df = df.dropna(subset=["X", "Y"]).copy()

    transformer = Transformer.from_crs(DEVELOPMENT_XY_CRS, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["X"].to_numpy(), df["Y"].to_numpy())
    df["lat"], df["lon"] = lat, lon
    # A handful of records have corrupt X/Y that reproject far outside the city; drop them.
    df = df[df["lat"].between(43.5, 43.9) & df["lon"].between(-79.7, -79.1)].copy()

    df["status"] = df["STATUS"].str.strip()
    df["app_type"] = df["APPLICATION_TYPE"].str.strip()
    df["app_type_label"] = df["app_type"].map(APPLICATION_TYPE_LABELS).fillna(df["app_type"])
    df["ward_num"] = pd.to_numeric(df["WARD_NUMBER"], errors="coerce")
    df["address"] = (
        df["STREET_NUM"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        + " " + df["STREET_NAME"].fillna("").astype(str)
        + " " + df["STREET_TYPE"].fillna("").astype(str)
    ).str.strip()

    return df[[
        "lat", "lon", "status", "app_type", "app_type_label", "ward_num",
        "WARD_NAME", "address", "DESCRIPTION", "DATE_SUBMITTED", "APPLICATION_URL",
    ]].rename(columns={
        "WARD_NAME": "ward_name", "DESCRIPTION": "description",
        "DATE_SUBMITTED": "date_submitted", "APPLICATION_URL": "url",
    })


def _fsa_to_ward() -> pd.Series:
    """Map each postal FSA (first 3 chars) to its most common ward, from dev-app records."""
    df = pd.read_csv(DEV_APPLICATIONS_PATH, low_memory=False)
    df = df.dropna(subset=["POSTAL", "WARD_NUMBER"]).copy()
    df["FSA"] = df["POSTAL"].str.strip().str.upper().str[:3]
    df["WARD_NUMBER"] = pd.to_numeric(df["WARD_NUMBER"], errors="coerce")
    return (
        df.dropna(subset=["WARD_NUMBER"])
        .groupby("FSA")["WARD_NUMBER"]
        .agg(lambda s: s.value_counts().index[0])
        .astype(int)
    )


def load_building_permit_ward_counts(statuses: set[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Approximate active-building-permit count per ward via FSA -> dominant-ward lookup.

    Returns (per-ward counts DataFrame[ward_num, permit_count], n_unmatched_permits).
    Pass a set of STATUS values to include only those; None keeps every active permit.
    """
    permits = pd.read_csv(BUILDING_PERMITS_PATH, low_memory=False)
    if statuses is not None:
        permits = permits[permits["STATUS"].str.strip().isin(statuses)]

    fsa_ward = _fsa_to_ward()
    fsa = permits["POSTAL"].astype(str).str.strip().str.upper().str[:3]
    permits = permits.assign(ward_num=fsa.map(fsa_ward))

    n_unmatched = int(permits["ward_num"].isna().sum())
    counts = (
        permits.dropna(subset=["ward_num"])
        .astype({"ward_num": int})
        .groupby("ward_num")
        .size()
        .rename("permit_count")
        .reset_index()
    )
    return counts, n_unmatched


def building_permit_statuses() -> list[str]:
    """Distinct permit statuses, most common first -- for the sidebar status filter."""
    permits = pd.read_csv(BUILDING_PERMITS_PATH, low_memory=False, usecols=["STATUS"])
    return permits["STATUS"].str.strip().value_counts().index.tolist()
