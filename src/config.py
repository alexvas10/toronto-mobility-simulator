from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"

# Verified against the live CKAN API on 2026-06-16.
# "Summary and Trip Data" holds both the per-ward-hour trip archives (trips_*.zip,
# the large file that motivates chunked/Dask loading) and a small pre-aggregated
# daily city-wide "summary_stats" resource used for city-wide context features.
# The separate "Vehicle Operating Data" package is per-vehicle-per-hour (vehid x hr,
# no ward, no useful geography) -- intentionally not ingested, see README.
PTC_TRIPS_PACKAGE = "private-transportation-companies-summary-and-trip-data"
TTC_DELAY_PACKAGES = {
    "subway": "ttc-subway-delay-data",
    "streetcar": "ttc-streetcar-delay-data",
    "bus": "ttc-bus-delay-data",
    "lrt": "ttc-lrt-delay-data",
}
WARD_BOUNDARY_PACKAGE = "city-wards"
GTFS_PACKAGE = "ttc-routes-and-schedules"

# Development pipeline datasets, for the "where is new construction happening" overlays.
# Development Applications: proposed/approved planning decisions (rezonings, site plans,
# subdivisions) -- has projected X/Y (EPSG:2952) and a 25-ward number, so it maps as points.
# Building Permits (Active): permits currently in-progress -- large (~211k rows) but has NO
# coordinates and no 25-ward field, only a postal FSA, so it's aggregated to wards via an
# FSA -> dominant-ward lookup borrowed from the Development Applications data (see
# src/development_geo.py). Both are on CKAN datastore, dumped as CSV.
DEVELOPMENT_APPLICATIONS_PACKAGE = "development-applications"
BUILDING_PERMITS_PACKAGE = "building-permits-active-permits"

# Development Applications X/Y are NAD83 MTM zone 10 metres; reprojected to WGS84 for the map.
DEVELOPMENT_XY_CRS = "EPSG:2952"

# PTC trip-level zip archives are published one per year.
PTC_TRIP_YEARS = range(2018, 2027)

RUSH_HOUR_RANGES = ((7, 10), (16, 19))  # (start_hour_inclusive, end_hour_exclusive)
