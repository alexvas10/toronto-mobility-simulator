# Toronto Urban Mobility & Transit-Delay Scenario Simulator

A historical scenario simulator — not real-time — that fuses Toronto rideshare trip data, TTC transit delay logs, and Environment Canada weather into a Streamlit dashboard. Pick a hypothetical day, hour, weather, and TTC-delay scenario; the app predicts rideshare trip duration per ward and highlights historical anomalies under similar conditions.

Built as a data-engineering and geospatial analysis project on real, messy open-data sources: ~20.5M rows of rideshare trips, a decade of TTC delay logs, and hourly weather spanning 2018–2026.

---

## What it does

- **Choropleth map** of Toronto's 25 wards, coloured by predicted rideshare trip duration for your chosen scenario — on an OpenStreetMap basemap with TTC subway/streetcar lines and subway station markers overlaid from GTFS.
- **XGBoost regressor** trained on time/weather/delay features, evaluated with a time-based holdout (not a shuffled split, to avoid future-data leakage).
- **Isolation Forest anomaly detector** flags ward-hours that were unusual given their time-of-day, weather, and TTC-delay context — framed as association, not causation.
- **SHAP waterfall** shows which features drove each ward's prediction.
- All model artifacts and processed data are regenerated locally — nothing is shipped in the repo (see [Setup](#setup)).

---

## Dashboard screenshot

> Run `streamlit run app.py` after completing the data pipeline. The sidebar lets you dial in day of week, hour, temperature, precipitation, and active TTC delay counts.

---

## Data sources

All data is publicly available with no API key required.

| Source | What it provides | How it's fetched |
|---|---|---|
| [Toronto Open Data — PTC Summary and Trip Data](https://open.toronto.ca/dataset/private-transportation-companies-summary-and-trip-data/) | Hourly rideshare trip counts, fares, durations, and wait times by pickup/dropoff ward | CKAN API, yearly `.zip` archives |
| [Toronto Open Data — TTC Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/) (subway, streetcar, bus, LRT) | Per-mode delay logs with cause codes and delay minutes | CKAN API, yearly XLSX + rolling CSV |
| [Toronto Open Data — TTC Routes and Schedules (GTFS)](https://open.toronto.ca/dataset/ttc-routes-and-schedules/) | Station coordinates and transit line geometry for the map overlay | CKAN API, static GTFS zip |
| [Toronto Open Data — City Wards](https://open.toronto.ca/dataset/city-wards/) | Ward boundary polygons (GeoJSON, WGS84) for the choropleth | CKAN API |
| [Environment Canada — Climate Data](https://climate.weather.gc.ca/) | Hourly temperature, precipitation, and weather description (Toronto Pearson Intl A, station 51459) | HTTP bulk download, one file per month |

**Geography note:** PTC trip data only has ward name/number — no lat/long or H3 granularity — so the map is a ward-level choropleth, not point clustering. Subway station coordinates come from GTFS and are used separately for delay-to-ward matching and the map overlay.

---

## Project layout

```
app.py                      # Streamlit dashboard
requirements.txt
src/
  config.py                 # paths and package IDs
  ckan_utils.py             # generic CKAN package/resource + download helpers
  ingest_ptc.py             # download PTC trip archives + city-wide summary_stats
  ingest_ttc_delays.py      # download TTC delay logs per mode
  ingest_weather.py         # download Environment Canada hourly weather
  ingest_wards.py           # download ward boundary GeoJSON
  ingest_gtfs.py            # download TTC GTFS feed
  ttc_station_wards.py      # subway station → ward lookup via GTFS + fuzzy text match
  transit_geometry.py       # subway/streetcar line geometry for the map overlay
  build_features.py         # join all sources → hourly ward fact table
  anomaly_model.py          # Isolation Forest anomaly detection
  forecast_model.py         # XGBoost regressor + time-based backtest
data/
  raw/                      # downloaded source files — gitignored, must be fetched
  processed/                # joined fact table (.parquet) — gitignored, must be built
models/                     # trained model artifacts — gitignored, must be trained
```

---

## Setup

**Requirements:** Python 3.11+, ~4 GB disk for raw data.

```bash
git clone <repo-url>
cd toronto-mobility-simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Download raw data

Each script is independently runnable:

```bash
python -m src.ingest_ptc          # rideshare trip archives (2018–2026, ~3.1 GB)
python -m src.ingest_ttc_delays   # TTC delay logs, all modes
python -m src.ingest_weather      # Environment Canada hourly weather
python -m src.ingest_wards        # Toronto ward boundary GeoJSON
python -m src.ingest_gtfs         # TTC GTFS feed (stops, routes, shapes)
```

All downloads are idempotent — already-present files are skipped unless `force=True`.

### 2. Build the feature table

```bash
python -m src.build_features
```

Joins PTC trips, weather, and TTC delay counts into a single hourly ward-level Parquet file. Uses Dask for the PTC load (~20.5M rows) to avoid materialising the full dataset in memory at once.

### 3. Train the models

```bash
python -m src.anomaly_model    # Isolation Forest — writes hourly_ward_facts_with_anomalies.parquet
python -m src.forecast_model   # XGBoost regressor — writes models/duration_forecast_xgb.json
```

`forecast_model` prints held-out test metrics (MAE and RMSE in minutes) before saving.

### 4. Run the dashboard

```bash
streamlit run app.py
```

---

## Key design decisions

**Why Dask for PTC ingestion?** The full 2018–2026 trip history is ~20.5M rows. Plain pandas loads it in ~32s; Dask takes ~26s. The difference is modest — the real win is that Dask never materialises the full frame in memory at once, which matters on RAM-constrained machines or if the dataset grows. This is documented as a measured trade-off, not a "had to use Spark" story.

**Why a time-based train/test split?** The last 60 days of data are held out. Random shuffling would leak future ward-hour observations into training through the `trips_total_last_week` lag feature. Time-based splitting mirrors real deployment conditions.

**How are subway delays matched to wards?** TTC delay logs only give a free-text `Station` field (e.g. `"BATHURST STATION (ENTE"`, truncated). `src/ttc_station_wards.py` uses GTFS station coordinates spatially joined to ward polygons, then matches messy delay-log text to canonical station names via word-boundary regex (validated at ~99% row match rate on subway data, ~96% on LRT data). Streetcar/bus delays use intersection-style free text that isn't reliably geocodable, so those are counted city-wide rather than per-ward.

**OpenStreetMap basemap — no API key needed.** The map uses Plotly's `choropleth_map` with `map_style="open-street-map"`, which renders tiles directly from the OSM CDN.

---

## What's gitignored

Raw data, processed Parquet files, the trained model, and the virtual environment are all excluded via `.gitignore`. You must run the full pipeline after cloning. The only committed data is `data/raw/.gitkeep` and `data/processed/.gitkeep` (placeholder files to preserve directory structure).

---

## Limitations

- **Historical only.** Every prediction comes from a model trained on 2018–2026 data. There is no real-time feed.
- **Ward-level geography only.** PTC data has no lat/long, so sub-ward granularity isn't possible with this dataset.
- **Anomaly flags are associative.** The Isolation Forest flags hours that were statistically unusual given their context — not hours where TTC delays *caused* rideshare spikes.
- **Weather is airport-station.** Environment Canada hourly data is from Toronto Pearson (station 51459), which is reasonable for city-wide conditions but not neighbourhood-level microclimate.
