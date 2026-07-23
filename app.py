"""Toronto Urban Mobility & Transit-Delay Scenario Simulator (Streamlit dashboard).

This is a HISTORICAL SCENARIO SIMULATOR, not a real-time feed: every number on this
page comes from a model trained on 2018-2026 historical data. Pick a hypothetical
day/hour/weather/TTC-delay scenario and see what the model would predict for trip
duration in each ward, and where the historical data shows similar conditions were
associated with anomalous demand spikes (not proof of causation).
"""
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from xgboost import XGBRegressor

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.development_geo import (
    ACTIVE_APPLICATION_STATUSES,
    building_permit_statuses,
    load_building_permit_ward_counts,
    load_development_applications,
)
from src.forecast_model import FEATURE_COLUMNS
from src.transit_geometry import MODE_COLORS, load_transit_lines
from src.ttc_station_wards import station_points_with_delay_counts

# Qualitative palette for development-application points, keyed by application type.
APP_TYPE_COLORS = {
    "Official Plan / Rezoning": "#8e44ad",
    "Site Plan Approval": "#2980b9",
    "Draft Plan of Condominium": "#d35400",
    "Draft Plan of Subdivision": "#16a085",
    "Part Lot Control": "#c0392b",
}

WARDS_PATH = RAW_DIR / "city_wards_4326.geojson"
FACTS_PATH = PROCESSED_DIR / "hourly_ward_facts_with_anomalies.parquet"
MODEL_PATH = MODELS_DIR / "duration_forecast_xgb.json"

st.set_page_config(page_title="Toronto Mobility Scenario Simulator", layout="wide")


@st.cache_resource
def load_model() -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_explainer(_model: XGBRegressor) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


@st.cache_data
def load_wards() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(WARDS_PATH)
    gdf["ward_num"] = gdf["AREA_SHORT_CODE"].astype(int)
    return gdf[["ward_num", "AREA_NAME", "geometry"]]


@st.cache_data
def load_facts() -> pd.DataFrame:
    df = pd.read_parquet(FACTS_PATH)
    df["dayofweek"] = pd.to_datetime(df["datetime"]).dt.dayofweek
    return df


@st.cache_data
def load_transit_overlay() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_transit_lines(), station_points_with_delay_counts()


@st.cache_data
def load_dev_applications() -> pd.DataFrame:
    return load_development_applications()


@st.cache_data
def load_permit_status_options() -> list[str]:
    return building_permit_statuses()


@st.cache_data
def load_permit_counts(statuses: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    return load_building_permit_ward_counts(set(statuses) if statuses else None)


def _bus_sort_key(name: str) -> tuple[int, str]:
    """Sort bus route labels ('29 Dufferin') by leading route number, then name."""
    head = name.split(" ", 1)[0]
    return (int(head) if head.isdigit() else 10**6, name)


@st.cache_data
def ward_hour_baseline(facts: pd.DataFrame) -> pd.DataFrame:
    """Historical median trips_total per (ward, hour, dayofweek), used as the lag
    feature when scoring a hypothetical scenario that has no real history of its own."""
    return (
        facts.groupby(["ward_num", "hour", "dayofweek"])["trips_total"]
        .median()
        .rename("trips_total_last_week")
        .reset_index()
    )


def build_scenario_frame(wards: gpd.GeoDataFrame, baseline: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    rows = wards[["ward_num"]].copy()
    for key, value in scenario.items():
        rows[key] = value
    matching_baseline = baseline[
        (baseline["hour"] == scenario["hour"]) & (baseline["dayofweek"] == scenario["dayofweek"])
    ][["ward_num", "trips_total_last_week"]]
    rows = rows.merge(matching_baseline, on="ward_num", how="left")
    rows["trips_total_last_week"] = rows["trips_total_last_week"].fillna(baseline["trips_total_last_week"].median())
    return rows


def add_route_lines(fig: go.Figure, rows: pd.DataFrame, mode: str) -> None:
    """Draw one line + one mid-point label per transit route in `rows`."""
    color = MODE_COLORS[mode]
    for _, row in rows.iterrows():
        fig.add_trace(
            go.Scattermap(
                lat=row["lats"], lon=row["lons"], mode="lines",
                line=dict(width=3, color=color), name=row["route_name"],
                legendgroup=mode, showlegend=False, hoverinfo="text", text=row["route_name"],
            )
        )
        mid = len(row["lats"]) // 2
        fig.add_trace(
            go.Scattermap(
                lat=[row["lats"][mid]], lon=[row["lons"][mid]], mode="text",
                text=[row["route_name"]], textfont=dict(size=11, color=color),
                hoverinfo="skip", showlegend=False,
            )
        )


st.title("Toronto Urban Mobility & Transit-Delay Scenario Simulator")
st.caption(
    "Historical scenario simulator -- not real-time. Predictions come from a model "
    "trained on 2018-2026 Toronto rideshare, TTC delay, and weather data."
)

wards = load_wards()
facts = load_facts()
baseline = ward_hour_baseline(facts)
model = load_model()
transit_lines, station_points = load_transit_overlay()
dev_apps = load_dev_applications()
bus_route_options = sorted(
    transit_lines.loc[transit_lines["mode"] == "bus", "route_name"], key=_bus_sort_key
)

with st.sidebar:
    st.header("Map layers")
    show_transit_lines = st.checkbox("Show subway/streetcar lines", value=True)
    show_delay_stations = st.checkbox("Show subway stations (sized by historical delay count)", value=True)

    st.markdown("**Bus routes**")
    selected_bus_routes = st.multiselect(
        "Show bus routes (search by number or name)",
        options=bus_route_options,
        default=[],
        help="All 200+ TTC bus routes are available. None are drawn until you pick some, "
        "to keep the map fast and readable.",
    )

    st.markdown("**New development**")
    show_dev_apps = st.checkbox("Show development applications", value=False)
    dev_status_options = sorted(dev_apps["status"].dropna().unique())
    dev_default_statuses = [s for s in dev_status_options if s in ACTIVE_APPLICATION_STATUSES]
    dev_statuses = st.multiselect(
        "Application status", dev_status_options, default=dev_default_statuses,
        help="Defaults to in-progress decisions (under review / approved but not closed).",
        disabled=not show_dev_apps,
    )
    dev_type_options = sorted(dev_apps["app_type_label"].dropna().unique())
    dev_types = st.multiselect(
        "Application type", dev_type_options, default=dev_type_options, disabled=not show_dev_apps,
    )

    show_permits = st.checkbox("Show active building permits (per ward)", value=False)
    permit_status_options = load_permit_status_options()
    permit_statuses = st.multiselect(
        "Permit status (blank = all active permits)", permit_status_options, default=[],
        disabled=not show_permits,
    )
    st.divider()
    st.header("Scenario controls")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dayofweek = day_names.index(st.selectbox("Day of week", day_names, index=1))
    hour = st.slider("Hour of day", 0, 23, 17)
    temp_c = st.slider("Temperature (°C)", -25, 35, 0)
    precip_mm = st.slider("Precipitation (mm)", 0.0, 30.0, 0.0, step=0.5)
    is_snowing = st.checkbox("Snowing", value=temp_c < 2 and precip_mm > 0)
    is_raining = st.checkbox("Raining", value=temp_c >= 2 and precip_mm > 0)
    active_ttc_delays_citywide = st.slider("Active streetcar/bus delays this hour (city-wide)", 0, 30, 0)
    st.divider()
    subway_delay_active = st.checkbox("Active subway delay in a specific ward")
    delay_ward_name = st.selectbox(
        "Ward with the subway delay", wards.sort_values("ward_num")["AREA_NAME"],
        disabled=not subway_delay_active,
    )
    subway_delay_count = st.slider("Number of subway delay incidents", 1, 10, 1, disabled=not subway_delay_active)

scenario = {
    "hour": hour,
    "dayofweek": dayofweek,
    "is_weekend": dayofweek >= 5,
    "is_rush_hour": (7 <= hour < 10) or (16 <= hour < 19),
    "temp_c": float(temp_c),
    "precip_mm": float(precip_mm),
    "is_snowing": is_snowing,
    "is_raining": is_raining,
    "active_ttc_delays_ward": 0,
    "active_ttc_delays_citywide": active_ttc_delays_citywide,
}

scenario_df = build_scenario_frame(wards, baseline, scenario)
if subway_delay_active:
    delay_ward_num = wards.loc[wards["AREA_NAME"] == delay_ward_name, "ward_num"].iloc[0]
    scenario_df.loc[scenario_df["ward_num"] == delay_ward_num, "active_ttc_delays_ward"] = subway_delay_count
scenario_df["predicted_duration_min"] = model.predict(scenario_df[FEATURE_COLUMNS])

map_df = wards.merge(scenario_df[["ward_num", "predicted_duration_min"]], on="ward_num")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Predicted trip duration by ward")
    centroid = map_df.geometry.union_all().centroid
    fig = px.choropleth_map(
        map_df,
        geojson=map_df.geometry.__geo_interface__,
        locations=map_df.index,
        color="predicted_duration_min",
        hover_name="AREA_NAME",
        color_continuous_scale="OrRd",
        opacity=0.55,
        map_style="open-street-map",
        center={"lat": centroid.y, "lon": centroid.x},
        zoom=10,
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color="black")

    shown_modes = []
    if show_transit_lines:
        for mode in ("subway", "streetcar"):
            add_route_lines(fig, transit_lines[transit_lines["mode"] == mode], mode)
        shown_modes += ["subway", "streetcar"]
    if selected_bus_routes:
        buses = transit_lines[
            (transit_lines["mode"] == "bus") & transit_lines["route_name"].isin(selected_bus_routes)
        ]
        add_route_lines(fig, buses, "bus")
        shown_modes.append("bus")
    # one dummy trace per shown mode just to carry a single legend entry
    for mode in shown_modes:
        fig.add_trace(
            go.Scattermap(
                lat=[None], lon=[None], mode="lines",
                line=dict(width=3, color=MODE_COLORS[mode]), name=mode, showlegend=True,
            )
        )

    if show_dev_apps:
        shown = dev_apps[dev_apps["status"].isin(dev_statuses) & dev_apps["app_type_label"].isin(dev_types)]
        for app_type_label, group in shown.groupby("app_type_label"):
            hover = (
                group["address"] + " — " + group["status"]
                + " (Ward " + group["ward_num"].astype("Int64").astype(str) + ")"
            )
            fig.add_trace(
                go.Scattermap(
                    lat=group["lat"], lon=group["lon"], mode="markers",
                    marker=dict(size=7, color=APP_TYPE_COLORS.get(app_type_label, "#555555")),
                    text=hover, hoverinfo="text", name=app_type_label, showlegend=True,
                )
            )

    if show_permits:
        permit_counts, n_unmatched = load_permit_counts(tuple(sorted(permit_statuses)))
        ward_centroids = wards.copy()
        ward_centroids["cx"] = ward_centroids.geometry.centroid.x
        ward_centroids["cy"] = ward_centroids.geometry.centroid.y
        pc = ward_centroids.merge(permit_counts, on="ward_num", how="left").fillna({"permit_count": 0})
        fig.add_trace(
            go.Scattermap(
                lat=pc["cy"], lon=pc["cx"], mode="markers",
                marker=dict(
                    size=(pc["permit_count"].clip(lower=1)) ** 0.5 * 0.9,
                    color="#1f3a5f", opacity=0.75,
                ),
                text=pc["AREA_NAME"] + ": " + pc["permit_count"].astype(int).astype(str) + " active permits",
                hoverinfo="text", name="Active building permits", showlegend=True,
            )
        )

    if show_delay_stations:
        fig.add_trace(
            go.Scattermap(
                lat=station_points["stop_lat"], lon=station_points["stop_lon"], mode="markers",
                marker=dict(
                    size=(station_points["delay_count"].clip(lower=1)) ** 0.5 * 1.5,
                    color="black",
                ),
                text=station_points["base_name"] + " (" + station_points["delay_count"].astype(str) + " delays)",
                hoverinfo="text", name="Subway stations", showlegend=True,
            )
        )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
    st.plotly_chart(fig, width="stretch")

    if show_dev_apps:
        st.caption(
            "Each development-application dot is a proposed or approved project, coloured by "
            "application type — this is where new city planning decisions are being made."
        )
    if show_permits:
        st.caption(
            f"Active building-permit counts are approximate: permits carry no coordinates, so "
            f"each is assigned to a ward by its postal FSA ({n_unmatched:,} could not be mapped). "
            "Bubble size is proportional to √(permit count), placed at the ward centroid."
        )

with col2:
    st.subheader("Predicted duration by ward")
    st.dataframe(
        map_df[["AREA_NAME", "predicted_duration_min"]]
        .sort_values("predicted_duration_min", ascending=False)
        .rename(columns={"AREA_NAME": "Ward", "predicted_duration_min": "Predicted duration (min)"})
        .reset_index(drop=True),
        height=600,
    )

st.divider()
st.subheader("Why this prediction?")
explainer = load_explainer(model)

imp_col, shap_col = st.columns(2)

with imp_col:
    st.caption("Global feature importance (gain) across the whole trained model.")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    st.plotly_chart(
        px.bar(importance, orientation="h", labels={"index": "Feature", "value": "Importance"}),
        width="stretch",
    )

with shap_col:
    explain_ward_name = st.selectbox(
        "Explain prediction for ward", wards.sort_values("ward_num")["AREA_NAME"], key="explain_ward"
    )
    explain_ward_num = wards.loc[wards["AREA_NAME"] == explain_ward_name, "ward_num"].iloc[0]
    explain_row = scenario_df[scenario_df["ward_num"] == explain_ward_num][FEATURE_COLUMNS]
    shap_values = explainer(explain_row)
    st.caption(
        f"SHAP contributions for {explain_ward_name}'s predicted duration "
        f"({scenario_df.loc[scenario_df['ward_num'] == explain_ward_num, 'predicted_duration_min'].iloc[0]:.1f} min)."
    )
    contrib = pd.Series(shap_values.values[0], index=FEATURE_COLUMNS).sort_values()
    st.plotly_chart(
        px.bar(contrib, orientation="h", labels={"index": "Feature", "value": "SHAP value (minutes)"}),
        width="stretch",
    )

st.divider()
st.subheader("Historical anomalies under similar conditions")
st.caption(
    "Ward-hours flagged by an Isolation Forest as unusual given their time-of-day, "
    "weather, and TTC-delay context -- associated with these conditions historically, "
    "not necessarily caused by them."
)
similar = facts[
    (facts["hour"] == hour)
    & (facts["is_snowing"] == is_snowing)
    & facts.get("is_anomaly", False)
]
if len(similar):
    st.dataframe(
        similar[["ward_num", "datetime", "trips_total", "duration_avg", "active_ttc_delays_ward", "active_ttc_delays_citywide"]]
        .sort_values("datetime", ascending=False)
        .head(20)
    )
else:
    st.write("No historical anomalies found for this exact hour/weather combination.")
