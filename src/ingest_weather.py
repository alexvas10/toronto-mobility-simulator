"""Download hourly historical weather for Toronto Pearson Airport from Environment Canada.

The bulk_data endpoint only returns one month of hourly data per request
(confirmed live: https://climate.weather.gc.ca/climate_data/bulk_data_e.html),
so we loop month by month and concatenate.
"""
import datetime as dt

from src.ckan_utils import download_file
from src.config import RAW_DIR

STATION_ID = 51459  # Toronto Pearson Intl A (current station, active since 2019)
BULK_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID={station_id}&Year={year}&Month={month}&Day=1"
    "&timeframe=1&submit=Download+Data"
)
WEATHER_DIR = RAW_DIR / "weather"

START_YEAR_MONTH = (2018, 1)


def _month_range(start: tuple[int, int], end: tuple[int, int]):
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def download_weather() -> None:
    today = dt.date.today()
    end = (today.year, today.month)
    for year, month in _month_range(START_YEAR_MONTH, end):
        url = BULK_URL.format(station_id=STATION_ID, year=year, month=month)
        dest = WEATHER_DIR / f"weather_{year}_{month:02d}.csv"
        download_file(url, dest)
    print(f"downloaded weather files into {WEATHER_DIR}")


if __name__ == "__main__":
    download_weather()
