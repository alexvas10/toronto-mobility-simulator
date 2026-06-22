"""Download PTC per-ward-hour trip archives and the city-wide daily summary_stats table."""
import os
import zipfile

import requests

from src.ckan_utils import datastore_dump_url, download_file, package_show
from src.config import PTC_TRIP_YEARS, PTC_TRIPS_PACKAGE, RAW_DIR

TRIPS_DIR = RAW_DIR / "ptc_trips"


def _safe_extractall(zf: zipfile.ZipFile, dest_dir) -> None:
    dest_dir = os.path.realpath(dest_dir)
    for member in zf.namelist():
        member_path = os.path.realpath(os.path.join(dest_dir, member))
        if not member_path.startswith(dest_dir + os.sep):
            raise ValueError(f"Zip slip blocked: {member!r}")
    zf.extractall(dest_dir)
SUMMARY_STATS_PATH = RAW_DIR / "ptc_summary_stats.csv"


def download_trip_archives() -> None:
    pkg = package_show(PTC_TRIPS_PACKAGE)
    resources_by_name = {r["name"]: r for r in pkg["resources"]}

    for year in PTC_TRIP_YEARS:
        resource = resources_by_name.get(f"trips_{year}.zip")
        if resource is None:
            continue  # not yet published for this year
        zip_path = TRIPS_DIR / f"trips_{year}.zip"
        try:
            download_file(resource["url"], zip_path)
        except requests.HTTPError as exc:
            print(f"skipping trips_{year}: {exc}")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extractall(zf, TRIPS_DIR)
        print(f"downloaded + extracted trips_{year}")


def download_summary_stats() -> None:
    pkg = package_show(PTC_TRIPS_PACKAGE)
    resource = next(r for r in pkg["resources"] if r["name"] == "summary_stats")
    download_file(datastore_dump_url(resource["id"]), SUMMARY_STATS_PATH)
    print("downloaded ptc_summary_stats.csv")


if __name__ == "__main__":
    download_trip_archives()
    download_summary_stats()
