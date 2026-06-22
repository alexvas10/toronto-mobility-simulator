"""Download TTC's static GTFS feed (routes, stops, shapes) -- used to locate subway
stations geographically and to draw transit line geometry on the map."""
import os
import zipfile

from src.ckan_utils import download_file, package_show
from src.config import GTFS_PACKAGE, RAW_DIR

GTFS_DIR = RAW_DIR / "gtfs"
GTFS_ZIP_PATH = GTFS_DIR / "opendata_ttc_schedules.zip"


def download_gtfs() -> None:
    pkg = package_show(GTFS_PACKAGE)
    resource = pkg["resources"][0]
    download_file(resource["url"], GTFS_ZIP_PATH)
    with zipfile.ZipFile(GTFS_ZIP_PATH) as zf:
        dest_dir = os.path.realpath(GTFS_DIR)
        for member in zf.namelist():
            member_path = os.path.realpath(os.path.join(dest_dir, member))
            if not member_path.startswith(dest_dir + os.sep):
                raise ValueError(f"Zip slip blocked: {member!r}")
        zf.extractall(GTFS_DIR)
    print(f"downloaded + extracted GTFS feed into {GTFS_DIR}")


if __name__ == "__main__":
    download_gtfs()
