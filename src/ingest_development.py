"""Download Toronto's Development Applications and Active Building Permits datasets.

These power the "where is new construction happening / where will new city decisions be
built" map overlays. Both are pulled from the CKAN datastore as CSV dumps. Idempotent --
already-present files are skipped unless force=True.
"""
from src.ckan_utils import datastore_dump_url, datastore_resource, download_file
from src.config import (
    BUILDING_PERMITS_PACKAGE,
    DEVELOPMENT_APPLICATIONS_PACKAGE,
    RAW_DIR,
)

DEV_APPLICATIONS_PATH = RAW_DIR / "development_applications.csv"
BUILDING_PERMITS_PATH = RAW_DIR / "building_permits_active.csv"


def download_development_data(force: bool = False) -> None:
    for package, dest in (
        (DEVELOPMENT_APPLICATIONS_PACKAGE, DEV_APPLICATIONS_PATH),
        (BUILDING_PERMITS_PACKAGE, BUILDING_PERMITS_PATH),
    ):
        resource = datastore_resource(package)
        download_file(datastore_dump_url(resource["id"]), dest, force=force)
        print(f"downloaded {dest.name}")


if __name__ == "__main__":
    download_development_data()
