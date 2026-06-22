"""Download Toronto's official ward boundary GeoJSON (WGS84 / EPSG:4326) for the choropleth map."""
from src.ckan_utils import download_file, package_show
from src.config import RAW_DIR, WARD_BOUNDARY_PACKAGE

WARDS_PATH = RAW_DIR / "city_wards_4326.geojson"


def download_ward_boundaries() -> None:
    pkg = package_show(WARD_BOUNDARY_PACKAGE)
    resource = next(
        r for r in pkg["resources"]
        if r["format"].upper() == "GEOJSON" and "4326" in r["name"]
    )
    download_file(resource["url"], WARDS_PATH)
    print("downloaded city_wards_4326.geojson")


if __name__ == "__main__":
    download_ward_boundaries()
