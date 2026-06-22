"""Download TTC delay logs for each transit mode (subway/streetcar/bus/lrt).

Each mode's package mixes yearly XLSX archives (older years) with a single
CSV/XML/JSON "since <year>" resource for recent data. We skip readme/code-lookup
resources and keep one file per distinct year/range, preferring CSV over XLSX/XML/JSON
when more than one format covers the same content.
"""
from src.ckan_utils import download_file, package_show, safe_filename
from src.config import RAW_DIR, TTC_DELAY_PACKAGES

DELAYS_DIR = RAW_DIR / "ttc_delays"

SKIP_NAME_SUBSTRINGS = ("readme", "code-descriptions", "delay-codes")
PREFERRED_FORMAT_ORDER = ("CSV", "XLSX", "JSON", "XML")


def _is_data_resource(resource: dict) -> bool:
    name = resource["name"].lower()
    return not any(skip in name for skip in SKIP_NAME_SUBSTRINGS)


def _dedupe_by_base_name(resources: list[dict]) -> list[dict]:
    """When the same dataset is offered in multiple formats, keep one (CSV first)."""
    best: dict[str, dict] = {}
    for resource in resources:
        base = resource["name"].rsplit(".", 1)[0].lower()
        fmt = resource["format"].upper()
        current = best.get(base)
        if current is None:
            best[base] = resource
            continue
        current_rank = PREFERRED_FORMAT_ORDER.index(current["format"].upper()) \
            if current["format"].upper() in PREFERRED_FORMAT_ORDER else len(PREFERRED_FORMAT_ORDER)
        new_rank = PREFERRED_FORMAT_ORDER.index(fmt) if fmt in PREFERRED_FORMAT_ORDER else len(PREFERRED_FORMAT_ORDER)
        if new_rank < current_rank:
            best[base] = resource
    return list(best.values())


def download_mode(mode: str, package_name: str) -> None:
    pkg = package_show(package_name)
    resources = [r for r in pkg["resources"] if _is_data_resource(r)]
    resources = _dedupe_by_base_name(resources)

    mode_dir = DELAYS_DIR / mode
    for resource in resources:
        ext = resource["format"].lower()
        dest = mode_dir / f"{safe_filename(resource['name'])}.{ext}"
        download_file(resource["url"], dest)
        print(f"downloaded {mode}: {resource['name']}")


if __name__ == "__main__":
    for mode, package_name in TTC_DELAY_PACKAGES.items():
        download_mode(mode, package_name)
