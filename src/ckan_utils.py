"""Helpers for resolving and downloading resources from Toronto's CKAN open data API."""
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.config import CKAN_BASE

API_BASE = f"{CKAN_BASE}/api/3/action"

_ALLOWED_HOSTS = {
    urlparse(CKAN_BASE).hostname,
    "climate.weather.gc.ca",
}


def _validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"Disallowed download host: {parsed.hostname!r}")


def safe_filename(name: str) -> str:
    """Strip path separators and non-safe characters from an API-supplied filename."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(name).name)


def package_show(package_name: str) -> dict:
    resp = requests.get(f"{API_BASE}/package_show", params={"id": package_name}, timeout=30)
    resp.raise_for_status()
    return resp.json()["result"]


def find_resource(package_name: str, name_contains: str) -> dict:
    """Return the first resource in a package whose name contains the given substring."""
    pkg = package_show(package_name)
    for resource in pkg["resources"]:
        if name_contains.lower() in resource["name"].lower():
            return resource
    raise ValueError(f"No resource matching '{name_contains}' in package '{package_name}'")


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    _validate_download_url(url)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return dest


def datastore_dump_url(resource_id: str) -> str:
    return f"{CKAN_BASE}/datastore/dump/{resource_id}"
