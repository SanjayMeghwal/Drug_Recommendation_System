"""Entry point for Module A (Data Ingestion).

Downloads the DrugBank-derived DDI dataset into data/raw/. See
docs/data_acquisition.md for the data source, license (CC BY-NC 4.0), and
required citation.
"""

import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

BASE_URL = "https://raw.githubusercontent.com/ozkantuncel/GAINET/master/data"

# filename -> minimum expected file size in bytes, a sanity check against a
# truncated or failed download.
FILES = {
    "drug_smiles.csv": 100_000,
    "file_drugs.csv": 25_000_000,
    "ddi_training.csv": 20_000_000,
    "ddi_validation.csv": 2_500_000,
    "ddi_test.csv": 2_500_000,
}


def download_file(filename: str) -> Path:
    """Download filename into data/raw/, skipping it if already present."""
    destination = RAW_DIR / filename
    if destination.exists():
        print(f"  already present, skipping: {filename}")
        return destination

    url = f"{BASE_URL}/{filename}"
    print(f"  downloading: {filename}")
    urllib.request.urlretrieve(url, destination)
    return destination


def verify_file(path: Path, min_size_bytes: int) -> bool:
    """Basic integrity check: the file exists and isn't suspiciously small,
    which would indicate a truncated or failed download.
    """
    return path.exists() and path.stat().st_size >= min_size_bytes


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ingesting DDI dataset into {RAW_DIR}")

    for filename, min_size in FILES.items():
        path = download_file(filename)
        if not verify_file(path, min_size):
            actual_size = path.stat().st_size if path.exists() else 0
            raise RuntimeError(
                f"Integrity check failed for {filename}: "
                f"expected at least {min_size} bytes, got {actual_size}."
            )

    print("All files downloaded and verified.")


if __name__ == "__main__":
    run()
