"""Download raw NBI delimited files from FHWA.

Only handles legacy-coding years (<= 2024) for now — see docs/data_sources.md
for why 2025+ (SNBI format) is deliberately out of scope until harmonization
is verified.
"""

from pathlib import Path

import requests

FHWA_BASE_URL = "https://www.fhwa.dot.gov/bridge/nbi"
LAST_LEGACY_YEAR = 2024

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_state_year(state_code: str, year: int, out_dir: Path = RAW_DATA_DIR) -> Path:
    """Download one state's NBI delimited file for one year.

    state_code: two-letter USPS code, e.g. "DE"
    year: inspection year, must be <= LAST_LEGACY_YEAR
    Returns the local path the file was saved to.
    """
    if year > LAST_LEGACY_YEAR:
        raise ValueError(
            f"Year {year} is SNBI-format (2025+); legacy download path only "
            f"supports up to {LAST_LEGACY_YEAR}. See docs/data_sources.md."
        )

    yy = str(year)[-2:]
    filename = f"{state_code.upper()}{yy}.txt"
    url = f"{FHWA_BASE_URL}/{year}/delimited/{filename}"

    dest_dir = out_dir / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest_path.write_bytes(response.content)

    return dest_path


if __name__ == "__main__":
    path = download_state_year("DE", 2023)
    print(f"Downloaded to: {path}")
