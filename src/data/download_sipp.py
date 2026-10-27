"""
src/data/download_sipp.py
Automates the retrieval of the 2014 SIPP SSA Supplement microdata.
"""

from pathlib import Path
import zipfile

import requests
from tqdm import tqdm

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SSA_SUPP_URL = "https://www2.census.gov/programs-surveys/sipp/data/datasets/pu2014ssa_dta.zip"
DTA_NAME = "pu2014ssa.dta"


def download_file(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_filename = dest_dir / url.split("/")[-1]

    if local_filename.exists() and local_filename.stat().st_size > 0:
        print(f"File already exists at {local_filename}")
        return local_filename

    print(f"Downloading {url}...")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(local_filename, "wb") as handle,
            tqdm(
                desc=local_filename.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = handle.write(chunk)
                bar.update(size)

    print(f"Saved raw file to {local_filename}")
    return local_filename


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract the Stata archive into data/raw without altering the zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        dta_members = [name for name in members if name.lower().endswith(".dta")]
        if len(dta_members) != 1:
            raise RuntimeError(f"Expected one .dta member in {archive_path}, found {dta_members or members}")

        member = dta_members[0]
        extracted = dest_dir / Path(member).name
        if extracted.exists() and extracted.stat().st_size > 0:
            print(f"Extracted file already exists at {extracted}")
            return extracted

        print(f"Extracting {member} to {dest_dir}...")
        archive.extract(member, path=dest_dir)
        extracted_path = dest_dir / member
        if extracted_path != extracted and extracted_path.exists():
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted_path.replace(extracted)
        print(f"Extracted {extracted}")
        return extracted


def main() -> None:
    archive_path = download_file(SSA_SUPP_URL, RAW_DATA_DIR)
    dta_path = extract_archive(archive_path, RAW_DATA_DIR)
    if dta_path.name != DTA_NAME:
        print(f"Warning: extracted Stata file is named {dta_path.name}, expected {DTA_NAME}")
    print(f"Ready: {dta_path} exists={dta_path.exists()} size={dta_path.stat().st_size}")


if __name__ == "__main__":
    main()
