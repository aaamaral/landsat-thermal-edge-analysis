"""Create a Zenodo DRAFT of the aggregated derived dataset (never publishes).

Steps: create a deposition -> upload README.md, MANIFEST.csv, verify_deposit.py and one zip per
data folder (stored, the parquet files are already compressed; folder paths are kept inside the
zip, so unzipping into one directory restores the layout described in README.md) -> set metadata
from dataset_metadata.json. Review and publish the draft on the Zenodo web page.

Requires a personal access token with scopes deposit:write and deposit:actions:
  Zenodo -> Account -> Applications -> Personal access tokens.
The token is read from the environment, never from the command line or a file:
  PowerShell:  $env:ZENODO_TOKEN = "..." ; python revision/11_zenodo/upload_dataset_zenodo.py
  Test first on the sandbox (separate account/token at sandbox.zenodo.org):  --sandbox

Usage:
  python revision/11_zenodo/upload_dataset_zenodo.py [--sandbox] [--deposit DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DEPOSIT = ROOT / "revision" / "data" / "zenodo_deposit"
LOOSE = ("README.md", "MANIFEST.csv", "verify_deposit.py")
FOLDERS = ("contrasts", "dtec", "polygons", "results", "polygon_ring")


def build_zips(deposit: Path, work: Path) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    out = []
    for name in FOLDERS:
        src = deposit / name
        if not src.is_dir():
            continue
        z = work / f"{name}.zip"
        with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(deposit).as_posix())
        out.append(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--deposit", type=Path, default=DEFAULT_DEPOSIT)
    ap.add_argument("--dry-run", action="store_true", help="build the zips and list the files only")
    a = ap.parse_args()

    files = [a.deposit / n for n in LOOSE]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing in {a.deposit}: {missing} (run export_derived_data.py first)")
    files += build_zips(a.deposit, a.deposit.parent / "zenodo_upload")
    for f in files:
        print(f"  {f.name:<22} {f.stat().st_size / 1e6:>10,.1f} MB")
    if a.dry_run:
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set the ZENODO_TOKEN environment variable")
    api = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    h = {"Authorization": f"Bearer {token}"}

    r = requests.post(f"{api}/deposit/depositions", json={}, headers=h, timeout=60)
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    print(f"draft created: id {dep['id']}")
    for f in files:
        with open(f, "rb") as fh:
            u = requests.put(f"{bucket}/{f.name}", data=fh, headers=h, timeout=None)
        u.raise_for_status()
        print(f"  uploaded {f.name} (checksum {u.json().get('checksum')})")
    meta = json.loads((HERE / "dataset_metadata.json").read_text(encoding="utf-8"))
    meta["prereserve_doi"] = True  # DOI known before publishing, so it can be cited in the manuscript
    r = requests.put(f"{api}/deposit/depositions/{dep['id']}", json={"metadata": meta},
                     headers=h, timeout=60)
    r.raise_for_status()
    print(f"metadata set. Review and publish at: {dep['links']['html']}")
    print("Reserved DOI (after publishing):",
          r.json().get("metadata", {}).get("prereserve_doi", {}).get("doi"))


if __name__ == "__main__":
    main()
