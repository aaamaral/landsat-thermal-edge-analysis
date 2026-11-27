"""Create (or resume) a Zenodo DRAFT of the aggregated derived dataset (never publishes).

Steps: create a deposition (or reuse one with --resume ID) -> upload README.md, MANIFEST.csv,
verify_deposit.py and the zips (stored, the parquet files are already compressed; folder paths are
kept inside each zip, so unzipping everything into one directory restores the layout described in
README.md) -> set metadata from dataset_metadata.json. Review and publish the draft on the web page.

Zips: one per small folder (contrasts, dtec, polygons, results) and one per year for the large
scene x polygon x ring table (polygon_ring_{year}.zip), so that a failed transfer of a large file
is retried on its own. On --resume, files already in the draft with the same MD5 are skipped and
remote files that are not part of the upload (e.g. a broken partial file) are deleted.

Requires a personal access token with scopes deposit:write and deposit:actions:
  Zenodo -> Account -> Applications -> Personal access tokens.
The token is read from the environment, never from the command line or a file:
  PowerShell:  $env:ZENODO_TOKEN = "..." ; python revision/11_zenodo/upload_dataset_zenodo.py
  Test first on the sandbox (separate account/token at sandbox.zenodo.org):  --sandbox

Usage:
  python revision/11_zenodo/upload_dataset_zenodo.py [--sandbox] [--deposit DIR] [--dry-run]
  python revision/11_zenodo/upload_dataset_zenodo.py --resume DEPOSITION_ID
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DEPOSIT = ROOT / "revision" / "data" / "zenodo_deposit"
LOOSE = ("README.md", "MANIFEST.csv", "verify_deposit.py")
FOLDERS = ("contrasts", "dtec", "polygons", "results")   # one zip per folder
PER_FILE = ("polygon_ring",)                             # one zip per file (large)
TRIES = 4


def _zip(z: Path, members: list[Path], deposit: Path) -> Path:
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in members:
            zf.write(f, f.relative_to(deposit).as_posix())
    return z


def build_zips(deposit: Path, work: Path) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.zip"):
        old.unlink()
    out = []
    for name in FOLDERS:
        src = deposit / name
        if src.is_dir():
            out.append(_zip(work / f"{name}.zip",
                            [f for f in sorted(src.rglob("*")) if f.is_file()], deposit))
    for name in PER_FILE:
        src = deposit / name
        if src.is_dir():
            for f in sorted(src.glob("*.parquet")):
                out.append(_zip(work / f"{f.stem}.zip", [f], deposit))
    return out


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def put_file(bucket: str, f: Path, h: dict) -> str:
    for attempt in range(1, TRIES + 1):
        try:
            with open(f, "rb") as fh:
                u = requests.put(f"{bucket}/{f.name}", data=fh, headers=h, timeout=None)
            u.raise_for_status()
            return u.json().get("checksum", "")
        except requests.RequestException as e:
            print(f"  {f.name}: attempt {attempt}/{TRIES} failed ({e})", flush=True)
            if attempt == TRIES:
                raise
            time.sleep(30 * attempt)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--deposit", type=Path, default=DEFAULT_DEPOSIT)
    ap.add_argument("--resume", type=int, help="id of an existing draft to complete")
    ap.add_argument("--dry-run", action="store_true", help="build the zips and list the files only")
    a = ap.parse_args()

    files = [a.deposit / n for n in LOOSE]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing in {a.deposit}: {missing} (run export_derived_data.py first)")
    print("building zips ...", flush=True)
    files += build_zips(a.deposit, a.deposit.parent / "zenodo_upload")
    for f in files:
        print(f"  {f.name:<26} {f.stat().st_size / 1e6:>10,.1f} MB")
    if a.dry_run:
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set the ZENODO_TOKEN environment variable")
    api = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    h = {"Authorization": f"Bearer {token}"}

    if a.resume:
        r = requests.get(f"{api}/deposit/depositions/{a.resume}", headers=h, timeout=60)
    else:
        r = requests.post(f"{api}/deposit/depositions", json={}, headers=h, timeout=60)
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"draft {'resumed' if a.resume else 'created'}: id {dep_id}", flush=True)

    remote = {}
    if a.resume:
        r = requests.get(f"{api}/deposit/depositions/{dep_id}/files", headers=h, timeout=60)
        r.raise_for_status()
        remote = {x["filename"]: x for x in r.json()}
        wanted = {f.name for f in files}
        for name, x in remote.items():
            if name not in wanted:
                requests.delete(f"{api}/deposit/depositions/{dep_id}/files/{x['id']}",
                                headers=h, timeout=60).raise_for_status()
                print(f"  removed remote file not in this upload: {name}", flush=True)

    for f in files:
        x = remote.get(f.name)
        if x and x.get("checksum", "").replace("md5:", "") == md5(f):
            print(f"  already in draft, same MD5: {f.name}", flush=True)
            continue
        if x:  # same name, different content: replace
            requests.delete(f"{api}/deposit/depositions/{dep_id}/files/{x['id']}",
                            headers=h, timeout=60).raise_for_status()
        print(f"  uploading {f.name} ({f.stat().st_size / 1e6:,.1f} MB) ...", flush=True)
        checksum = put_file(bucket, f, h)
        local = md5(f)
        status = "OK" if checksum.replace("md5:", "") == local else f"MISMATCH (local {local})"
        print(f"  uploaded {f.name} (checksum {checksum}) {status}", flush=True)

    meta = json.loads((HERE / "dataset_metadata.json").read_text(encoding="utf-8"))
    meta["prereserve_doi"] = True  # DOI known before publishing, so it can be cited in the manuscript
    r = requests.put(f"{api}/deposit/depositions/{dep_id}", json={"metadata": meta},
                     headers=h, timeout=60)
    r.raise_for_status()
    print(f"metadata set. Review and publish at: {dep['links']['html']}")
    print("Reserved DOI (after publishing):",
          r.json().get("metadata", {}).get("prereserve_doi", {}).get("doi"))


if __name__ == "__main__":
    main()
