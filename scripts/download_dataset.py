#!/usr/bin/env python3
"""Download, verify, and extract the complete GTSRB archive."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/13741936/files/data.zip/content"
MD5 = "7faafd0d78d831c7eb4d555641a83a62"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path):
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as output:
                total = int(response.headers.get("Content-Length", "0"))
                received = 0
                while block := response.read(1024 * 1024):
                    output.write(block); received += len(block)
                    if total: print(f"\rDownloaded {received / total:.1%}", end="", flush=True)
            print()
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 5:
                raise
            delay = 5 * attempt
            print(f"Download attempt {attempt} failed; retrying in {delay}s...", file=sys.stderr)
            time.sleep(delay)


def validate_extracted(root: Path) -> bool:
    return any(
        (p / "Train").is_dir() and (p / "Test").is_dir()
        or (p / "Final_Training" / "Images").is_dir()
        for p in [root, root / "GTSRB"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/gtsrb"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "data.zip"
    if archive.exists() and md5(archive) != MD5:
        print("Existing archive checksum mismatch; replacing it.", file=sys.stderr)
        archive.unlink()
    if not archive.exists():
        print(f"Downloading GTSRB from {args.url}")
        download(args.url, archive)
    actual = md5(archive)
    if actual != MD5:
        raise RuntimeError(f"MD5 mismatch: expected {MD5}, got {actual}")
    print(f"Verified {archive} (MD5 {actual})")
    if not validate_extracted(args.output_dir):
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad: raise RuntimeError(f"Corrupt archive member: {bad}")
            zf.extractall(args.output_dir)
        print(f"Extracted complete dataset to {args.output_dir}")
    else:
        print("Dataset is already extracted; nothing to do.")
    if not validate_extracted(args.output_dir):
        raise RuntimeError("Extraction completed but no supported GTSRB Train/Test layout was found")


if __name__ == "__main__":
    main()
