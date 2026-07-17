#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REPO_ID = "Dmmm997/DenseUAV"
ARCHIVE_NAME = "DenseUAV.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and optionally extract the DenseUAV dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/DenseUAV"), help="Extracted dataset directory.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/downloads"),
        help="Directory used for the downloaded DenseUAV.zip archive.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Only download the archive.")
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    args = parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}/{ARCHIVE_NAME} ...")
    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ARCHIVE_NAME,
            local_dir=args.download_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    )
    print(f"Downloaded archive: {archive_path}")

    if args.no_extract:
        return

    marker = args.output_dir / ".denseuav_extracted"
    if marker.exists():
        print(f"Extraction marker exists, skipping: {marker}")
        return

    print(f"Extracting to {args.output_dir} ...")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(args.output_dir)
    marker.write_text(str(archive_path), encoding="utf-8")
    print(f"Extracted DenseUAV to: {args.output_dir}")


if __name__ == "__main__":
    main()
