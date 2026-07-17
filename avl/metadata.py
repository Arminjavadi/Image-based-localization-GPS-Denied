from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("image_path", "latitude", "longitude")
OPTIONAL_COLUMNS = ("altitude_m", "heading_deg", "image_id")


@dataclass(frozen=True)
class ReferenceRecord:
    image_path: str
    latitude: float
    longitude: float
    altitude_m: float | None = None
    heading_deg: float | None = None
    image_id: str | None = None


def load_reference_metadata(csv_path: Path, base_dir: Path | None = None) -> list[ReferenceRecord]:
    """Load geo-tagged reference metadata from CSV."""
    df = pd.read_csv(csv_path, dtype={"image_path": "string", "image_id": "string"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {missing}")
    if "expected_image_id" in df.columns and "image_id" not in df.columns:
        raise ValueError(
            "This is a query metadata CSV, not a reference database. "
            "Choose a reference CSV containing an image_id column."
        )

    records: list[ReferenceRecord] = []
    for row in df.itertuples(index=False):
        image_path = str(getattr(row, "image_path"))
        if base_dir is not None and not Path(image_path).is_absolute():
            image_path = str((base_dir / image_path).resolve())

        records.append(
            ReferenceRecord(
                image_path=image_path,
                latitude=float(getattr(row, "latitude")),
                longitude=float(getattr(row, "longitude")),
                altitude_m=_optional_float(row, "altitude_m"),
                heading_deg=_optional_float(row, "heading_deg"),
                image_id=_optional_str(row, "image_id"),
            )
        )
    return records


def _optional_float(row, name: str) -> float | None:
    if name not in row._fields:
        return None
    value = getattr(row, name)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(row, name: str) -> str | None:
    if name not in row._fields:
        return None
    value = getattr(row, name)
    if value is None or pd.isna(value):
        return None
    return str(value)
