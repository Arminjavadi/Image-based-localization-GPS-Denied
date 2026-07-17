#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DenseRecord:
    image_path: Path
    latitude: float
    longitude: float
    altitude_m: float | None
    image_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DenseUAV metadata to AVL CSV files.")
    parser.add_argument("--dense-root", type=Path, default=Path("data/DenseUAV"), help="Extracted DenseUAV directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/denseuav_avl"), help="Output CSV directory.")
    parser.add_argument(
        "--reference-split",
        choices=["gallery", "train", "all"],
        default="gallery",
        help="Which DenseUAV records to use as the AVL reference database.",
    )
    parser.add_argument(
        "--reference-view",
        choices=["drone", "satellite", "both"],
        default="satellite",
        help="Which view to include in the AVL reference database.",
    )
    parser.add_argument(
        "--query-view",
        choices=["drone", "satellite", "both"],
        default="drone",
        help="Which test query view to export.",
    )
    parser.add_argument("--limit-references", type=int, default=None, help="Optional small subset for smoke tests.")
    parser.add_argument("--limit-queries", type=int, default=None, help="Optional small query subset for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_root = find_dense_root(args.dense_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gps_files = find_gps_files(dense_root)
    if not gps_files:
        raise SystemExit(f"No Dense_GPS*.txt files found under {dense_root}")

    print(f"DenseUAV root: {dense_root}")
    print("GPS files:")
    for key, path in gps_files.items():
        print(f"  {key}: {path}")

    all_records = load_records(dense_root, gps_files)
    if not all_records:
        raise SystemExit("No DenseUAV records could be resolved to existing images")

    references = select_records(
        all_records,
        split=args.reference_split,
        view=args.reference_view,
        limit=args.limit_references,
        for_query=False,
    )
    queries = select_records(
        all_records,
        split="test",
        view=args.query_view,
        limit=args.limit_queries,
        for_query=True,
    )

    ref_csv = args.output_dir / f"references_{args.reference_split}_{args.reference_view}.csv"
    query_csv = args.output_dir / f"queries_test_{args.query_view}.csv"
    write_reference_csv(ref_csv, references)
    write_query_csv(query_csv, queries)

    print(f"Wrote {len(references)} reference rows -> {ref_csv}")
    print(f"Wrote {len(queries)} query rows -> {query_csv}")
    print("\nExample benchmark command:")
    print(
        "python scripts/benchmark_kpis.py "
        f"--metadata {ref_csv} "
        "--base-dir / "
        f"--query-metadata {query_csv} "
        "--query-base-dir / "
        "--model-config mixvpr:4096 "
        "--index-types hnsw flat "
        "--top-k 5 "
        "--repeats 3 "
        "--output-json artifacts/denseuav_kpis.json "
        "--output-csv artifacts/denseuav_kpis.csv"
    )


def find_dense_root(path: Path) -> Path:
    path = path.resolve()
    candidates = [path, path / "DenseUAV"]
    candidates.extend(p for p in path.rglob("Dense_GPS_ALL.txt") if p.is_file())
    for candidate in candidates:
        if candidate.is_file():
            candidate = candidate.parent
        if (candidate / "Dense_GPS_ALL.txt").exists() or any(candidate.glob("Dense_GPS*.txt")):
            return candidate
    return path


def find_gps_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("Dense_GPS*.txt")):
        name = path.stem.lower()
        if "all" in name:
            files["all"] = path
        elif "train" in name:
            files["train"] = path
        elif "test" in name:
            files["test"] = path
        else:
            files[name] = path
    return files


def load_records(root: Path, gps_files: dict[str, Path]) -> list[DenseRecord]:
    records_by_path: dict[Path, DenseRecord] = {}
    ordered_files = [gps_files[key] for key in ("all", "train", "test") if key in gps_files]
    ordered_files.extend(path for key, path in gps_files.items() if key not in {"all", "train", "test"})

    for gps_file in ordered_files:
        for raw_line in gps_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for record in parse_gps_line(root, line):
                records_by_path[record.image_path] = record

    return sorted(records_by_path.values(), key=lambda record: str(record.image_path))


def parse_gps_line(root: Path, line: str) -> list[DenseRecord]:
    parts = re.split(r"[\s,]+", line)
    if len(parts) < 3:
        return []

    numeric_start = None
    for index, value in enumerate(parts):
        if is_coordinate(value):
            numeric_start = index
            break
    if numeric_start is None or numeric_start < 1 or len(parts) < numeric_start + 2:
        return []

    image_token = " ".join(parts[:numeric_start])
    first_coord = parse_coordinate(parts[numeric_start])
    second_coord = parse_coordinate(parts[numeric_start + 1])
    if first_coord is None or second_coord is None:
        return []

    latitude, longitude = lat_lon_from_coordinates(first_coord, second_coord)
    altitude = float(parts[numeric_start + 2]) if len(parts) > numeric_start + 2 and is_float(parts[numeric_start + 2]) else None

    image_paths = resolve_image_paths(root, image_token)
    return [
        DenseRecord(
            image_path=image_path,
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude,
            image_id=make_location_id(image_path),
        )
        for image_path in image_paths
    ]


def resolve_image_paths(root: Path, image_token: str) -> list[Path]:
    candidate = Path(image_token)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        token = image_token.lstrip("./")
        candidates.extend([root / candidate, root / token])
        candidates.extend(denseuav_path_aliases(root, token))

    resolved: list[Path] = []
    for path in candidates:
        if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
            resolved.append(path.resolve())

    resolved.extend(resolve_companion_drone_images(root, image_token))

    deduped = sorted(set(resolved), key=lambda path: str(path))
    return deduped


def denseuav_path_aliases(root: Path, token: str) -> list[Path]:
    aliases = []
    replacements = {
        "test/satellite/": "test/gallery_satellite/",
        "test/drone/": "test/query_drone/",
    }
    for old, new in replacements.items():
        if old in token:
            aliases.append(root / token.replace(old, new, 1))
    return aliases


def resolve_companion_drone_images(root: Path, image_token: str) -> list[Path]:
    token_parts = Path(image_token).parts
    if len(token_parts) < 3 or "satellite" not in image_token.lower():
        return []

    split = token_parts[0].lower()
    location_id = token_parts[2]
    if split == "train":
        drone_dir = root / "train" / "drone" / location_id
    elif split == "test":
        drone_dir = root / "test" / "query_drone" / location_id
    else:
        return []

    if not drone_dir.exists():
        return []
    return sorted(path.resolve() for path in drone_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def make_location_id(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"drone", "satellite", "query_drone", "gallery_satellite"} and index + 1 < len(parts):
            return parts[index + 1]
    return path.stem


def select_records(
    records: list[DenseRecord],
    split: str,
    view: str,
    limit: int | None,
    for_query: bool,
) -> list[DenseRecord]:
    selected = [record for record in records if matches_split(record.image_path, split, for_query)]
    selected = [record for record in selected if matches_view(record.image_path, view)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def matches_split(path: Path, split: str, for_query: bool) -> bool:
    parts = {part.lower() for part in path.parts}
    path_text = str(path).lower()
    if split == "all":
        return True
    if split == "train":
        return "train" in parts
    if split == "gallery":
        return "gallery" in path_text or ("test" in parts and not for_query and "query" not in path_text)
    if split == "test":
        return "query" in path_text if for_query else "test" in parts
    return False


def matches_view(path: Path, view: str) -> bool:
    if view == "both":
        return True
    path_text = str(path).lower()
    if view == "drone":
        return "drone" in path_text or "uav" in path_text
    if view == "satellite":
        return "satellite" in path_text
    return False


def write_reference_csv(path: Path, records: list[DenseRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "latitude", "longitude", "altitude_m", "image_id"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "altitude_m": record.altitude_m,
                    "image_id": record.image_id,
                }
            )


def write_query_csv(path: Path, records: list[DenseRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "latitude", "longitude", "expected_image_id"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "expected_image_id": record.image_id,
                }
            )


def is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def is_coordinate(value: str) -> bool:
    return parse_coordinate(value) is not None


def parse_coordinate(value: str) -> tuple[str | None, float] | None:
    if not value:
        return None
    axis = value[0].upper()
    if axis in {"N", "S", "E", "W"}:
        number = value[1:]
        if not is_float(number):
            return None
        parsed = float(number)
        if axis in {"S", "W"}:
            parsed = -parsed
        return axis, parsed
    if is_float(value):
        return None, float(value)
    return None


def lat_lon_from_coordinates(
    first: tuple[str | None, float],
    second: tuple[str | None, float],
) -> tuple[float, float]:
    first_axis, first_value = first
    second_axis, second_value = second
    coords = {first_axis: first_value, second_axis: second_value}

    if first_axis in {"E", "W"} and second_axis in {"N", "S"}:
        return second_value, first_value
    if first_axis in {"N", "S"} and second_axis in {"E", "W"}:
        return first_value, second_value
    if "N" in coords or "S" in coords or "E" in coords or "W" in coords:
        latitude = coords.get("N", coords.get("S"))
        longitude = coords.get("E", coords.get("W"))
        if latitude is not None and longitude is not None:
            return latitude, longitude

    return first_value, second_value


if __name__ == "__main__":
    main()
