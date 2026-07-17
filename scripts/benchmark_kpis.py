#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from avl.config import AVLConfig, IndexType, ModelName
from avl.encoder import VPREncoder
from avl.geo import haversine_m, weighted_geo_fusion
from avl.index import HashIndex, SearchResult
from avl.metadata import ReferenceRecord, load_reference_metadata


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GEO_FUSION_TOP_K = 5


@dataclass(frozen=True)
class ModelSpec:
    model: ModelName
    descriptor_dim: int
    cosplace_backbone: str = "ResNet101"

    @property
    def label(self) -> str:
        if self.model == "cosplace":
            return f"{self.model}:{self.descriptor_dim}:{self.cosplace_backbone}"
        return f"{self.model}:{self.descriptor_dim}"


@dataclass(frozen=True)
class QueryRecord:
    image_path: str
    latitude: float | None = None
    longitude: float | None = None
    expected_image_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure AVL KPIs for offline feature/index construction and online "
            "query feature/search latency."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Reference CSV with geo-tagged images.")
    parser.add_argument("--base-dir", type=Path, default=None, help="Base directory for relative reference paths.")
    parser.add_argument(
        "--query",
        type=Path,
        action="append",
        default=[],
        help="Query image file or directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=None,
        help=(
            "Optional query CSV. Requires image_path and may include latitude, "
            "longitude, expected_image_id."
        ),
    )
    parser.add_argument(
        "--query-base-dir",
        type=Path,
        default=None,
        help="Base directory for relative query paths. Defaults to --base-dir.",
    )
    parser.add_argument(
        "--model-config",
        action="append",
        default=[],
        metavar="MODEL:DIM[:BACKBONE]",
        help=(
            "Model benchmark target, for example mixvpr:4096, mixvpr:512, "
            "or cosplace:2048:ResNet101. Can be repeated."
        ),
    )
    parser.add_argument(
        "--index-types",
        nargs="+",
        choices=["hnsw", "ivfpq", "flat"],
        default=["hnsw", "flat"],
        help="FAISS index types to benchmark for each model config.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches used for search/fusion.")
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Recall cutoffs to evaluate when expected_image_id is available.",
    )
    parser.add_argument(
        "--query-rotations",
        type=int,
        choices=[1, 4],
        default=4,
        help="Encode one orientation or search 0/90/180/270-degree query rotations.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device. Falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--batch-size", type=int, default=16, help="Reference encoding batch size.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed query feature extraction runs before measuring online latency.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed repeats per query for online feature and search latency.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write full KPI report as JSON.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Write flattened KPI summary as CSV.")
    parser.add_argument(
        "--no-localizations",
        action="store_true",
        help="Omit per-query visual matches from reports generated for large query CSVs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable reference encoding progress bars.")
    return parser.parse_args()


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("Expected MODEL:DIM or MODEL:DIM:BACKBONE")

    model = parts[0].lower()
    if model not in {"denseuav-vit", "mixvpr", "cosplace"}:
        raise argparse.ArgumentTypeError("MODEL must be denseuav-vit, mixvpr, or cosplace")

    try:
        descriptor_dim = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("DIM must be an integer") from exc

    backbone = parts[2] if len(parts) == 3 else "ResNet101"
    return ModelSpec(model=model, descriptor_dim=descriptor_dim, cosplace_backbone=backbone)  # type: ignore[arg-type]


def resolve_path(path_value: str | Path, base_dir: Path | None) -> str:
    path = Path(path_value)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_query_records(args: argparse.Namespace) -> list[QueryRecord]:
    query_base_dir = args.query_base_dir or args.base_dir

    if args.query_metadata is not None:
        df = pd.read_csv(
            args.query_metadata,
            dtype={"image_path": "string", "expected_image_id": "string"},
        )
        if "image_path" not in df.columns:
            raise ValueError("Query metadata CSV must contain an image_path column")

        records: list[QueryRecord] = []
        for row in df.itertuples(index=False):
            records.append(
                QueryRecord(
                    image_path=resolve_path(getattr(row, "image_path"), query_base_dir),
                    latitude=_optional_float(row, "latitude"),
                    longitude=_optional_float(row, "longitude"),
                    expected_image_id=_optional_str(row, "expected_image_id"),
                )
            )
        return records

    query_paths: list[Path] = []
    for query in args.query:
        query_path = query
        if query_base_dir is not None and not query_path.is_absolute():
            query_path = query_base_dir / query_path

        if query_path.is_dir():
            query_paths.extend(p for p in sorted(query_path.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            query_paths.append(query_path)

    if not query_paths:
        raise ValueError("Pass at least one --query image/directory or --query-metadata CSV")

    return [QueryRecord(image_path=str(path.resolve())) for path in query_paths]


def _optional_float(row: Any, name: str) -> float | None:
    if name not in row._fields:
        return None
    value = getattr(row, name)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(row: Any, name: str) -> str | None:
    if name not in row._fields:
        return None
    value = getattr(row, name)
    if value is None or pd.isna(value):
        return None
    return str(value)


def timed_seconds(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def stats_ms(samples_s: list[float]) -> dict[str, float | int | None]:
    if not samples_s:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    samples_ms = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return {
        "count": int(samples_ms.size),
        "mean": float(samples_ms.mean()),
        "median": float(np.median(samples_ms)),
        "p95": float(np.percentile(samples_ms, 95)),
        "min": float(samples_ms.min()),
        "max": float(samples_ms.max()),
    }


def summarize_error_m(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def reference_id(record: ReferenceRecord) -> str:
    return record.image_id or Path(record.image_path).stem


def evaluate_quality(
    records: list[ReferenceRecord],
    queries: list[QueryRecord],
    search_results: list[SearchResult],
    recall_ks: list[int],
    primary_k: int,
) -> dict[str, Any]:
    recall_hits = {k: 0 for k in recall_ks}
    id_eval_count = 0
    top1_errors_m: list[float] = []
    fused_errors_m: list[float] = []

    for query, search in zip(queries, search_results):
        matched_records = [records[int(idx)] for idx in search.indices]
        if not matched_records:
            continue

        if query.expected_image_id is not None:
            id_eval_count += 1
            matched_ids = [reference_id(record) for record in matched_records]
            for k in recall_ks:
                recall_hits[k] += int(query.expected_image_id in matched_ids[:k])

        if query.latitude is not None and query.longitude is not None:
            best = matched_records[0]
            top1_errors_m.append(haversine_m(query.latitude, query.longitude, best.latitude, best.longitude))

            fusion_records = matched_records[:GEO_FUSION_TOP_K]
            fusion_scores = search.scores[:GEO_FUSION_TOP_K]
            latitudes = np.array([record.latitude for record in fusion_records], dtype=np.float64)
            longitudes = np.array([record.longitude for record in fusion_records], dtype=np.float64)
            weights = np.array(fusion_scores, dtype=np.float64)
            pose = weighted_geo_fusion(latitudes, longitudes, weights)
            fused_errors_m.append(haversine_m(query.latitude, query.longitude, pose.latitude, pose.longitude))

    recall_at = {
        str(k): (recall_hits[k] / id_eval_count) if id_eval_count else None
        for k in recall_ks
    }
    quality: dict[str, Any] = {
        "id_eval_count": id_eval_count,
        "top1_hit_rate": recall_at.get("1"),
        "recall_at_k": recall_at.get(str(primary_k)),
        "recall_at": recall_at,
        "top1_error_m": summarize_error_m(top1_errors_m),
        "fused_error_m": summarize_error_m(fused_errors_m),
    }
    return quality


def build_localization_results(
    records: list[ReferenceRecord],
    queries: list[QueryRecord],
    search_results: list[SearchResult],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for query, search in zip(queries, search_results):
        matches: list[dict[str, Any]] = []
        matched_records: list[ReferenceRecord] = []
        indices = search.indices[:top_k] if top_k is not None else search.indices
        scores = search.scores[:top_k] if top_k is not None else search.scores
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            record = records[int(idx)]
            matched_records.append(record)
            matches.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "image_path": record.image_path,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "altitude_m": record.altitude_m,
                    "heading_deg": record.heading_deg,
                    "image_id": reference_id(record),
                }
            )

        estimated_position = None
        error_m = None
        if matched_records:
            fusion_records = matched_records[:GEO_FUSION_TOP_K]
            fusion_scores = search.scores[:GEO_FUSION_TOP_K]
            latitudes = np.array([record.latitude for record in fusion_records], dtype=np.float64)
            longitudes = np.array([record.longitude for record in fusion_records], dtype=np.float64)
            altitudes = np.array(
                [
                    record.altitude_m if record.altitude_m is not None else np.nan
                    for record in fusion_records
                ],
                dtype=np.float64,
            )
            pose = weighted_geo_fusion(
                latitudes,
                longitudes,
                np.asarray(fusion_scores, dtype=np.float64),
                altitudes=altitudes,
            )
            estimated_position = {
                "latitude": pose.latitude,
                "longitude": pose.longitude,
                "altitude_m": pose.altitude_m,
            }
            if query.latitude is not None and query.longitude is not None:
                error_m = haversine_m(
                    query.latitude,
                    query.longitude,
                    pose.latitude,
                    pose.longitude,
                )

        results.append(
            {
                "query_image_path": query.image_path,
                "query_ground_truth": (
                    {
                        "latitude": query.latitude,
                        "longitude": query.longitude,
                        "expected_image_id": query.expected_image_id,
                    }
                    if query.latitude is not None and query.longitude is not None
                    else None
                ),
                "estimated_position": estimated_position,
                "fusion_match_count": min(len(matches), GEO_FUSION_TOP_K),
                "localization_error_m": error_m,
                "matches": matches,
            }
        )

    return results


def make_config(spec: ModelSpec, index_type: IndexType, args: argparse.Namespace) -> AVLConfig:
    return AVLConfig(
        model=spec.model,
        descriptor_dim=spec.descriptor_dim,
        cosplace_backbone=spec.cosplace_backbone,
        device=args.device,
        batch_size=args.batch_size,
        query_rotations=args.query_rotations,
        index_type=index_type,
        top_k=args.top_k,
    )


def benchmark_model(
    spec: ModelSpec,
    index_types: list[IndexType],
    reference_records: list[ReferenceRecord],
    query_records: list[QueryRecord],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    config = make_config(spec, index_types[0], args)
    encoder, model_load_s = timed_seconds(VPREncoder, config)

    for _ in range(max(0, args.warmup)):
        encoder.encode_query(query_records[0].image_path)

    reference_paths = [record.image_path for record in reference_records]
    descriptors, reference_encode_s = timed_seconds(
        encoder.encode_paths,
        reference_paths,
        show_progress=not args.quiet,
    )

    query_descriptors: list[np.ndarray] = []
    query_encode_times_s: list[float] = []
    for query in query_records:
        query_descriptor: np.ndarray | None = None
        for repeat in range(max(1, args.repeats)):
            descriptor, elapsed_s = timed_seconds(encoder.encode_query, query.image_path)
            query_encode_times_s.append(elapsed_s)
            if repeat == 0:
                query_descriptor = descriptor
        if query_descriptor is None:
            raise RuntimeError("No query descriptor was produced")
        query_descriptors.append(query_descriptor)

    runs: list[dict[str, Any]] = []
    for index_type in index_types:
        run: dict[str, Any] = {
            "model": spec.model,
            "descriptor_dim": spec.descriptor_dim,
            "model_label": spec.label,
            "cosplace_backbone": spec.cosplace_backbone,
            "index_type": index_type,
            "error": None,
            "model_load_s": model_load_s,
            "offline": {
                "reference_count": len(reference_records),
                "reference_encode_s": reference_encode_s,
                "reference_encode_ms_per_image": (reference_encode_s / len(reference_records)) * 1000.0,
                "reference_images_per_s": len(reference_records) / reference_encode_s if reference_encode_s else None,
                "descriptor_memory_mb": descriptors.nbytes / (1024.0 * 1024.0),
            },
            "online": {
                "query_count": len(query_records),
                "query_encode_ms": stats_ms(query_encode_times_s),
            },
            "quality": {},
        }

        try:
            index_config = make_config(spec, index_type, args)
            index = HashIndex(index_config)
            _, build_s = timed_seconds(index.build, descriptors.copy(), reference_records)
            run["offline"]["faiss_build_s"] = build_s
            run["offline"]["faiss_build_ms_per_image"] = (build_s / len(reference_records)) * 1000.0
            run["offline"]["faiss_index_size_mb"] = faiss.serialize_index(index.index).nbytes / (1024.0 * 1024.0)

            search_times_s: list[float] = []
            search_results: list[SearchResult] = []
            retrieval_k = max(args.top_k, *args.recall_k)
            for descriptor in query_descriptors:
                first_result: SearchResult | None = None
                for repeat in range(max(1, args.repeats)):
                    result, elapsed_s = timed_seconds(
                        index.search_orientations,
                        descriptor,
                        retrieval_k,
                    )
                    search_times_s.append(elapsed_s)
                    if repeat == 0:
                        first_result = result
                if first_result is None:
                    raise RuntimeError("No search result was produced")
                search_results.append(first_result)

            search_stats = stats_ms(search_times_s)
            run["online"]["search_ms"] = search_stats
            query_encode_mean = run["online"]["query_encode_ms"]["mean"] or 0.0
            search_mean = search_stats["mean"] or 0.0
            run["online"]["encode_plus_search_ms_mean"] = query_encode_mean + search_mean
            run["online"]["query_feature_images_per_s"] = (
                1000.0 / query_encode_mean if query_encode_mean else None
            )
            run["online"]["search_queries_per_s"] = 1000.0 / search_mean if search_mean else None
            total_mean = query_encode_mean + search_mean
            run["online"]["online_queries_per_s"] = 1000.0 / total_mean if total_mean else None
            run["quality"] = evaluate_quality(
                reference_records,
                query_records,
                search_results,
                args.recall_k,
                args.top_k,
            )
            run["localizations"] = (
                []
                if args.no_localizations
                else build_localization_results(
                    reference_records,
                    query_records,
                    search_results,
                    top_k=args.top_k,
                )
            )
        except Exception as exc:  # Keep benchmarking other index/model combinations.
            run["error"] = f"{type(exc).__name__}: {exc}"

        runs.append(run)

    return runs


def flatten_run(run: dict[str, Any]) -> dict[str, Any]:
    offline = run.get("offline", {})
    online = run.get("online", {})
    query_encode = online.get("query_encode_ms", {})
    search = online.get("search_ms", {})
    quality = run.get("quality", {})
    recall_at = quality.get("recall_at", {})
    top1_error = quality.get("top1_error_m", {})
    fused_error = quality.get("fused_error_m", {})

    return {
        "model_label": run.get("model_label"),
        "model": run.get("model"),
        "descriptor_dim": run.get("descriptor_dim"),
        "index_type": run.get("index_type"),
        "error": run.get("error"),
        "reference_count": offline.get("reference_count"),
        "query_count": online.get("query_count"),
        "model_load_s": run.get("model_load_s"),
        "reference_encode_s": offline.get("reference_encode_s"),
        "reference_encode_ms_per_image": offline.get("reference_encode_ms_per_image"),
        "reference_images_per_s": offline.get("reference_images_per_s"),
        "faiss_build_s": offline.get("faiss_build_s"),
        "faiss_build_ms_per_image": offline.get("faiss_build_ms_per_image"),
        "descriptor_memory_mb": offline.get("descriptor_memory_mb"),
        "faiss_index_size_mb": offline.get("faiss_index_size_mb"),
        "query_encode_ms_mean": query_encode.get("mean"),
        "query_encode_ms_p95": query_encode.get("p95"),
        "query_feature_images_per_s": online.get("query_feature_images_per_s"),
        "search_ms_mean": search.get("mean"),
        "search_ms_p95": search.get("p95"),
        "search_queries_per_s": online.get("search_queries_per_s"),
        "encode_plus_search_ms_mean": online.get("encode_plus_search_ms_mean"),
        "online_queries_per_s": online.get("online_queries_per_s"),
        "top1_hit_rate": quality.get("top1_hit_rate"),
        "recall_at_k": quality.get("recall_at_k"),
        "recall_at_1": recall_at.get("1"),
        "recall_at_5": recall_at.get("5"),
        "recall_at_10": recall_at.get("10"),
        "top1_error_m_mean": top1_error.get("mean"),
        "fused_error_m_mean": fused_error.get("mean"),
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_run(run) for run in runs]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(runs: list[dict[str, Any]]) -> None:
    header = (
        "model",
        "index",
        "ref enc ms/img",
        "build s",
        "query enc ms",
        "search ms",
        "online ms",
        "R@1",
        "R@5",
        "R@10",
        "error",
    )
    print("\nKPI summary")
    print(
        f"{header[0]:<24} {header[1]:<7} {header[2]:>14} {header[3]:>9} "
        f"{header[4]:>13} {header[5]:>10} {header[6]:>10} {header[7]:>8} "
        f"{header[8]:>8} {header[9]:>8} {header[10]}"
    )

    for run in runs:
        row = flatten_run(run)
        print(
            f"{str(row['model_label']):<24} {str(row['index_type']):<7} "
            f"{format_number(row['reference_encode_ms_per_image']):>14} "
            f"{format_number(row['faiss_build_s']):>9} "
            f"{format_number(row['query_encode_ms_mean']):>13} "
            f"{format_number(row['search_ms_mean']):>10} "
            f"{format_number(row['encode_plus_search_ms_mean']):>10} "
            f"{format_number(row['recall_at_1']):>8} "
            f"{format_number(row['recall_at_5']):>8} "
            f"{format_number(row['recall_at_10']):>8} "
            f"{row['error'] or ''}"
        )


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    args = parse_args()
    args.recall_k = sorted(set([*args.recall_k, args.top_k]))
    if not args.recall_k or any(k < 1 for k in args.recall_k):
        raise ValueError("--recall-k values must be positive integers")
    model_specs = [parse_model_spec(value) for value in args.model_config] or [
        ModelSpec("denseuav-vit", 512)
    ]
    index_types = [index_type for index_type in args.index_types]

    reference_records = load_reference_metadata(args.metadata, base_dir=args.base_dir)
    query_records = load_query_records(args)
    if not reference_records:
        raise ValueError("Reference CSV does not contain any rows")
    if not query_records:
        raise ValueError("Query selection does not contain any images")

    runs: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for spec in model_specs:
        try:
            runs.extend(benchmark_model(spec, index_types, reference_records, query_records, args))
        except Exception as exc:
            for index_type in index_types:
                runs.append(
                    {
                        "model": spec.model,
                        "descriptor_dim": spec.descriptor_dim,
                        "model_label": spec.label,
                        "cosplace_backbone": spec.cosplace_backbone,
                        "index_type": index_type,
                        "error": f"{type(exc).__name__}: {exc}",
                        "model_load_s": None,
                        "offline": {"reference_count": len(reference_records)},
                        "online": {"query_count": len(query_records)},
                        "quality": {},
                    }
                )

    report = {
        "started_at": started_at,
        "metadata": str(args.metadata),
        "query_metadata": str(args.query_metadata) if args.query_metadata else None,
        "reference_count": len(reference_records),
        "query_count": len(query_records),
        "top_k": args.top_k,
        "recall_k": args.recall_k,
        "query_rotations": args.query_rotations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runs": runs,
    }

    print_summary(runs)
    if args.output_json is not None:
        write_json(args.output_json, report)
        print(f"\nWrote JSON KPI report -> {args.output_json}")
    if args.output_csv is not None:
        write_csv(args.output_csv, runs)
        print(f"Wrote CSV KPI summary -> {args.output_csv}")


if __name__ == "__main__":
    main()
