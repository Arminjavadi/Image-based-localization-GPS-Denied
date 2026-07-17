#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.geo import weighted_geo_fusion
from avl.index import HashIndex, SearchResult


GEO_FUSION_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online-only localization against an existing offline AVL index."
    )
    parser.add_argument("--index", type=Path, required=True, help="Saved offline index directory.")
    parser.add_argument("--query", type=Path, required=True, help="Single aerial query image.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-rotations", type=int, choices=[1, 4], default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def timed_seconds(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def stats_ms(samples_s: list[float]) -> dict[str, float | int | None]:
    if not samples_s:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    samples_ms = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return {
        "count": int(samples_ms.size),
        "mean": float(samples_ms.mean()),
        "median": float(np.median(samples_ms)),
        "p95": float(np.percentile(samples_ms, 95)),
        "min": float(samples_ms.min()),
        "max": float(samples_ms.max()),
    }


def model_label(config: AVLConfig) -> str:
    label = f"{config.model}:{config.descriptor_dim}"
    if config.model == "cosplace":
        label = f"{label}:{config.cosplace_backbone}"
    return label


def reference_id(record, index: int) -> str:
    return record.image_id or Path(record.image_path).stem or f"index:{index}"


def build_localization(query_path: Path, index: HashIndex, search: SearchResult) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    matched_records = []
    for rank, (idx, score) in enumerate(zip(search.indices, search.scores), start=1):
        record_index = int(idx)
        record = index.records[record_index]
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
                "image_id": reference_id(record, record_index),
            }
        )

    estimated_position = None
    if matched_records:
        fusion_records = matched_records[:GEO_FUSION_TOP_K]
        fusion_scores = search.scores[:GEO_FUSION_TOP_K]
        altitudes = np.asarray(
            [
                record.altitude_m if record.altitude_m is not None else np.nan
                for record in fusion_records
            ],
            dtype=np.float64,
        )
        pose = weighted_geo_fusion(
            np.asarray([record.latitude for record in fusion_records], dtype=np.float64),
            np.asarray([record.longitude for record in fusion_records], dtype=np.float64),
            np.asarray(fusion_scores, dtype=np.float64),
            altitudes=altitudes,
        )
        estimated_position = {
            "latitude": pose.latitude,
            "longitude": pose.longitude,
            "altitude_m": pose.altitude_m,
        }

    return {
        "query_image_path": str(query_path.resolve()),
        "query_ground_truth": None,
        "estimated_position": estimated_position,
        "fusion_match_count": min(len(matches), GEO_FUSION_TOP_K),
        "localization_error_m": None,
        "matches": matches,
    }


def load_offline_metrics(index_dir: Path, index: HashIndex) -> dict[str, Any]:
    report_path = index_dir / "build_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            runs = report.get("runs", [])
            if runs:
                return dict(runs[0].get("offline", {}))
        except (OSError, json.JSONDecodeError):
            pass
    index_size = (index_dir / "index.faiss").stat().st_size / (1024.0 * 1024.0)
    return {
        "reference_count": index.size,
        "faiss_index_size_mb": index_size,
    }


def main() -> None:
    args = parse_args()
    if not args.query.is_file():
        raise FileNotFoundError(f"Query image does not exist: {args.query}")

    index, index_load_s = timed_seconds(HashIndex.load, args.index)
    config = index.config
    config.device = args.device
    config.query_rotations = args.query_rotations
    config.top_k = args.top_k
    encoder, model_load_s = timed_seconds(VPREncoder, config)

    for _ in range(max(0, args.warmup)):
        encoder.encode_query(args.query)

    descriptor = None
    query_times_s: list[float] = []
    for repeat in range(max(1, args.repeats)):
        current, elapsed_s = timed_seconds(encoder.encode_query, args.query)
        query_times_s.append(elapsed_s)
        if repeat == 0:
            descriptor = current
    if descriptor is None:
        raise RuntimeError("No query descriptor was produced")

    first_search = None
    search_times_s: list[float] = []
    for repeat in range(max(1, args.repeats)):
        current, elapsed_s = timed_seconds(index.search_orientations, descriptor, args.top_k)
        search_times_s.append(elapsed_s)
        if repeat == 0:
            first_search = current
    if first_search is None:
        raise RuntimeError("No search result was produced")

    query_stats = stats_ms(query_times_s)
    search_stats = stats_ms(search_times_s)
    query_mean = query_stats["mean"] or 0.0
    search_mean = search_stats["mean"] or 0.0
    online = {
        "query_count": 1,
        "query_encode_ms": query_stats,
        "query_feature_images_per_s": (1000.0 / query_mean) if query_mean else None,
        "search_ms": search_stats,
        "search_queries_per_s": (1000.0 / search_mean) if search_mean else None,
        "encode_plus_search_ms_mean": query_mean + search_mean,
        "online_queries_per_s": (1000.0 / (query_mean + search_mean)) if (query_mean + search_mean) else None,
    }
    run = {
        "model": config.model,
        "descriptor_dim": config.descriptor_dim,
        "model_label": model_label(config),
        "cosplace_backbone": config.cosplace_backbone,
        "index_type": config.index_type,
        "error": None,
        "model_load_s": model_load_s,
        "index_load_s": index_load_s,
        "offline": load_offline_metrics(args.index, index),
        "online": online,
        "quality": {},
        "localizations": [build_localization(args.query, index, first_search)],
    }
    report = {
        "mode": "interactive_online",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "index_dir": str(args.index),
        "query_count": 1,
        "top_k": args.top_k,
        "query_rotations": args.query_rotations,
        "repeats": args.repeats,
        "runs": [run],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Online localization complete: feature {query_mean:.3f} ms, "
        f"search {search_mean:.3f} ms -> {args.output_json}"
    )


if __name__ == "__main__":
    main()
