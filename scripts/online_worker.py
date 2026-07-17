#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.geo import weighted_geo_fusion
from avl.index import HashIndex, SearchResult


EVENT_PREFIX = "AVL_EVENT "
GEO_FUSION_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent online AVL engine that keeps the model and index loaded."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-rotations", type=int, choices=[1, 4], default=4)
    parser.add_argument("--warmup-image", type=Path, default=None)
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(
        EVENT_PREFIX + json.dumps({"event": event, **payload}, separators=(",", ":")),
        flush=True,
    )


def timed_seconds(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def single_sample_stats(elapsed_s: float) -> dict[str, float | int]:
    elapsed_ms = elapsed_s * 1000.0
    return {
        "count": 1,
        "mean": elapsed_ms,
        "median": elapsed_ms,
        "p95": elapsed_ms,
        "min": elapsed_ms,
        "max": elapsed_ms,
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
    return {
        "reference_count": index.size,
        "faiss_index_size_mb": (index_dir / "index.faiss").stat().st_size
        / (1024.0 * 1024.0),
    }


class OnlineEngine:
    def __init__(
        self,
        index_dir: Path,
        device: str,
        query_rotations: int,
    ) -> None:
        self.index_dir = index_dir
        self.index, self.index_load_s = timed_seconds(HashIndex.load, index_dir)
        self.config = self.index.config
        self.config.device = device
        self.config.query_rotations = query_rotations
        self.encoder, self.model_load_s = timed_seconds(VPREncoder, self.config)
        self.offline = load_offline_metrics(index_dir, self.index)

    def warmup(self, image_path: Path) -> float:
        _, elapsed_s = timed_seconds(self.encoder.encode_query, image_path)
        return elapsed_s

    def query(
        self,
        query_path: Path,
        top_k: int,
        output_json: Path,
        request_id: str,
    ) -> dict[str, Any]:
        request_start = time.perf_counter()
        descriptor, query_s = timed_seconds(self.encoder.encode_query, query_path)
        search, search_s = timed_seconds(
            self.index.search_orientations,
            descriptor,
            top_k,
        )
        localization = build_localization(query_path, self.index, search)
        request_ms = (time.perf_counter() - request_start) * 1000.0

        query_ms = query_s * 1000.0
        search_ms = search_s * 1000.0
        pipeline_ms = query_ms + search_ms
        online = {
            "query_count": 1,
            "query_encode_ms": single_sample_stats(query_s),
            "query_feature_images_per_s": 1000.0 / query_ms if query_ms else None,
            "search_ms": single_sample_stats(search_s),
            "search_queries_per_s": 1000.0 / search_ms if search_ms else None,
            "encode_plus_search_ms_mean": pipeline_ms,
            "online_queries_per_s": 1000.0 / pipeline_ms if pipeline_ms else None,
            "end_to_end_ms": request_ms,
        }
        run = {
            "model": self.config.model,
            "descriptor_dim": self.config.descriptor_dim,
            "model_label": model_label(self.config),
            "cosplace_backbone": self.config.cosplace_backbone,
            "index_type": self.config.index_type,
            "error": None,
            "model_load_s": self.model_load_s,
            "index_load_s": self.index_load_s,
            "engine_reused": True,
            "offline": self.offline,
            "online": online,
            "quality": {},
            "localizations": [localization],
        }
        report = {
            "mode": "interactive_online_persistent",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "index_dir": str(self.index_dir),
            "query_count": 1,
            "top_k": top_k,
            "query_rotations": self.config.query_rotations,
            "repeats": 1,
            "request_id": request_id,
            "runs": [run],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main() -> None:
    args = parse_args()
    engine = OnlineEngine(args.index, args.device, args.query_rotations)

    warmup_s = None
    if args.warmup_image is not None and args.warmup_image.is_file():
        warmup_s = engine.warmup(args.warmup_image)

    emit(
        "ready",
        model_label=model_label(engine.config),
        index_type=engine.config.index_type,
        reference_count=engine.index.size,
        model_load_s=engine.model_load_s,
        index_load_s=engine.index_load_s,
        warmup_s=warmup_s,
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id = ""
        try:
            command = json.loads(line)
            request_id = str(command.get("request_id", ""))
            action = command.get("command")
            if action == "shutdown":
                emit("stopped")
                return
            if action != "query":
                raise ValueError(f"Unsupported command: {action}")

            query_path = Path(command["query"])
            if not query_path.is_file():
                raise FileNotFoundError(f"Query image does not exist: {query_path}")
            output_json = Path(command["output_json"])
            report = engine.query(
                query_path=query_path,
                top_k=int(command.get("top_k", 5)),
                output_json=output_json,
                request_id=request_id,
            )
            online = report["runs"][0]["online"]
            emit(
                "result",
                request_id=request_id,
                output_json=str(output_json),
                pipeline_ms=online["encode_plus_search_ms_mean"],
                end_to_end_ms=online["end_to_end_ms"],
            )
        except Exception as exc:
            emit(
                "error",
                request_id=request_id,
                message=f"{type(exc).__name__}: {exc}",
            )


if __name__ == "__main__":
    main()
