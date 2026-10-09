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
from avl.geo import weighted_geo_fusion
from avl.index import HashIndex
from avl.localizer import AVLLocalizer
from avl.pipeline import QueryTelemetry
from avl.rerank import (
    DEFAULT_BACKEND,
    RERANK_BACKENDS,
    RerankConfig,
    RerankedList,
    Reranker,
    apply_rerank,
)


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
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable geometric re-ranking by default (each query can override it).",
    )
    parser.add_argument("--rerank-backend", choices=list(RERANK_BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--rerank-candidates", type=int, default=10)
    parser.add_argument("--rerank-min-inliers", type=int, default=12)
    parser.add_argument("--rerank-blend", type=float, default=0.5)
    parser.add_argument("--rerank-max-features", type=int, default=2048)
    parser.add_argument("--rerank-image-size", type=int, default=640)
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


def build_localization(query_path: Path, index: HashIndex, ranked: RerankedList) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    matched_records = []
    for rank, (idx, score) in enumerate(zip(ranked.indices, ranked.scores), start=1):
        record_index = int(idx)
        record = index.records[record_index]
        matched_records.append(record)
        match: dict[str, Any] = {
            "rank": rank,
            "score": float(score),
            "image_path": record.image_path,
            "latitude": record.latitude,
            "longitude": record.longitude,
            "altitude_m": record.altitude_m,
            "heading_deg": record.heading_deg,
            "image_id": reference_id(record, record_index),
        }
        position = rank - 1
        if position < len(ranked.per_match) and ranked.per_match[position] is not None:
            match["rerank"] = ranked.per_match[position]
            match["retrieval_rank"] = ranked.retrieval_ranks[position]
        matches.append(match)

    estimated_position = None
    if matched_records:
        fusion_records = matched_records[:GEO_FUSION_TOP_K]
        # Weights already carry the geometric evidence when re-ranking is on.
        fusion_scores = ranked.weights[:GEO_FUSION_TOP_K]
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
        "rerank": ranked.summary,
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
        rerank: RerankConfig,
    ) -> None:
        self.index_dir = index_dir
        # AVLLocalizer loads the index with its recipe (ensemble, centering, heading,
        # altitude crop) and encodes frames the way the index expects.
        self.localizer = AVLLocalizer(AVLConfig(device=device))
        _, self.index_load_s = timed_seconds(
            self.localizer.load_index,
            index_dir,
            {"query_rotations": query_rotations},
            load_encoder=False,
        )
        _, self.model_load_s = timed_seconds(self.localizer.load_encoder)
        self.index = self.localizer.index
        self.config = self.localizer.config
        self.offline = load_offline_metrics(index_dir, self.index)
        self.rerank_defaults = rerank
        # Keyed by the settings that decide how an image is prepared, so toggling
        # thresholds or the blend in the GUI keeps the warm feature cache.
        self._rerankers: dict[tuple, Reranker] = {}

    def reranker_for(self, settings: dict[str, Any]) -> Reranker:
        base = self.rerank_defaults
        config = RerankConfig(
            enabled=bool(settings.get("enabled", base.enabled)),
            backend=str(settings.get("backend", base.backend)),
            candidates=int(settings.get("candidates", base.candidates)),
            max_features=int(settings.get("max_features", base.max_features)),
            image_size=int(settings.get("image_size", base.image_size)),
            ratio_test=float(settings.get("ratio_test", base.ratio_test)),
            ransac_threshold_px=float(
                settings.get("ransac_threshold_px", base.ransac_threshold_px)
            ),
            min_inliers=int(settings.get("min_inliers", base.min_inliers)),
            blend=float(settings.get("blend", base.blend)),
            device=base.device,
        )
        key = (config.backend, config.max_features, config.image_size, config.device)
        existing = self._rerankers.get(key)
        if existing is None:
            self._rerankers[key] = Reranker(config)
            return self._rerankers[key]
        # Same prepared features, cheap knobs updated in place.
        existing.config = config
        return existing

    def warmup(self, image_path: Path) -> float:
        _, elapsed_s = timed_seconds(self.localizer.encode_frame, image_path)
        # a warm-up frame is not part of the flight
        self.localizer.reset_flight()
        return elapsed_s

    def query(
        self,
        query_path: Path,
        top_k: int,
        output_json: Path,
        request_id: str,
        rerank_settings: dict[str, Any] | None = None,
        telemetry: QueryTelemetry | None = None,
    ) -> dict[str, Any]:
        request_start = time.perf_counter()
        reranker = self.reranker_for(rerank_settings or {})
        candidate_k = max(top_k, reranker.config.candidates) if reranker.enabled else top_k

        (descriptor, plan, view), query_s = timed_seconds(
            self.localizer.encode_frame, query_path, telemetry
        )
        search, search_s = timed_seconds(
            self.index.search_orientations,
            descriptor,
            candidate_k,
            self.localizer.window_for(telemetry, candidate_k),
        )
        rerank_start = time.perf_counter()
        ranked = apply_rerank(
            reranker,
            view,
            search.indices,
            search.scores,
            [self.index.records[int(i)].image_path for i in search.indices],
            top_k,
        )
        rerank_s = time.perf_counter() - rerank_start

        localization = build_localization(query_path, self.index, ranked)
        localization["plan"] = plan
        request_ms = (time.perf_counter() - request_start) * 1000.0

        query_ms = query_s * 1000.0
        search_ms = search_s * 1000.0
        rerank_ms = rerank_s * 1000.0
        pipeline_ms = query_ms + search_ms + rerank_ms
        online = {
            "query_count": 1,
            "query_encode_ms": single_sample_stats(query_s),
            "query_feature_images_per_s": 1000.0 / query_ms if query_ms else None,
            "search_ms": single_sample_stats(search_s),
            "search_queries_per_s": 1000.0 / search_ms if search_ms else None,
            "rerank_ms": single_sample_stats(rerank_s) if reranker.enabled else None,
            "rerank": ranked.summary,
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
            "rerank": reranker.config.as_dict(),
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
    rerank_defaults = RerankConfig(
        enabled=args.rerank,
        backend=args.rerank_backend,
        candidates=args.rerank_candidates,
        max_features=args.rerank_max_features,
        image_size=args.rerank_image_size,
        min_inliers=args.rerank_min_inliers,
        blend=args.rerank_blend,
        device=args.device,
    )
    engine = OnlineEngine(args.index, args.device, args.query_rotations, rerank_defaults)

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
            telemetry = command.get("telemetry") or {}
            report = engine.query(
                query_path=query_path,
                top_k=int(command.get("top_k", 5)),
                output_json=output_json,
                request_id=request_id,
                rerank_settings=command.get("rerank") or {},
                telemetry=QueryTelemetry(**telemetry) if telemetry else None,
            )
            online = report["runs"][0]["online"]
            rerank_stats = online.get("rerank_ms") or {}
            emit(
                "result",
                request_id=request_id,
                output_json=str(output_json),
                pipeline_ms=online["encode_plus_search_ms_mean"],
                end_to_end_ms=online["end_to_end_ms"],
                rerank_ms=rerank_stats.get("mean"),
                rerank=online.get("rerank"),
            )
        except Exception as exc:
            emit(
                "error",
                request_id=request_id,
                message=f"{type(exc).__name__}: {exc}",
            )


if __name__ == "__main__":
    main()
