#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss

from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.index import HashIndex
from avl.metadata import load_reference_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a geo-tagged AVL FAISS index.")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with image_path, latitude, longitude")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save index.faiss + metadata.json")
    parser.add_argument("--base-dir", type=Path, default=None, help="Base directory for relative image paths")
    parser.add_argument(
        "--model",
        choices=["denseuav-vit", "mixvpr", "cosplace"],
        default="denseuav-vit",
    )
    parser.add_argument("--descriptor-dim", type=int, default=512)
    parser.add_argument("--cosplace-backbone", default="ResNet101")
    parser.add_argument("--index-type", choices=["hnsw", "ivfpq", "flat"], default="hnsw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-rotations", type=int, choices=[1, 4], default=4)
    parser.add_argument("--report-json", type=Path, default=None, help="Optional offline KPI JSON output.")
    parser.add_argument("--quiet", action="store_true", help="Disable reference encoding progress bars.")
    return parser.parse_args()


def timed_seconds(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    config = AVLConfig(
        model=args.model,
        descriptor_dim=args.descriptor_dim,
        cosplace_backbone=args.cosplace_backbone,
        index_type=args.index_type,
        device=args.device,
        batch_size=args.batch_size,
        query_rotations=args.query_rotations,
    )
    records = load_reference_metadata(args.metadata, base_dir=args.base_dir)
    if not records:
        raise ValueError("Reference CSV does not contain any rows")

    encoder, model_load_s = timed_seconds(VPREncoder, config)
    descriptors, reference_encode_s = timed_seconds(
        encoder.encode_paths,
        [record.image_path for record in records],
        show_progress=not args.quiet,
    )
    index = HashIndex(config)
    _, faiss_build_s = timed_seconds(index.build, descriptors, records)
    _, index_save_s = timed_seconds(index.save, args.output)

    model_label = f"{config.model}:{config.descriptor_dim}"
    if config.model == "cosplace":
        model_label = f"{model_label}:{config.cosplace_backbone}"
    offline = {
        "reference_count": len(records),
        "reference_encode_s": reference_encode_s,
        "reference_encode_ms_per_image": (reference_encode_s / len(records)) * 1000.0,
        "reference_images_per_s": len(records) / reference_encode_s if reference_encode_s else None,
        "descriptor_memory_mb": descriptors.nbytes / (1024.0 * 1024.0),
        "faiss_build_s": faiss_build_s,
        "faiss_build_ms_per_image": (faiss_build_s / len(records)) * 1000.0,
        "faiss_index_size_mb": faiss.serialize_index(index.index).nbytes / (1024.0 * 1024.0),
        "index_save_s": index_save_s,
    }
    run = {
        "model": config.model,
        "descriptor_dim": config.descriptor_dim,
        "model_label": model_label,
        "cosplace_backbone": config.cosplace_backbone,
        "index_type": config.index_type,
        "error": None,
        "model_load_s": model_load_s,
        "offline": offline,
        "online": {},
        "quality": {},
        "localizations": [],
    }
    report = {
        "mode": "interactive_offline",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metadata": str(args.metadata),
        "index_dir": str(args.output),
        "reference_count": len(records),
        "runs": [run],
    }
    build_report_path = args.output / "build_report.json"
    build_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report_json is not None and args.report_json.resolve() != build_report_path.resolve():
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Built and saved offline index with {index.size} references -> {args.output}")
    print(f"Offline KPI report -> {build_report_path}")


if __name__ == "__main__":
    main()
