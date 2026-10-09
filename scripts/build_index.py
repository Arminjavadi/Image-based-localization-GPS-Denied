#!/usr/bin/env python3
"""Build a geo-tagged AVL index, with the localisation recipe stored next to it.

The recipe (avl.pipeline.PRESETS) decides the encoder and the steps every query of
this index will run: heading alignment, domain centering, altitude crop. By default
the recommended pipeline is built — MegaLoc + heading + flight centering + gated
altitude crop; pass --camera-k for your lens (k = 2 tan(HFOV / 2)).

python scripts/build_index.py --metadata data/visloc_avl/r05_t250/references.csv \
    --output artifacts/index_r05 --camera-k 0.971
python scripts/build_index.py --metadata refs.csv --output idx --preset none \
    --model denseuav-vit          # the historical behaviour: one encoder, no recipe
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss

from avl.config import AVLConfig
from avl.localizer import AVLLocalizer
from avl.pipeline import PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with image_path, latitude, longitude")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save index.faiss + metadata.json")
    parser.add_argument("--base-dir", type=Path, default=None, help="Base directory for relative image paths")
    parser.add_argument(
        "--preset",
        choices=["none", *PRESETS],
        default="recommended",
        help="Localisation recipe (avl.pipeline.PRESETS); 'none' = one encoder, no recipe steps.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Encoder or ensemble 'a+b+c'; overrides the preset's (default with --preset none: denseuav-vit).",
    )
    parser.add_argument("--descriptor-dim", type=int, default=None,
                        help="Single encoders with several widths (MixVPR, CosPlace); default native.")
    parser.add_argument("--camera-k", type=float, default=None,
                        help="Camera constant for the altitude crop: full-frame ground width / AGL.")
    parser.add_argument("--tile-m", type=float, default=None,
                        help="Tile ground size; default parsed from '_t<metres>' in --metadata.")
    parser.add_argument("--cosplace-backbone", default="ResNet101")
    parser.add_argument("--index-type", choices=["hnsw", "ivfpq", "flat"], default=None,
                        help="Default: flat (exact) with a preset, hnsw with --preset none.")
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
    if args.preset == "none":
        config = AVLConfig(
            model=args.model or "denseuav-vit",
            index_type=args.index_type or "hnsw",
            query_rotations=args.query_rotations,
        )
    else:
        config = AVLConfig.from_preset(args.preset, index_type=args.index_type or "flat")
        if args.model:
            config.model = args.model
    config.cosplace_backbone = args.cosplace_backbone
    config.device = args.device
    config.batch_size = args.batch_size
    config.camera_k = args.camera_k
    config.tile_m = args.tile_m
    if config.scale_from_agl and config.camera_k is None:
        print("note: no --camera-k — the altitude crop stays off until one is given "
              "(AVLLocalizer.load_index(..., overrides={'camera_k': k}))")

    from avl.ensemble import build_encoder
    from avl.metadata import load_reference_metadata

    records = load_reference_metadata(args.metadata, base_dir=args.base_dir)
    if not records:
        raise ValueError("Reference CSV does not contain any rows")
    paths = [record.image_path for record in records]

    localizer = AVLLocalizer(config)
    vocab = args.output / f"vocab_{config.model}.npz"
    encoder, model_load_s = timed_seconds(
        build_encoder, config.model, config, vocab, paths, args.descriptor_dim
    )
    descriptors, reference_encode_s = timed_seconds(
        encoder.encode_paths, paths, show_progress=not args.quiet
    )
    _, faiss_build_s = timed_seconds(
        localizer.build_index, args.metadata, args.output, args.base_dir, descriptors, encoder
    )
    index = localizer.index
    config = localizer.config
    index_save_s = 0.0  # included in faiss_build_s (build_index saves)

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
        "preset": args.preset,
        "recipe": {
            "heading_mode": config.heading_mode,
            "center_mode": config.center_mode,
            "scale_from_agl": config.scale_from_agl,
            "camera_k": config.camera_k,
            "tile_m": config.tile_m,
        },
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
