#!/usr/bin/env python3
"""Localize a UAV frame against an AVL index built by scripts/build_index.py.

The index carries its recipe (heading alignment, centering, altitude crop); the
frame's telemetry goes on the command line. A step whose input is missing is
skipped: no --heading-deg means the four-rotation search, no --altitude-asl (or
--agl) means no altitude crop, no --prior-sigma means a search of the whole map.

python scripts/localize.py --index artifacts/index_r05 \
    --query data/UAVVisLoc/05/drone/05_0001.JPG \
    --heading-deg 91.1 --altitude-asl 2315 --prior-lat 24.6659 --prior-lon 102.3413 --prior-sigma 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avl.config import AVLConfig
from avl.localizer import AVLLocalizer
from avl.pipeline import HEADING_MODES, QueryTelemetry
from avl.rerank import DEFAULT_BACKEND, RERANK_BACKENDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="Directory containing index.faiss")
    parser.add_argument("--query", type=Path, required=True, help="Query image path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Reject fixes whose best cosine is below this; default: the index's "
                             "(0.35 for raw descriptors, off for centred recipes).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Verify the retrieval short-list with local features + RANSAC.",
    )
    parser.add_argument("--rerank-backend", choices=list(RERANK_BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--rerank-candidates", type=int, default=10)
    parser.add_argument("--rerank-min-inliers", type=int, default=12)
    parser.add_argument("--rerank-blend", type=float, default=0.5)
    # -- telemetry for this frame (avl.pipeline.QueryTelemetry) -------------------
    parser.add_argument("--heading-deg", type=float, default=None,
                        help="Compass direction of the image's top edge, degrees clockwise from north.")
    parser.add_argument("--altitude-asl", type=float, default=None,
                        help="Altitude above sea level; the terrain is read at the prior position.")
    parser.add_argument("--agl", type=float, default=None, help="Height above ground, if known directly.")
    parser.add_argument("--prior-lat", type=float, default=None)
    parser.add_argument("--prior-lon", type=float, default=None)
    parser.add_argument("--prior-sigma", type=float, default=None,
                        help="1-sigma of the prior (m); search only tiles within prior_k sigma of it.")
    # -- recipe overrides for this session ----------------------------------------
    parser.add_argument("--camera-k", type=float, default=None, help="Override the index's camera constant.")
    parser.add_argument("--heading-mode", choices=list(HEADING_MODES), default=None,
                        help="Override the index's heading mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AVLConfig(
        device=args.device,
        top_k=args.top_k,
        rerank_enabled=args.rerank,
        rerank_backend=args.rerank_backend,
        rerank_candidates=args.rerank_candidates,
        rerank_min_inliers=args.rerank_min_inliers,
        rerank_blend=args.rerank_blend,
    )
    overrides = {}
    if args.score_threshold is not None:
        overrides["score_threshold"] = args.score_threshold
    if args.camera_k is not None:
        overrides["camera_k"] = args.camera_k
    if args.heading_mode is not None:
        overrides["heading_mode"] = args.heading_mode
    localizer = AVLLocalizer(config)
    localizer.load_index(args.index, overrides)
    telemetry = QueryTelemetry(
        heading_deg=args.heading_deg,
        altitude_asl_m=args.altitude_asl,
        agl_m=args.agl,
        prior_lat=args.prior_lat,
        prior_lon=args.prior_lon,
        prior_sigma_m=args.prior_sigma,
    )
    result = localizer.localize(args.query, top_k=args.top_k, telemetry=telemetry)

    output = {
        "latitude": result.pose.latitude,
        "longitude": result.pose.longitude,
        "altitude_m": result.pose.altitude_m,
        "confidence": result.confidence,
        "model": localizer.config.model,
        "plan": result.plan,
        "rerank": result.rerank,
        "best_match": {
            "score": result.best_match.score,
            "image_path": result.best_match.record.image_path,
            "latitude": result.best_match.record.latitude,
            "longitude": result.best_match.record.longitude,
        },
        "matches": [
            {
                "rank": match.rank,
                "score": match.score,
                "image_path": match.record.image_path,
                "latitude": match.record.latitude,
                "longitude": match.record.longitude,
                **(
                    {"rerank": match.rerank, "retrieval_rank": match.retrieval_rank}
                    if match.rerank
                    else {}
                ),
            }
            for match in result.matches
        ],
    }

    print(json.dumps(output, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
