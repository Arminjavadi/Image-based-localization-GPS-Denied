#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avl.config import AVLConfig
from avl.localizer import AVLLocalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize a UAV query image against a geo-tagged AVL index.")
    parser.add_argument("--index", type=Path, required=True, help="Directory containing index.faiss")
    parser.add_argument("--query", type=Path, required=True, help="Query image path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AVLConfig(device=args.device, top_k=args.top_k, score_threshold=args.score_threshold)
    localizer = AVLLocalizer(config)
    localizer.load_index(args.index)
    result = localizer.localize(args.query, top_k=args.top_k)

    output = {
        "latitude": result.pose.latitude,
        "longitude": result.pose.longitude,
        "altitude_m": result.pose.altitude_m,
        "confidence": result.confidence,
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
