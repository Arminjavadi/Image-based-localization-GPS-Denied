#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from avl.config import AVLConfig
from avl.localizer import AVLLocalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a geo-tagged AVL FAISS index.")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with image_path, latitude, longitude")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save index.faiss + metadata.json")
    parser.add_argument("--base-dir", type=Path, default=None, help="Base directory for relative image paths")
    parser.add_argument("--model", choices=["mixvpr", "cosplace"], default="mixvpr")
    parser.add_argument("--descriptor-dim", type=int, default=4096)
    parser.add_argument("--cosplace-backbone", default="ResNet101")
    parser.add_argument("--index-type", choices=["hnsw", "ivfpq", "flat"], default="hnsw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AVLConfig(
        model=args.model,
        descriptor_dim=args.descriptor_dim,
        cosplace_backbone=args.cosplace_backbone,
        index_type=args.index_type,
        device=args.device,
        batch_size=args.batch_size,
    )
    localizer = AVLLocalizer(config)
    localizer.build_index(args.metadata, args.output, base_dir=args.base_dir)
    print(f"Built index with {localizer.index.size} references -> {args.output}")


if __name__ == "__main__":
    main()
