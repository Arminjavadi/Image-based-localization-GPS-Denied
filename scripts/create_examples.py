#!/usr/bin/env python3
"""Create synthetic example images for a quick AVL smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    refs_dir = Path("examples/refs")
    queries_dir = Path("examples/queries")
    refs_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for i in range(5):
        base = rng.integers(30, 220, size=3, dtype=np.uint8)
        noise = rng.integers(0, 40, size=(320, 320, 3), dtype=np.uint8)
        image = np.clip(base + noise, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(refs_dir / f"ref_{i:02d}.jpg")

    query_noise = rng.integers(0, 35, size=(320, 320, 3), dtype=np.uint8)
    query = np.clip(base + query_noise, 0, 255).astype(np.uint8)
    Image.fromarray(query).save(queries_dir / "query.jpg")
    print("Created examples/refs/*.jpg and examples/queries/query.jpg")


if __name__ == "__main__":
    main()
