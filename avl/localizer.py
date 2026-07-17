from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.geo import GeoPose, haversine_m, weighted_geo_fusion
from avl.index import HashIndex
from avl.metadata import ReferenceRecord, load_reference_metadata


@dataclass
class Match:
    rank: int
    score: float
    record: ReferenceRecord


@dataclass
class LocalizationResult:
    pose: GeoPose
    confidence: float
    best_match: Match
    matches: list[Match]
    query_descriptor: np.ndarray


class AVLLocalizer:
    """Absolute Visual Localization pipeline: encode -> hash search -> geo fusion."""

    def __init__(self, config: AVLConfig | None = None) -> None:
        self.config = config or AVLConfig()
        self.encoder = VPREncoder(self.config)
        self.index = HashIndex(self.config)

    def build_index(
        self,
        metadata_csv: Path,
        output_dir: Path,
        base_dir: Path | None = None,
    ) -> None:
        records = load_reference_metadata(metadata_csv, base_dir=base_dir)
        image_paths = [record.image_path for record in records]
        descriptors = self.encoder.encode_paths(image_paths)
        self.index.build(descriptors, records)
        self.index.save(output_dir)

    def load_index(self, index_dir: Path) -> None:
        runtime_device = self.config.device
        runtime_top_k = self.config.top_k
        runtime_threshold = self.config.score_threshold
        runtime_rotations = self.config.query_rotations
        self.index = HashIndex.load(index_dir)
        self.config = self.index.config
        self.config.device = runtime_device
        self.config.top_k = runtime_top_k
        self.config.score_threshold = runtime_threshold
        self.config.query_rotations = runtime_rotations
        self.encoder = VPREncoder(self.config)

    def localize(self, query_image: Path, top_k: int | None = None) -> LocalizationResult:
        if self.index.size == 0:
            raise RuntimeError("Index is empty. Build or load an index first.")

        top_k = top_k or self.config.top_k
        descriptor = self.encoder.encode_query(query_image)
        search = self.index.search_orientations(descriptor, top_k=top_k)

        matches: list[Match] = []
        for rank, (idx, score) in enumerate(zip(search.indices, search.scores), start=1):
            matches.append(Match(rank=rank, score=float(score), record=self.index.records[int(idx)]))

        if not matches:
            raise RuntimeError("No valid matches returned by the index")

        if matches[0].score < self.config.score_threshold:
            raise RuntimeError(
                f"Best match score {matches[0].score:.3f} is below threshold "
                f"{self.config.score_threshold:.3f}"
            )

        fusion_matches = matches[:5]
        latitudes = np.array([m.record.latitude for m in fusion_matches], dtype=np.float64)
        longitudes = np.array([m.record.longitude for m in fusion_matches], dtype=np.float64)
        altitudes = np.array(
            [
                m.record.altitude_m if m.record.altitude_m is not None else np.nan
                for m in fusion_matches
            ],
            dtype=np.float64,
        )
        weights = np.array([m.score for m in fusion_matches], dtype=np.float64)
        pose = weighted_geo_fusion(latitudes, longitudes, weights, altitudes=altitudes)

        best = matches[0]
        spread_m = np.mean(
            [
                haversine_m(pose.latitude, pose.longitude, m.record.latitude, m.record.longitude)
                for m in fusion_matches
            ]
        )
        confidence = float(best.score * np.exp(-spread_m / 500.0))

        return LocalizationResult(
            pose=pose,
            confidence=confidence,
            best_match=best,
            matches=matches,
            query_descriptor=descriptor,
        )
