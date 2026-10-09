from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from avl.centering import DomainCentering
from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.ensemble import EnsembleEncoder, blocks_of, build_encoder
from avl.geo import GeoPose, haversine_m, search_window, weighted_geo_fusion
from avl.index import HashIndex
from avl.metadata import ReferenceRecord, load_reference_metadata
from avl.pipeline import (
    QueryTelemetry,
    is_ensemble,
    plan_query,
    rotation_for_heading,
    tile_m_of,
)
from avl.rerank import apply_rerank, build_reranker
from avl.retrieval import query_variants

Image.MAX_IMAGE_PIXELS = None

#: file next to index.faiss holding the map's mean descriptor (domain centering)
CENTER_FILE = "centering_mean.npy"


@dataclass
class Match:
    rank: int
    score: float
    record: ReferenceRecord
    #: Verification evidence, present only when re-ranking ran.
    rerank: dict | None = None
    retrieval_rank: int | None = None


@dataclass
class LocalizationResult:
    pose: GeoPose
    confidence: float
    best_match: Match
    matches: list[Match]
    query_descriptor: np.ndarray
    rerank: dict | None = None
    #: what the recipe did to this frame: rotation, crop, AGL, window, centering
    plan: dict = field(default_factory=dict)


class AVLLocalizer:
    """Absolute Visual Localization pipeline.

    frame -> heading rotation + altitude crop -> encode (one encoder or an ensemble)
    -> domain centering -> hash search (optionally inside the prior's window)
    -> [geometric re-rank] -> geo fusion of the top-5

    Every step comes from avl.pipeline and is measured in docs/AVL_Progress_Report.
    The recipe is chosen when the index is built (``AVLConfig.from_preset``) and
    stored with it; telemetry (:class:`avl.pipeline.QueryTelemetry`) arrives with
    each frame, and a step whose input is missing is skipped for that frame.
    """

    def __init__(self, config: AVLConfig | None = None) -> None:
        self.config = config or AVLConfig()
        self.encoder: VPREncoder | EnsembleEncoder | None = None
        self.index = HashIndex(self.config)
        self.reranker = build_reranker(self.config)
        self.centering = DomainCentering(self.config.center_mode, self.config.center_warmup)
        self._terrain = None
        self._index_dir: Path | None = None

    # ------------------------------------------------------------------ build --
    def _vocab_path(self, directory: Path) -> Path:
        # avl.ensemble.member_path turns this into one vocabulary per AnyLoc member
        return Path(directory) / f"vocab_{self.config.model}.npz"

    def build_index(
        self,
        metadata_csv: Path,
        output_dir: Path,
        base_dir: Path | None = None,
        descriptors: np.ndarray | None = None,
        encoder: VPREncoder | EnsembleEncoder | None = None,
    ) -> None:
        """Encode the reference map and save the index with its recipe.

        ``descriptors`` are precomputed raw reference descriptors in the encoder's
        layout (an ensemble's members combined by avl.ensemble.combine_blocks), for
        maps a benchmark already encoded; ``encoder`` an already-built encoder for
        ``config.model`` (its AnyLoc vocabularies must sit in ``output_dir``).
        """
        output_dir = Path(output_dir)
        self._index_dir = output_dir
        records = load_reference_metadata(metadata_csv, base_dir=base_dir)
        paths = [record.image_path for record in records]
        if self.config.tile_m is None:
            self.config.tile_m = tile_m_of(metadata_csv)
        self.encoder = encoder or build_encoder(
            self.config.model, self.config, self._vocab_path(output_dir), paths
        )
        if descriptors is None:
            descriptors = self.encoder.encode_paths(paths)
        if descriptors.shape[1] != self.encoder.config.descriptor_dim:
            raise ValueError(
                f"descriptors are {descriptors.shape[1]}-d, {self.config.model} gives "
                f"{self.encoder.config.descriptor_dim}-d"
            )
        self.config.descriptor_dim = self.encoder.config.descriptor_dim

        blocks = blocks_of(self.encoder)
        self.centering = DomainCentering(self.config.center_mode, self.config.center_warmup, blocks)
        indexed = self.centering.fit_refs(np.ascontiguousarray(descriptors, dtype=np.float32))

        self.index = HashIndex(self.config)
        self.index.blocks = blocks
        self.index.build(indexed, records)
        self.index.save(output_dir)
        if self.centering.map_mean is not None:
            np.save(output_dir / CENTER_FILE, self.centering.map_mean.astype(np.float32))

    #: Query-time settings the caller chose, which the index's stored config must
    #: not silently overwrite when it is loaded.
    _RUNTIME_FIELDS = (
        "device",
        "top_k",
        "rerank_enabled",
        "rerank_backend",
        "rerank_candidates",
        "rerank_max_features",
        "rerank_image_size",
        "rerank_ratio_test",
        "rerank_ransac_threshold_px",
        "rerank_min_inliers",
        "rerank_blend",
    )

    def load_index(
        self, index_dir: Path, overrides: dict | None = None, load_encoder: bool = True
    ) -> None:
        """Load an index with the recipe it was built with. ``overrides`` change
        recipe fields for this session (e.g. ``{"camera_k": 1.0}`` for another lens).
        With ``load_encoder=False`` the caller loads the model later (and can time it)
        with :meth:`load_encoder`."""
        index_dir = Path(index_dir)
        runtime = {name: getattr(self.config, name) for name in self._RUNTIME_FIELDS}
        self.index = HashIndex.load(index_dir)
        self.config = self.index.config
        for name, value in {**runtime, **(overrides or {})}.items():
            setattr(self.config, name, value)
        self.config.__post_init__()
        self._index_dir = index_dir
        self.encoder = None
        if load_encoder:
            self.load_encoder()
        self.centering = DomainCentering(
            self.config.center_mode, self.config.center_warmup, self.index.blocks
        )
        if self.config.center_mode != "off":
            mean_path = index_dir / CENTER_FILE
            if not mean_path.exists():
                raise FileNotFoundError(
                    f"{index_dir} was built without centering; rebuild it or load with "
                    "overrides={'center_mode': 'off'}"
                )
            self.centering.map_mean = np.load(mean_path)
        self.reranker = build_reranker(self.config)

    def load_encoder(self) -> None:
        """Build the encoder (or ensemble) the loaded index was made with."""
        vocab = self._vocab_path(self._index_dir) if self._index_dir is not None else None
        single = not is_ensemble(self.config.model) and not self.config.model.startswith("anyloc")
        # a single encoder can be built at a non-native width (MixVPR-512)
        width = self.config.descriptor_dim if single and self.config.descriptor_dim else None
        self.encoder = build_encoder(self.config.model, self.config, vocab, descriptor_dim=width)

    def reset_flight(self) -> None:
        """Start a new flight: forget the running flight-mean descriptor."""
        self.centering.reset_flight()

    # ------------------------------------------------------------- localize --
    def _terrain_elevation(self, latitude: float, longitude: float) -> float:
        if self._terrain is None:
            from avl.terrain import TerrainModel

            self._terrain = TerrainModel(self.config.dem_dir)  # on-disk tiles only, no download
        return float(self._terrain.elevation(latitude, longitude)[0])

    def encode_frame(
        self, query_image: Path, telemetry: QueryTelemetry | None = None
    ) -> tuple[np.ndarray, dict, Image.Image]:
        """The recipe's query side for one frame: heading rotation and altitude crop,
        encode, centre. Returns (descriptors of every searched view, what was done,
        the upright view as retrieval saw it). Updates the flight mean."""
        if self.encoder is None:
            self.load_encoder()
        config = self.config
        telemetry = telemetry or QueryTelemetry()
        with Image.open(query_image) as handle:
            image = handle.convert("RGB")
        width, height = image.size

        # -- heading + altitude: how the frame is turned into encoder input ------
        prior = (
            (telemetry.prior_lat, telemetry.prior_lon)
            if telemetry.prior_lat is not None and telemetry.prior_lon is not None
            else None
        )
        agl = telemetry.agl_m
        if agl is None and config.scale_from_agl and telemetry.altitude_asl_m is not None and prior:
            # in flight the terrain is read where the navigation prior puts the vehicle
            agl = telemetry.altitude_asl_m - self._terrain_elevation(*prior)
        scale_on = config.scale_from_agl and config.camera_k is not None and config.tile_m is not None
        plan = plan_query(
            width,
            height,
            rotate_deg=rotation_for_heading(telemetry.heading_deg),
            heading_mode=config.heading_mode,
            agl_m=agl,
            camera_k=config.camera_k if scale_on else None,
            tile_m=config.tile_m if scale_on else None,
            agl_gate=config.agl_gate,
            min_agl_m=config.min_agl_m,
        )
        rotations = config.query_rotations
        if config.heading_mode != "off" and plan.rotate_deg is None:
            rotations = 4  # the recipe expected a heading; without one, search all four
        views = query_variants(image, rotations, config.query_crop, plan.scales, plan.rotate_deg)

        descriptor = self.centering.query(self.encoder.encode_images(views))
        info = {
            **plan.as_dict(),
            "rotations": rotations,
            "center": config.center_mode,
            "flight_frames": self.centering.frames_seen,
        }
        return descriptor, info, views[0]

    def window_for(self, telemetry: QueryTelemetry | None, min_keep: int) -> np.ndarray | None:
        """Tile indices inside the prior's search window; None = search everything."""
        if telemetry is None or not telemetry.prior_sigma_m:
            return None
        if telemetry.prior_lat is None or telemetry.prior_lon is None:
            return None
        lat = np.asarray([r.latitude for r in self.index.records])
        lon = np.asarray([r.longitude for r in self.index.records])
        window = search_window(
            lat, lon, telemetry.prior_lat, telemetry.prior_lon,
            self.config.prior_k * telemetry.prior_sigma_m, min_keep=min_keep,
        )
        return np.flatnonzero(window)

    def localize(
        self,
        query_image: Path,
        top_k: int | None = None,
        telemetry: QueryTelemetry | None = None,
    ) -> LocalizationResult:
        if self.index.size == 0:
            raise RuntimeError("Index is empty. Build or load an index first.")
        config = self.config
        top_k = top_k or config.top_k
        candidate_k = (
            max(top_k, self.reranker.config.candidates) if self.reranker.enabled else top_k
        )
        descriptor, info, view = self.encode_frame(query_image, telemetry)

        # -- search, inside the prior's window when there is one -----------------
        allowed = self.window_for(telemetry, candidate_k)
        search = self.index.search_orientations(descriptor, top_k=candidate_k, allowed=allowed)
        ranked = apply_rerank(
            self.reranker,
            view,  # the frame exactly as retrieval saw it
            search.indices,
            search.scores,
            [self.index.records[int(i)].image_path for i in search.indices],
            top_k,
        )

        matches: list[Match] = []
        for rank, (idx, score) in enumerate(zip(ranked.indices, ranked.scores), start=1):
            position = rank - 1
            matches.append(
                Match(
                    rank=rank,
                    score=float(score),
                    record=self.index.records[int(idx)],
                    rerank=(
                        ranked.per_match[position] if position < len(ranked.per_match) else None
                    ),
                    retrieval_rank=(
                        ranked.retrieval_ranks[position]
                        if position < len(ranked.retrieval_ranks)
                        else None
                    ),
                )
            )

        if not matches:
            raise RuntimeError("No valid matches returned by the index")

        if matches[0].score < config.score_threshold:
            raise RuntimeError(
                f"Best match score {matches[0].score:.3f} is below threshold "
                f"{config.score_threshold:.3f}"
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
        # fusion_weight zeroes out candidates geometry rejected; without the stage
        # it is absent and the descriptor score is used, as before.
        weights = np.array(
            [(m.rerank or {}).get("fusion_weight", m.score) for m in fusion_matches],
            dtype=np.float64,
        )
        pose = weighted_geo_fusion(
            latitudes,
            longitudes,
            weights,
            altitudes=altitudes,
            method=config.fusion_method,
            softmax_temp=config.fusion_softmax_temp,
            cluster_radius_m=config.fusion_cluster_radius_m,
        )

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
            rerank=ranked.summary,
            plan={**info, "window_tiles": int(len(allowed)) if allowed is not None else None},
        )
