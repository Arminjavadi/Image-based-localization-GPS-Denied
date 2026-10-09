"""Single-query retrieval against a prepared reference map.

scripts/visloc_eval.py answers "how good is this encoder over a whole query set".
This module answers the other question the console needs: "where is *this* image",
using the same encoding, the same four-orientation search, the same per-location
de-duplication and the same top-5 geo fusion, so a single lookup and a benchmark
row cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avl.centering import DomainCentering
from avl.config import AVLConfig
from avl.encoder import VPREncoder
from avl.ensemble import (
    NATIVE_DIM,
    EnsembleEncoder,
    build_encoder as build_spec_encoder,
    combine_blocks,
    member_path,
    members_of,
)
from avl.geo import DEFAULT_FUSION_METHOD, GeoPose, haversine_m, weighted_geo_fusion
from avl.rerank import Reranker

Image.MAX_IMAGE_PIXELS = None

__all__ = ["NATIVE_DIM"]  # re-exported: scripts import the encoder list from here


@dataclass
class ReferenceMap:
    csv_path: Path
    paths: list[Path]
    ids: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


@dataclass
class QueryMatch:
    rank: int
    image_id: str
    latitude: float
    longitude: float
    score: float
    image_path: str
    error_m: float | None = None
    # Filled in only when the geometric re-ranking stage ran.
    retrieval_rank: int | None = None
    rerank: dict | None = None

    def as_dict(self) -> dict:
        payload = {
            "rank": self.rank,
            "image_id": self.image_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "score": self.score,
            "image_path": self.image_path,
            "error_m": self.error_m,
        }
        if self.rerank is not None:
            payload["retrieval_rank"] = self.retrieval_rank
            payload["rerank"] = self.rerank
        return payload


@dataclass
class QueryResult:
    query_image: str
    model: str
    refs_csv: str
    n_refs: int
    rotations: int
    query_crop: str
    matches: list[QueryMatch]
    fused: GeoPose
    spread_m: float
    confidence: float
    fusion_method: str = DEFAULT_FUSION_METHOD
    ground_truth: tuple[float, float] | None = None
    fused_error_m: float | None = None
    timings: dict = field(default_factory=dict)
    rerank: dict | None = None
    #: how the frame was prepared (avl.pipeline.QueryPlan.as_dict) when a recipe ran
    plan: dict | None = None

    def as_dict(self) -> dict:
        return {
            "query_image": self.query_image,
            "model": self.model,
            "refs_csv": self.refs_csv,
            "n_refs": self.n_refs,
            "rotations": self.rotations,
            "query_crop": self.query_crop,
            "fusion_method": self.fusion_method,
            "matches": [m.as_dict() for m in self.matches],
            "fused": {
                "latitude": self.fused.latitude,
                "longitude": self.fused.longitude,
                "altitude_m": self.fused.altitude_m,
            },
            "spread_m": self.spread_m,
            "confidence": self.confidence,
            "ground_truth": (
                {"latitude": self.ground_truth[0], "longitude": self.ground_truth[1]}
                if self.ground_truth
                else None
            ),
            "fused_error_m": self.fused_error_m,
            "timings": self.timings,
            "rerank": self.rerank,
            "plan": self.plan,
        }


def resolve_paths(paths: pd.Series, csv_path: Path) -> list[Path]:
    base = csv_path.parent
    return [Path(p) if Path(p).is_absolute() else (base / p) for p in paths.astype(str)]


def load_reference_map(refs_csv: Path) -> ReferenceMap:
    frame = pd.read_csv(refs_csv)
    missing = {"image_path", "latitude", "longitude"} - set(frame.columns)
    if missing:
        raise ValueError(f"{refs_csv} is missing column(s): {', '.join(sorted(missing))}")
    ids = (
        frame["image_id"].astype(str).to_numpy()
        if "image_id" in frame.columns
        else np.array([f"row_{i}" for i in range(len(frame))])
    )
    return ReferenceMap(
        csv_path=refs_csv,
        paths=resolve_paths(frame["image_path"], refs_csv),
        ids=ids,
        latitude=frame["latitude"].to_numpy(dtype=np.float64),
        longitude=frame["longitude"].to_numpy(dtype=np.float64),
    )


def build_encoder(
    model: str,
    references: ReferenceMap,
    vocab_path: Path | None = None,
    batch_size: int = 16,
    rotations: int = 4,
    head_weights: Path | None = None,
) -> VPREncoder | EnsembleEncoder:
    """The encoder for ``model`` — one name, or an ensemble spec ``a+b+c``.

    AnyLoc's vocabulary is map-specific: the one fitted for this map by the
    benchmark is reused when ``vocab_path`` exists, otherwise it is fitted on the
    map and cached there (per member for an ensemble, see avl.ensemble.member_path).
    """
    config = AVLConfig(
        model="denseuav-vit",
        batch_size=batch_size,
        query_rotations=rotations,
        index_type="flat",
    )
    if head_weights is not None:
        config.head_weights = Path(head_weights)
    return build_spec_encoder(model, config, vocab_path, references.paths)


def encode_images(
    encoder: VPREncoder | EnsembleEncoder, images: list[Image.Image], batch_size: int = 16
) -> np.ndarray:
    return encoder.encode_images(images, batch_size)


def encode_reference_map(
    encoder: VPREncoder | EnsembleEncoder,
    references: ReferenceMap,
    cache_path: Path | None = None,
    batch_size: int = 16,
) -> tuple[np.ndarray, bool]:
    """Descriptors for every reference tile, reusing the benchmark's cache when it fits.

    An ensemble reads and writes one cache per member, so it reuses what the
    single-encoder runs on this map already computed.
    """
    spec = encoder.config.model
    parts: list[np.ndarray] = []
    all_cached = True
    for member in members_of(encoder):
        path = member_path(cache_path, spec, member.config.model)
        cached = None
        if path is not None and path.exists():
            cached = np.load(path).astype(np.float32)
            if len(cached) != len(references):
                cached = None
        if cached is None:
            all_cached = False
            cached = np.ascontiguousarray(
                member.encode_paths([str(p) for p in references.paths]), dtype=np.float32
            )
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, cached)
        parts.append(np.ascontiguousarray(cached))
    return combine_blocks(parts), all_cached


def rotate_no_padding(image: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate the full frame counter-clockwise by ``angle_deg``, then crop the
    largest centred square that contains no padding.

    Cropping to a square *before* rotating (the old north-align path) leaves black
    corners in most orientations, and the encoder then sees those corners. A
    centred square of side ``a`` fits inside a ``w x h`` frame rotated by ``t``
    when ``a * (|cos t| + |sin t|) <= min(w, h)``.
    """
    width, height = image.size
    theta = math.radians(angle_deg)
    side = min(width, height) / (abs(math.cos(theta)) + abs(math.sin(theta)))
    rotated = image.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
    cx, cy = rotated.size[0] / 2.0, rotated.size[1] / 2.0
    half = math.floor(side / 2.0) - 1  # stay a pixel clear of interpolated edges
    return rotated.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))


def centre_crop(image: Image.Image, fraction: float) -> Image.Image:
    if fraction >= 1.0:
        return image
    width, height = image.size
    w, h = max(8, round(width * fraction)), max(8, round(height * fraction))
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def query_variants(
    image: Image.Image,
    rotations: int,
    query_crop: str,
    scales: tuple[float, ...] = (1.0,),
    heading_deg: float | None = None,
) -> list[Image.Image]:
    """The query as retrieval sees it.

    With ``heading_deg`` the frame is first turned north-up (see
    :func:`rotate_no_padding`); otherwise ``query_crop`` applies. Each of
    ``scales`` is a centre crop (1.0 = the whole view), searched like the
    orientations because the frame's ground footprint relative to the tiles is
    usually unknown. Returns ``len(scales) * rotations`` images, scale-major.
    """
    if heading_deg is not None and np.isfinite(heading_deg):
        image = rotate_no_padding(image, heading_deg)
    elif query_crop == "square":
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    out: list[Image.Image] = []
    for fraction in scales:
        view = centre_crop(image, fraction)
        if rotations == 1:
            out.append(view)
        else:
            out.extend(view.rotate(90 * r, expand=True) for r in range(rotations))
    return out


def query_scores(
    encoder: VPREncoder | EnsembleEncoder,
    image: Image.Image,
    ref_descriptors: np.ndarray,
    rotations: int = 4,
    query_crop: str = "square",
    batch_size: int = 16,
    scales: tuple[float, ...] = (1.0,),
    heading_deg: float | None = None,
    centering: DomainCentering | None = None,
) -> np.ndarray:
    """Cosine score of the query against every tile, best variant per tile.

    With ``centering`` the reference descriptors must already be centred by the same
    instance (:meth:`DomainCentering.fit_refs`); the query is centred here.
    """
    variants = query_variants(image, rotations, query_crop, scales, heading_deg)
    descriptors = encode_images(encoder, variants, batch_size)
    if centering is not None:
        descriptors = centering.query(descriptors)
    return (descriptors @ ref_descriptors.T).max(axis=0)


def localize(
    encoder: VPREncoder | EnsembleEncoder,
    image_path: Path,
    references: ReferenceMap,
    ref_descriptors: np.ndarray,
    rotations: int = 4,
    query_crop: str = "square",
    top_k: int = 5,
    ground_truth: tuple[float, float] | None = None,
    batch_size: int = 16,
    fusion_method: str = DEFAULT_FUSION_METHOD,
    reranker: Reranker | None = None,
    window: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    scales: tuple[float, ...] = (1.0,),
    heading_deg: float | None = None,
    centering: DomainCentering | None = None,
) -> QueryResult:
    """Localise one image.

    ``window`` is an optional boolean tile mask (see :func:`avl.geo.search_window`):
    only tiles inside it can be returned. ``scores`` are per-tile scores from
    :func:`query_scores` for this image, so a caller that searches several windows
    for the same frame encodes it once. ``heading_deg`` is the PIL angle that turns
    the frame north-up and ``scales`` the centre crop (see avl.pipeline.plan_query);
    ``centering`` must be the instance that centred ``ref_descriptors``.
    """
    import time

    reranking = reranker is not None and reranker.enabled
    image: Image.Image | None = None
    if scores is None or reranking:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

    started = time.perf_counter()
    if scores is None:
        scores = query_scores(
            encoder, image, ref_descriptors, rotations, query_crop, batch_size, scales, heading_deg,
            centering,
        )
    encode_s = time.perf_counter() - started
    if window is not None:
        scores = np.where(window, scores, -np.inf)

    # Re-ranking only re-orders what retrieval hands it, so the short-list has to
    # be deeper than the answer when the stage is on.
    candidate_k = max(top_k, reranker.config.candidates) if reranking else top_k

    # One entry per physical location: several tiles can share an image_id.
    order = np.argsort(-scores)
    seen: set[str] = set()
    chosen: list[int] = []
    for index in order:
        if not np.isfinite(scores[index]):
            break  # the rest are outside the search window
        image_id = str(references.ids[index])
        if image_id in seen:
            continue
        seen.add(image_id)
        chosen.append(int(index))
        if len(chosen) >= candidate_k:
            break

    rerank_summary: dict | None = None
    rerank_stats: dict[int, dict] = {}
    retrieval_rank: dict[int, int] = {}
    if reranking:
        # Verify against the query exactly as retrieval saw it (same crop), so a
        # candidate is judged on the pixels that produced its descriptor score.
        query_view = query_variants(image, 1, query_crop, heading_deg=heading_deg)[0]
        outcome = reranker.rerank(
            query_view,
            [str(references.paths[i]) for i in chosen],
            [float(scores[i]) for i in chosen],
        )
        rerank_summary = outcome.summary()
        retrieval_rank = {index: position + 1 for position, index in enumerate(chosen)}
        rerank_stats = {
            chosen[stat.candidate]: stat.as_dict() for stat in outcome.stats
        }
        chosen = [chosen[position] for position in outcome.order]

    chosen = chosen[:top_k]

    matches: list[QueryMatch] = []
    for rank, index in enumerate(chosen, start=1):
        latitude = float(references.latitude[index])
        longitude = float(references.longitude[index])
        matches.append(
            QueryMatch(
                rank=rank,
                image_id=str(references.ids[index]),
                latitude=latitude,
                longitude=longitude,
                score=float(scores[index]),
                image_path=str(references.paths[index]),
                error_m=(
                    haversine_m(ground_truth[0], ground_truth[1], latitude, longitude)
                    if ground_truth
                    else None
                ),
                retrieval_rank=retrieval_rank.get(index),
                rerank=rerank_stats.get(index),
            )
        )

    # With re-ranking on, fusion_weight already encodes the verification verdict:
    # a candidate geometry rejected contributes nothing to the pose, so the fused
    # position agrees with the ranking shown above it.
    weights = np.array(
        [
            (m.rerank or {}).get("fusion_weight", m.score) if m.rerank else m.score
            for m in matches
        ],
        dtype=np.float64,
    )
    pose = weighted_geo_fusion(
        np.array([m.latitude for m in matches]),
        np.array([m.longitude for m in matches]),
        weights,
        method=fusion_method,
    )
    spread = float(
        np.mean([haversine_m(pose.latitude, pose.longitude, m.latitude, m.longitude) for m in matches])
    )
    return QueryResult(
        query_image=str(image_path),
        model=encoder.config.model,
        refs_csv=str(references.csv_path),
        n_refs=len(references),
        rotations=rotations,
        query_crop=query_crop,
        matches=matches,
        fused=pose,
        spread_m=spread,
        confidence=float(matches[0].score * np.exp(-spread / 500.0)),
        fusion_method=fusion_method,
        ground_truth=ground_truth,
        fused_error_m=(
            haversine_m(ground_truth[0], ground_truth[1], pose.latitude, pose.longitude)
            if ground_truth
            else None
        ),
        timings={
            "query_encode_ms": 1000.0 * encode_s,
            "rerank_ms": (rerank_summary or {}).get("elapsed_ms", 0.0),
        },
        rerank=rerank_summary,
    )
