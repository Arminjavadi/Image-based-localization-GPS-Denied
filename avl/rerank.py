"""Geometric re-ranking of the retrieval short-list.

Global descriptors alone plateau on aerial imagery: repetitive fields, roof grids
and tree canopy all look alike to a single vector, so the correct tile often sits
at rank 3-10 rather than rank 1. Every aerial VPR study we follow (Moskalenko et
al. 2024; Ye et al. 2024) therefore treats retrieval as a *candidate generator*
and settles the ranking with local features plus RANSAC.

This module is that second stage. Retrieval hands it N candidates, each backend
matches local features between the query and each candidate, and the number of
geometrically consistent inliers re-orders the list. A candidate that shares a
consistent homography with the query is the same place; one that merely shares a
texture statistic is not.

Backends
--------
``sift-ransac``   SIFT + ratio test + MAGSAC. No extra dependency, rotation and
                  scale invariant, the sensible default.
``orb-ransac``    ORB + Hamming matching. Much faster, noticeably weaker.
``akaze-ransac``  AKAZE. Middle ground, good on low-texture terrain.
``superpoint-lightglue``  The pairing that won most rows in Moskalenko et al.
                  Needs ``pip install lightglue``.
``loftr``         Detector-free dense matching, strongest across viewpoint
                  change and the slowest. Needs ``pip install kornia``.

The stage is pure re-ordering: it never invents a candidate retrieval did not
return, so with ``blend=0`` it is a no-op and the pipeline degrades exactly to
descriptor-only behaviour.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

RERANK_BACKENDS = (
    "sift-ransac",
    "orb-ransac",
    "akaze-ransac",
    "superpoint-lightglue",
    "loftr",
)

DEFAULT_BACKEND = "sift-ransac"

#: Backends that need a package beyond the project's base requirements.
OPTIONAL_BACKEND_HINT = {
    "superpoint-lightglue": "pip install lightglue",
    "loftr": "pip install kornia",
}

Image.MAX_IMAGE_PIXELS = None


@dataclass
class RerankStat:
    """Verification outcome for one candidate."""

    candidate: int
    descriptor_score: float
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    rerank_score: float = 0.0
    combined_score: float = 0.0
    #: Weight this candidate carries into geo fusion. Zero once geometry has
    #: confirmed other candidates and rejected this one.
    fusion_weight: float = 0.0
    verified: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "rerank_score": self.rerank_score,
            "combined_score": self.combined_score,
            "fusion_weight": self.fusion_weight,
            "verified": self.verified,
            "error": self.error,
        }


@dataclass
class RerankOutcome:
    """The re-ordered short-list plus the evidence behind it."""

    backend: str
    order: list[int]
    stats: list[RerankStat]
    elapsed_ms: float
    candidates: int
    verified: int
    blend: float
    min_inliers: int
    skipped: bool = False
    error: str | None = None

    def stat_for(self, candidate: int) -> RerankStat | None:
        for stat in self.stats:
            if stat.candidate == candidate:
                return stat
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "candidates": self.candidates,
            "verified": self.verified,
            "elapsed_ms": self.elapsed_ms,
            "blend": self.blend,
            "min_inliers": self.min_inliers,
            "skipped": self.skipped,
            "error": self.error,
        }


def load_gray(source: str | Path | Image.Image, long_side: int) -> np.ndarray:
    """Grayscale uint8 array, longest side capped at ``long_side``."""
    if isinstance(source, Image.Image):
        image = source.convert("L")
    else:
        with Image.open(source) as handle:
            image = handle.convert("L")
    width, height = image.size
    longest = max(width, height)
    if long_side > 0 and longest > long_side:
        scale = long_side / float(longest)
        image = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.BILINEAR,
        )
    return np.asarray(image, dtype=np.uint8)


class _Backend:
    """Two-phase matcher: prepare each image once, then match pairs."""

    #: LoFTR-style dense matchers cannot reuse a per-image representation.
    cacheable = True

    def __init__(self, config: "RerankConfig") -> None:
        self.config = config

    def prepare(self, source: str | Path | Image.Image) -> Any:
        raise NotImplementedError

    def match(self, query: Any, reference: Any) -> tuple[int, int]:
        """Return ``(matches, inliers)`` for one pair."""
        raise NotImplementedError


def _is_plausible_homography(
    homography: np.ndarray,
    shape: tuple[int, int],
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 50.0,
) -> bool:
    """Reject the degenerate warps RANSAC happily fits to random correspondences.

    On a cross-domain pair with ~70 near-random matches, RANSAC can always find
    some homography that "explains" 15 of them -- typically one that folds the
    image onto a line or flips it inside out. Without this check those fake
    inliers are indistinguishable from a true match's, and the stage ranks on
    noise. A real match maps the query outline to a convex quad of comparable
    area; anything else is discarded.
    """
    import cv2

    if homography is None or not np.all(np.isfinite(homography)):
        return False
    height, width = shape
    corners = np.float32(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
    ).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(warped)):
        return False

    edges = np.roll(warped, -1, axis=0) - warped
    following = np.roll(edges, -1, axis=0)
    # 2-D cross product; np.cross on 2-vectors is deprecated in NumPy 2.
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False  # self-intersecting or folded quad

    area = 0.5 * abs(
        float(np.dot(warped[:, 0], np.roll(warped[:, 1], -1)))
        - float(np.dot(np.roll(warped[:, 0], -1), warped[:, 1]))
    )
    ratio = area / float(max(width * height, 1))
    return min_area_ratio <= ratio <= max_area_ratio


def _homography_inliers(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    threshold_px: float,
    source_shape: tuple[int, int] | None = None,
) -> int:
    import cv2

    if len(source_points) < 4:
        return 0
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    try:
        homography, mask = cv2.findHomography(
            source_points.reshape(-1, 1, 2),
            destination_points.reshape(-1, 1, 2),
            method,
            threshold_px,
            maxIters=2000,
            confidence=0.999,
        )
    except cv2.error:
        return 0
    if mask is None or homography is None:
        return 0
    if source_shape is not None and not _is_plausible_homography(homography, source_shape):
        return 0
    return int(mask.sum())


class _ClassicBackend(_Backend):
    """Detect-describe-match-RANSAC over an OpenCV feature detector."""

    def __init__(self, config: "RerankConfig", kind: str) -> None:
        super().__init__(config)
        import cv2

        self.cv2 = cv2
        self.kind = kind
        if kind == "sift":
            self.detector = cv2.SIFT_create(nfeatures=config.max_features)
            # Brute force over a few thousand 128-D float descriptors costs more
            # than the detection itself; approximate KD-trees are ~5x cheaper and
            # the ratio test absorbs the small loss in match quality.
            self.matcher = cv2.FlannBasedMatcher(
                {"algorithm": 1, "trees": 4}, {"checks": 32}
            )
        elif kind == "orb":
            self.detector = cv2.ORB_create(nfeatures=config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif kind == "akaze":
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown classic detector: {kind}")

    def prepare(self, source: str | Path | Image.Image) -> Any:
        gray = load_gray(source, self.config.image_size)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 2:
            return None
        points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        if self.kind == "sift":
            descriptors = np.asarray(descriptors, dtype=np.float32)
        return points, descriptors, gray.shape

    def match(self, query: Any, reference: Any) -> tuple[int, int]:
        if query is None or reference is None:
            return 0, 0
        query_points, query_desc, query_shape = query
        ref_points, ref_desc, _ = reference
        if len(query_desc) < 2 or len(ref_desc) < 2:
            return 0, 0
        try:
            knn = self.matcher.knnMatch(query_desc, ref_desc, k=2)
        except self.cv2.error:
            return 0, 0

        ratio = self.config.ratio_test
        source: list[np.ndarray] = []
        destination: list[np.ndarray] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            best, second = pair[0], pair[1]
            if best.distance < ratio * second.distance:
                source.append(query_points[best.queryIdx])
                destination.append(ref_points[best.trainIdx])
        if not source:
            return 0, 0
        inliers = _homography_inliers(
            np.asarray(source, dtype=np.float32),
            np.asarray(destination, dtype=np.float32),
            self.config.ransac_threshold_px,
            query_shape,
        )
        return len(source), inliers


class _LightGlueBackend(_Backend):
    """SuperPoint keypoints matched by LightGlue, verified with RANSAC."""

    def __init__(self, config: "RerankConfig") -> None:
        super().__init__(config)
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "The 'superpoint-lightglue' re-ranking backend needs LightGlue: "
                f"{OPTIONAL_BACKEND_HINT['superpoint-lightglue']}"
            ) from exc

        self.torch = torch
        self.device = torch.device(
            config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.extractor = (
            SuperPoint(max_num_keypoints=config.max_features).eval().to(self.device)
        )
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _tensor(self, gray: np.ndarray):
        array = gray.astype(np.float32) / 255.0
        return self.torch.from_numpy(array)[None, None].to(self.device)

    def prepare(self, source: str | Path | Image.Image) -> Any:
        gray = load_gray(source, self.config.image_size)
        with self.torch.inference_mode():
            return self.extractor.extract(self._tensor(gray)), gray.shape

    def match(self, query: Any, reference: Any) -> tuple[int, int]:
        if query is None or reference is None:
            return 0, 0
        from lightglue.utils import rbd

        query_feats, query_shape = query
        ref_feats, _ = reference
        with self.torch.inference_mode():
            pairs = self.matcher({"image0": query_feats, "image1": ref_feats})
        feats0, feats1, pairs = (rbd(query_feats), rbd(ref_feats), rbd(pairs))
        indices = pairs["matches"].cpu().numpy()
        if len(indices) == 0:
            return 0, 0
        points0 = feats0["keypoints"].cpu().numpy()[indices[:, 0]]
        points1 = feats1["keypoints"].cpu().numpy()[indices[:, 1]]
        inliers = _homography_inliers(
            points0.astype(np.float32),
            points1.astype(np.float32),
            self.config.ransac_threshold_px,
            query_shape,
        )
        return len(indices), inliers


class _LoFTRBackend(_Backend):
    """Detector-free dense matching; no per-image representation to cache."""

    cacheable = False

    def __init__(self, config: "RerankConfig") -> None:
        super().__init__(config)
        try:
            import torch
            import kornia.feature as KF
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "The 'loftr' re-ranking backend needs Kornia: "
                f"{OPTIONAL_BACKEND_HINT['loftr']}"
            ) from exc

        self.torch = torch
        self.device = torch.device(
            config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.matcher = KF.LoFTR(pretrained="outdoor").eval().to(self.device)

    def prepare(self, source: str | Path | Image.Image) -> Any:
        gray = load_gray(source, self.config.image_size)
        # LoFTR wants both sides divisible by 8.
        height = (gray.shape[0] // 8) * 8
        width = (gray.shape[1] // 8) * 8
        if height < 8 or width < 8:
            return None
        gray = gray[:height, :width]
        array = gray.astype(np.float32) / 255.0
        return self.torch.from_numpy(array)[None, None].to(self.device), gray.shape

    def match(self, query: Any, reference: Any) -> tuple[int, int]:
        if query is None or reference is None:
            return 0, 0
        query_tensor, query_shape = query
        ref_tensor, _ = reference
        with self.torch.inference_mode():
            output = self.matcher({"image0": query_tensor, "image1": ref_tensor})
        points0 = output["keypoints0"].cpu().numpy()
        points1 = output["keypoints1"].cpu().numpy()
        if len(points0) == 0:
            return 0, 0
        inliers = _homography_inliers(
            points0.astype(np.float32),
            points1.astype(np.float32),
            self.config.ransac_threshold_px,
            query_shape,
        )
        return len(points0), inliers


@dataclass
class RerankConfig:
    """Everything the stage needs, independent of AVLConfig so scripts can pass a subset."""

    enabled: bool = False
    backend: str = DEFAULT_BACKEND
    candidates: int = 10
    max_features: int = 2048
    image_size: int = 640
    ratio_test: float = 0.8
    ransac_threshold_px: float = 4.0
    min_inliers: int = 12
    blend: float = 0.5
    cache_size: int = 512
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.backend not in RERANK_BACKENDS:
            raise ValueError(
                f"rerank_backend must be one of {RERANK_BACKENDS}, got {self.backend!r}"
            )
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("rerank_blend must be between 0 and 1")
        if self.candidates < 1:
            raise ValueError("rerank_candidates must be >= 1")
        if self.min_inliers < 0:
            raise ValueError("rerank_min_inliers must be >= 0")

    @classmethod
    def from_avl_config(cls, config: Any) -> "RerankConfig":
        return cls(
            enabled=getattr(config, "rerank_enabled", False),
            backend=getattr(config, "rerank_backend", DEFAULT_BACKEND),
            candidates=getattr(config, "rerank_candidates", 10),
            max_features=getattr(config, "rerank_max_features", 2048),
            image_size=getattr(config, "rerank_image_size", 640),
            ratio_test=getattr(config, "rerank_ratio_test", 0.8),
            ransac_threshold_px=getattr(config, "rerank_ransac_threshold_px", 4.0),
            min_inliers=getattr(config, "rerank_min_inliers", 12),
            blend=getattr(config, "rerank_blend", 0.5),
            device=getattr(config, "device", "cuda"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "candidates": self.candidates,
            "max_features": self.max_features,
            "image_size": self.image_size,
            "ratio_test": self.ratio_test,
            "ransac_threshold_px": self.ransac_threshold_px,
            "min_inliers": self.min_inliers,
            "blend": self.blend,
        }


_BACKEND_FACTORY = {
    "sift-ransac": lambda cfg: _ClassicBackend(cfg, "sift"),
    "orb-ransac": lambda cfg: _ClassicBackend(cfg, "orb"),
    "akaze-ransac": lambda cfg: _ClassicBackend(cfg, "akaze"),
    "superpoint-lightglue": _LightGlueBackend,
    "loftr": _LoFTRBackend,
}


class Reranker:
    """Re-orders a retrieval short-list by geometric consistency.

    The backend is built lazily so constructing a disabled re-ranker (or one
    whose optional dependency is missing) costs nothing until it is used.
    """

    def __init__(self, config: RerankConfig) -> None:
        self.config = config
        self._backend: _Backend | None = None
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def backend(self) -> _Backend:
        if self._backend is None:
            self._backend = _BACKEND_FACTORY[self.config.backend](self.config)
        return self._backend

    def _prepare_reference(self, path: str | Path) -> Any:
        backend = self.backend()
        if not backend.cacheable or self.config.cache_size <= 0:
            return backend.prepare(path)
        key = str(path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        prepared = backend.prepare(path)
        self._cache[key] = prepared
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)
        return prepared

    def clear_cache(self) -> None:
        self._cache.clear()

    def rerank(
        self,
        query: str | Path | Image.Image,
        candidate_paths: Sequence[str | Path],
        descriptor_scores: Sequence[float],
    ) -> RerankOutcome:
        """Score every candidate against the query and return the new order."""
        scores = [float(s) for s in descriptor_scores]
        identity = list(range(len(candidate_paths)))

        if not self.config.enabled or not candidate_paths:
            return RerankOutcome(
                backend=self.config.backend,
                order=identity,
                stats=[
                    RerankStat(
                        candidate=i,
                        descriptor_score=s,
                        combined_score=s,
                        fusion_weight=s,
                    )
                    for i, s in enumerate(scores)
                ],
                elapsed_ms=0.0,
                candidates=len(candidate_paths),
                verified=0,
                blend=self.config.blend,
                min_inliers=self.config.min_inliers,
                skipped=True,
            )

        started = time.perf_counter()
        try:
            backend = self.backend()
            query_prepared = backend.prepare(query)
        except Exception as exc:
            # A missing optional dependency or a broken model must not take the
            # localization down: fall back to the descriptor order and say why.
            return RerankOutcome(
                backend=self.config.backend,
                order=identity,
                stats=[
                    RerankStat(
                        candidate=i,
                        descriptor_score=s,
                        combined_score=s,
                        fusion_weight=s,
                    )
                    for i, s in enumerate(scores)
                ],
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=len(candidate_paths),
                verified=0,
                blend=self.config.blend,
                min_inliers=self.config.min_inliers,
                skipped=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        stats: list[RerankStat] = []
        for position, path in enumerate(candidate_paths):
            stat = RerankStat(
                candidate=position,
                descriptor_score=scores[position] if position < len(scores) else 0.0,
            )
            try:
                reference = self._prepare_reference(path)
                stat.matches, stat.inliers = backend.match(query_prepared, reference)
            except Exception as exc:  # One unreadable tile must not sink the query.
                stat.error = f"{type(exc).__name__}: {exc}"
            if stat.matches:
                stat.inlier_ratio = stat.inliers / float(stat.matches)
            stat.verified = stat.inliers >= self.config.min_inliers
            stats.append(stat)

        # Normalise against the strongest candidate, but never let a short-list
        # where nothing reached min_inliers look like a perfect match.
        best_inliers = max((s.inliers for s in stats), default=0)
        denominator = float(max(best_inliers, self.config.min_inliers, 1))
        blend = self.config.blend
        for stat in stats:
            stat.rerank_score = stat.inliers / denominator
            if stat.verified:
                stat.combined_score = (
                    1.0 - blend
                ) * stat.descriptor_score + blend * stat.rerank_score
            else:
                # Geometry had nothing to say about this candidate, so it keeps the
                # descriptor's opinion untouched rather than being scored on noise.
                stat.combined_score = stat.descriptor_score

        # Fusion has to agree with the ranking. Once geometry has confirmed some
        # candidates, a rejected one must not pull the fused pose toward itself
        # just because its descriptor score was high -- that is precisely the
        # outlier verification was added to catch. When nothing verified there is
        # no evidence to act on, so every candidate keeps its descriptor weight
        # and fusion behaves exactly as it did before this stage existed.
        any_verified = any(stat.verified for stat in stats)
        for stat in stats:
            if not any_verified:
                stat.fusion_weight = stat.descriptor_score
            elif stat.verified:
                stat.fusion_weight = stat.combined_score
            else:
                stat.fusion_weight = 0.0

        # Verification is a gate, not a vote: candidates that passed rank above
        # those that did not, and within each group the order is by score. The
        # consequence that matters is the failure mode -- when nothing verifies
        # (a cross-domain pair local features cannot bridge) the order is exactly
        # the retrieval order, so the stage can help or abstain but never scramble.
        order = sorted(
            range(len(stats)),
            key=lambda i: (
                stats[i].verified,
                stats[i].combined_score,
                stats[i].inliers,
                stats[i].descriptor_score,
            ),
            reverse=True,
        )
        return RerankOutcome(
            backend=self.config.backend,
            order=order,
            stats=stats,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            candidates=len(candidate_paths),
            verified=sum(1 for s in stats if s.verified),
            blend=blend,
            min_inliers=self.config.min_inliers,
        )


def build_reranker(config: Any) -> Reranker:
    """Reranker from an :class:`avl.config.AVLConfig` (or anything with the fields)."""
    if isinstance(config, RerankConfig):
        return Reranker(config)
    return Reranker(RerankConfig.from_avl_config(config))


@dataclass
class RerankedList:
    """A short-list after the stage ran, trimmed to the requested depth."""

    indices: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    per_match: list[dict | None] = field(default_factory=list)
    retrieval_ranks: list[int | None] = field(default_factory=list)
    summary: dict | None = None
    elapsed_ms: float = 0.0


def apply_rerank(
    reranker: Reranker | None,
    query: str | Path | Image.Image,
    indices: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    candidate_paths: Sequence[str | Path],
    top_k: int,
) -> RerankedList:
    """Re-order one retrieval short-list and trim it to ``top_k``.

    Shared by the interactive worker and the batch benchmark so a KPI row and a
    live query cannot rank the same candidates differently. With no re-ranker,
    or a disabled one, this is just a trim.
    """
    indices = np.asarray(indices)
    scores = np.asarray(scores, dtype=np.float64)

    if reranker is None or not reranker.enabled or len(indices) == 0:
        return RerankedList(
            indices=indices[:top_k],
            scores=scores[:top_k],
            weights=scores[:top_k],
            per_match=[None] * len(indices[:top_k]),
            retrieval_ranks=[None] * len(indices[:top_k]),
            summary=None,
        )

    outcome = reranker.rerank(query, list(candidate_paths), scores)
    order = outcome.order[:top_k]
    stats_by_candidate = {stat.candidate: stat for stat in outcome.stats}
    weights = np.asarray(
        [stats_by_candidate[i].fusion_weight for i in order], dtype=np.float64
    )
    return RerankedList(
        indices=indices[order],
        scores=scores[order],
        weights=weights,
        per_match=[stats_by_candidate[i].as_dict() for i in order],
        retrieval_ranks=[i + 1 for i in order],
        summary=outcome.summary(),
        elapsed_ms=outcome.elapsed_ms,
    )
