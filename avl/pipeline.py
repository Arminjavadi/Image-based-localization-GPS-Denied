"""The localisation recipe — every training-free step the progress report measured.

A recipe names the encoder (one model, or an ensemble of several) and the steps
around it. Each step was measured on UAV-VisLoc (docs/AVL_Progress_Report):

  heading    turn the frame north-up from the compass before encoding, so one
             encode replaces the four-rotation search          (r05 MegaLoc +11 pts)
  centering  subtract the map's and the flight's mean descriptor (r05 +6, r10 +9)
  agl scale  crop the frame to the map tile's ground size, from height above
             ground; only when it covers > ``agl_gate`` x a tile (r11 +39)
  window     search only tiles within k sigma of the navigation prior (+12-19)
  fusion     collapse the top-5 into one position             (r05 ensemble +6)

scripts/visloc_eval.py, scripts/visloc_query.py, avl.localizer.AVLLocalizer and the
benchmark console all take these definitions from here, so a console row, a CLI run
and the on-board localizer cannot drift apart. Pure Python + NumPy: the console
imports it without Torch.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from avl.centering import CENTER_MODES
from avl.geo import DEFAULT_FUSION_METHOD, FUSION_METHODS
from avl.scale import CameraFootprint, frame_ground_m, query_scale

# ── model specs ─────────────────────────────────────────────────────────────────
# An ensemble is written as its members joined by "+", e.g. "megaloc+game4loc+anyloc-l".
ENSEMBLE_SEP = "+"


def split_model_spec(spec: str) -> list[str]:
    members = [m.strip() for m in str(spec).split(ENSEMBLE_SEP) if m.strip()]
    if not members:
        raise ValueError(f"empty model spec {spec!r}")
    if len(set(members)) != len(members):
        raise ValueError(f"a model is listed twice in {spec!r}")
    return members


def join_model_spec(members: list[str] | tuple[str, ...]) -> str:
    return ENSEMBLE_SEP.join(members)


def is_ensemble(spec: str) -> bool:
    return len(split_model_spec(spec)) > 1


# ── heading ─────────────────────────────────────────────────────────────────────
HEADING_MODES = ("off", "snap90", "exact", "auto")


def snap90(angle_deg: float) -> float:
    """The multiple of 90 degrees nearest ``angle_deg`` (a full square, no lost corners)."""
    return float(90.0 * np.round(angle_deg / 90.0))


def rotation_for_heading(heading_deg: float | None) -> float | None:
    """PIL (counter-clockwise) angle that turns a frame north-up.

    ``heading_deg`` is the compass direction the image's top edge points to,
    clockwise from north. UAV-VisLoc's ``yaw_deg`` follows this convention, which is
    why scripts/visloc_eval.py rotates by ``yaw_sign * yaw`` with ``yaw_sign = -1``.
    """
    if heading_deg is None or not np.isfinite(heading_deg):
        return None
    return -float(heading_deg)


@dataclass(frozen=True)
class QueryPlan:
    """How one frame becomes the image(s) the encoder sees."""

    #: PIL counter-clockwise angle applied before encoding; None = as captured.
    rotate_deg: float | None
    #: centre-crop fractions searched (1.0 = the whole view).
    scales: tuple[float, ...]
    agl_m: float | None = None
    #: ground side of the frame's full short-side square, metres.
    footprint_m: float | None = None

    @property
    def crop_fraction(self) -> float:
        return float(self.scales[0]) if len(self.scales) == 1 else float("nan")

    @property
    def cropped(self) -> bool:
        return len(self.scales) == 1 and self.scales[0] < 0.9999

    def as_dict(self) -> dict[str, Any]:
        return {
            "rotate_deg": self.rotate_deg,
            "crop_fraction": self.crop_fraction,
            "agl_m": self.agl_m,
            "footprint_m": self.footprint_m,
        }


def plan_query(
    width: int,
    height: int,
    *,
    rotate_deg: float | None = None,
    heading_mode: str = "off",
    agl_m: float | None = None,
    camera_k: float | None = None,
    tile_m: float | None = None,
    agl_gate: float | None = None,
    min_agl_m: float = 100.0,
    scales: tuple[float, ...] = (1.0,),
) -> QueryPlan:
    """Rotation and crop for one frame.

    ``rotate_deg`` is the PIL angle that would turn the frame north-up (see
    :func:`rotation_for_heading`). Heading modes:

      off     never rotate
      exact   rotate by the heading; the largest padding-free square is kept, which
              shrinks to ~0.71 of the short side at 45 deg
      snap90  rotate by the nearest multiple of 90 deg; the full square is kept
      auto    exact where the altitude crop crops anyway (the crop fits inside the
              rotated square), snap90 everywhere else

    Altitude scale runs when ``camera_k`` and ``tile_m`` are both given: one crop
    that makes the encoded square cover ``tile_m`` metres, applied only when the
    frame covers more than ``agl_gate`` x a tile (see :func:`avl.scale.query_scale`).
    Otherwise ``scales`` is used as given.
    """
    if heading_mode not in HEADING_MODES:
        raise ValueError(f"heading_mode must be one of {HEADING_MODES}")
    angle: float | None = None
    if heading_mode != "off" and rotate_deg is not None and np.isfinite(rotate_deg):
        angle = float(rotate_deg)
        if heading_mode == "snap90":
            angle = snap90(angle)

    footprint: float | None = None
    scale_on = camera_k is not None and tile_m is not None
    if scale_on:
        camera = CameraFootprint(float(camera_k), int(width))
        scales = query_scale(camera, agl_m, width, height, angle, tile_m, min_agl_m, agl_gate)
        if agl_m is not None and np.isfinite(agl_m):
            footprint = frame_ground_m(camera, float(agl_m), width, height)
    plan = QueryPlan(angle, tuple(float(s) for s in scales), agl_m, footprint)
    if heading_mode == "auto" and angle is not None and not (scale_on and plan.cropped):
        plan = replace(plan, rotate_deg=snap90(angle))
    return plan


@dataclass
class QueryTelemetry:
    """What the vehicle knows about a frame besides its pixels. Every field is optional;
    a step whose input is missing is skipped for that frame."""

    #: compass direction of the image's top edge, degrees clockwise from north
    heading_deg: float | None = None
    #: altitude above sea level (barometer); AGL = this - terrain at the prior
    altitude_asl_m: float | None = None
    #: height above ground, when known directly (radar altimeter); wins over ASL
    agl_m: float | None = None
    #: navigation prior (IMU / filter prediction); also where the terrain is read
    prior_lat: float | None = None
    prior_lon: float | None = None
    #: 1-sigma of the prior in metres; enables the search window
    prior_sigma_m: float | None = None


# ── datasets ───────────────────────────────────────────────────────────────────
_REGION_RE = re.compile(r"r(\d+)_t\d+")
_TILE_RE = re.compile(r"_t(\d+)")


def region_of(path: str | Path | None) -> str | None:
    """UAV-VisLoc region ('05') of a prepared map / query folder such as r05_t250."""
    if path is None:
        return None
    match = _REGION_RE.search(str(path))
    return match.group(1).zfill(2) if match else None


def tile_m_of(path: str | Path | None) -> float | None:
    """Tile ground size from a prepared map folder name ('r05_t250' -> 250 m)."""
    if path is None:
        return None
    path = Path(path)
    name = path.parent.name if path.suffix else path.name
    match = _TILE_RE.search(name)
    return float(match.group(1)) if match else None


# Camera constant k = full-frame ground width / AGL = 2 tan(HFOV / 2), measured on
# UAV-VisLoc by matching frames to the map at 11 patch sizes (report §5.7). Camera B
# (3976 px; regions 01-04, 06, 08, 09, 11) needs one constant. The 3000 px frames of
# 05 and 10 look like one lens too. r10's patch-size scan was unresolved (MegaLoc 1.48
# vs DenseUAV-ViT 1.20); frame-to-frame visual odometry measures the ground scale
# directly, and at k = 1.197 its steps come out 1.21x the GPS steps, so
# k = 1.197 / 1.21 = 0.988 (r05: 0.971). Retrieval is unchanged by this: the
# altitude gate already left every r10 frame uncropped at 1.197, and a smaller k only
# shrinks the footprint further. See docs/VO_AVL_Fusion_Experiment.md. On a real drone
# k comes from the lens datasheet.
UAV_VISLOC_CAMERA_K: dict[str, float] = {"05": 0.971, "06": 1.050, "10": 0.988, "11": 0.959}
UAV_VISLOC_CAMERA_B = ("01", "02", "03", "04", "06", "08", "09", "11")
CAMERA_B_K = 1.00


def camera_k_for(region: str | None) -> tuple[float | None, str]:
    """(k, where it came from) for a UAV-VisLoc region; (None, reason) when unknown."""
    if region is None:
        return None, "not a UAV-VisLoc region — enter the lens's k"
    region = region.zfill(2)
    if region in UAV_VISLOC_CAMERA_K:
        return UAV_VISLOC_CAMERA_K[region], f"calibrated for r{region}"
    if region in UAV_VISLOC_CAMERA_B:
        return CAMERA_B_K, "camera B constant"
    return None, f"no calibration for r{region}"


# ── recipe ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Recipe:
    """One configuration of the localisation pipeline."""

    encoders: tuple[str, ...] = ("megaloc",)
    heading: str = "off"
    center: str = "off"
    agl_scale: bool = False
    agl_gate: float | None = 1.25
    #: None = look it up for the region (:func:`camera_k_for`)
    camera_k: float | None = None
    #: simulated navigation prior; None = global search
    window_sigma_m: float | None = None
    window_k: float = 3.0
    fusion: str = DEFAULT_FUSION_METHOD
    #: only matters with heading off ("square" squares the frame first)
    query_crop: str = "square"
    #: None = 1 with heading, 4 without
    rotations: int | None = None
    #: sign that turns the query CSV's yaw into a PIL angle (UAV-VisLoc: -1)
    yaw_sign: float = -1.0

    def __post_init__(self) -> None:
        if not self.encoders:
            raise ValueError("a recipe needs at least one encoder")
        if self.heading not in HEADING_MODES:
            raise ValueError(f"heading must be one of {HEADING_MODES}")
        if self.center not in CENTER_MODES:
            raise ValueError(f"center must be one of {CENTER_MODES}")
        if self.fusion not in FUSION_METHODS:
            raise ValueError(f"fusion must be one of {FUSION_METHODS}")

    # -- derived -------------------------------------------------------------
    @property
    def model_spec(self) -> str:
        return join_model_spec(self.encoders)

    @property
    def is_ensemble(self) -> bool:
        return len(self.encoders) > 1

    @property
    def effective_rotations(self) -> int:
        if self.rotations is not None:
            return self.rotations
        return 1 if self.heading != "off" else 4

    def key(self) -> tuple:
        """What makes two runs the same configuration (fine parameters such as the
        camera constant or the heading sub-mode are deliberately left out)."""
        return (
            tuple(sorted(self.encoders)),
            self.heading != "off",
            self.effective_rotations,
            self.query_crop if self.heading == "off" else "-",
            self.center,
            bool(self.agl_scale),
            round(self.window_sigma_m) if self.window_sigma_m else None,
            self.fusion,
        )

    def tag_suffix(self) -> str:
        """File-name suffix after the model spec. The baseline configuration (and the
        centering-only variants) keep the tags the console has always written."""
        parts: list[str] = []
        if self.center != "off":
            parts.append({"map": "cmap", "map+flight": "cflight"}[self.center])
        if self.heading != "off":
            parts.append(f"h{self.heading}")
        elif self.effective_rotations != 4:
            parts.append(f"r{self.effective_rotations}")
        if self.heading == "off" and self.query_crop != "square":
            parts.append(f"crop{self.query_crop}")
        if self.agl_scale:
            parts.append(f"agl{self.agl_gate:g}" if self.agl_gate else "agl")
        if self.window_sigma_m:
            parts.append(f"w{self.window_sigma_m:g}")
        if self.fusion != DEFAULT_FUSION_METHOD:
            parts.append(f"f{self.fusion}")
        return "".join(f"__{p}" for p in parts)

    def eval_args(self, camera_k: float | None = None) -> list[str]:
        """scripts/visloc_eval.py flags for this recipe (map, queries, caches excluded).
        ``camera_k`` overrides :attr:`camera_k` (the console resolves it per region)."""
        args = [
            "--model", self.model_spec,
            "--rotations", str(self.effective_rotations),
            "--query-crop", self.query_crop,
            "--fusion", self.fusion,
            "--center", self.center,
        ]
        if self.heading != "off":
            args += [
                "--north-align",
                "--north-align-mode", self.heading,
                "--yaw-sign", f"{self.yaw_sign:g}",
            ]
        if self.agl_scale:
            k = camera_k if camera_k is not None else self.camera_k
            if k is None:
                raise ValueError("altitude scale needs a camera constant k")
            args += ["--scale-from-agl", "--camera-k", f"{k:g}"]
            if self.agl_gate:
                args += ["--agl-gate", f"{self.agl_gate:g}"]
        if self.window_sigma_m:
            args += ["--prior-sigma", f"{self.window_sigma_m:g}", "--prior-k", f"{self.window_k:g}"]
        return args

    def chips(self) -> list[str]:
        """Short labels of the steps that are on, for tables and headers."""
        out: list[str] = []
        if self.is_ensemble:
            out.append(f"ensemble ×{len(self.encoders)}")
        if self.heading != "off":
            out.append("heading")
        elif self.effective_rotations == 4:
            out.append("4-rot")
        if self.center != "off":
            out.append("center: flight" if self.center == "map+flight" else "center: map")
        if self.agl_scale:
            out.append(f"AGL ×{self.agl_gate:g}" if self.agl_gate else "AGL")
        if self.window_sigma_m:
            out.append(f"window σ{self.window_sigma_m:g}")
        if self.fusion != DEFAULT_FUSION_METHOD:
            out.append(f"fuse: {self.fusion}")
        return out

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["encoders"] = list(self.encoders)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Recipe":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in payload.items() if k in known}
        data["encoders"] = tuple(data.get("encoders", ("megaloc",)))
        return cls(**data)

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> "Recipe":
        """The recipe a saved scripts/visloc_eval.py run used, read from its summary."""
        encoders = tuple(split_model_spec(summary.get("model", "megaloc")))
        heading = "off"
        if summary.get("north_align"):
            heading = summary.get("north_align_mode") or "exact"
        scale = summary.get("scale_from_agl") or {}
        prior = summary.get("prior") or {}
        fusion = summary.get("fusion", DEFAULT_FUSION_METHOD)
        center = (summary.get("center") or {}).get("mode", "off")
        return cls(
            encoders=encoders,
            heading=heading if heading in HEADING_MODES else "exact",
            center=center if center in CENTER_MODES else "off",
            agl_scale=bool(scale.get("enabled")),
            agl_gate=scale.get("gate"),
            camera_k=scale.get("camera_k"),
            window_sigma_m=prior.get("sigma_m") if prior.get("enabled") else None,
            window_k=float(prior.get("k", 3.0)) if prior.get("enabled") else 3.0,
            fusion=fusion if fusion in FUSION_METHODS else DEFAULT_FUSION_METHOD,
            query_crop=summary.get("query_crop", "square"),
            rotations=int(summary.get("rotations", 4)),
            yaw_sign=float(summary.get("yaw_sign") or -1.0),
        )


ENSEMBLE_REPORT = ("megaloc", "game4loc", "anyloc-l")

#: name -> (label, one-line description, recipe)
PRESETS: dict[str, tuple[str, str, Recipe]] = {
    "baseline": (
        "Baseline",
        "MegaLoc, four-rotation search, raw descriptors — where the report started "
        "(r05 56.9 %).",
        Recipe(("megaloc",), query_crop="square"),
    ),
    "recommended": (
        "Recommended",
        "MegaLoc + heading + flight centering + gated altitude crop — the shipped "
        "pipeline. Mean of r05 / r10 / r06 / r11: 45 → 55 %; 46 GMAC per frame.",
        Recipe(("megaloc",), heading="auto", center="map+flight", agl_scale=True, agl_gate=1.25),
    ),
    "accuracy": (
        "Best accuracy",
        "MegaLoc + Game4Loc + AnyLoc-L ensemble with heading, flight centering and the "
        "altitude crop. r05: 82.6 % top-1, 88.2 % fused; 173 GMAC (3.8× MegaLoc).",
        Recipe(ENSEMBLE_REPORT, heading="auto", center="map+flight", agl_scale=True, agl_gate=1.25),
    ),
}
DEFAULT_PRESET = "recommended"

_BASE = PRESETS["baseline"][2]
_REC = PRESETS["recommended"][2]
_ACC = PRESETS["accuracy"][2]

#: The report's "progress step by step", as recipes: each step adds one method.
LADDER: list[tuple[str, Recipe]] = [
    ("MegaLoc, 4-rotation search", _BASE),
    ("+ heading (1 rotation)", Recipe(("megaloc",), heading="auto")),
    ("+ flight-mean centering", Recipe(("megaloc",), heading="auto", center="map+flight")),
    ("+ altitude scale (gated)", _REC),
    ("3-encoder ensemble", _ACC),
    ("+ IMU window σ 100 m", replace(_ACC, window_sigma_m=100.0)),
]


def describe_spec(spec: str) -> str:
    members = split_model_spec(spec)
    if len(members) == 1:
        return members[0]
    return f"ensemble: {' + '.join(members)}"


def gmac_estimate(encoders: tuple[str, ...] | list[str], gmac: dict[str, float]) -> float | None:
    """Sum of the members' per-image compute; None when any member is unmeasured."""
    total = 0.0
    for name in encoders:
        if name not in gmac or not math.isfinite(gmac[name]):
            return None
        total += gmac[name]
    return total
