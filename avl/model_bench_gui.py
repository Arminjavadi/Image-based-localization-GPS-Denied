"""AVL benchmark console — which encoder, and which recipe around it, localizes best.

The left panel builds a **recipe** (avl.pipeline): the map and the flight, the
encoder (one model, an ensemble, or several compared side by side), and the measured,
training-free steps around it — heading alignment, domain centering, the altitude
crop, the IMU search window and top-5 fusion. Each step shows what it measured in
docs/AVL_Progress_Report (avl.findings) and can be switched off; three presets
(Baseline, Recommended, Best accuracy) set them all at once.

Tabs. **Pipeline** draws the flow with the live shapes on every stage. **Comparison**
lists every saved run with the steps it used. **Findings** rebuilds the report's
"progress, step by step" from your own runs on the selected map — and runs the
missing steps — next to the story of every method. **Localize** drills into one
frame (from a run, or any image via scripts/visloc_query.py). **Trajectory** and
**Studio** fuse fixes with a simulated IMU. Every number comes from
scripts/visloc_eval.py, so the console and the command line cannot disagree; the
subprocess log sits in a drawer under every tab.

The look follows the "Nocturne" design system: a dark blue-grey ground, Inter,
a single blurple accent used as a line and a glow rather than a flood.

Run with: python scripts/run_model_bench.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QPointF, QProcess, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from avl.console_widgets import (
    ACCENT,
    ACCENT_300,
    BAD,
    BG,
    CHIPS_ROLE,
    CHROME,
    FAINT,
    FRACTION_ROLE,
    GOOD,
    LINE,
    MONO,
    MUTED,
    PANEL,
    RAISED,
    TEXT,
    TEXT_2,
    WARN,
    ActionButton,
    BarDelegate,
    BrandMark,
    ChipsDelegate,
    FindingCard,
    FlowNode,
    FlowStrip,
    IconButton,
    LadderChart,
    LadderStep,
    LogDrawer,
    MethodCard,
    NavRail,
    Segmented,
    StatTile,
    delta_html,
    ui_font,
)
from avl.findings import ENCODER_FACTS, FINDINGS, FINDINGS_BY_KEY, GMAC
from avl.geo import DEFAULT_FUSION_METHOD
from avl.light_gui_app import ImagePreviewLabel, find_project_root
from avl.nav.frames import LocalFrame
from avl.nav.raster import RegionRaster
from avl.pipeline import (
    DEFAULT_PRESET,
    ENSEMBLE_REPORT,
    LADDER,
    PRESETS,
    Recipe,
    camera_k_for,
    gmac_estimate,
    region_of,
    split_model_spec,
    tile_m_of,
)

# (dropdown label, --fusion value). Mirrors avl.geo.FUSION_METHODS; default first.
FUSION_CHOICES: list[tuple[str, str]] = [
    ("spatial cluster (robust, default)", "cluster"),
    ("softmax weighted", "softmax"),
    ("geometric median", "median"),
    ("weighted mean (legacy)", "weighted_mean"),
    ("top-1 only", "top1"),
]

# (dropdown label, avl.centering mode) — "off" is the step's switch
CENTER_CHOICES: list[tuple[str, str]] = [
    ("map + flight mean (causal)", "map+flight"),
    ("map mean only (works for one image)", "map"),
]
# (dropdown label, avl.pipeline heading mode) — "off" is the step's switch
HEADING_CHOICES: list[tuple[str, str]] = [
    ("auto · exact if cropped", "auto"),
    ("90° steps · full frame", "snap90"),
    ("exact angle · loses corners", "exact"),
]
# Rough CPU cost of one encode per GMAC on the development laptop (MegaLoc:
# 1.76 s / 45.9 GMAC, report §4) — only for "this will take ~N min" notes.
SECONDS_PER_GMAC = 1.76 / 45.9
# On-board projection (report §9, TensorRT FP16 on a Jetson Orin Nano 8 GB Super):
# MegaLoc ≈ 55 ms per encode at 45.9 GMAC. A projection, not a measurement.
ORIN_MS_PER_GMAC = 55.0 / 45.9

#: (nav icon, label, tooltip) per page, in rail order
PAGES = [
    ("overview", "Overview", "The selected recipe on the selected flight: results, pipeline, progress"),
    ("runs", "Runs", "Every saved run, side by side"),
    ("findings", "Findings", "What each method measured — report and your runs"),
    ("inspect", "Inspect", "One frame at a time: its top-5 tiles and how the recipe prepared it"),
    ("models", "Models", "The 18 encoders: compute and raw accuracy"),
    ("trajectory", "Flight", "Fuse fixes with a simulated IMU (error-state Kalman filter)"),
    ("studio", "Studio", "Draw a route on the satellite mosaic and fly it"),
]

# short label + pipeline-node blurb per fusion method
FUSION_NODE_TEXT: dict[str, tuple[str, str]] = {
    "cluster": ("spatial cluster", "largest geo-consistent group of the top-5, softmax-weighted"),
    "softmax": ("softmax weighted", "top-5 averaged, weight ∝ exp(score / T)"),
    "median": ("geometric median", "Weiszfeld median of the top-5, outlier-robust"),
    "weighted_mean": ("weighted mean", "score-weighted average of the top-5 (legacy)"),
    "top1": ("top-1 only", "best match, no fusion"),
}

# Trajectory tab — IMU error model and fusion filter (mirrors avl.nav)
TRAJ_IMU_CHOICES: list[tuple[str, str]] = [
    ("consumer MEMS", "consumer"),
    ("tactical grade", "tactical"),
    ("none — perfect IMU", "perfect"),
]
TRAJ_FILTER_CHOICES: list[tuple[str, str]] = [
    ("error-state EKF · IMU (15-state)", "eskf"),
    ("hybrid PF→KF · visual odometry", "hybrid"),
    ("particle filter · visual odometry", "pf"),
    ("Kalman + re-anchor · visual odometry", "kf_reanchor"),
    ("Kalman + search window · visual odometry", "kf_window"),
    ("robust pose graph · visual odometry", "pgo"),
]
#: filters that run on real frame-to-frame visual odometry (avl.nav.vo_fusion)
TRAJ_VO_FILTERS = {value for _, value in TRAJ_FILTER_CHOICES if value != "eskf"}

# Strongest first (report §4): the dropdown and the ensemble spec follow this order.
MODELS: list[tuple[str, str]] = [
    ("megaloc", "MegaLoc — general-purpose VPR (street, indoor, aerial), DINOv2 + SALAD"),
    ("game4loc", "Game4Loc — UAV↔satellite, cross-area split (GTA-UAV)"),
    ("anyloc-l", "AnyLoc (ViT-L) — DINOv2 value facet + VLAD, vocab fitted on the map"),
    ("denseuav-vit", "DenseUAV ViT — UAV↔satellite, trained on 14 Hangzhou campuses"),
    ("infogeo-dense", "InfoGeo — same model, DenseUAV-trained weights"),
    ("infogeo", "InfoGeo — UAV↔satellite, built for unseen regions (GTA-UAV weights, 448 px)"),
    ("anyloc-lite", "AnyLoc-lite — DINOv2 ViT-B + VLAD, vocabulary fitted on the selected map"),
    ("anyloc-sat", "AnyLoc-sat — DINOv3 ViT-L pre-trained on satellite imagery + VLAD"),
    ("anyloc-g", "AnyLoc (ViT-G) — paper backbone, DINOv2 value facet + VLAD (slow on CPU)"),
    ("dinov3-b", "DINOv3 ViT-B + GeM — frozen foundation model, nothing fitted"),
    ("dinov3-l", "DINOv3 ViT-L + GeM — frozen foundation model, nothing fitted"),
    ("sample4geo", "Sample4Geo — ConvNeXt cross-view retrieval, University-1652"),
    ("boq", "BoQ — DINOv2 + Bag-of-Queries, general VPR (GSV-Cities)"),
    ("dinov2", "DINOv2 — control: no cross-view training"),
    ("dinov2-ft", "DINOv2-FT — frozen DINOv2-B + head trained on UAV-VisLoc (non-test regions)"),
    ("mixvpr", "MixVPR — ground-level VPR baseline"),
    ("cosplace", "CosPlace — ground-level VPR baseline"),
    ("eigenplaces", "EigenPlaces — ground-level VPR, CosPlace successor"),
]

# Descriptor width + provenance for every encoder, mirrors avl/retrieval.py::NATIVE_DIM.
# (dim, family, training data, is cross-view). anyloc-lite's width is fitted per map.
ENCODER_INFO: dict[str, tuple[str, str, str, bool]] = {
    "denseuav-vit": ("512", "UAV↔satellite ViT", "DenseUAV — 14 Hangzhou campuses", True),
    "game4loc": ("768", "UAV↔satellite", "GTA-UAV — cross-area split", True),
    "sample4geo": ("1024", "UAV↔satellite ConvNeXt", "University-1652", True),
    "anyloc-lite": ("VLAD→512", "DINOv2-B + VLAD (token)", "vocabulary fit on the selected map", True),
    "anyloc-l": ("VLAD→512", "DINOv2-L + VLAD (value)", "vocabulary fit on the selected map", True),
    "anyloc-g": ("VLAD→512", "DINOv2-G + VLAD (value)", "vocabulary fit on the selected map", True),
    "anyloc-sat": ("VLAD→512", "DINOv3-L SAT + VLAD (value)", "SAT-493M SSL; vocab fit on the map", True),
    "megaloc": ("8448", "general VPR (DINOv2 + SALAD)", "street + landmark + indoor + aerial", False),
    "boq": ("12288", "general VPR (DINOv2 + BoQ)", "GSV-Cities — street level", False),
    "dinov3-b": ("768", "DINOv3-B + GeM (foundation)", "LVD-1689M web SSL — no VPR training", False),
    "dinov3-l": ("1024", "DINOv3-L + GeM (foundation)", "LVD-1689M web SSL — no VPR training", False),
    "infogeo": ("4096", "UAV↔satellite, cross-dataset (DINOv2-B)", "GTA-UAV", True),
    "infogeo-dense": ("4096", "UAV↔satellite, cross-dataset (DINOv2-B)", "DenseUAV", True),
    "dinov2": ("768", "DINOv2 (control)", "ImageNet SSL — no cross-view", False),
    "dinov2-ft": ("512", "DINOv2-B + trained head", "UAV-VisLoc 01-04,07-09,11", True),
    "mixvpr": ("4096", "ground-level VPR", "GSV-Cities", False),
    "cosplace": ("2048", "ground-level VPR", "SF-XL — ResNet101", False),
    "eigenplaces": ("2048", "ground-level VPR", "SF-XL — ResNet50", False),
}

ENCODER_INFO = {name: ENCODER_INFO[name] for name, _ in MODELS}

METRIC_COLUMNS = [
    ("config", "Encoder"),
    ("steps", "Steps"),
    ("map", "Map"),
    ("n_queries", "Frames"),
    ("within100", "Top-1 <100 m"),
    ("fused100", "Fused"),
    ("top5_within100", "Top-5"),
    ("top1_median", "Median"),
    ("chance100", "Chance"),
    ("gmac", "GMAC"),
    ("ms_per_query", "ms"),
    ("date", "Date · Jalali"),
]
#: header tooltips: what each compact column means
METRIC_TIPS = {
    "within100": "Top-1 within 100 m: the rank-1 tile's centre lies within 100 m of the GPS position",
    "fused100": "The top-5 fused into one position (the recipe's fusion), within 100 m",
    "top5_within100": "At least one of the top-5 tiles within 100 m",
    "top1_median": "Median distance of the rank-1 tile from the truth",
    "chance100": "A random tile from the map within 100 m — what guessing scores",
    "gmac": "Encoder compute per frame (billions of multiply-accumulates), all views",
    "ms_per_query": "Measured ms per frame on this machine; 'cached' = re-scored, not a latency",
}
COL = {key: i for i, (key, _) in enumerate(METRIC_COLUMNS)}
HEADLINE_COL = COL["within100"]

TOPK_COLUMNS = ["#", "Tile id", "Latitude", "Longitude", "Score", "Error"]

# Persian (Jalali / Shamsi) month names, index 1..12.
JALALI_MONTHS = [
    "", "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Convert a Gregorian date to the Jalali calendar (Pournader/Toossi algorithm)."""
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        365 * gy
        + (gy2 + 3) // 4
        - (gy2 + 99) // 100
        + (gy2 + 399) // 400
        - 80
        + gd
        + g_d_m[gm - 1]
    )
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def jalali_datetime_text(when: datetime) -> str:
    """'1403/03/12 14:07' — numeric Jalali, kept LTR so it reads cleanly in a table."""
    jy, jm, jd = gregorian_to_jalali(when.year, when.month, when.day)
    return f"{jy}/{jm:02d}/{jd:02d} {when:%H:%M}"


def jalali_long_text(when: datetime) -> str:
    """'12 خرداد 1403 · 14:07' — month-name form for single-line detail text."""
    jy, jm, jd = gregorian_to_jalali(when.year, when.month, when.day)
    return f"{jd} {JALALI_MONTHS[jm]} {jy} · {when:%H:%M}"

# Nocturne design tokens live in avl.console_widgets (one source for both modules).


@dataclass
class Match:
    """One retrieved reference tile, however it was produced."""

    image_id: str
    latitude: float | None
    longitude: float | None
    score: float | None
    error_m: float | None
    image_path: Path | None


def coords_text(latitude: Any, longitude: Any) -> str:
    if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
        return "-"
    return f"{float(latitude):.6f}, {float(longitude):.6f}"


def id_text(value: Any) -> str:
    """Tile ids read back from CSV can arrive as floats (2256.0); ids are strings."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def split_floats(value: Any) -> list[float]:
    if value is None or pd.isna(value):
        return []
    out: list[float] = []
    for part in str(value).split("|"):
        try:
            out.append(float(part))
        except ValueError:
            break
    return out


def plan_text(rotate_deg: Any, crop_fraction: Any, agl_m: Any, extra: str = "") -> str:
    """'prepared: north-up −90° · crop 0.54 · AGL 736 m' for one frame's QueryPlan."""
    def finite(v: Any) -> bool:
        try:
            return v is not None and float(v) == float(v)
        except (TypeError, ValueError):
            return False

    bits = []
    if finite(rotate_deg):
        bits.append(f"north-up {float(rotate_deg):+.0f}°")
    if finite(crop_fraction):
        bits.append("full frame" if float(crop_fraction) >= 0.9999 else f"crop {float(crop_fraction):.2f}")
    if finite(agl_m):
        bits.append(f"AGL {float(agl_m):.0f} m")
    if extra:
        bits.append(extra)
    return ("prepared:  " + "  ·  ".join(bits)) if bits else ""


def resolve_path(image_path: Any, base: Path | None) -> Path | None:
    if image_path is None or not str(image_path) or pd.isna(image_path):
        return None
    path = Path(str(image_path))
    if path.is_absolute() or base is None:
        return path
    return base / path


class SortableItem(QTableWidgetItem):
    """A table cell that sorts by an explicit numeric key rather than its text,
    so '81 m' sorts before '1258 m' and '9.6%' before '54.2%'. Missing / NaN keys
    sort to the bottom on an ascending sort."""

    def __init__(self, text: str, sort_key: Any) -> None:
        super().__init__(text)
        try:
            key = float(sort_key)
        except (TypeError, ValueError):
            key = float("inf")
        self._key = float("inf") if key != key else key  # NaN -> +inf

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableItem):
            return self._key < other._key
        return super().__lt__(other)


class RetrievalPanel(QWidget):
    """Query frame, estimated position, and the top-5 tiles with their coordinates.

    Shared by the inspector and the single-image tab so a saved benchmark row and a
    live lookup are always presented identically.
    """

    def __init__(self, empty_text: str = "Select a query") -> None:
        super().__init__()
        self.setObjectName("retrievalPane")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 6)
        layout.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(14)
        frame_card = QFrame()
        frame_card.setObjectName("card")
        fv = QVBoxLayout(frame_card)
        fv.setContentsMargins(16, 13, 16, 16)
        fv.setSpacing(10)
        cap = QLabel("QUERY FRAME")
        cap.setObjectName("panelTitle")
        fv.addWidget(cap)
        self.query_preview = ImagePreviewLabel(empty_text, 360, 300)
        fv.addWidget(self.query_preview, 1)
        top.addWidget(frame_card, 3)

        position = QFrame()
        position.setObjectName("card")
        pv = QVBoxLayout(position)
        pv.setContentsMargins(18, 13, 18, 16)
        pv.setSpacing(10)
        cap2 = QLabel("POSITION")
        cap2.setObjectName("panelTitle")
        pv.addWidget(cap2)
        self.error_label = QLabel("-")
        self.error_label.setObjectName("errorHero")
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pv.addWidget(self.error_label)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.gt_label = QLabel("-")
        self.fused_label = QLabel("-")
        self.top1_label = QLabel("-")
        for row, (label, widget) in enumerate(
            [
                ("Ground truth", self.gt_label),
                ("Fused estimate", self.fused_label),
                ("Top-1 tile", self.top1_label),
            ]
        ):
            name = QLabel(label)
            name.setObjectName("posLabel")
            grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
            widget.setObjectName("coord")
            widget.setWordWrap(True)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(widget, row, 1)
        grid.setColumnStretch(1, 1)
        pv.addLayout(grid)
        self.plan_label = QLabel("")
        self.plan_label.setObjectName("planLine")
        self.plan_label.setWordWrap(True)
        self.plan_label.setToolTip("How the recipe prepared this frame before encoding")
        pv.addWidget(self.plan_label)
        pv.addStretch(1)
        top.addWidget(position, 2)
        layout.addLayout(top)

        tiles_card = QFrame()
        tiles_card.setObjectName("card")
        tv = QVBoxLayout(tiles_card)
        tv.setContentsMargins(16, 13, 16, 12)
        tv.setSpacing(10)
        cap3 = QLabel("RETRIEVED TILES  ·  rank 1 → 5")
        cap3.setObjectName("panelTitle")
        tv.addWidget(cap3)
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_previews: list[ImagePreviewLabel] = []
        self.tile_labels: list[QLabel] = []
        for rank in range(5):
            column = QVBoxLayout()
            column.setSpacing(6)
            thumb = ImagePreviewLabel(f"#{rank + 1}", 110, 130)
            caption = QLabel("-")
            caption.setObjectName("tileCaption")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(thumb)
            column.addWidget(caption)
            tiles.addLayout(column, 1)
            self.tile_previews.append(thumb)
            self.tile_labels.append(caption)
        tv.addLayout(tiles)

        self.topk_table = QTableWidget(0, len(TOPK_COLUMNS))
        self.topk_table.setHorizontalHeaderLabels(TOPK_COLUMNS)
        self.topk_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.topk_table.verticalHeader().setVisible(False)
        self.topk_table.setMinimumHeight(70)
        tv.addWidget(self.topk_table)
        layout.addWidget(tiles_card)
        layout.addStretch(1)

    # ------------------------------------------------------------ rendering --
    def clear(self) -> None:
        for label in (self.gt_label, self.fused_label, self.top1_label, self.error_label):
            label.setText("-")
        self.plan_label.setText("")
        self.topk_table.setRowCount(0)
        self._fit_table()
        self.query_preview.set_path(None)
        for thumb, caption in zip(self.tile_previews, self.tile_labels):
            thumb.set_path(None)
            caption.setText("-")

    def set_result(
        self,
        query_image: Path | None,
        matches: list[Match],
        ground_truth: tuple[Any, Any] | None = None,
        fused: tuple[Any, Any] | None = None,
        error_parts: list[str] | None = None,
        plan_text: str | None = None,
    ) -> None:
        self.query_preview.set_path(query_image)
        self.plan_label.setText(plan_text or "")
        self.gt_label.setText(coords_text(*ground_truth) if ground_truth else "-")
        self.fused_label.setText(coords_text(*fused) if fused else "-")

        if matches:
            best = matches[0]
            suffix = f", score {best.score:.3f}" if best.score is not None else ""
            self.top1_label.setText(
                f"{coords_text(best.latitude, best.longitude)}   ({best.image_id}{suffix})"
            )
        else:
            self.top1_label.setText("-")
        self.error_label.setText("  ·  ".join(error_parts or []) or "—")

        self.topk_table.setRowCount(0)
        for rank, match in enumerate(matches, start=1):
            values = [
                str(rank),
                match.image_id,
                f"{match.latitude:.6f}" if match.latitude is not None else "-",
                f"{match.longitude:.6f}" if match.longitude is not None else "-",
                f"{match.score:.4f}" if match.score is not None else "-",
                f"{match.error_m:.0f} m" if match.error_m is not None else "-",
            ]
            row = self.topk_table.rowCount()
            self.topk_table.insertRow(row)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 5 and match.error_m is not None:
                    item.setForeground(
                        QColor("#7ee0a8") if match.error_m <= 100 else QColor("#f07a7a")
                    )
                self.topk_table.setItem(row, column, item)
        self._fit_table()

        for rank, (thumb, caption) in enumerate(zip(self.tile_previews, self.tile_labels)):
            if rank >= len(matches):
                thumb.set_path(None)
                caption.setText("-")
                continue
            match = matches[rank]
            thumb.set_path(match.image_path)
            lines = [f"#{rank + 1}" + (f" · {match.error_m:.0f} m" if match.error_m is not None else "")]
            if match.latitude is not None and match.longitude is not None:
                lines.append(f"{match.latitude:.5f}")
                lines.append(f"{match.longitude:.5f}")
            caption.setText("\n".join(lines))

    def _fit_table(self) -> None:
        """Size the table to its rows: five results should never need scrolling."""
        rows = sum(self.topk_table.rowHeight(row) for row in range(self.topk_table.rowCount()))
        self.topk_table.setFixedHeight(self.topk_table.horizontalHeader().height() + rows + 6)


class TrajectoryPlot(QWidget):
    """A minimal multi-series line plot (no matplotlib / pyqtgraph).

    Draws one or more polylines in a shared data rectangle with auto-fit,
    light gridlines and tick labels, an optional legend, optional shaded
    vertical bands (fix dropouts), and an optional equal-aspect mode for the
    ground-track view. Series with ``dots=True`` are rendered as points.
    """

    BG = "#101118"
    AXIS = "#2b2d3a"
    TICK = "#6f7385"
    BAND = QColor(240, 192, 74, 26)

    def __init__(self, xlabel: str = "", ylabel: str = "", equal_aspect: bool = False) -> None:
        super().__init__()
        self.setMinimumHeight(140)
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._equal = equal_aspect
        self._series: list[dict] = []
        self._bands: list[tuple[float, float]] = []
        self._empty = "no run yet"

    def set_empty(self, text: str) -> None:
        self._empty = text
        self._series = []
        self.update()

    def set_series(self, series: list[dict], bands: list[tuple[float, float]] | None = None) -> None:
        """``series`` items: {name, pts: [(x, y), ...], color, dots?, width?}."""
        self._series = [s for s in series if s.get("pts")]
        self._bands = list(bands or [])
        self.update()

    # -- painting -------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(self.BG))

        m_l, m_r, m_t, m_b = 54, 12, 10, 34
        w, h = self.width(), self.height()
        plot = QRectF(m_l, m_t, max(1, w - m_l - m_r), max(1, h - m_t - m_b))

        if not self._series:
            p.setPen(QColor(self.TICK))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty)
            return

        xs = [x for s in self._series for x, _ in s["pts"]]
        ys = [y for s in self._series for _, y in s["pts"]]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax - xmin < 1e-9:
            xmax += 1.0
        if ymax - ymin < 1e-9:
            ymax += 1.0
        pad_x = 0.03 * (xmax - xmin)
        pad_y = 0.06 * (ymax - ymin)
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y

        if self._equal:
            sx = plot.width() / (xmax - xmin)
            sy = plot.height() / (ymax - ymin)
            s = min(sx, sy)
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            xmin, xmax = cx - plot.width() / (2 * s), cx + plot.width() / (2 * s)
            ymin, ymax = cy - plot.height() / (2 * s), cy + plot.height() / (2 * s)

        def X(v: float) -> float:
            return plot.left() + (v - xmin) / (xmax - xmin) * plot.width()

        def Y(v: float) -> float:
            return plot.bottom() - (v - ymin) / (ymax - ymin) * plot.height()

        # dropout bands
        for a, b in self._bands:
            p.fillRect(QRectF(X(a), plot.top(), max(1.0, X(b) - X(a)), plot.height()), self.BAND)

        # grid + ticks
        p.setFont(QFont(self.font().family(), 8))
        for i in range(5):
            gx = plot.left() + i / 4 * plot.width()
            gy = plot.bottom() - i / 4 * plot.height()
            p.setPen(QPen(QColor(self.AXIS), 1))
            p.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))
            p.drawLine(QPointF(plot.left(), gy), QPointF(plot.right(), gy))
            p.setPen(QColor(self.TICK))
            p.drawText(QRectF(gx - 40, plot.bottom() + 4, 80, 14),
                       Qt.AlignmentFlag.AlignCenter, self._fmt(xmin + i / 4 * (xmax - xmin)))
            p.drawText(QRectF(2, gy - 7, m_l - 8, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self._fmt(ymin + i / 4 * (ymax - ymin)))

        p.setPen(QColor(self.TICK))
        if self._xlabel:
            p.drawText(QRectF(plot.left(), h - 15, plot.width(), 14),
                       Qt.AlignmentFlag.AlignCenter, self._xlabel)

        # series
        for s in self._series:
            col = QColor(s.get("color", "#b5abfc"))
            pts = [QPointF(X(x), Y(y)) for x, y in s["pts"]]
            if s.get("dots"):
                p.setPen(QPen(col, 1))
                p.setBrush(col)
                r = float(s.get("width", 2.6))
                for pt in pts:
                    p.drawEllipse(pt, r, r)
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(QPen(col, float(s.get("width", 1.8))))
                p.drawPolyline(QPolygonF(pts))

        # legend
        p.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(p.font())
        lx, ly = plot.left() + 8, plot.top() + 6
        for s in self._series:
            col = QColor(s.get("color", "#b5abfc"))
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(lx, ly + 6), QPointF(lx + 16, ly + 6))
            p.setPen(QColor("#cfd2de"))
            p.drawText(QPointF(lx + 22, ly + 10), s["name"])
            lx += 34 + fm.horizontalAdvance(s["name"])

    @staticmethod
    def _fmt(v: float) -> str:
        a = abs(v)
        if a >= 1000:
            return f"{v/1000:.1f}k"
        if a >= 10:
            return f"{v:.0f}"
        if a >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"


class MosaicView(QGraphicsView):
    """A pan/zoom view of a region satellite thumbnail for Trajectory Studio.

    In *draw mode* a left click appends a vertex to the planned path (dashed
    amber). ``draw_polyline`` / ``draw_markers`` add result overlays; the scene
    coordinate system is the thumbnail's own pixels, so callers convert with the
    thumbnail scale + the ``RegionRaster`` transform.
    """

    def __init__(self, on_vertex) -> None:
        super().__init__()
        self._on_vertex = on_vertex
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#101118"))
        self.setMinimumHeight(240)
        self._bg = None
        self._draw = False
        self._verts: list[QPointF] = []
        self._path_items: list = []
        self._overlay_items: list = []
        self._fit_target: QRectF | None = None
        self.setToolTip("double-click to zoom back out to the whole mosaic")

    # -- mosaic -------------------------------------------------------
    def set_mosaic(self, pil_image) -> None:
        self._scene.clear()
        self._bg = None
        self._verts.clear()
        self._path_items.clear()
        self._overlay_items.clear()
        self._fit_target = None
        data = pil_image.tobytes("raw", "RGB")
        qim = QImage(data, pil_image.width, pil_image.height,
                     3 * pil_image.width, QImage.Format.Format_RGB888).copy()
        self._bg = self._scene.addPixmap(QPixmap.fromImage(qim))
        self._scene.setSceneRect(QRectF(0, 0, pil_image.width, pil_image.height))
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_content(self, rect: QRectF | None) -> None:
        self._fit_target = rect
        target = rect if (rect is not None and rect.isValid()) else self._scene.sceneRect()
        self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)

    def fit_overlays(self, pad_frac: float = 0.18) -> None:
        rect: QRectF | None = None
        for it in self._overlay_items:
            br = it.sceneBoundingRect()
            rect = br if rect is None else rect.united(br)
        if rect is None or not rect.isValid():
            return
        pad = max(rect.width(), rect.height()) * pad_frac + 24
        self.fit_content(rect.adjusted(-pad, -pad, pad, pad))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit_content(None)
        super().mouseDoubleClickEvent(event)

    def clear_mosaic(self) -> None:
        self._scene.clear()
        self._bg = None
        self._verts.clear()

    # -- planned path ---------------------------------------------
    def set_draw_mode(self, on: bool) -> None:
        self._draw = bool(on)
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def reset_path(self) -> None:
        for it in self._path_items:
            self._scene.removeItem(it)
        self._path_items.clear()
        self._verts.clear()

    def path_vertices(self) -> list[QPointF]:
        return list(self._verts)

    def clear_overlay(self) -> None:
        for it in self._overlay_items:
            self._scene.removeItem(it)
        self._overlay_items.clear()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._draw and event.button() == Qt.MouseButton.LeftButton and self._bg is not None:
            self._verts.append(self.mapToScene(event.position().toPoint()))
            self._redraw_path()
            return
        super().mousePressEvent(event)

    def _redraw_path(self) -> None:
        for it in self._path_items:
            self._scene.removeItem(it)
        self._path_items.clear()
        if not self._verts:
            return
        r = max(3.0, self._scene.width() / 400)
        pen = QPen(QColor("#f0c04a"), max(1.5, r / 2))
        pen.setStyle(Qt.PenStyle.DashLine)
        if len(self._verts) >= 2:
            path = QPainterPath(self._verts[0])
            for v in self._verts[1:]:
                path.lineTo(v)
            self._path_items.append(self._scene.addPath(path, pen))
        dot = QBrush(QColor("#f0c04a"))
        for v in self._verts:
            self._path_items.append(
                self._scene.addEllipse(v.x() - r, v.y() - r, 2 * r, 2 * r, QPen(Qt.PenStyle.NoPen), dot)
            )

    # -- overlays -------------------------------------------------
    def draw_polyline(self, pts: list[QPointF], color: str, width: float, dashed: bool = False) -> None:
        if len(pts) < 2:
            return
        pen = QPen(QColor(color), width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        self._overlay_items.append(self._scene.addPath(path, pen))

    def draw_markers(self, pts: list[QPointF], color: str, radius: float) -> None:
        brush = QBrush(QColor(color))
        pen = QPen(Qt.PenStyle.NoPen)
        for p in pts:
            self._overlay_items.append(
                self._scene.addEllipse(p.x() - radius, p.y() - radius, 2 * radius, 2 * radius, pen, brush)
            )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._bg is not None:
            self.fit_content(self._fit_target)


class ModelBenchConsole(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project_root = find_project_root()
        self.out_dir = self.project_root / "artifacts" / "visloc"
        self.queue: list[tuple[str, list[str]]] = []
        self.process: QProcess | None = None
        self.current_tag: str | None = None
        self.single_process: QProcess | None = None
        self.single_result_path: Path | None = None
        self.traj_process: QProcess | None = None
        self.traj_result_path: Path | None = None
        self._traj_dest: str = "tab"          # "tab" | "studio"
        self._studio_raster: RegionRaster | None = None
        self._studio_scale: float = 1.0
        self._studio_subset: list[str] = []
        self._studio_subset_path: Path | None = None
        self._stopping: bool = False
        self._pending_inspect_tag: str | None = None
        self._queue_total: int = 0
        self._run_index: int = 0
        self._eval_total: int | None = None
        self._per_query: pd.DataFrame | None = None
        self._ref_lookup: pd.DataFrame | None = None
        self._queries_lookup: pd.DataFrame | None = None
        self._queries_frame: pd.DataFrame | None = None
        self._tile_dir: Path | None = None
        self._queries_dir: Path | None = None
        self._query_stride: int = 1
        self._sections: dict[str, QWidget] = {}
        self._pipe_phase: str | None = None
        # progress / ETA bookkeeping
        self._run_t0: float | None = None          # wall clock when the current run started
        self._run_walltimes: list[float] = []      # finished-run durations, for queue ETA
        self._phase: str | None = None             # "encode-ref" | "queries"
        self._phase_t0: float | None = None        # wall clock when this phase's counter began
        self._phase_done0: int = 0                 # counter value at _phase_t0
        # recipe panel
        self._summaries: list[tuple[Path, dict]] = []  # every saved run, newest first
        self._recipe_guard = False      # True while a preset fills the panel
        self._flight_cols: set[str] = set()  # columns of the selected flight CSV
        self._k_entered = False         # camera constant typed by hand
        self._k_region: str | None = None  # region the camera constant belongs to
        self._missing_steps: list[Recipe] = []
        self._finding_cards: dict[str, QWidget] = {}

        self.setWindowTitle("AVL Console")
        self.resize(1560, 960)
        # keep the window shrinkable on smaller screens — tall pages scroll instead
        self.setMinimumSize(1180, 640)
        self._build_ui()
        self._apply_style()
        self.refresh_databases()
        self._apply_preset(DEFAULT_PRESET)
        self.reload_results()
        self._refresh_pipeline()
        self.set_status("READY")

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setFont(ui_font(13))
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.log_drawer = LogDrawer()  # first: every builder may log
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_progress_strip())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.nav = NavRail()
        # the inspector first: pages connect to its controls while they are built
        self.inspector = self._build_controls()
        body.addWidget(self.nav)
        body.addWidget(self._build_results(), 1)
        body.addWidget(self.inspector)
        outer.addLayout(body, 1)
        outer.addWidget(self.log_drawer)
        outer.addWidget(self._build_statusbar())

    def _build_header(self) -> QWidget:
        """Top bar: brand · map + flight · the next run · Run / Stop · inspector."""
        header = QWidget()
        header.setObjectName("topbar")
        header.setFixedHeight(56)
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 0, 14, 0)
        row.setSpacing(10)

        row.addWidget(BrandMark(26))
        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("AVL Console")
        title.setObjectName("appTitle")
        subtitle = QLabel("GPS-denied localization")
        subtitle.setObjectName("appSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        row.addLayout(brand)
        row.addSpacing(14)
        row.addWidget(self._vrule())
        row.addSpacing(6)

        self.refs_combo = self._dataset_combo()
        self.refs_combo.setObjectName("ctxCombo")
        self.refs_combo.currentIndexChanged.connect(self._refs_changed)
        self.queries_combo = self._dataset_combo()
        self.queries_combo.setObjectName("ctxCombo")
        self.queries_combo.currentIndexChanged.connect(self._queries_changed)
        for caption, combo in (("MAP", self.refs_combo), ("FLIGHT", self.queries_combo)):
            label = QLabel(caption)
            label.setObjectName("ctxLabel")
            row.addWidget(label)
            row.addWidget(combo)
        rescan = IconButton("refresh", "Rescan data/ for maps and flights")
        rescan.clicked.connect(self.refresh_databases)
        row.addWidget(rescan)
        row.addStretch(1)

        # the recipe the next run will use, always in view
        self.recipe_line = QLabel("")
        self.recipe_line.setObjectName("recipeLine")
        self.recipe_line.setTextFormat(Qt.TextFormat.RichText)
        self.recipe_line.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recipe_line.setToolTip("The recipe the next run uses — click to edit it")
        self.recipe_line.mousePressEvent = lambda _e: self._focus_section("recipe")
        row.addWidget(self.recipe_line)
        row.addSpacing(4)

        self.stop_button = ActionButton("stop", "Stop", "stopBtn")
        self.stop_button.clicked.connect(self.stop_benchmark)
        self.stop_button.setEnabled(False)
        self.run_button = ActionButton("play", "Run benchmark", "runBtn")
        self.run_button.clicked.connect(self.run_benchmark)
        self.run_button.setToolTip("Run the recipe on the selected map + flight (scripts/visloc_eval.py)")
        row.addWidget(self.stop_button)
        row.addWidget(self.run_button)
        row.addSpacing(4)
        self.inspector_toggle = IconButton("panel", "Show / hide the recipe inspector", checkable=True)
        self.inspector_toggle.setChecked(True)
        self.inspector_toggle.toggled.connect(lambda on: self.inspector.setVisible(on))
        row.addWidget(self.inspector_toggle)
        return header

    @staticmethod
    def _vrule() -> QFrame:
        rule = QFrame()
        rule.setObjectName("vrule")
        rule.setFixedSize(1, 26)
        return rule

    def _build_statusbar(self) -> QWidget:
        """Bottom status bar: state · job + ETA · queue · output panel."""
        bar = QWidget()
        bar.setObjectName("statusbar")
        bar.setFixedHeight(28)
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 0, 8, 0)
        row.setSpacing(12)

        self.status_pill = QFrame()
        self.status_pill.setObjectName("statusPill")
        pill = QHBoxLayout(self.status_pill)
        pill.setContentsMargins(8, 2, 10, 2)
        pill.setSpacing(7)
        self.status_dot = QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setFixedSize(7, 7)
        self.status = QLabel("READY")
        self.status.setObjectName("statusText")
        pill.addWidget(self.status_dot)
        pill.addWidget(self.status)
        row.addWidget(self.status_pill)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progressLabel")
        self.progress_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.progress_label, 1)

        self.queue_line = QLabel("")
        self.queue_line.setObjectName("statusText2")
        row.addWidget(self.queue_line)
        device = QLabel("GPU" if shutil.which("nvidia-smi") else "CPU")
        device.setObjectName("statusText2")
        device.setToolTip("Encoders run on the GPU when CUDA is available, otherwise on the CPU")
        row.addWidget(device)
        self.log_toggle = QPushButton("Output")
        self.log_toggle.setObjectName("statusBtn")
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self.log_drawer.set_open)
        self.log_drawer.toggled.connect(
            lambda on: (self.log_toggle.blockSignals(True), self.log_toggle.setChecked(on),
                        self.log_toggle.blockSignals(False))
        )
        row.addWidget(self.log_toggle)
        return bar

    def _build_progress_strip(self) -> QWidget:
        """A 2 px activity line under the top bar; its words go to the status bar."""
        strip = QWidget()
        strip.setObjectName("progressStrip")
        strip.setFixedHeight(2)
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setObjectName("progress")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(2)
        lay.addWidget(self.progress)
        self._progress_strip = self.progress
        self.progress.hide()
        return strip

    # ----------------------------------------------------- progress bar ----
    @staticmethod
    def _fmt_eta(seconds: float | None) -> str:
        if seconds is None or seconds < 0 or seconds != seconds:  # None / negative / NaN
            return "—"
        seconds = int(round(seconds))
        if seconds >= 3600:
            return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
        if seconds >= 60:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        return f"{seconds}s"

    def _phase_reset(self, phase: str) -> None:
        """Anchor a fresh rate estimate for a new counted phase."""
        self._phase = phase
        self._phase_t0 = None
        self._phase_done0 = 0

    def _phase_eta(self, done: int, total: int) -> float | None:
        """Seconds left in the current phase from the rate seen since it started."""
        now = time.monotonic()
        if self._phase_t0 is None or done <= self._phase_done0:
            self._phase_t0, self._phase_done0 = now, done
            return None
        rate = (done - self._phase_done0) / max(now - self._phase_t0, 1e-6)  # items/sec
        if rate <= 0:
            return None
        return (total - done) / rate

    def _queue_eta(self, this_run_left: float | None) -> float | None:
        """This run's remaining time plus a projection for the still-queued runs."""
        if this_run_left is None:
            return None
        runs_left = len(self.queue)
        if runs_left == 0:
            return this_run_left
        if self._run_walltimes:
            per_run = sum(self._run_walltimes) / len(self._run_walltimes)
        elif self._run_t0 is not None:
            per_run = (time.monotonic() - self._run_t0) + this_run_left  # projected total
        else:
            return this_run_left
        return this_run_left + per_run * runs_left

    def _progress_start(self, label: str, total: int | None = None) -> None:
        self._eval_total = total if total and total > 0 else None
        self.progress.setRange(0, self._eval_total or 0)  # (0, 0) = indeterminate
        self.progress.setValue(0)
        self.progress_label.setText(label)
        self._progress_strip.show()
        self._set_node("result", "—", "fused latitude / longitude")
        self._set_pipe_phase("encode")

    def _progress_update(
        self, done: int | None = None, total: int | None = None, label: str | None = None
    ) -> None:
        if total and total > 0:
            self._eval_total = total
            self.progress.setRange(0, total)
        if done is not None and self._eval_total:
            self.progress.setValue(min(done, self._eval_total))
        if label is not None:
            self.progress_label.setText(label)

    def _progress_phase(self, phase: str, done: int, total: int, note: str) -> None:
        """Bar + '<prefix> · <note> N/M · P% · ETA … (run k/N, ~… total)'."""
        if total <= 0:
            return
        if self._phase != phase:
            self._phase_reset(phase)
        self._eval_total = total
        self.progress.setRange(0, total)
        self.progress.setValue(min(done, total))
        run_left = self._phase_eta(done, total)
        pct = int(round(100 * done / total))
        bits = [f"{self.current_tag or 'run'}  {note} {done}/{total}  ({pct}%)",
                f"ETA {self._fmt_eta(run_left)}"]
        if self._queue_total > 1:
            bits.append(
                f"run {self._run_index}/{self._queue_total}  ·  "
                f"~{self._fmt_eta(self._queue_eta(run_left))} left overall"
            )
        self.progress_label.setText("   ·   ".join(bits))

    def _progress_done(self) -> None:
        self._eval_total = None
        self._phase = None
        self.progress.reset()
        self.progress_label.setText("")
        self._progress_strip.hide()

    def _section(
        self, number: int, title: str, key: str | None = None
    ) -> tuple[QWidget, QVBoxLayout]:
        """An inspector section: a small caps header over a body layout."""
        container = QWidget()
        wrap = QVBoxLayout(container)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        wrap.addWidget(label)
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        wrap.addLayout(body)
        if key is not None:
            container.setObjectName("pipeSection")
            self._sections[key] = container
        return container, body

    def _focus_section(self, key: str) -> None:
        """Open the inspector at a section and flash it."""
        target = self._sections.get(key)
        if target is None:
            return
        if not self.inspector.isVisible():
            self.inspector_toggle.setChecked(True)
        self._panel_scroll.ensureWidgetVisible(target, 0, 40)
        target.setProperty("flash", "1")
        self._repolish(target)
        QTimer.singleShot(
            900,
            lambda: (target.setProperty("flash", "0"), self._repolish(target)),
        )

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _labeled(self, title: str, hint: str | None, widget: QWidget) -> QVBoxLayout:
        """Field label (with an optional right-aligned hint) stacked over its control."""
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        lab = QLabel(title)
        lab.setObjectName("fieldLabel")
        head.addWidget(lab)
        if hint:
            head.addStretch(1)
            extra = QLabel(hint)
            extra.setObjectName("fieldHint")
            head.addWidget(extra)
        box.addLayout(head)
        box.addWidget(widget)
        return box

    @staticmethod
    def _encoder_blurb(name: str | None) -> str:
        for model_name, description in MODELS:
            if model_name == name:
                return description.split("—", 1)[1].strip()
        return "-"

    def _dataset_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setMinimumContentsLength(20)
        combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        combo.view().setMinimumWidth(460)
        return combo

    def _build_controls(self) -> QWidget:
        """The inspector (right): the data's capabilities, then the recipe."""
        panel = QWidget()
        panel.setObjectName("inspectorBody")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(16, 14, 16, 18)
        outer.setSpacing(18)

        # data: what the selected map + flight carry (the pickers are in the top bar)
        sec1, b1 = self._section(1, "DATA", key="search")
        data = QFrame()
        data.setObjectName("insetCard")
        dv = QVBoxLayout(data)
        dv.setContentsMargins(12, 10, 12, 10)
        dv.setSpacing(5)
        self.refs_info = QLabel("-")
        self.refs_info.setObjectName("dataLine")
        self.refs_info.setWordWrap(True)
        self.queries_info = QLabel("-")
        self.queries_info.setObjectName("dataLine")
        self.queries_info.setWordWrap(True)
        self.caps_line = QLabel("")
        self.caps_line.setObjectName("capsLine")
        self.caps_line.setTextFormat(Qt.TextFormat.RichText)
        self.caps_line.setWordWrap(True)
        self.caps_line.setToolTip(
            "What this flight carries for the recipe: a heading per frame (heading "
            "alignment), altitude above sea level (altitude scale), the UAV-VisLoc "
            "region (camera constant) and the map's tile size."
        )
        dv.addWidget(self.refs_info)
        dv.addWidget(self.queries_info)
        dv.addWidget(self.caps_line)
        btns = QHBoxLayout()
        btns.setSpacing(6)
        browse_refs = QPushButton("Map CSV…")
        browse_refs.setObjectName("mini")
        browse_refs.clicked.connect(lambda: self._browse_csv(self.refs_combo, "reference"))
        browse_q = QPushButton("Flight CSV…")
        browse_q.setObjectName("mini")
        browse_q.clicked.connect(lambda: self._browse_csv(self.queries_combo, "query"))
        self.pair_button = QPushButton("Pair with map")
        self.pair_button.setObjectName("mini")
        self.pair_button.setToolTip("Select the flight CSV sitting beside the selected map.")
        self.pair_button.clicked.connect(self._pair_queries_with_refs)
        btns.addWidget(browse_refs)
        btns.addWidget(browse_q)
        btns.addWidget(self.pair_button)
        btns.addStretch(1)
        dv.addLayout(btns)
        b1.addWidget(data)
        outer.addWidget(sec1)

        # 2 · Recipe ----------------------------------------------------------
        sec2, b2 = self._section(2, "PRESET", key="recipe")
        self.preset_seg = Segmented(
            [(label, name) for name, (label, _desc, _r) in PRESETS.items()],
            tooltips={name: desc for name, (_label, desc, _r) in PRESETS.items()},
        )
        self.preset_seg.changed.connect(self._apply_preset)
        b2.addWidget(self.preset_seg)
        self.preset_note = QLabel("")
        self.preset_note.setObjectName("presetNote")
        self.preset_note.setWordWrap(True)
        b2.addWidget(self.preset_note)

        outer.addWidget(sec2)

        # encoder
        sec_enc, b_enc = self._section(3, "ENCODER", key="encoder")
        enc = QFrame()
        enc.setObjectName("insetCard")
        ev = QVBoxLayout(enc)
        ev.setContentsMargins(12, 11, 12, 12)
        ev.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(7)
        title = QLabel("Model")
        title.setObjectName("stepTitle")
        head.addWidget(title)
        head.addStretch(1)
        badge = QLabel(f"ensemble {FINDINGS_BY_KEY['ensemble'].badge}")
        badge.setObjectName("impactBadge")
        badge.setProperty("status", "accuracy")
        head.addWidget(badge)
        info = QPushButton("i")
        info.setObjectName("infoBtn")
        info.setFixedSize(18, 18)
        info.setToolTip("Encoders and ensembles — opens Findings")
        info.clicked.connect(lambda: self._show_finding("ensemble"))
        head.addWidget(info)
        ev.addLayout(head)
        enc_note = QLabel("One encoder, an ensemble of the ticked ones, or each ticked one compared.")
        enc_note.setObjectName("stepSummary")
        enc_note.setWordWrap(True)
        ev.addWidget(enc_note)
        self.enc_mode = Segmented(
            [("Single", "single"), ("Ensemble", "ensemble"), ("Compare", "compare")],
            tooltips={
                "single": "One encoder.",
                "ensemble": "The ticked encoders as one: their cosine scores are averaged "
                            "(avl.ensemble). Costs the sum of the members.",
                "compare": "One run per ticked encoder, same recipe — rows side by side "
                           "in Comparison.",
            },
        )
        self.enc_mode.changed.connect(self._encoder_mode_changed)
        ev.addWidget(self.enc_mode)
        self.encoder_combo = QComboBox()
        for name, description in MODELS:
            self.encoder_combo.addItem(name, name)
            self.encoder_combo.setItemData(
                self.encoder_combo.count() - 1, description, Qt.ItemDataRole.ToolTipRole
            )
        self.encoder_combo.currentIndexChanged.connect(self._recipe_changed)
        ev.addWidget(self.encoder_combo)
        self.enc_grid = QWidget()
        grid = QGridLayout(self.enc_grid)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.model_checks: dict[str, QCheckBox] = {}
        for i, (name, description) in enumerate(MODELS):
            check = QCheckBox(name)
            check.setObjectName("encCheck")
            check.setToolTip(description)
            check.toggled.connect(self._recipe_changed)
            grid.addWidget(check, i // 2, i % 2)
            self.model_checks[name] = check
        ev.addWidget(self.enc_grid)
        self.enc_cost = QLabel("")
        self.enc_cost.setObjectName("methodSummary")
        self.enc_cost.setWordWrap(True)
        ev.addWidget(self.enc_cost)
        b_enc.addWidget(enc)
        outer.addWidget(sec_enc)

        # the measured steps
        sec_steps, b_steps = self._section(4, "STEPS  ·  measured on UAV-VisLoc", key="steps")
        steps_box = QFrame()
        steps_box.setObjectName("insetCard")
        steps_layout = QVBoxLayout(steps_box)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(0)
        self.card_heading = MethodCard(FINDINGS_BY_KEY["heading"])
        self.heading_mode = QComboBox()
        for label, value in HEADING_CHOICES:
            self.heading_mode.addItem(label, value)
        self.card_heading.body.addWidget(self.heading_mode)

        self.card_center = MethodCard(FINDINGS_BY_KEY["center"])
        self.center_mode = QComboBox()
        for label, value in CENTER_CHOICES:
            self.center_mode.addItem(label, value)
        self.card_center.body.addWidget(self.center_mode)

        self.card_agl = MethodCard(FINDINGS_BY_KEY["agl"])
        self.agl_k = QDoubleSpinBox()
        self.agl_k.setRange(0.2, 3.0)
        self.agl_k.setDecimals(3)
        self.agl_k.setSingleStep(0.01)
        self.agl_k.setPrefix("k ")
        self.agl_k.setToolTip(
            "Camera constant: full-frame ground width / height above ground = 2 tan(HFOV / 2). "
            "Pre-filled from the UAV-VisLoc calibration (report §5.7); for your drone, "
            "take it from the lens datasheet."
        )
        self.agl_gate = QDoubleSpinBox()
        self.agl_gate.setRange(1.0, 3.0)
        self.agl_gate.setDecimals(2)
        self.agl_gate.setSingleStep(0.05)
        self.agl_gate.setValue(1.25)
        self.agl_gate.setPrefix("gate ×")
        self.agl_gate.setToolTip("Crop only frames covering more than this × a tile's ground.")
        agl_row = QHBoxLayout()
        agl_row.setSpacing(7)
        agl_row.addWidget(self.agl_k, 1)
        agl_row.addWidget(self.agl_gate, 1)
        self.card_agl.body.addLayout(agl_row)
        self.agl_k_source = QLabel("")
        self.agl_k_source.setObjectName("hint")
        self.card_agl.body.addWidget(self.agl_k_source)

        self.card_window = MethodCard(FINDINGS_BY_KEY["window"])
        self.window_sigma = QSpinBox()
        self.window_sigma.setRange(10, 2000)
        self.window_sigma.setSingleStep(10)
        self.window_sigma.setValue(100)
        self.window_sigma.setPrefix("σ ")
        self.window_sigma.setSuffix(" m")
        self.window_sigma.setToolTip(
            "1-sigma of the simulated navigation prior (truth + Gaussian noise, seeded so "
            "every encoder sees the same priors). Tiles within 3σ are searched."
        )
        self.window_note = QLabel("")
        self.window_note.setObjectName("hint")
        win_row = QHBoxLayout()
        win_row.setSpacing(9)
        win_row.addWidget(self.window_sigma, 1)
        win_row.addWidget(self.window_note, 1)
        self.card_window.body.addLayout(win_row)

        self.card_fusion = MethodCard(FINDINGS_BY_KEY["fusion"], checkable=False)
        self.fusion = QComboBox()
        for label, value in FUSION_CHOICES:
            self.fusion.addItem(label, value)
        self.fusion.setToolTip(
            "How the top-5 matches become one lat/lon.\n"
            "spatial cluster: keep the largest geo-consistent group, drop the strays (default)\n"
            "softmax weighted: average, weight ∝ exp(score / T)\n"
            "geometric median: Weiszfeld median of the 5, robust to a minority of outliers\n"
            "weighted mean: the old score-weighted average (kept for comparison)\n"
            "top-1 only: the single best match, no fusion"
        )
        self.card_fusion.body.addWidget(self.fusion)

        self._step_cards = {
            "heading": self.card_heading,
            "center": self.card_center,
            "agl": self.card_agl,
            "window": self.card_window,
            "fusion": self.card_fusion,
        }
        for i, card in enumerate(self._step_cards.values()):
            card.toggled.connect(self._recipe_changed)
            card.info_requested.connect(self._show_finding)
            if i:
                sep = QFrame()
                sep.setObjectName("hrule")
                sep.setFixedHeight(1)
                steps_layout.addWidget(sep)
            steps_layout.addWidget(card)
        b_steps.addWidget(steps_box)
        for combo in (self.heading_mode, self.center_mode, self.fusion):
            combo.currentIndexChanged.connect(self._recipe_changed)
        self.agl_k.valueChanged.connect(self._k_edited)
        self.agl_gate.valueChanged.connect(self._recipe_changed)
        self.window_sigma.valueChanged.connect(self._recipe_changed)
        for combo in (self.encoder_combo, self.heading_mode, self.center_mode, self.fusion):
            # never wider than the inspector: long item texts are elided, not a minimum width
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.view().setMinimumWidth(300)
        outer.addWidget(sec_steps)

        # 3 · Scope -----------------------------------------------------------
        sec3, b3 = self._section(5, "SCOPE", key="protocol")
        self.max_queries = QSpinBox()
        self.max_queries.setRange(1, 10000)
        self.max_queries.setValue(144)
        self.stride = QSpinBox()
        self.stride.setRange(1, 100)
        self.stride.setValue(1)
        self.crop = QComboBox()
        self.crop.addItems(["square", "none"])
        self.crop.setToolTip(
            "Frames encoded without a heading: 'square' centre-crops them to 1:1 first "
            "(tiles are square). North-aligned frames are always square."
        )
        for widget in (self.max_queries, self.stride):
            widget.valueChanged.connect(self._recipe_changed)
        self.crop.currentIndexChanged.connect(self._recipe_changed)
        scope = QGridLayout()
        scope.setHorizontalSpacing(9)
        scope.setVerticalSpacing(9)
        scope.addLayout(self._labeled("Frames", "first N", self.max_queries), 0, 0)
        scope.addLayout(self._labeled("Stride", "every Nth", self.stride), 0, 1)
        scope.addLayout(self._labeled("Crop", "no heading", self.crop), 0, 2)
        b3.addLayout(scope)
        scope_note = QLabel("The report evaluated the first 144 frames of each flight.")
        scope_note.setObjectName("caption")
        scope_note.setWordWrap(True)
        b3.addWidget(scope_note)
        outer.addWidget(sec3)

        outer.addStretch(1)

        scroller = QScrollArea()
        scroller.setObjectName("inspectorScroll")
        scroller.setWidget(panel)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._panel_scroll = scroller

        column = QWidget()
        column.setObjectName("inspector")
        column.setFixedWidth(384)
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        head = QWidget()
        head.setObjectName("inspectorHead")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(16, 12, 12, 10)
        title = QLabel("Recipe")
        title.setObjectName("inspectorTitle")
        hint = QLabel("what the next run does")
        hint.setObjectName("inspectorHint")
        hl.addWidget(title)
        hl.addWidget(hint)
        hl.addStretch(1)
        col.addWidget(head)
        col.addWidget(scroller, 1)
        return column

    # ------------------------------------------------------------ recipe ----
    def _recipes(self, intended: bool = False) -> list[Recipe]:
        """The configurations the next run will evaluate (several in Compare mode).
        ``intended`` keeps the steps that are switched on but that the selected flight
        cannot feed (no heading / altitude column) — what the preset asked for."""
        mode = self.enc_mode.value()
        if mode == "single":
            groups = [(self.encoder_combo.currentData(),)]
        else:
            ticked = tuple(name for name, _ in MODELS if self.model_checks[name].isChecked())
            if not ticked:
                return []
            groups = [ticked] if mode == "ensemble" else [(name,) for name in ticked]
        def on(card: MethodCard) -> bool:
            return card.is_checked() if intended else card.is_on()

        agl = on(self.card_agl)
        steps = dict(
            heading=self.heading_mode.currentData() if on(self.card_heading) else "off",
            center=self.center_mode.currentData() if on(self.card_center) else "off",
            agl_scale=agl,
            agl_gate=float(self.agl_gate.value()),
            camera_k=float(self.agl_k.value()) if agl else None,
            window_sigma_m=float(self.window_sigma.value()) if on(self.card_window) else None,
            fusion=self.fusion.currentData(),
            query_crop=self.crop.currentText(),
        )
        return [Recipe(encoders=group, **steps) for group in groups]

    def _primary_recipe(self) -> Recipe:
        recipes = self._recipes()
        return recipes[0] if recipes else Recipe()

    def _primary_model(self) -> str:
        """The single encoder for the tabs that take one (Trajectory, Studio)."""
        return self._primary_recipe().encoders[0]

    def _apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        label, description, recipe = PRESETS[name]
        self._recipe_guard = True
        try:
            if recipe.is_ensemble:
                for model, check in self.model_checks.items():
                    check.setChecked(model in recipe.encoders)
                self.enc_mode.set_value("ensemble")
            else:
                self.encoder_combo.setCurrentIndex(max(0, self.encoder_combo.findData(recipe.encoders[0])))
                self.enc_mode.set_value("single")
            self._sync_encoder_mode()
            self.card_heading.set_checked(recipe.heading != "off")
            if recipe.heading != "off":
                self.heading_mode.setCurrentIndex(max(0, self.heading_mode.findData(recipe.heading)))
            self.card_center.set_checked(recipe.center != "off")
            if recipe.center != "off":
                self.center_mode.setCurrentIndex(max(0, self.center_mode.findData(recipe.center)))
            self.card_agl.set_checked(recipe.agl_scale)
            if recipe.agl_gate:
                self.agl_gate.setValue(recipe.agl_gate)
            self.card_window.set_checked(bool(recipe.window_sigma_m))
            if recipe.window_sigma_m:
                self.window_sigma.setValue(int(recipe.window_sigma_m))
            self.fusion.setCurrentIndex(max(0, self.fusion.findData(recipe.fusion)))
            self.crop.setCurrentText(recipe.query_crop)
        finally:
            self._recipe_guard = False
        self.preset_seg.set_value(name)
        self._recipe_changed()

    def _encoder_mode_changed(self, mode: str) -> None:
        if mode == "ensemble" and sum(c.isChecked() for c in self.model_checks.values()) < 2:
            self._recipe_guard = True
            for model in ENSEMBLE_REPORT:
                self.model_checks[model].setChecked(True)
            self._recipe_guard = False
        elif mode == "compare" and not any(c.isChecked() for c in self.model_checks.values()):
            self.model_checks[self.encoder_combo.currentData()].setChecked(True)
        self._sync_encoder_mode()
        self._recipe_changed()

    def _sync_encoder_mode(self) -> None:
        single = self.enc_mode.value() == "single"
        self.encoder_combo.setVisible(single)
        self.enc_grid.setVisible(not single)

    def _k_edited(self, *_args) -> None:
        if self._recipe_guard:
            return
        self._k_entered = True
        self.agl_k_source.setText("entered by hand")
        self._recipe_changed()

    def _recipe_changed(self, *_args) -> None:
        """Every control of the recipe panel lands here."""
        if self._recipe_guard or not hasattr(self, "card_fusion"):
            return
        recipes = self._recipes()
        recipe = recipes[0] if recipes else None
        # preset: highlight the one the switches still match, else "custom"
        intended = self._recipes(intended=True)
        match = None
        if len(intended) == 1:
            key = intended[0].key()
            match = next((n for n, (_l, _d, r) in PRESETS.items() if r.key() == key), None)
        self.preset_seg.set_value(match)
        idle = [
            card.finding.title.lower() for card in self._step_cards.values()
            if card.is_checked() and not card.available
        ]
        note = PRESETS[match][1] if match else "Custom recipe — pick a preset above to reset every step."
        if idle:
            note += f"  This flight cannot feed: {', '.join(idle)} — skipped."
        self.preset_note.setText(note)

        mode = self.enc_mode.value()
        if recipe is None:
            self.enc_cost.setText("Tick at least one encoder.")
            self.recipe_line.setText("")
            self.run_button.setEnabled(False)
            return
        if self.process is None:
            self.run_button.setEnabled(True)
        per_frame = [gmac_estimate(r.encoders, GMAC) for r in recipes]
        rot = recipe.effective_rotations
        if mode == "compare":
            self.enc_cost.setText(
                f"{len(recipes)} runs, one per encoder, each with the steps below."
            )
            self.run_button.setText(f"Run {len(recipes)} encoders")
        else:
            gm = per_frame[0]
            dims = [ENCODER_INFO.get(m, ("?",))[0] for m in recipe.encoders]
            width = sum(int(d) for d in dims) if all(str(d).isdigit() for d in dims) else None
            cost = f"≈ {gm * rot:.0f} GMAC / frame" if gm is not None else "compute not measured"
            self.enc_cost.setText(
                f"{cost}  ·  "
                + (f"{width:,}-d descriptor" if width else "AnyLoc width fitted on the map")
                + (f"  ·  {len(recipe.encoders)} encoders, equal vote" if recipe.is_ensemble else "")
            )
            self.run_button.setText("Run ensemble" if recipe.is_ensemble else "Run benchmark")
        sigma = self.window_sigma.value()
        self.window_note.setText(f"radius 3σ = {3 * sigma:,} m")

        # header line: what the next run is (short; the full recipe is the tooltip)
        if mode == "compare":
            who = f"compare ×{len(recipes)}"
        elif recipe.is_ensemble:
            who = f"ensemble ×{len(recipe.encoders)}"
        else:
            who = recipe.encoders[0]
        chips = [c for c in recipe.chips() if not c.startswith("ensemble")]
        gm = per_frame[0]
        self.recipe_line.setText(
            f"<span style='color:{FAINT}'>NEXT RUN</span>&nbsp;&nbsp;"
            f"<span style='color:{ACCENT_300}'>{who}</span>"
            + "".join(f"&nbsp;&nbsp;·&nbsp;&nbsp;{c}" for c in chips)
        )
        self.recipe_line.setToolTip(
            "The next run — click to edit\n"
            + "\n".join(
                f"{' + '.join(r.encoders)}:  {', '.join(r.chips()) or 'no steps'}"
                + (f"  ·  ≈ {g * r.effective_rotations:.0f} GMAC / frame" if g is not None else "")
                for r, g in zip(recipes, per_frame)
            )
        )
        self._refresh_pipeline()
        self._refresh_overview()
        if hasattr(self, "single_encoder_label"):
            self._single_image_changed()

    def _update_availability(self) -> None:
        """Grey out the steps the selected flight cannot feed, and say why."""
        refs, queries = self._current_paths()
        cols: set[str] = set()
        if queries is not None and queries.exists():
            try:
                with queries.open() as handle:
                    cols = {c.strip() for c in handle.readline().split(",")}
            except OSError:
                cols = set()
        self._flight_cols = cols
        has_heading = "yaw_deg" in cols
        has_alt = "height_m" in cols
        tile = tile_m_of(refs) if refs is not None else None
        region = region_of(queries) or region_of(refs)
        k, source = camera_k_for(region)
        if region != self._k_region:
            # another region may be another camera: drop a hand-typed constant
            self._k_entered = False
            self._k_region = region

        self.card_heading.set_available(
            has_heading,
            "" if has_heading else "This flight has no heading column (yaw_deg): its frames "
            "are searched in four rotations.",
        )
        if not has_alt:
            reason = "This flight has no altitude column (height_m)."
        elif tile is None:
            reason = "The map folder name carries no tile size ('_t<metres>')."
        else:
            reason = ""
        self.card_agl.set_available(not reason, reason)
        if not self._k_entered:
            self._recipe_guard = True
            self.agl_k.setValue(k if k is not None else 1.0)
            self._recipe_guard = False
            self.agl_k_source.setText(source if k is not None else f"{source} — 1.0 is a guess")

        def mark(ok: bool) -> str:
            return f"<span style='color:{GOOD if ok else MUTED}'>{'✓' if ok else '✕'}</span>"

        self.caps_line.setText(
            f"heading {mark(has_heading)}&nbsp;&nbsp;·&nbsp;&nbsp;altitude {mark(has_alt)}"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;region {('r' + region) if region else '—'}"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;tiles {f'{tile:.0f} m' if tile else '—'}"
        )

    def _build_results(self) -> QWidget:
        """The workspace: one page per rail destination, each with page margins."""
        self.tabs = QStackedWidget()
        self.tabs.setObjectName("workspace")
        built = [
            self._build_overview_page(),
            self._build_comparison_tab(),
            self._build_findings_tab(),
            self._build_localize_tab(),
            self._build_models_page(),
            self._build_trajectory_tab(),
            self._build_studio_tab(),
        ]
        wrapped = []
        for (icon, label, tip), page in zip(PAGES, built):
            holder = QWidget()
            holder.setObjectName("pageHolder")
            lay = QVBoxLayout(holder)
            lay.setContentsMargins(24, 18, 24, 14)
            lay.setSpacing(0)
            lay.addWidget(page)
            self.tabs.addWidget(holder)
            self.nav.add(icon, label, tip)
            wrapped.append(holder)
        (self.overview_page, self.runs_page, self.findings_page, self.localize_page,
         self.models_page, self.trajectory_page, self._studio_page) = wrapped
        self._page_names = [label for _icon, label, _tip in PAGES]
        self.nav.changed.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.nav.set_current)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        return self.tabs

    def _page_header(self, title: str, subtitle: str = "") -> tuple[QHBoxLayout, QLabel]:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        sub = QLabel(subtitle)
        sub.setObjectName("pageSubtitle")
        sub.setWordWrap(True)
        col.addWidget(heading)
        col.addWidget(sub)
        row.addLayout(col, 1)
        return row, sub

    def _on_tab_changed(self, index: int) -> None:
        name = self._page_names[index] if 0 <= index < len(self._page_names) else ""
        if name == "Studio":
            self._studio_sync()
        elif name == "Findings":
            self._refresh_findings()
        elif name == "Overview":
            self._refresh_overview()

    # ---------------------------------------------------------- pipeline ----

    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 8, 8)
        layout.setSpacing(16)

        head, self.overview_sub = self._page_header("Overview", "")
        open_findings = QPushButton("Open findings")
        open_findings.setObjectName("secondary")
        open_findings.clicked.connect(lambda: self.tabs.setCurrentWidget(self.findings_page))
        head.addWidget(open_findings, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)

        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.tile_top1 = StatTile("Top-1 within 100 m", hero=True)
        self.tile_fused = StatTile("Fused top-5 within 100 m")
        self.tile_median = StatTile("Median error, rank-1 tile")
        self.tile_board = StatTile("On the drone — Jetson Orin Nano Super")
        for tile in (self.tile_top1, self.tile_fused, self.tile_median, self.tile_board):
            tiles.addWidget(tile, 1)
        self.tile_top1.clicked.connect(lambda: self.tabs.setCurrentWidget(self.runs_page))
        layout.addLayout(tiles)

        flow_card = QFrame()
        flow_card.setObjectName("card")
        fc = QVBoxLayout(flow_card)
        fc.setContentsMargins(18, 14, 18, 14)
        fc.setSpacing(10)
        fhead = QHBoxLayout()
        flabel = QLabel("PIPELINE")
        flabel.setObjectName("panelTitle")
        fhint = QLabel("hollow stages are off in this recipe · the running stage glows · click a stage to edit it")
        fhint.setObjectName("panelHint")
        fhead.addWidget(flabel)
        fhead.addWidget(fhint)
        fhead.addStretch(1)
        fc.addLayout(fhead)
        self.flow = FlowStrip([
            FlowNode("prepare", "Prepare frame"),
            FlowNode("encode", "Encode"),
            FlowNode("center", "Center"),
            FlowNode("search", "Search"),
            FlowNode("dedup", "Top-5"),
            FlowNode("fuse", "Fuse"),
            FlowNode("result", "Result", "—", "fused latitude / longitude"),
        ])
        self.flow.clickable = {"prepare", "encode", "center", "search", "fuse"}
        self.flow.clicked.connect(self._flow_clicked)
        fc.addWidget(self.flow)
        self.shape_strip = QLabel("-")
        self.shape_strip.setObjectName("shapeStrip")
        self.shape_strip.setWordWrap(True)
        fc.addWidget(self.shape_strip)
        layout.addWidget(flow_card)

        lower = QHBoxLayout()
        lower.setSpacing(14)
        ladder_card = QFrame()
        ladder_card.setObjectName("card")
        lc = QVBoxLayout(ladder_card)
        lc.setContentsMargins(18, 14, 18, 12)
        lc.setSpacing(6)
        lhead = QHBoxLayout()
        llabel = QLabel("PROGRESS ON THIS MAP")
        llabel.setObjectName("panelTitle")
        lhead.addWidget(llabel)
        lhead.addStretch(1)
        more = QPushButton("Findings ›")
        more.setObjectName("ghost")
        more.clicked.connect(lambda: self.tabs.setCurrentWidget(self.findings_page))
        lhead.addWidget(more)
        lc.addLayout(lhead)
        self.mini_ladder = LadderChart(compact=True)
        lc.addWidget(self.mini_ladder)
        lc.addStretch(1)
        lower.addWidget(ladder_card, 3)

        recent_card = QFrame()
        recent_card.setObjectName("card")
        rc = QVBoxLayout(recent_card)
        rc.setContentsMargins(18, 14, 18, 12)
        rc.setSpacing(4)
        rhead = QHBoxLayout()
        rlabel = QLabel("RECENT RUNS")
        rlabel.setObjectName("panelTitle")
        rhead.addWidget(rlabel)
        rhead.addStretch(1)
        all_runs = QPushButton("All runs ›")
        all_runs.setObjectName("ghost")
        all_runs.clicked.connect(lambda: self.tabs.setCurrentWidget(self.runs_page))
        rhead.addWidget(all_runs)
        rc.addLayout(rhead)
        self.recent_list = QVBoxLayout()
        self.recent_list.setSpacing(0)
        rc.addLayout(self.recent_list)
        rc.addStretch(1)
        lower.addWidget(recent_card, 2)
        layout.addLayout(lower, 1)
        layout.addStretch(1)
        return self._scrolled(page)

    def _flow_clicked(self, key: str) -> None:
        target = {"prepare": "steps", "center": "steps", "search": "steps", "fuse": "steps",
                  "encode": "encoder"}.get(key, "recipe")
        self._focus_section(target)

    def _build_models_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 8, 8)
        layout.setSpacing(16)
        head, _sub = self._page_header(
            "Models",
            "18 encoders: compute per image and raw accuracy on UAV-VisLoc (report §4 — top-1 "
            "within 100 m, four rotations, no recipe). Click a row to use the encoder.",
        )
        layout.addLayout(head)
        table_card = QFrame()
        table_card.setObjectName("card")
        tc = QVBoxLayout(table_card)
        tc.setContentsMargins(1, 6, 1, 6)
        tc.setSpacing(0)
        cols = ["Encoder", "GMAC", "r05", "r10", "r06", "Descriptor", "Family", "Training data", "Cross-view"]
        self.encoder_table = QTableWidget(len(ENCODER_INFO), len(cols))
        self.encoder_table.setHorizontalHeaderLabels(cols)
        self.encoder_table.verticalHeader().setVisible(False)
        header = self.encoder_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.encoder_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.encoder_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.encoder_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        def pct(value: Any) -> str:
            return f"{value:.1f}%" if value is not None else "—"

        for row, (name, (dim, family, train, cross)) in enumerate(ENCODER_INFO.items()):
            facts = ENCODER_FACTS.get(name, {})
            gm = facts.get("gmac")
            cells = [
                name,
                f"{gm:g}" if gm is not None else "—",
                pct(facts.get("r05")), pct(facts.get("r10")), pct(facts.get("r06")),
                dim, family, train, "yes" if cross else "no",
            ]
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setToolTip(f"{name}: {ENCODER_INFO[name][1]} · {train}" if col == 0 else value)
                if col == 8:
                    item.setForeground(QColor(GOOD if cross else FAINT))
                elif 1 <= col <= 4 and value == "—":
                    item.setForeground(QColor(FAINT))
                self.encoder_table.setItem(row, col, item)
        self.encoder_table.cellClicked.connect(self._pick_encoder_from_table)
        self.encoder_table.verticalHeader().setDefaultSectionSize(34)
        for r in range(self.encoder_table.rowCount()):
            self.encoder_table.setRowHeight(r, 34)
        self.encoder_table.setFixedHeight(
            self.encoder_table.horizontalHeader().height()
            + sum(self.encoder_table.rowHeight(r) for r in range(len(ENCODER_INFO)))
            + 14
        )
        tc.addWidget(self.encoder_table)
        layout.addWidget(table_card)
        layout.addStretch(1)
        return self._scrolled(page)

    def _pick_encoder_from_table(self, row: int, _col: int) -> None:
        item = self.encoder_table.item(row, 0)
        if item is None:
            return
        name = item.text()
        if self.enc_mode.value() == "single":
            self.encoder_combo.setCurrentIndex(max(0, self.encoder_combo.findData(name)))
        else:
            check = self.model_checks.get(name)
            if check is not None:
                check.setChecked(not check.isChecked())

    @staticmethod
    def _folder_tile_metres(path: Path | None) -> str:
        if path is None:
            return ""
        match = re.search(r"_t(\d+)", path.parent.name)
        return f"{match.group(1)} m tiles" if match else ""

    @staticmethod
    def _csv_rows(path: Path | None) -> int | None:
        if path is None or not path.exists():
            return None
        try:
            return sum(1 for _ in path.open()) - 1
        except OSError:
            return None

    def _refresh_pipeline(self) -> None:
        if not hasattr(self, "flow") or not hasattr(self, "card_fusion"):
            return
        refs, queries = self._current_paths()
        n_refs = self._csv_rows(refs)
        n_queries_total = self._csv_rows(queries)
        cap = self.max_queries.value()
        stride = self.stride.value()
        n_queries = None
        if n_queries_total is not None:
            n_queries = min(-(-n_queries_total // stride), cap)  # ceil(total/stride), capped

        recipes = self._recipes()
        recipe = recipes[0] if recipes else Recipe()
        members = recipe.encoders
        dims = [ENCODER_INFO.get(m, ("?",))[0] for m in members]
        width = sum(int(d) for d in dims) if all(str(d).isdigit() for d in dims) else None
        dim = f"{width:,}-d" if width else "VLAD-d"
        rot = recipe.effective_rotations
        gm = gmac_estimate(members, GMAC)
        compare = self.enc_mode.value() == "compare" and len(recipes) > 1

        heading_on = recipe.heading != "off"
        prepare = []
        if heading_on:
            prepare.append("north-up")
        if recipe.agl_scale:
            prepare.append(f"AGL crop >{recipe.agl_gate:g}×")
        self._set_node(
            "prepare",
            " + ".join(p.split(" ")[0] if p.startswith("AGL") else p for p in prepare)
            or f"as captured ×{rot}",
            ("heading turns it north-up; " if heading_on else "")
            + (f"altitude crops it above {recipe.agl_gate:g}× a tile" if recipe.agl_scale else "")
            if prepare else "searched in every rotation",
            off=not prepare,
        )
        self._set_node(
            "encode", f"{rot} × {dim}",
            " + ".join(members) if len(members) > 1 else "one descriptor per view",
        )
        self._set_node(
            "center",
            {"map+flight": "− flight mean", "map": "− map mean"}.get(recipe.center, "off"),
            "map mean on tiles, running mean on frames" if recipe.center == "map+flight"
            else "map mean on both sides" if recipe.center == "map" else "raw descriptors",
            off=recipe.center == "off",
        )
        if recipe.window_sigma_m:
            self._set_node(
                "search", f"{3 * recipe.window_sigma_m:.0f} m window",
                f"tiles within 3σ of the IMU prior, of {n_refs or 'N'}",
            )
        else:
            self._set_node(
                "search",
                f"{n_refs} tiles" if n_refs is not None else "whole map",
                "cosine, best view per tile",
            )
        self._set_node("dedup", "→ top-5", "one hit per physical location")
        fuse_label, fuse_detail = FUSION_NODE_TEXT.get(
            recipe.fusion, FUSION_NODE_TEXT[DEFAULT_FUSION_METHOD]
        )
        self._set_node("fuse", fuse_label, fuse_detail)
        result = self.flow.node("result")
        if result is not None and result.value in ("-", ""):
            self._set_node("result", "—", "fused latitude / longitude")

        steps = [f"[{n_queries if n_queries is not None else 'M'} frames]"]
        if prepare:
            steps.append(f"[{' + '.join(prepare)}]")
        steps.append(f"[{rot} × {dim}]")
        if recipe.center != "off":
            steps.append("[− mean]")
        steps.append(f"[· {n_refs if n_refs is not None else 'N'} tiles"
                     + (f", window {3 * recipe.window_sigma_m:.0f} m]" if recipe.window_sigma_m else "]"))
        steps += ["[top-5 unique]", "[1 lat/lon]"]
        self.shape_strip.setText("  →  ".join(steps))
        self._highlight_encoder_row(set(m for r in recipes for m in r.encoders))

    def _pipeline_result_from_summary(self, tag: str | None) -> None:
        if not tag:
            return
        try:
            data = json.loads((self.out_dir / f"{tag}_summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            return
        recall = data.get("recall_top1_within_m", {}).get("100")
        median = data.get("top1_error_m", {}).get("median")
        bits = []
        if recall is not None:
            bits.append(f"top-1 <100 m: {100 * recall:.0f}%")
        if median is not None:
            bits.append(f"median {median:.0f} m")
        self._set_node("result", f"{data.get('n_queries', '?')} queries", "  ·  ".join(bits))

    def _set_node(self, key: str, value: str, detail: str = "", off: bool | None = None) -> None:
        if hasattr(self, "flow"):
            self.flow.update_node(key, value=value or "-", detail=detail, off=off)

    def _highlight_encoder_row(self, encoders: set[str]) -> None:
        if not hasattr(self, "encoder_table"):
            return
        for row, name in enumerate(ENCODER_INFO):
            on = name in encoders
            for col in range(self.encoder_table.columnCount()):
                item = self.encoder_table.item(row, col)
                if item is None:
                    continue
                font = item.font()
                font.setBold(on)
                item.setFont(font)
                if col == 0:
                    item.setForeground(QColor(ACCENT_300 if on else TEXT))

    def _set_pipe_phase(self, phase: str | None) -> None:
        """encode → search → fuse → done: mark the flow's stages done / active / pending."""
        self._pipe_phase = phase
        order = ["prepare", "encode", "center", "search", "dedup", "fuse", "result"]
        active = {"encode": 1, "search": 3, "fuse": 5}.get(phase or "", -1)
        for i, key in enumerate(order):
            if phase == "done":
                state = "done"
            elif phase is None:
                state = "idle"
            elif active < 0:
                state = "pending"
            elif i < active or (phase == "fuse" and key == "dedup"):
                state = "done"
            elif i == active:
                state = "active"
            else:
                state = "pending"
            if hasattr(self, "flow"):
                self.flow.update_node(key, phase=state)

    def _build_comparison_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        page_head, _sub = self._page_header(
            "Runs", "Every saved evaluation. Filter by map or text, sort by any column, "
                    "click a row to inspect its frames."
        )
        layout.addLayout(page_head)

        card = QWidget()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(1, 1, 1, 1)
        cl.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(14, 12, 14, 8)
        head.setSpacing(10)
        heading = QLabel("All runs")
        heading.setObjectName("panelTitleLarge")
        sub = QLabel("headline: the rank-1 tile within 100 m of the truth  ·  fused = top-5 fusion")
        sub.setObjectName("panelHint")
        head.addWidget(heading)
        head.addWidget(sub)
        head.addStretch(1)
        reload_button = IconButton("refresh", "Reload saved results from artifacts/visloc/")
        reload_button.clicked.connect(self.reload_results)
        head.addWidget(reload_button)
        self.filter_map = QComboBox()
        self.filter_map.setMinimumContentsLength(14)
        self.filter_map.addItem("All maps", None)
        self.filter_map.currentIndexChanged.connect(self._apply_comparison_filter)
        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("filter: encoder, step, map…")
        self.filter_text.setClearButtonEnabled(True)
        self.filter_text.setMinimumWidth(220)
        self.filter_text.textChanged.connect(self._apply_comparison_filter)
        head.addWidget(self.filter_map)
        head.addWidget(self.filter_text)
        cl.addLayout(head)

        self.table = QTableWidget(0, len(METRIC_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in METRIC_COLUMNS])
        for col, (key, _label) in enumerate(METRIC_COLUMNS):
            if key in METRIC_TIPS:
                self.table.horizontalHeaderItem(col).setToolTip(METRIC_TIPS[key])
        self.table.setItemDelegateForColumn(COL["steps"], ChipsDelegate(self.table))
        self.table.setItemDelegateForColumn(COL["within100"], BarDelegate(self.table))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL["config"], QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        widths = {"steps": 236, "map": 96, "n_queries": 64, "within100": 168, "fused100": 70,
                  "top5_within100": 68, "top1_median": 74, "chance100": 70, "gmac": 60,
                  "ms_per_query": 64}
        for key, width in widths.items():
            self.table.setColumnWidth(COL[key], width)
        header.setSortIndicator(COL["date"], Qt.SortOrder.DescendingOrder)
        self.table.setSortingEnabled(True)  # click a header to sort; click again to flip
        self.table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._open_run_from_comparison)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_context_menu)
        cl.addWidget(self.table)
        foot = QHBoxLayout()
        foot.setContentsMargins(14, 6, 14, 10)
        self.run_count = QLabel("")
        self.run_count.setObjectName("caption")
        row_hint = QLabel("click a header to sort · click a row to open it in Localize · right-click to delete")
        row_hint.setObjectName("caption")
        self.delete_run_button = QPushButton("Delete selected run…")
        self.delete_run_button.setObjectName("ghost")
        self.delete_run_button.clicked.connect(self._delete_selected_run)
        foot.addWidget(self.run_count)
        foot.addSpacing(14)
        foot.addWidget(row_hint)
        foot.addStretch(1)
        foot.addWidget(self.delete_run_button)
        cl.addLayout(foot)
        layout.addWidget(card, 1)
        return page

    def _apply_comparison_filter(self, *_args) -> None:
        wanted_map = self.filter_map.currentData()
        needle = self.filter_text.text().strip().lower()
        shown = 0
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            row_map = item.data(Qt.ItemDataRole.UserRole + 1) if item is not None else None
            haystack = item.data(Qt.ItemDataRole.UserRole + 2) if item is not None else ""
            hide = (wanted_map is not None and row_map != wanted_map) or (
                bool(needle) and needle not in (haystack or "")
            )
            self.table.setRowHidden(row, hide)
            shown += not hide
        self.run_count.setText(f"{shown} of {self.table.rowCount()} runs")

    # ---------------------------------------------------------- findings ----
    USABLE_FINDINGS = ("ensemble", "heading", "center", "agl", "window", "fusion")

    def _build_findings_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 8, 8)
        layout.setSpacing(16)

        head, _sub = self._page_header(
            "Findings",
            "What each method measured — in the progress report, and on your own runs of the "
            "selected map + flight.",
        )
        layout.addLayout(head)

        ladder_card = QWidget()
        ladder_card.setObjectName("card")
        lc = QVBoxLayout(ladder_card)
        lc.setContentsMargins(18, 14, 18, 14)
        lc.setSpacing(6)
        top = QHBoxLayout()
        label = QLabel("PROGRESS ON THIS MAP  ·  each step adds one method")
        label.setObjectName("panelTitle")
        top.addWidget(label)
        top.addStretch(1)
        self.ladder_title = QLabel("")
        self.ladder_title.setObjectName("runDetails")
        top.addWidget(self.ladder_title)
        lc.addLayout(top)
        self.ladder = LadderChart()
        lc.addWidget(self.ladder)
        foot = QHBoxLayout()
        self.ladder_note = QLabel("")
        self.ladder_note.setObjectName("caption")
        self.ladder_note.setWordWrap(True)
        self.ladder_run_button = QPushButton("Run missing steps")
        self.ladder_run_button.setObjectName("primary")
        self.ladder_run_button.clicked.connect(self._run_missing_steps)
        foot.addWidget(self.ladder_note, 1)
        foot.addWidget(self.ladder_run_button, 0, Qt.AlignmentFlag.AlignTop)
        lc.addLayout(foot)
        layout.addWidget(ladder_card)

        methods = QLabel("METHODS  ·  report §5  ·  “Use in recipe” switches the step on in the inspector")
        methods.setObjectName("panelTitle")
        layout.addWidget(methods)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for i, finding in enumerate(FINDINGS):
            card = FindingCard(finding, can_use=finding.key in self.USABLE_FINDINGS)
            card.use_requested.connect(self._use_finding)
            grid.addWidget(card, i // 2, i % 2)
            self._finding_cards[finding.key] = card
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

        self.findings_scroll = self._scrolled(page)
        return self.findings_scroll

    def _show_finding(self, key: str) -> None:
        card = self._finding_cards.get(key)
        self.tabs.setCurrentWidget(self.findings_page)
        if card is None:
            return
        QTimer.singleShot(30, lambda: self.findings_scroll.ensureWidgetVisible(card, 0, 30))
        card.setProperty("flash", "1")
        self._repolish(card)
        QTimer.singleShot(1100, lambda: (card.setProperty("flash", "0"), self._repolish(card)))

    def _use_finding(self, key: str) -> None:
        if key == "ensemble":
            self._recipe_guard = True
            for model, check in self.model_checks.items():
                check.setChecked(model in ENSEMBLE_REPORT)
            self.enc_mode.set_value("ensemble")
            self._sync_encoder_mode()
            self._recipe_guard = False
        elif key == "fusion":
            self.fusion.setCurrentIndex(max(0, self.fusion.findData(DEFAULT_FUSION_METHOD)))
        else:
            card = self._step_cards.get(key)
            if card is None:
                return
            if card.check is not None and not card.check.isEnabled():
                self.append_log(f"[findings] {card.finding.title}: {card.note.text()}")
            card.set_checked(True)
        self._recipe_changed()
        self._focus_section("recipe")

    def _expected_frames(self, queries: Path | None) -> int | None:
        total = self._csv_rows(queries)
        if total is None:
            return None
        return min(-(-total // self.stride.value()), self.max_queries.value())

    def _same_csv(self, recorded: Any, wanted: Path | None) -> bool:
        if wanted is None or not recorded:
            return False
        path = resolve_path(recorded, self.project_root)
        try:
            return path is not None and path.resolve() == wanted.resolve()
        except OSError:
            return False

    def _find_run(self, refs: Path, queries: Path, recipe: Recipe, frames: int | None):
        """Newest saved run of ``recipe`` on this map + flight, preferring the current scope."""
        fallback = None
        key = recipe.key()
        for path, data in self._summaries:
            if not (self._same_csv(data.get("refs_csv"), refs)
                    and self._same_csv(data.get("queries_csv"), queries)):
                continue
            try:
                if Recipe.from_summary(data).key() != key:
                    continue
            except ValueError:
                continue
            if frames is None or data.get("n_queries") == frames:
                return path, data
            fallback = fallback or (path, data)
        return fallback

    def _step_blocker(self, recipe: Recipe) -> str:
        if recipe.heading != "off" and "yaw_deg" not in self._flight_cols:
            return "needs a heading per frame — this flight has none"
        if recipe.agl_scale and ("height_m" not in self._flight_cols or tile_m_of(self._current_paths()[0]) is None):
            return "needs altitude and the map's tile size"
        return ""

    def _refresh_findings(self) -> None:
        if not hasattr(self, "ladder"):
            return
        refs, queries = self._current_paths()
        if refs is None or queries is None:
            self.ladder_title.setText("")
            self.ladder.empty_text = "select a map and a flight in the top bar"
            self.ladder.set_steps([])
            if hasattr(self, "mini_ladder"):
                self.mini_ladder.set_steps([])
            self.ladder_run_button.setEnabled(False)
            return
        frames = self._expected_frames(queries)
        self.ladder_title.setText(
            f"{refs.parent.name}  ·  {queries.parent.name}  ·  {frames or '?'} frames"
        )
        steps: list[LadderStep] = []
        self._missing_steps = []
        for label, recipe in LADDER:
            found = self._find_run(refs, queries, recipe, frames)
            if found is not None:
                path, data = found
                fused = (data.get("fused_within_m") or {}).get("100")
                steps.append(LadderStep(
                    label=label,
                    top1=data["recall_top1_within_m"]["100"],
                    fused=fused,
                    ensemble=recipe.is_ensemble,
                    detail=(
                        f"{data.get('tag')}\n{data.get('n_queries')} frames · "
                        f"top-1 median {data['top1_error_m']['median']:.0f} m · "
                        f"fused median {(data.get('fused_error_m') or {}).get('median', float('nan')):.0f} m\n"
                        f"{self._run_jalali_text(data, path)}"
                    ),
                ))
            else:
                blocker = self._step_blocker(recipe)
                steps.append(LadderStep(label=label, note=blocker or "not run yet", ensemble=recipe.is_ensemble))
                if not blocker:
                    self._missing_steps.append(recipe)
        self.ladder.set_steps(steps)
        if hasattr(self, "mini_ladder"):
            self.mini_ladder.set_steps(steps)

        n = len(self._missing_steps)
        self.ladder_run_button.setText(f"Run missing steps ({n})" if n else "All steps measured")
        self.ladder_run_button.setEnabled(bool(n) and self.process is None and not self.queue)
        if n:
            minutes = self._estimate_minutes(self._missing_steps, frames or 144)
            self.ladder_note.setText(
                f"{n} step(s) have no run on this map + flight at {frames} frames. Running them "
                f"queues visloc_eval with each step's recipe — at most ~{minutes:.0f} min on this "
                "CPU; frames and members already encoded are reused from the caches."
            )
        else:
            self.ladder_note.setText(
                "Every step is measured on this map + flight. Hover a bar for the run behind it; "
                "the bars are top-1 within 100 m, the dots the fused top-5."
            )

    def _refresh_overview(self) -> None:
        """KPI tiles, recent runs and the subtitle for the selected map + flight."""
        if not hasattr(self, "tile_top1") or not hasattr(self, "card_fusion"):
            return
        refs, queries = self._current_paths()
        recipes = self._recipes()
        recipe = recipes[0] if recipes else None
        frames = self._expected_frames(queries)
        self.overview_sub.setText(
            f"{refs.parent.name if refs else '—'} map  ·  {queries.parent.name if queries else '—'} flight"
            f"  ·  {frames or '—'} frames  ·  "
            + (" + ".join(recipe.encoders) if recipe else "no encoder")
        )
        run = base = None
        if refs is not None and queries is not None and recipe is not None:
            run = self._find_run(refs, queries, recipe, frames)
            base = self._find_run(refs, queries, PRESETS["baseline"][2], frames)
        base_label = "Baseline preset"
        if run is None:
            msg = ("This recipe has no run on this map + flight yet — press Run benchmark."
                   if recipe is not None else "Tick at least one encoder.")
            self.tile_top1.set("—", "", "", msg)
            self.tile_fused.set("—", "", "", "")
            self.tile_median.set("—", "", "", "")
        else:
            path, data = run
            top1 = 100 * data["recall_top1_within_m"]["100"]
            fused = (data.get("fused_within_m") or {}).get("100")
            median = data["top1_error_m"]["median"]
            when = self._run_jalali_text(data, path)
            same = base is not None and base[1].get("tag") == data.get("tag")
            b = base[1] if base is not None and not same else None
            self.tile_top1.set(
                f"{top1:.1f}", "%",
                delta_html(top1 - 100 * b["recall_top1_within_m"]["100"], " pts", base_label) if b else "",
                f"{data['n_queries']} frames  ·  run {when}",
            )
            b_fused = (b.get("fused_within_m") or {}).get("100") if b else None
            self.tile_fused.set(
                f"{100 * fused:.1f}" if fused is not None else "—", "%" if fused is not None else "",
                delta_html(100 * (fused - b_fused), " pts", base_label) if fused is not None and b_fused is not None else "",
                f"fusion: {data.get('fusion', '?')}",
            )
            self.tile_median.set(
                f"{median:.0f}", "m",
                delta_html(median - b["top1_error_m"]["median"], " m", base_label, higher_is_better=False) if b else "",
                f"p95 {data['top1_error_m']['p95']:.0f} m",
            )
        gm = gmac_estimate(recipe.encoders, GMAC) if recipe else None
        if gm is not None:
            per_frame = gm * recipe.effective_rotations
            ms = per_frame * ORIN_MS_PER_GMAC
            self.tile_board.set(
                f"≈ {ms:.0f}", "ms / fix",
                f"<span style='color:{MUTED}'>{per_frame:.0f} GMAC per frame  ·  ≈ {1000 / ms:.1f} fixes/s</span>",
                "Projected from published TensorRT FP16 numbers (report §9), not measured. "
                "With the IMU bridging, ~1 fix/s is enough.",
            )
        else:
            self.tile_board.set("—", "", "", "Compute not measured for this encoder.")

        # recent runs
        while self.recent_list.count():
            item = self.recent_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        for path, data in self._summaries[:7]:
            try:
                rec = Recipe.from_summary(data)
                chips = [c for c in rec.chips() if not c.startswith("ensemble")]
                who = " + ".join(rec.encoders)
            except ValueError:
                chips, who = [], data.get("model", "?")
            row = QFrame()
            row.setObjectName("recentRow")
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 7, 8, 7)
            rl.setSpacing(10)
            col = QVBoxLayout()
            col.setSpacing(1)
            name = QLabel(who)
            name.setObjectName("recentName")
            meta = QLabel(f"{Path(data.get('refs_csv', '')).parent.name}  ·  " + ("  ·  ".join(chips) or "no steps"))
            meta.setObjectName("recentMeta")
            col.addWidget(name)
            col.addWidget(meta)
            rl.addLayout(col, 1)
            value = QLabel(f"{100 * data['recall_top1_within_m']['100']:.1f}%")
            value.setObjectName("recentValue")
            rl.addWidget(value)
            tag = data.get("tag")
            row.mousePressEvent = lambda _e, t=tag: self._open_run(t)
            row.setToolTip(f"{tag}\nclick to inspect its frames")
            self.recent_list.addWidget(row)
        if not self._summaries:
            empty = QLabel("No runs yet.")
            empty.setObjectName("panelHint")
            self.recent_list.addWidget(empty)

    def _open_run(self, tag: str | None) -> None:
        if not tag:
            return
        index = self.run_combo.findText(tag)
        if index >= 0:
            self.run_combo.setCurrentIndex(index)
        self._show_set_mode()

    @staticmethod
    def _estimate_minutes(recipes: list[Recipe], frames: int) -> float:
        seconds = 0.0
        for recipe in recipes:
            gm = gmac_estimate(recipe.encoders, GMAC) or 100.0
            seconds += frames * recipe.effective_rotations * gm * SECONDS_PER_GMAC
        return seconds / 60.0

    def _run_missing_steps(self) -> None:
        if self._busy() or not self._missing_steps:
            return
        refs, queries = self._current_paths()
        frames = self._expected_frames(queries) or 144
        minutes = self._estimate_minutes(self._missing_steps, frames)
        names = [label for label, recipe in LADDER if recipe in self._missing_steps]
        answer = QMessageBox.question(
            self,
            "Run missing steps",
            f"Queue {len(names)} run(s) on {refs.parent.name} with {frames} frames:\n\n"
            + "\n".join(f"  · {n}" for n in names)
            + f"\n\nAt most ~{minutes:.0f} min on this CPU (cached frames and ensemble members "
            "are reused, so usually much less).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._queue_runs(list(self._missing_steps), "[findings]")

    # ---------------------------------------------------------- localize ----
    def _build_localize_tab(self) -> QWidget:
        outer_page = QWidget()
        outer = QVBoxLayout(outer_page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)
        head, _sub = self._page_header(
            "Inspect",
            "One frame at a time: where the recipe put it, its five best tiles, and how the "
            "frame was prepared. Pick a saved run on the left, or localize any image.",
        )
        outer.addLayout(head)
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        outer.addWidget(page, 1)

        left = QWidget()
        left.setFixedWidth(360)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(12)

        seg = QFrame()
        seg.setObjectName("seg")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(3)
        self.mode_set_btn = QPushButton("From query set")
        self.mode_img_btn = QPushButton("Any image")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, btn in enumerate((self.mode_set_btn, self.mode_img_btn)):
            btn.setObjectName("segBtn")
            btn.setCheckable(True)
            group.addButton(btn, i)
            sl.addWidget(btn, 1)
        self.mode_set_btn.setChecked(True)
        self.mode_set_btn.toggled.connect(
            lambda on: on and self.localize_stack.setCurrentIndex(0)
        )
        self.mode_img_btn.toggled.connect(
            lambda on: on and self.localize_stack.setCurrentIndex(1)
        )
        lv.addWidget(seg)

        self.localize_stack = QStackedWidget()
        self.localize_stack.addWidget(self._build_from_set_page())
        self.localize_stack.addWidget(self._build_any_image_page())
        lv.addWidget(self.localize_stack, 1)
        layout.addWidget(left, 0)

        self.retrieval_panel = RetrievalPanel("Select a frame on the left, or localize any image")
        layout.addWidget(self._scrolled(self.retrieval_panel), 1)
        return outer_page

    def _build_from_set_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(11)

        card = QWidget()
        card.setObjectName("cardAccent")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(11, 11, 11, 11)
        cv.setSpacing(9)
        blurb = QLabel(
            "Run the recipe from the inspector on the selected map + flight, or open any "
            "saved run below to see each frame's top-5 tiles."
        )
        blurb.setObjectName("hint")
        blurb.setWordWrap(True)
        cv.addWidget(blurb)
        self.inspector_run_button = QPushButton("Run recipe on this flight")
        self.inspector_run_button.setObjectName("primary")
        self.inspector_run_button.clicked.connect(self.run_inspector)
        cv.addWidget(self.inspector_run_button)
        self.inspector_force = QCheckBox("force re-run")
        self.inspector_force.setToolTip(
            "Recompute even when a saved per-query result already exists."
        )
        cv.addWidget(self.inspector_force)
        v.addWidget(card)

        self.run_combo = QComboBox()
        self.run_combo.view().setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.run_combo.view().setMinimumWidth(460)
        self.run_combo.currentIndexChanged.connect(self._load_run_queries)
        v.addLayout(self._labeled("Run", None, self.run_combo))

        self.run_details = QLabel("-")
        self.run_details.setObjectName("runDetails")
        self.run_details.setWordWrap(True)
        v.addWidget(self.run_details)

        list_card = QWidget()
        list_card.setObjectName("card")
        lc = QVBoxLayout(list_card)
        lc.setContentsMargins(1, 1, 1, 1)
        lc.setSpacing(0)
        list_head = QLabel("EVALUATED QUERIES")
        list_head.setObjectName("logHeading")
        list_head.setContentsMargins(12, 9, 12, 6)
        lc.addWidget(list_head)
        self.query_table = QTableWidget(0, 4)
        self.query_table.setHorizontalHeaderLabels(["Query", "Top-1 err", "Best top-5", "Score"])
        self.query_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.query_table.horizontalHeader().setSortIndicatorShown(True)
        self.query_table.setSortingEnabled(True)
        self.query_table.verticalHeader().setVisible(False)
        self.query_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.query_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.query_table.itemSelectionChanged.connect(self._show_query)
        lc.addWidget(self.query_table)
        v.addWidget(list_card, 1)
        return page

    def _build_any_image_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(11)

        card = QWidget()
        card.setObjectName("card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(12, 12, 12, 12)
        cv.setSpacing(10)
        note = QLabel(
            "For a frame that is not part of any prepared query set — a new drone "
            "capture, a screenshot."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        cv.addWidget(note)

        self.single_image_edit = QLineEdit()
        self.single_image_edit.setPlaceholderText("Pick any image — a query frame, or one of your own")
        self.single_image_edit.textChanged.connect(self._single_image_changed)
        cv.addLayout(self._labeled("Image", None, self.single_image_edit))
        browse = QPushButton("Browse…  jpg · png · tif")
        browse.setObjectName("mini")
        browse.clicked.connect(self._browse_single_image)
        cv.addWidget(browse, 0, Qt.AlignmentFlag.AlignLeft)

        self.single_heading = QLineEdit()
        self.single_heading.setPlaceholderText("unknown — four rotations")
        self.single_heading.setToolTip(
            "Compass direction of the image's top edge, degrees clockwise from north. "
            "Filled from the flight CSV when the image is one of its frames."
        )
        self.single_altitude = QLineEdit()
        self.single_altitude.setPlaceholderText("unknown — no crop")
        self.single_altitude.setToolTip(
            "Altitude above sea level (m). With the altitude step on, the terrain is read "
            "at the frame's ground-truth position to get height above ground."
        )
        tele = QHBoxLayout()
        tele.setSpacing(8)
        tele.addLayout(self._labeled("Heading °", None, self.single_heading), 1)
        tele.addLayout(self._labeled("Altitude ASL m", None, self.single_altitude), 1)
        cv.addLayout(tele)

        self.single_encoder_label = QLabel("-")
        self.single_encoder_label.setObjectName("encoderMirror")
        self.single_encoder_label.setWordWrap(True)
        self.single_encoder_label.setToolTip("Set on the inspector (Recipe).")
        cv.addLayout(self._labeled("Recipe", "from the inspector", self.single_encoder_label))

        self.single_button = QPushButton("Localize image")
        self.single_button.setObjectName("primary")
        self.single_button.clicked.connect(self.run_single_query)
        cv.addWidget(self.single_button)

        self.single_hint = QLabel("-")
        self.single_hint.setObjectName("hint")
        self.single_hint.setWordWrap(True)
        cv.addWidget(self.single_hint)
        v.addWidget(card)
        v.addStretch(1)
        return page

    # -------------------------------------------------------- trajectory ----
    TRAJ_METRIC_COLS = ["Track", "median", "p95", "max", "RMSE", "final", "drift %", "fix avail", "gated"]

    def _build_trajectory_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 8, 4)
        layout.setSpacing(12)

        head = QHBoxLayout()
        heading = QLabel("Flight  ·  VPR + odometry")
        heading.setObjectName("pageTitle")
        sub = QLabel("fuse per-frame VPR with a simulated IMU or with real visual odometry")
        sub.setObjectName("hint")
        head.addWidget(heading)
        head.addWidget(sub)
        head.addStretch(1)
        layout.addLayout(head)

        # -- controls --------------------------------------------------
        ctl = QWidget()
        ctl.setObjectName("card")
        cv = QVBoxLayout(ctl)
        cv.setContentsMargins(13, 12, 13, 12)
        cv.setSpacing(10)

        box = QGroupBox("Fusion")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)

        self.traj_imu = QComboBox()
        for label, value in TRAJ_IMU_CHOICES:
            self.traj_imu.addItem(label, value)
        self.traj_filter = QComboBox()
        for label, value in TRAJ_FILTER_CHOICES:
            self.traj_filter.addItem(label, value)
        self.traj_filter.setToolTip(
            "EKF: VPR + simulated IMU. The visual-odometry filters measure frame-to-frame motion "
            "from the images themselves (IMU only where VO fails) and fuse it with the full "
            "per-tile similarity map; hybrid = particle filter supervising a windowed Kalman "
            "filter, the best in docs/VO_AVL_Fusion_Experiment.md."
        )
        self.traj_initvel = QComboBox()
        self.traj_initvel.addItem("known (from GT)", "known")
        self.traj_initvel.addItem("zero", "zero")
        self.traj_vpr_every = QSpinBox()
        self.traj_vpr_every.setRange(1, 20)
        self.traj_vpr_every.setValue(1)
        self.traj_imu_scale = QDoubleSpinBox()
        self.traj_imu_scale.setRange(0.1, 10.0)
        self.traj_imu_scale.setSingleStep(0.1)
        self.traj_imu_scale.setValue(1.0)
        self.traj_stride = QSpinBox()
        self.traj_stride.setRange(1, 20)
        self.traj_stride.setValue(1)
        self.traj_max_frames = QSpinBox()
        self.traj_max_frames.setRange(2, 2000)
        self.traj_max_frames.setValue(120)
        self.traj_seed = QSpinBox()
        self.traj_seed.setRange(0, 99999)
        self.traj_seed.setValue(0)
        self.traj_dropout = QLineEdit()
        self.traj_dropout.setPlaceholderText("seconds, e.g. 120-180, 300-330")
        self.traj_fuse_alt = QCheckBox("fuse altitude")
        self.traj_real_time = QCheckBox("real frame timestamps")
        self.traj_real_time.setChecked(True)
        self.traj_real_time.setToolTip(
            "Source per-frame times from the matching UAV-VisLoc region CSV; "
            "otherwise assume a uniform 3 s spacing."
        )

        grid.addLayout(self._labeled("IMU error model", None, self.traj_imu), 0, 0)
        grid.addLayout(self._labeled("Fusion filter", None, self.traj_filter), 0, 1)
        grid.addLayout(self._labeled("Initial velocity", None, self.traj_initvel), 0, 2)
        grid.addLayout(self._labeled("VPR fix rate", "every N frames", self.traj_vpr_every), 1, 0)
        grid.addLayout(self._labeled("IMU noise ×", "scales the preset", self.traj_imu_scale), 1, 1)
        grid.addLayout(self._labeled("RNG seed", None, self.traj_seed), 1, 2)
        grid.addLayout(self._labeled("Frame stride", None, self.traj_stride), 2, 0)
        grid.addLayout(self._labeled("Max frames", None, self.traj_max_frames), 2, 1)
        grid.addLayout(self._labeled("Fix dropouts", None, self.traj_dropout), 2, 2)
        opts = QHBoxLayout()
        opts.addWidget(self.traj_fuse_alt)
        opts.addSpacing(16)
        opts.addWidget(self.traj_real_time)
        opts.addStretch(1)
        grid.addLayout(opts, 3, 0, 1, 3)
        cv.addWidget(box)

        inherit = QLabel(
            "Uses the inspector's map, flight, first encoder and top-5 fusion. Trajectory mode "
            "does not apply heading, centering or the altitude crop yet; with heading on, its "
            "frames are searched in four rotations."
        )
        inherit.setObjectName("caption")
        inherit.setWordWrap(True)
        cv.addWidget(inherit)

        run_row = QHBoxLayout()
        self.traj_run_button = QPushButton("Run trajectory")
        self.traj_run_button.setObjectName("primary")
        self.traj_run_button.clicked.connect(self.run_trajectory)
        self.traj_hint = QLabel("-")
        self.traj_hint.setObjectName("hint")
        self.traj_hint.setWordWrap(True)
        run_row.addWidget(self.traj_run_button, 0)
        run_row.addWidget(self.traj_hint, 1)
        cv.addLayout(run_row)
        layout.addWidget(ctl)

        # -- plots ---------------------------------------------------
        self.traj_map = TrajectoryPlot(xlabel="east (m) →", equal_aspect=True)
        self.traj_map.setMinimumHeight(180)
        self.traj_map.set_empty("run a trajectory to see the ground track")
        self.traj_err = TrajectoryPlot(xlabel="mission time (s) →")
        self.traj_err.setMinimumHeight(140)
        self.traj_err.set_empty("horizontal error vs time")
        layout.addWidget(self.traj_map, 3)
        layout.addWidget(self.traj_err, 2)

        # -- metrics table -----------------------------------------
        self.traj_table = QTableWidget(4, len(self.TRAJ_METRIC_COLS))
        self.traj_table.setHorizontalHeaderLabels(self.TRAJ_METRIC_COLS)
        self.traj_table.verticalHeader().setVisible(False)
        self.traj_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.traj_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.traj_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.traj_table.setFixedHeight(152)
        for r, name in enumerate(("IMU-only", "VPR-only", "Fused", "VO-only")):
            self.traj_table.setItem(r, 0, QTableWidgetItem(name))
            for c in range(1, len(self.TRAJ_METRIC_COLS)):
                self.traj_table.setItem(r, c, QTableWidgetItem("–"))
        layout.addWidget(self.traj_table, 0)
        return self._scrolled(page)

    def run_trajectory(self) -> None:
        if self._busy():
            return
        refs, queries = self._current_paths()
        if refs is None or queries is None:
            self.traj_hint.setText("Select both a reference database and a query set in the top bar.")
            return
        model = self._primary_model()
        tag = f"{self._db_tag(refs, queries)}__{model}__traj"
        self.traj_result_path = self.out_dir / f"{tag}_traj.json"
        self._traj_dest = "tab"
        args = self._traj_args(refs, queries, model, tag,
                               real_timestamps=self.traj_real_time.isChecked())

        self.current_tag = tag
        self.traj_run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_status(f"TRAJECTORY {tag}", busy=True)
        self._progress_start(f"trajectory {tag} · loading {model}…")
        self.append_log(f"[traj] {queries.name} over {refs.parent.name} map with {model}")

        self.traj_process = self._make_process()
        self.traj_process.readyReadStandardOutput.connect(self._read_traj_output)
        self.traj_process.finished.connect(self._traj_finished)
        self._spawn(self.traj_process, args)

    def _traj_args(
        self, refs: Path, queries: Path, model: str, tag: str,
        subset_file: Path | None = None, real_timestamps: bool = True,
    ) -> list[str]:
        recipe = self._primary_recipe()
        args = [
            str(self.project_root / "scripts" / "visloc_traj.py"),
            "--refs", str(refs),
            "--queries", str(queries),
            "--model", model,
            "--rotations", "4" if recipe.heading != "off" else str(recipe.effective_rotations),
            "--query-crop", recipe.query_crop,
            "--fusion", recipe.fusion,
            "--imu-grade", self.traj_imu.currentData(),
            "--imu-scale", f"{self.traj_imu_scale.value():g}",
            "--filter", self.traj_filter.currentData(),
            "--vpr-every", str(self.traj_vpr_every.value()),
            "--init-vel", self.traj_initvel.currentData(),
            "--seed", str(self.traj_seed.value()),
            "--tag", tag,
            "--out-dir", str(self.out_dir),
            "--ref-cache", str(self._ref_cache_path(refs, model)),
        ]
        # per-frame scores and VO steps do not depend on the filter, so switching
        # filters re-runs only the fusion (visloc_traj.py checks the frame list)
        frames = (subset_file.stem if subset_file is not None
                  else f"s{self.traj_stride.value()}_m{self.traj_max_frames.value()}")
        rot = args[args.index("--rotations") + 1]
        cache_dir = self.out_dir / "traj_cache"
        args += ["--scores-cache",
                 str(cache_dir / f"scores_{self._db_tag(refs, queries)}__{model}_r{rot}_{recipe.query_crop}_{frames}.npy")]
        if self.traj_filter.currentData() in TRAJ_VO_FILTERS:
            args += ["--vo-cache", str(cache_dir / f"vo_{queries.parent.name}_{frames}.npz")]
        if subset_file is not None:
            args += ["--frame-subset", str(subset_file)]
        else:
            args += ["--frame-stride", str(self.traj_stride.value()),
                     "--max-frames", str(self.traj_max_frames.value())]
        if self.traj_fuse_alt.isChecked():
            args.append("--fuse-altitude")
        spec = self.traj_dropout.text().strip()
        if spec:
            args += ["--dropout", spec]
        if model.startswith("anyloc"):
            args += ["--vocab", str(self.out_dir / f"vocab_{refs.parent.name}__{model}.npz")]
        # pass the real per-frame timestamps; visloc_traj.py itself falls back to
        # uniform spacing if they turn out non-monotonic (a re-ordered subset)
        if real_timestamps:
            region_csv = self._region_csv_for(queries)
            if region_csv is not None:
                args += ["--region-csv", str(region_csv)]
        return args

    def _region_csv_for(self, queries: Path) -> Path | None:
        """The UAV-VisLoc region CSV (…/UAVVisLoc/NN/NN.csv) for a prepared query set."""
        match = re.search(r"r(\d+)_t\d+", str(queries))
        if not match:
            return None
        region = match.group(1)
        candidate = self.project_root / "data" / "UAVVisLoc" / region / f"{region}.csv"
        return candidate if candidate.is_file() else None

    _TRAJ_FRAME_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+frames localized")

    def _read_traj_output(self) -> None:
        if self.traj_process is None or self._stopping:
            return
        text = bytes(self.traj_process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.replace("\r", "\n").splitlines():
            stripped = line.strip()
            if not stripped or "it/s]" in line or stripped.startswith("RESULT_JSON"):
                continue
            self.append_log(line.rstrip())
            step = self._TRAJ_FRAME_RE.search(stripped)
            if step:
                done, whole = int(step.group(1)), int(step.group(2))
                self._progress_phase("queries", done, whole, "frames")
                self._set_pipe_phase("search")
            elif "ref tiles" in stripped:
                self._set_pipe_phase("search")
            elif stripped.startswith("[traj]") and "ready on" in stripped:
                self._progress_update(label="encoding / localizing frames…")

    def _traj_finished(self, code: int, _status: Any) -> None:
        if self._stopping:
            return
        self.traj_process = None
        self.traj_run_button.setEnabled(True)
        self.studio_run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_status("READY")
        self._progress_done()
        studio = self._traj_dest == "studio"
        if code != 0:
            self.append_log(f"[traj] failed with code {code}")
            msg = "Trajectory run failed — open the LOG drawer at the bottom."
            (self.studio_hint if studio else self.traj_hint).setText(msg)
            return
        try:
            payload = json.loads(self.traj_result_path.read_text())
        except (OSError, json.JSONDecodeError, AttributeError):
            self.append_log("[traj] could not read the result JSON")
            return
        if studio:
            self._render_studio_result(payload)
        else:
            self._render_trajectory(payload)

    def _render_trajectory(self, payload: dict) -> None:
        series = payload.get("series", {})
        gt = series.get("gt_en", [])
        ins = series.get("ins_en", [])
        vpr = [p for p in series.get("vpr_en", []) if p]
        fused = series.get("fused_en", [])
        vo = series.get("vo_en") or []
        # with visual odometry the IMU only bridges VO failures; its free-running
        # track (often km off) would set the map's scale, so VO-only replaces it
        motion = ({"name": "VO-only", "pts": vo, "color": "#5ad1e6", "width": 1.4} if vo else
                  {"name": "IMU-only", "pts": ins, "color": "#f07a7a", "width": 1.4})
        self.traj_map.set_series([
            {"name": "ground truth", "pts": gt, "color": "#cfd2de", "width": 1.6},
            motion,
            {"name": "VPR fixes", "pts": vpr, "color": "#f0c04a", "dots": True, "width": 2.4},
            {"name": "fused", "pts": fused, "color": "#b5abfc", "width": 2.2},
        ])

        t = series.get("t_s", [])
        e_ins = series.get("err_ins_m", [])
        if vo:
            e_ins = [None if g is None or v is None else ((v[0] - g[0]) ** 2 + (v[1] - g[1]) ** 2) ** 0.5
                     for v, g in zip(vo, gt)]
        motion_name = "VO-only" if vo else "IMU-only"
        motion_colour = "#5ad1e6" if vo else "#f07a7a"
        e_vpr = series.get("err_vpr_m", [])
        e_fused = series.get("err_fused_m", [])
        ref_max = max([v for v in (e_fused + [v for v in e_vpr if v is not None]) if v is not None]
                     or [50.0])
        cap = max(50.0, 4.0 * ref_max)
        ins_clipped = any(v is not None and v > cap for v in e_ins)
        err_series = [
            {"name": f"{motion_name}{' (clipped)' if ins_clipped else ''}",
             "pts": [(t[i], min(e_ins[i], cap)) for i in range(len(e_ins)) if e_ins[i] is not None],
             "color": motion_colour, "width": 1.3},
            {"name": "VPR-only",
             "pts": [(t[i], e_vpr[i]) for i in range(len(e_vpr)) if e_vpr[i] is not None],
             "color": "#f0c04a", "dots": True, "width": 2.2},
            {"name": "fused",
             "pts": [(t[i], e_fused[i]) for i in range(len(e_fused)) if e_fused[i] is not None],
             "color": "#b5abfc", "width": 2.0},
        ]
        self.traj_err.set_series(err_series, bands=[tuple(b) for b in payload.get("dropouts_s", [])])

        self._fill_traj_row(0, payload.get("ins_only", {}))
        self._fill_traj_row(1, payload.get("vpr_only", {}))
        self._fill_traj_row(2, payload.get("fused", {}))
        self._fill_traj_row(3, payload.get("vo_only") or {})

        imu = payload.get("imu", {})
        self.traj_hint.setText(
            f"{payload.get('n_frames')} frames · {payload.get('duration_s', 0):.0f} s · "
            f"path {payload.get('path_length_m', 0) / 1000:.2f} km · "
            f"{payload.get('filter', 'eskf')} · "
            + (f"VO {100 * payload['vo_only'].get('steps_ok', 0):.0f}% steps ok · "
               if payload.get("vo_only") else f"IMU {imu.get('grade')}×{imu.get('scale', 1)} · ")
            + f"{payload.get('query_ms_per_frame', 0):.0f} ms/frame · "
            + (f"fixes {payload.get('fused', {}).get('fixes_accepted', '?')}"
               if payload.get("vo_only") else
               f"fixes acc {payload.get('fused', {}).get('fixes_accepted', '?')} / "
               f"gated {payload.get('fused', {}).get('fixes_gated', '?')}")
            + self._gate_warning(payload)
        )
        self._set_pipe_phase("done")

    def _fill_traj_row(self, row: int, m: dict) -> None:
        he = m.get("horiz_error_m", {})

        def put(col: int, value, tone: bool = False) -> None:
            if value is None or (isinstance(value, float) and value != value):
                text = "–"
            elif isinstance(value, float):
                text = f"{value:.1f}"
            else:
                text = str(value)
            item = QTableWidgetItem(text)
            if tone and isinstance(value, (int, float)) and value == value:
                colour = "#7ee0a8" if value < 50 else "#f0c04a" if value < 150 else "#f07a7a"
                item.setForeground(QColor(colour))
            self.traj_table.setItem(row, col, item)

        put(1, he.get("median"))
        put(2, he.get("p95"), tone=True)
        put(3, he.get("max"))
        put(4, m.get("rmse_m"))
        put(5, m.get("final_error_m"), tone=True)
        put(6, m.get("drift_rate_pct"))
        avail = m.get("fix_availability")
        put(7, f"{avail:.2f}" if isinstance(avail, (int, float)) else None)
        put(8, m.get("fixes_gated"))

    # ---------------------------------------------------- trajectory studio ----
    def _build_studio_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 8, 4)
        layout.setSpacing(10)

        head = QHBoxLayout()
        heading = QLabel("Studio")
        heading.setObjectName("pageTitle")
        sub = QLabel("draw a route → it becomes the flight path → simulate IMU + VPR (error from this region's benchmark) → fuse")
        sub.setObjectName("hint")
        head.addWidget(heading)
        head.addWidget(sub)
        head.addStretch(1)
        layout.addLayout(head)

        bar = QWidget()
        bar.setObjectName("card")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 9, 12, 9)
        bl.setSpacing(9)
        self.studio_region_label = QLabel("region —")
        self.studio_region_label.setObjectName("coord")
        self.studio_speed = QSpinBox()
        self.studio_speed.setRange(2, 120)
        self.studio_speed.setValue(20)
        self.studio_speed.setSuffix(" m/s")
        self.studio_fix_dt = QDoubleSpinBox()
        self.studio_fix_dt.setRange(0.5, 30.0)
        self.studio_fix_dt.setSingleStep(0.5)
        self.studio_fix_dt.setValue(3.0)
        self.studio_fix_dt.setSuffix(" s")
        self.studio_vpr_sigma = QSpinBox()
        self.studio_vpr_sigma.setRange(5, 1000)
        self.studio_vpr_sigma.setValue(80)
        self.studio_vpr_sigma.setSuffix(" m")
        self.studio_vpr_sigma.setToolTip(
            "VPR fix noise used when there is no benchmark on disk for this "
            "region + encoder. Run the encoder in the Comparison tab for a "
            "measured error distribution instead."
        )
        self.studio_draw_btn = QPushButton("Draw route")
        self.studio_draw_btn.setCheckable(True)
        self.studio_draw_btn.toggled.connect(self._studio_toggle_draw)
        self.studio_clear_btn = QPushButton("Clear")
        self.studio_clear_btn.clicked.connect(self._studio_clear)
        self.studio_run_button = QPushButton("Run mission")
        self.studio_run_button.setObjectName("primary")
        self.studio_run_button.clicked.connect(self._studio_run)
        bl.addWidget(self.studio_region_label)
        bl.addStretch(1)
        bl.addWidget(QLabel("speed"))
        bl.addWidget(self.studio_speed)
        bl.addWidget(QLabel("fix every"))
        bl.addWidget(self.studio_fix_dt)
        bl.addWidget(QLabel("VPR ±"))
        bl.addWidget(self.studio_vpr_sigma)
        bl.addWidget(self.studio_draw_btn)
        bl.addWidget(self.studio_clear_btn)
        bl.addWidget(self.studio_run_button)
        layout.addWidget(bar)

        self.studio_hint = QLabel("open this tab with a UAV-VisLoc region query set selected in the top bar")
        self.studio_hint.setObjectName("hint")
        self.studio_hint.setWordWrap(True)
        layout.addWidget(self.studio_hint)

        layout.addWidget(self._legend_row([
            ("#f0c04a", "drawn route", "line"),
            ("#cfd2de", "ground truth (= route)", "line"),
            ("#f07a7a", "IMU-only", "line"),
            ("#5ad1e6", "VPR fixes", "dot"),
            ("#b5abfc", "fused", "line"),
        ]))

        self.studio_view = MosaicView(self._on_studio_vertex)
        layout.addWidget(self.studio_view, 1)

        self._studio_page = page
        # a query-set change invalidates the loaded mosaic; re-sync now if the
        # Studio tab is what the user is looking at, otherwise on its next visit
        self.refs_combo.currentIndexChanged.connect(self._studio_invalidate)
        self.queries_combo.currentIndexChanged.connect(self._studio_invalidate)
        self.encoder_combo.currentIndexChanged.connect(
            lambda *_: self.tabs.currentWidget() is self._studio_page and self._studio_update_ready_hint()
        )
        return page

    def _studio_invalidate(self, *_a) -> None:
        self._studio_raster = None
        if getattr(self, "tabs", None) is not None and self.tabs.currentWidget() is self._studio_page:
            self._studio_sync()

    @staticmethod
    def _legend_row(items: list[tuple[str, str, str]]) -> QWidget:
        """A horizontal key: [(colour, label, 'line'|'dot'), ...]."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(6)
        for colour, label, kind in items:
            swatch = QLabel()
            if kind == "dot":
                swatch.setFixedSize(9, 9)
                swatch.setStyleSheet(f"background:{colour}; border-radius:4px;")
            else:
                swatch.setFixedSize(18, 4)
                swatch.setStyleSheet(f"background:{colour}; border-radius:2px;")
            text = QLabel(label)
            text.setObjectName("hint")
            h.addWidget(swatch)
            h.addWidget(text)
            h.addSpacing(10)
        h.addStretch(1)
        return w

    def _studio_region_id(self) -> str | None:
        _, queries = self._current_paths()
        if queries is None:
            return None
        match = re.search(r"r(\d+)_t\d+", str(queries))
        return match.group(1) if match else None

    def _studio_bench_csv(self) -> Path | None:
        """The visloc_eval per-query CSV for the current region + active encoder."""
        refs, queries = self._current_paths()
        if refs is None or queries is None:
            return None
        model = self._primary_model()
        csv = self.out_dir / f"{self._db_tag(refs, queries)}__{model}_per_query.csv"
        return csv if csv.is_file() else None

    def _studio_set_enabled(self, on: bool) -> None:
        for w in (self.studio_draw_btn, self.studio_clear_btn, self.studio_run_button,
                  self.studio_speed, self.studio_fix_dt, self.studio_vpr_sigma):
            w.setEnabled(on)

    def _studio_sync(self) -> None:
        region = self._studio_region_id()
        if region is None:
            _, queries = self._current_paths()
            name = queries.parent.name if queries is not None else "none"
            self._studio_raster = None
            self.studio_view.clear_mosaic()
            self.studio_region_label.setText("region —")
            self.studio_hint.setText(
                f"Studio needs a UAV-VisLoc region query set — a folder named like 'r05_t250' "
                f"with a satellite mosaic on disk. Current query set: '{name}'."
            )
            self._studio_set_enabled(False)
            return
        if self._studio_raster is not None and self._studio_raster.region == region.zfill(2):
            return
        raster = RegionRaster.for_region(region, self.project_root)
        if raster is None:
            self._studio_raster = None
            self.studio_view.clear_mosaic()
            self.studio_region_label.setText(f"region {region} — no mosaic")
            self.studio_hint.setText(
                f"No satellite mosaic found for region {region} under data/UAVVisLoc/."
            )
            self._studio_set_enabled(False)
            return
        self.studio_hint.setText(f"loading region {raster.region} mosaic…")
        QApplication.processEvents()
        thumb, scale = raster.thumbnail(1600)
        self._studio_raster = raster
        self._studio_scale = scale
        self._studio_subset = []
        self.studio_view.set_mosaic(thumb)
        self.studio_region_label.setText(
            f"region {raster.region}  ·  {raster.width}×{raster.height} px"
        )
        self._studio_set_enabled(True)
        self.studio_draw_btn.setChecked(False)
        self._studio_update_ready_hint()

    def _studio_update_ready_hint(self) -> None:
        bench = self._studio_bench_csv()
        model = self._primary_model()
        has_route = len(self.studio_view.path_vertices()) >= 2
        self.studio_run_button.setEnabled(has_route and not self._busy())
        if bench is not None:
            src = f"VPR error sampled from {bench.name}"
        else:
            src = (f"no benchmark for '{model}' on this region — VPR error ≈ ±"
                   f"{self.studio_vpr_sigma.value()} m (run '{model}' in the Comparison "
                   f"tab for a measured distribution)")
        pre = "click 'Draw route', lay a flight path, then 'Run mission'. " if not has_route else ""
        self.studio_hint.setText(f"{pre}{src}. IMU / filter settings from the Trajectory tab.")

    def _studio_toggle_draw(self, on: bool) -> None:
        self.studio_draw_btn.setText("Drawing… (click to finish)" if on else "Draw route")
        self.studio_view.set_draw_mode(on)
        if on:
            self.studio_view.reset_path()
            self.studio_view.clear_overlay()
        else:
            self._studio_update_ready_hint()

    def _studio_clear(self) -> None:
        self.studio_draw_btn.setChecked(False)
        self.studio_view.reset_path()
        self.studio_view.clear_overlay()
        self.studio_view.fit_content(None)
        self._studio_update_ready_hint()

    def _on_studio_vertex(self, _scene_pt) -> None:
        n = len(self.studio_view.path_vertices())
        self.studio_run_button.setEnabled(n >= 2 and not self._busy())
        self.studio_hint.setText(
            f"{n} vertex/vertices — click 'Draw route' again to finish, then 'Run mission'"
        )

    def _studio_route_geo(self) -> list[list[float]] | None:
        raster = self._studio_raster
        verts = self.studio_view.path_vertices()
        if raster is None or len(verts) < 2:
            return None
        route = []
        for v in verts:
            lat, lon = raster.px_to_geo(v.x() / self._studio_scale, v.y() / self._studio_scale)
            route.append([float(lat), float(lon)])
        return route

    def _studio_run(self) -> None:
        if self._busy():
            return
        refs, queries = self._current_paths()
        route = self._studio_route_geo()
        bench = self._studio_bench_csv()
        model = self._primary_model()
        if route is None:
            self.studio_hint.setText("draw a route with at least two points first")
            return

        spec = {
            "waypoints": route,
            "speed_mps": float(self.studio_speed.value()),
            "alt_m": 0.0,
            "fix_dt_s": float(self.studio_fix_dt.value()),
        }
        self._studio_subset_path = Path(tempfile.gettempdir()) / f"avl_studio_route_{model}.json"
        self._studio_subset_path.write_text(json.dumps(spec))
        tag = f"{self._db_tag(refs, queries) if refs and queries else 'studio'}__{model}__studio"
        self.traj_result_path = self.out_dir / f"{tag}_traj.json"
        self._traj_dest = "studio"

        args = [
            str(self.project_root / "scripts" / "visloc_traj.py"),
            "--synthetic-path", str(self._studio_subset_path),
            "--model", model,
            "--imu-grade", self.traj_imu.currentData(),
            "--imu-scale", f"{self.traj_imu_scale.value():g}",
            # a drawn mission has no images, so no visual odometry: always the IMU EKF
            "--filter", "eskf",
            "--vpr-every", str(self.traj_vpr_every.value()),
            "--init-vel", self.traj_initvel.currentData(),
            "--seed", str(self.traj_seed.value()),
            "--tag", tag,
            "--out-dir", str(self.out_dir),
        ]
        if bench is not None:
            args += ["--vpr-error-csv", str(bench)]
            src = bench.name
        else:
            args += ["--vpr-sigma", str(self.studio_vpr_sigma.value())]
            src = f"±{self.studio_vpr_sigma.value()} m (synthetic — no benchmark)"
        if refs is not None:
            args += ["--refs", str(refs)]
        if self.traj_fuse_alt.isChecked():
            args.append("--fuse-altitude")
        spec_drop = self.traj_dropout.text().strip()
        if spec_drop:
            args += ["--dropout", spec_drop]

        self.current_tag = tag
        self.traj_run_button.setEnabled(False)
        self.studio_run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_status(f"STUDIO {tag}", busy=True)
        self._progress_start(f"studio mission · {len(route)} waypoints…")
        self.append_log(f"[studio] mission · {len(route)} waypoints · VPR error {src}")

        self.traj_process = self._make_process()
        self.traj_process.readyReadStandardOutput.connect(self._read_traj_output)
        self.traj_process.finished.connect(self._traj_finished)
        self._spawn(self.traj_process, args)

    def _render_studio_result(self, payload: dict) -> None:
        raster = self._studio_raster
        if raster is None:
            return
        origin = payload.get("origin", {})
        lf = LocalFrame(lat0=origin.get("lat", 0.0), lon0=origin.get("lon", 0.0))
        series = payload.get("series", {})

        def to_view(en_list) -> list[QPointF]:
            out = []
            for e, n in en_list:
                lat, lon, _ = lf.enu_to_geo(e, n, 0.0)
                px, py = raster.geo_to_px(lat, lon)
                out.append(QPointF(float(px) * self._studio_scale, float(py) * self._studio_scale))
            return out

        self.studio_view.clear_overlay()
        self.studio_view.draw_polyline(to_view(series.get("gt_en", [])), "#cfd2de", 1.8)
        self.studio_view.draw_polyline(to_view(series.get("ins_en", [])), "#f07a7a", 1.5)
        vpr = [p for p in series.get("vpr_en", []) if p]
        self.studio_view.draw_markers(
            to_view(vpr), "#5ad1e6", max(2.0, raster.width * self._studio_scale / 650)
        )
        self.studio_view.draw_polyline(to_view(series.get("fused_en", [])), "#b5abfc", 2.4)
        self.studio_view.fit_overlays()

        fused = payload.get("fused", {})
        vprm = payload.get("vpr_only", {})
        ins = payload.get("ins_only", {})
        f_med = fused.get("horiz_error_m", {}).get("median", float("nan"))
        v_med = vprm.get("horiz_error_m", {}).get("median", float("nan"))
        t = series.get("t_s", [])
        gap = max((t[i + 1] - t[i] for i in range(len(t) - 1)), default=0.0)
        gap_note = f"  ·  ⚠ {gap:.0f} s data gap in this leg (IMU coasts)" if gap > 12 else ""
        self.studio_hint.setText(
            f"{payload.get('n_frames')} frames · {payload.get('path_length_m', 0)/1000:.2f} km  ·  "
            f"VPR-only median {v_med:.0f} m  →  Fused median {f_med:.0f} m "
            f"(RMSE {fused.get('rmse_m', float('nan')):.0f} m, final {fused.get('final_error_m', float('nan')):.0f} m)  ·  "
            f"IMU-only drift {ins.get('drift_rate_pct', float('nan')):.1f}%  ·  "
            f"gated {fused.get('fixes_gated', '?')}/{(fused.get('fixes_gated', 0) or 0) + (fused.get('fixes_accepted', 0) or 0)}"
            + gap_note + self._gate_warning(payload)
        )
        self._set_pipe_phase("done")

    @staticmethod
    def _gate_warning(payload: dict) -> str:
        """Appended when the χ² gate rejected most VPR fixes — the fused solution is
        then little more than dead-reckoning."""
        f = payload.get("fused", {})
        gated = f.get("fixes_gated", 0) or 0
        acc = f.get("fixes_accepted", 0) or 0
        total = gated + acc
        if total and gated / total > 0.4:
            return (
                f"   ⚠ {100 * gated / total:.0f}% of VPR fixes rejected as inconsistent — "
                "the encoder is likely a poor match for this region "
                "(try anyloc-l / anyloc-lite), or the route does not follow the flight."
            )
        return ""

    def _open_run_from_comparison(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        tag = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not tag:
            return
        index = self.run_combo.findText(tag)
        if index >= 0:
            self.run_combo.setCurrentIndex(index)
        self._show_set_mode()

    # ------------------------------------------------------- delete a run --
    def _row_tag(self, row: int) -> str | None:
        item = self.table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _table_context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        tag = self._row_tag(row) if row >= 0 else None
        if not tag:
            return
        menu = QMenu(self.table)
        menu.addAction(f"Delete run  “{tag}”…", lambda: self._delete_run(tag))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _delete_selected_run(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        tag = self._row_tag(rows[0].row()) if rows else None
        if not tag:
            self.append_log("Select a run in the table first, then Delete.")
            return
        self._delete_run(tag)

    def _delete_run(self, tag: str) -> None:
        if self.process is not None:
            self.append_log("A benchmark is running — stop it before deleting a run.")
            return
        targets = [
            self.out_dir / f"{tag}_summary.json",
            self.out_dir / f"{tag}_per_query.csv",
            self.out_dir / f"vocab_{tag}.npz",  # anyloc vocab, if any
        ]
        present = [p for p in targets if p.exists()]
        if not present:
            self.append_log(f"Nothing on disk for run '{tag}'.")
            self.reload_results()
            return
        answer = QMessageBox.question(
            self,
            "Delete run",
            f"Delete the saved report for\n\n  {tag}\n\n"
            + "\n".join(f"  · {p.name}" for p in present)
            + "\n\nThe encoded-reference cache is kept. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for path in present:
            try:
                path.unlink()
                self.append_log(f"[delete] removed {path.name}")
            except OSError as error:
                self.append_log(f"[delete] could not remove {path.name}: {error}")
        self.reload_results()

    def _show_set_mode(self) -> None:
        self.mode_set_btn.setChecked(True)
        self.localize_stack.setCurrentIndex(0)
        self.tabs.setCurrentWidget(self.localize_page)

    def _show_image_mode(self) -> None:
        self.mode_img_btn.setChecked(True)
        self.localize_stack.setCurrentIndex(1)
        self.tabs.setCurrentWidget(self.localize_page)

    @staticmethod
    def _scrolled(widget: QWidget) -> QScrollArea:
        scroller = QScrollArea()
        scroller.setWidget(widget)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroller

    def _apply_style(self) -> None:
        """Nocturne Console: tokens from avl.console_widgets, one accent, hairlines."""
        t = dict(
            BG=BG, CHROME=CHROME, PANEL=PANEL, RAISED=RAISED, LINE=LINE, LINE2="#30364a",
            TEXT=TEXT, TEXT2=TEXT_2, MUTED=MUTED, FAINT=FAINT, ACCENT=ACCENT,
            A300=ACCENT_300, A400="#b0a7fb", A800="#3b3470", GOOD=GOOD, WARN=WARN, BAD=BAD,
            MONO=MONO,
        )
        css = """
        QWidget { color: %(TEXT2)s; font-size: 13px; }
        QLabel { background: transparent; }
        #root, #workspace, #pageHolder, QScrollArea, #tabBody, #retrievalPane { background: %(BG)s; }
        QScrollArea > QWidget > QWidget { background: transparent; }

        /* ── top bar ── */
        #topbar { background: %(CHROME)s; border-bottom: 1px solid %(LINE)s; }
        #appTitle { color: %(TEXT)s; font-size: 15px; font-weight: 600; }
        #appSubtitle { color: %(FAINT)s; font-size: 11px; }
        #vrule { background: %(LINE)s; }
        #ctxLabel { color: %(FAINT)s; font-size: 10px; font-weight: 700; letter-spacing: 1px;
                    padding-left: 8px; }
        QComboBox#ctxCombo { background: %(RAISED)s; border: 1px solid %(LINE)s; border-radius: 8px;
                    padding: 6px 10px; min-width: 170px; color: %(TEXT)s; font-weight: 500; }
        QComboBox#ctxCombo:hover { border-color: %(LINE2)s; }
        #recipeLine { color: %(TEXT2)s; background: %(PANEL)s; border: 1px solid %(LINE)s;
                      border-radius: 16px; padding: 7px 14px; font-size: 12px; }
        #recipeLine:hover { border-color: %(A800)s; }
        QPushButton#runBtn { color: #ffffff; border: 0; border-radius: 9px;
                    padding: 8px 18px 8px 34px; font-weight: 600; font-size: 13px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #9d92f8, stop:1 #6c5fe0); }
        QPushButton#runBtn:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #aea5fa, stop:1 #7a6dea); }
        QPushButton#runBtn:pressed { background: #5f52d2; }
        QPushButton#runBtn:disabled { background: %(RAISED)s; color: %(FAINT)s; }
        QPushButton#stopBtn { background: transparent; color: %(TEXT2)s; border: 1px solid %(LINE2)s;
                    border-radius: 9px; padding: 8px 14px 8px 32px; font-weight: 500; }
        QPushButton#stopBtn:hover { border-color: %(BAD)s; color: %(TEXT)s; }
        QPushButton#stopBtn:disabled { color: %(FAINT)s; border-color: %(LINE)s; }
        QPushButton#iconBtn { background: transparent; border: 0; border-radius: 8px; padding: 0; }
        QPushButton#iconBtn:hover { background: rgba(255,255,255,0.06); }
        QPushButton#iconBtn:checked { background: rgba(139,127,240,0.16); }
        #progressStrip { background: %(CHROME)s; }
        QProgressBar#progress { background: transparent; border: 0; }
        QProgressBar#progress::chunk { background: %(ACCENT)s; }

        /* ── rail, inspector, status bar ── */
        #navRail { background: %(CHROME)s; border-right: 1px solid %(LINE)s; }
        #inspector { background: %(CHROME)s; border-left: 1px solid %(LINE)s; }
        #inspectorHead { background: %(CHROME)s; border-bottom: 1px solid %(LINE)s; }
        #inspectorTitle { color: %(TEXT)s; font-size: 15px; font-weight: 600; }
        #inspectorHint { color: %(FAINT)s; font-size: 11px; padding-top: 3px; }
        QScrollArea#inspectorScroll, #inspectorBody { background: %(CHROME)s; }
        #sectionTitle { color: %(FAINT)s; font-size: 10px; font-weight: 700; letter-spacing: 1.3px; }
        #pipeSection[flash="1"] { background: rgba(139,127,240,0.12); border-radius: 10px; }
        #insetCard { background: %(PANEL)s; border: 1px solid %(LINE)s; border-radius: 11px; }
        #dataLine { color: %(TEXT2)s; font-size: 12px; }
        #capsLine { color: %(MUTED)s; font-size: 12px; padding-top: 2px; }
        #presetNote { color: %(MUTED)s; font-size: 12px; padding: 0 2px; }
        #statusbar { background: %(CHROME)s; border-top: 1px solid %(LINE)s; }
        #progressLabel { color: %(MUTED)s; font-family: %(MONO)s; font-size: 11px; }
        #statusText2 { color: %(FAINT)s; font-size: 11px; }
        QPushButton#statusBtn { border: 0; color: %(MUTED)s; font-size: 11px; padding: 3px 10px;
                    border-radius: 6px; background: transparent; }
        QPushButton#statusBtn:hover { color: %(TEXT)s; background: rgba(255,255,255,0.06); }
        QPushButton#statusBtn:checked { color: %(A300)s; background: rgba(139,127,240,0.16); }

        /* ── recipe steps ── */
        #seg { background: %(BG)s; border: 1px solid %(LINE)s; border-radius: 10px; }
        QPushButton#segBtn { border: 0; border-radius: 7px; padding: 6px 8px; color: %(MUTED)s;
                    background: transparent; font-weight: 500; font-size: 12px; }
        QPushButton#segBtn:hover { color: %(TEXT)s; }
        QPushButton#segBtn:checked { background: rgba(139,127,240,0.20); color: %(A300)s; }
        #stepRow { background: transparent; border: 0; }
        #stepRow[on="1"] { background: rgba(139,127,240,0.045); }
        #stepTitle { color: %(TEXT)s; font-size: 13px; font-weight: 600; }
        #stepRow[on="0"] #stepTitle { color: %(MUTED)s; }
        #stepSummary { color: %(MUTED)s; font-size: 12px; }
        #stepRow[on="0"] #stepSummary { color: %(FAINT)s; }
        #stepDot { color: %(ACCENT)s; font-size: 9px; }
        #hrule { background: %(LINE)s; }
        #impactBadge { border-radius: 9px; color: %(GOOD)s; font-size: 10px; font-weight: 600;
                    padding: 2px 8px; background: rgba(95,211,154,0.10); }
        #impactBadge[status="accuracy"] { color: %(A300)s; background: rgba(139,127,240,0.16); }
        #impactBadge[status="not adopted"] { color: %(BAD)s; background: rgba(242,114,122,0.10); }
        #impactBadge[status="prototype"] { color: %(WARN)s; background: rgba(242,193,78,0.10); }
        QPushButton#infoBtn { border: 1px solid %(LINE2)s; border-radius: 9px; color: %(MUTED)s;
                    font-size: 10px; font-weight: 700; padding: 0; background: transparent; }
        QPushButton#infoBtn:hover { color: %(A300)s; border-color: %(ACCENT)s; }
        #methodNote { color: %(WARN)s; font-size: 11px; }
        QCheckBox#encCheck { color: %(TEXT2)s; font-size: 12px; padding: 3px 0; }

        /* ── pages ── */
        #pageTitle { color: %(TEXT)s; font-size: 22px; font-weight: 600; }
        #pageSubtitle { color: %(MUTED)s; font-size: 13px; }
        #cardHeading { color: %(TEXT)s; font-size: 16px; font-weight: 600; }
        #card { background: %(PANEL)s; border: 1px solid %(LINE)s; border-radius: 12px; }
        #cardAccent { background: #17152c; border: 1px solid %(A800)s;
                      border-radius: 12px; }
        #panelTitle { color: %(MUTED)s; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; }
        #panelTitleLarge { color: %(TEXT)s; font-size: 15px; font-weight: 600; }
        #panelHint { color: %(FAINT)s; font-size: 11px; }
        #logHeading { color: %(MUTED)s; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; }
        #shapeStrip { color: %(A400)s; font-family: %(MONO)s; font-size: 12px; padding-top: 2px; }
        #statTile { background: %(PANEL)s; border: 1px solid %(LINE)s; border-radius: 14px; }
        #statTile[hero="1"] { border: 1px solid %(A800)s; background: qlineargradient(x1:0, y1:0,
                    x2:1, y2:1, stop:0 #221d45, stop:0.65 #171a28, stop:1 #161a24); }
        #tileLabel { color: %(MUTED)s; font-size: 12px; font-weight: 500; }
        #tileValue { color: %(TEXT)s; font-size: 32px; font-weight: 600; }
        #tileValueHero { color: #ffffff; font-size: 42px; font-weight: 600; }
        #tileUnit { color: %(MUTED)s; font-size: 15px; font-weight: 500; }
        #tileDelta { font-size: 12px; }
        #tileFoot { color: %(FAINT)s; font-size: 11px; }
        #recentRow { border-radius: 8px; background: transparent; }
        #recentRow:hover { background: %(RAISED)s; }
        #recentName { color: %(TEXT)s; font-size: 13px; font-weight: 500; }
        #recentMeta { color: %(FAINT)s; font-size: 11px; }
        #recentValue { color: %(TEXT)s; font-size: 15px; font-weight: 600; }

        /* findings */
        #findingCard { background: %(PANEL)s; border: 1px solid %(LINE)s; border-radius: 14px; }
        #findingCard[flash="1"] { border: 1px solid %(ACCENT)s; background: #1b1934; }
        #findingTitle { color: %(TEXT)s; font-size: 16px; font-weight: 600; }
        #findingBadge { color: %(A300)s; font-size: 22px; font-weight: 600; }
        #overline { color: %(FAINT)s; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
                    padding-top: 4px; }
        #findingText { color: %(TEXT2)s; font-size: 13px; }
        #findingMeta { color: %(MUTED)s; font-size: 12px; }
        #findingNumbers { background: %(BG)s; border: 1px solid %(LINE)s; border-radius: 9px; }
        #numSetting { color: %(MUTED)s; font-size: 12px; }
        #numResult { color: %(TEXT)s; font-size: 12px; font-weight: 600; }
        #statusTag { border-radius: 9px; font-size: 11px; font-weight: 600; padding: 3px 9px;
                     color: %(GOOD)s; background: rgba(95,211,154,0.10); }
        #statusTag[status="accuracy"] { color: %(A300)s; background: rgba(139,127,240,0.16); }
        #statusTag[status="not adopted"] { color: %(BAD)s; background: rgba(242,114,122,0.10); }
        #statusTag[status="prototype"] { color: %(WARN)s; background: rgba(242,193,78,0.10); }

        /* output panel */
        #logDrawer { background: %(CHROME)s; border-top: 1px solid %(LINE)s; }
        #logLast { color: %(FAINT)s; font-family: %(MONO)s; font-size: 11px; }
        #log { background: #0a0c11; color: %(TEXT2)s; border: 0; border-top: 1px solid %(LINE)s;
               font-family: %(MONO)s; font-size: 12px; padding: 6px 12px; }

        /* inspect / flight pages */
        #runDetails { color: %(MUTED)s; font-family: %(MONO)s; font-size: 11px; padding: 2px; }
        #encoderMirror { color: %(A300)s; font-size: 12px; border: 1px dashed %(LINE2)s;
                         border-radius: 8px; padding: 7px 10px; background: %(PANEL)s; }
        #coord { color: %(TEXT)s; font-family: %(MONO)s; }
        #planLine { color: %(A400)s; font-family: %(MONO)s; font-size: 11px; padding: 0 2px; }
        #errorHero { color: %(TEXT)s; font-size: 17px; font-weight: 600; }
        #posLabel { color: %(MUTED)s; font-size: 12px; }
        #tileCaption { color: %(MUTED)s; font-size: 11px; font-family: %(MONO)s; }
        #imagePreview { background: #0a0c11; border: 1px solid %(LINE)s; border-radius: 10px;
                        color: %(FAINT)s; font-size: 12px; }
        #hint { color: %(MUTED)s; font-size: 12px; }
        #caption { color: %(FAINT)s; font-size: 11px; }
        #fieldLabel { color: %(TEXT2)s; font-size: 12px; font-weight: 500; }
        #fieldHint { color: %(FAINT)s; font-size: 10px; }

        /* ── controls ── */
        QPushButton { background: %(RAISED)s; color: %(TEXT2)s; border: 1px solid %(LINE)s;
                      border-radius: 8px; padding: 7px 13px; font-size: 12px; font-weight: 500; }
        QPushButton:hover { border-color: %(LINE2)s; color: %(TEXT)s; }
        QPushButton:pressed { background: #252a3a; }
        QPushButton:disabled { color: %(FAINT)s; border-color: %(LINE)s; background: transparent; }
        QPushButton#primary { color: #ffffff; border: 0; background: qlineargradient(x1:0, y1:0,
                    x2:1, y2:1, stop:0 #9d92f8, stop:1 #6c5fe0); font-weight: 600; padding: 8px 14px; }
        QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #aea5fa, stop:1 #7a6dea); }
        QPushButton#primary:disabled { background: %(RAISED)s; color: %(FAINT)s; }
        QPushButton#secondary { background: rgba(139,127,240,0.10); color: %(A300)s;
                    border: 1px solid %(A800)s; }
        QPushButton#secondary:hover { background: rgba(139,127,240,0.18); }
        QPushButton#ghost { border: 0; background: transparent; color: %(A400)s; padding: 5px 8px; }
        QPushButton#ghost:hover { background: rgba(139,127,240,0.10); }
        QPushButton#mini { padding: 4px 10px; font-size: 11px; }
        QPushButton:checked { background: rgba(139,127,240,0.18); color: %(A300)s; border-color: %(A800)s; }

        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: %(RAISED)s; color: %(TEXT)s;
                    border: 1px solid %(LINE)s; border-radius: 8px; padding: 6px 9px; font-size: 12px;
                    selection-background-color: %(ACCENT)s; }
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: %(LINE2)s; }
        QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: %(ACCENT)s; }
        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
                    color: %(FAINT)s; background: %(PANEL)s; }
        QComboBox::drop-down { border: 0; width: 20px; }
        QComboBox QAbstractItemView { background: %(RAISED)s; color: %(TEXT)s; border: 1px solid %(LINE2)s;
                    outline: 0; padding: 4px; selection-background-color: rgba(139,127,240,0.30);
                    selection-color: #ffffff; }
        QComboBox QAbstractItemView::item { min-height: 26px; padding: 4px 8px; border-radius: 6px; }
        QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button,
        QDoubleSpinBox::down-button { width: 14px; border: 0; background: transparent; }
        QCheckBox { color: %(TEXT2)s; spacing: 8px; }
        QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid %(LINE2)s;
                    border-radius: 4px; background: %(RAISED)s; }
        QCheckBox::indicator:checked { background: %(ACCENT)s; border-color: %(ACCENT)s; }
        QCheckBox::indicator:disabled { background: %(PANEL)s; border-color: %(LINE)s; }
        QGroupBox { color: %(TEXT2)s; border: 1px solid %(LINE)s; border-radius: 10px;
                    margin-top: 12px; padding-top: 12px; font-size: 12px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
        QGraphicsView { background: #0a0c11; border: 1px solid %(LINE)s; border-radius: 12px; }

        QTableWidget { background: transparent; color: %(TEXT2)s; gridline-color: transparent;
                    border: 0; font-size: 13px; outline: 0; }
        QTableWidget::item { padding: 8px 10px; border-bottom: 1px solid %(LINE)s; }
        QTableWidget::item:hover { background: rgba(255,255,255,0.025); }
        QTableWidget::item:selected { background: rgba(139,127,240,0.15); color: #ffffff; }
        QHeaderView { background: transparent; border: 0; }
        QHeaderView::section { background: transparent; color: %(FAINT)s; border: 0;
                    border-bottom: 1px solid %(LINE2)s; padding: 9px 10px; font-size: 11px;
                    font-weight: 700; }
        QTableCornerButton::section { background: transparent; border: 0; }

        QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
        QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
        QScrollBar::handle { background: rgba(255,255,255,0.12); border-radius: 3px; }
        QScrollBar::handle:hover { background: rgba(255,255,255,0.24); }
        QScrollBar::handle:vertical { min-height: 30px; }
        QScrollBar::handle:horizontal { min-width: 30px; }
        QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
        QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
        QToolTip { background: %(RAISED)s; color: %(TEXT)s; border: 1px solid %(LINE2)s;
                   padding: 6px 8px; border-radius: 6px; font-size: 12px; }
        QMenu { background: %(RAISED)s; color: %(TEXT)s; border: 1px solid %(LINE2)s; padding: 4px; }
        QMenu::item { padding: 6px 14px; border-radius: 6px; }
        QMenu::item:selected { background: rgba(139,127,240,0.25); }
        QMessageBox { background: %(PANEL)s; }
        """ % t
        self.setStyleSheet(css)

    # ----------------------------------------------------------- databases --
    def refresh_databases(self) -> None:
        """Find every prepared reference map and query set under data/."""
        refs_found: list[tuple[str, Path]] = []
        queries_found: list[tuple[str, Path]] = []

        denseuav = self.project_root / "data" / "denseuav_avl"
        refs = denseuav / "references_gallery_satellite.csv"
        queries = denseuav / "queries_test_drone.csv"
        if refs.exists():
            refs_found.append(("DenseUAV satellite gallery", refs))
        if queries.exists():
            queries_found.append(("DenseUAV test drone queries", queries))

        for folder in sorted((self.project_root / "data" / "visloc_avl").glob("*/")):
            refs = folder / "references.csv"
            queries = folder / "queries.csv"
            if refs.exists():
                refs_found.append((f"UAV-VisLoc {folder.name} tiles", refs))
            if queries.exists():
                queries_found.append((f"UAV-VisLoc {folder.name} drone frames", queries))

        first = self.refs_combo.count() == 0
        self._fill_dataset_combo(self.refs_combo, refs_found, "references")
        self._fill_dataset_combo(self.queries_combo, queries_found, "queries")
        if first:
            # open on the report's main region, where every recipe step has its inputs
            for combo, found in ((self.refs_combo, refs_found), (self.queries_combo, queries_found)):
                for i, (_label, path) in enumerate(found):
                    if path.parent.name == "r05_t250":
                        combo.blockSignals(True)
                        combo.setCurrentIndex(i)
                        combo.blockSignals(False)
                        break
        self._update_dataset_info()

    def _fill_dataset_combo(
        self, combo: QComboBox, found: list[tuple[str, Path]], kind: str
    ) -> None:
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, path in found:
            combo.addItem(label, str(path))
            combo.setItemData(combo.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        if not found:
            combo.addItem(f"No {kind} CSV found — run scripts/visloc_prepare.py", None)
        if previous:
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _browse_csv(self, combo: QComboBox, kind: str) -> None:
        start = str(self.project_root / "data")
        selected, _ = QFileDialog.getOpenFileName(
            self, f"Choose a {kind} CSV", start, "CSV files (*.csv);;All files (*)"
        )
        if not selected:
            return
        index = combo.findData(selected)
        if index < 0:
            combo.addItem(f"{Path(selected).parent.name}/{Path(selected).name}", selected)
            combo.setItemData(combo.count() - 1, selected, Qt.ItemDataRole.ToolTipRole)
            index = combo.count() - 1
        combo.setCurrentIndex(index)

    def _refs_changed(self) -> None:
        """The query set is chosen independently; use 'Pair with reference folder' to link them."""
        self._update_dataset_info()

    def _queries_changed(self) -> None:
        self._update_dataset_info()

    def _pair_queries_with_refs(self) -> None:
        """Opt-in convenience: select the query CSV sitting beside the reference database."""
        refs = self.refs_combo.currentData()
        if not refs:
            return
        sibling = Path(refs).parent
        for index in range(self.queries_combo.count()):
            data = self.queries_combo.itemData(index)
            if data and Path(data).parent == sibling:
                self.queries_combo.setCurrentIndex(index)
                return
        self.append_log(f"No query CSV found next to {sibling}")

    def _active_encoder_changed(self) -> None:
        self._recipe_changed()

    def _update_dataset_info(self) -> None:
        for combo, label in ((self.refs_combo, self.refs_info), (self.queries_combo, self.queries_info)):
            data = combo.currentData()
            if not data:
                label.setText("-")
                continue
            path = Path(data)
            combo.setToolTip(str(path))
            try:
                rows = sum(1 for _ in path.open()) - 1
            except OSError:
                label.setText(f"unreadable: {path}")
                continue
            noun = "tiles" if combo is self.refs_combo else "frames"
            label.setText(f"{rows} {noun} · {path.parent.name}/{path.name}")
        if hasattr(self, "card_agl"):
            self._update_availability()
            self._recipe_changed()
        self._single_image_changed()
        self._refresh_pipeline()
        self._refresh_findings()

    def _current_paths(self) -> tuple[Path | None, Path | None]:
        refs = self.refs_combo.currentData()
        queries = self.queries_combo.currentData()
        return (Path(refs) if refs else None, Path(queries) if queries else None)

    def _db_tag(self, refs: Path, queries: Path) -> str:
        if refs.parent == queries.parent:
            return refs.parent.name
        return f"{refs.parent.name}~{queries.parent.name}"

    # ------------------------------------------------------------ running --
    def _busy(self) -> bool:
        if (
            self.process is not None
            or self.single_process is not None
            or self.traj_process is not None
        ):
            self.append_log("A run is still active — press Stop first.")
            return True
        return False

    def run_benchmark(self) -> None:
        if self._busy():
            return
        recipes = self._recipes()
        if not recipes:
            self.append_log("Tick at least one encoder.")
            return
        self._queue_runs(recipes, "")

    def _queue_runs(self, recipes: list[Recipe], prefix: str) -> None:
        refs, queries = self._current_paths()
        if refs is None or queries is None:
            self.append_log("Select both a map and a flight.")
            return
        self.queue = []
        for recipe in recipes:
            tag = self._tag_for(refs, queries, recipe)
            try:
                args = self._eval_args(refs, queries, recipe, tag)
            except ValueError as error:
                self.append_log(f"{prefix} skipped {tag}: {error}".strip())
                continue
            self.queue.append((tag, args))
        if not self.queue:
            return
        self._pending_inspect_tag = None
        self._queue_total = len(self.queue)
        self._run_index = 0
        self._run_walltimes = []
        self.run_button.setEnabled(False)
        self.inspector_run_button.setEnabled(False)
        self.ladder_run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.append_log(
            f"{prefix} Queued {len(self.queue)} run(s): {refs.parent.name} map, "
            f"{queries.parent.name} flight.".strip()
        )
        self._start_next()

    def _tag_for(self, refs: Path, queries: Path, recipe: Recipe) -> str:
        return f"{self._db_tag(refs, queries)}__{recipe.model_spec}{recipe.tag_suffix()}"

    def _eval_args(self, refs: Path, queries: Path, recipe: Recipe, tag: str) -> list[str]:
        """scripts/visloc_eval.py arguments for one recipe. Caches are per encoder, so an
        ensemble reuses every member's descriptors (avl.ensemble.member_path)."""
        k = float(self.agl_k.value()) if recipe.agl_scale else None
        spec = recipe.model_spec
        return [
            str(self.project_root / "scripts" / "visloc_eval.py"),
            "--refs", str(refs),
            "--queries", str(queries),
            *recipe.eval_args(camera_k=k),
            "--max-queries", str(self.max_queries.value()),
            "--query-stride", str(self.stride.value()),
            "--tag", tag,
            "--out-dir", str(self.out_dir),
            "--ref-cache", str(self._ref_cache_path(refs, spec)),
            "--query-cache", str(self._query_cache_path(queries, recipe, k)),
            # AnyLoc's vocabulary is fitted per map and shared by every run on it
            *(
                ["--vocab", str(self.out_dir / f"vocab_{self._db_tag(refs, queries)}__{spec}.npz")]
                if any(m.startswith("anyloc") for m in recipe.encoders)
                else []
            ),
        ]

    def _query_cache_path(self, queries: Path, recipe: Recipe, camera_k: float | None) -> Path:
        """Raw query descriptors, keyed by everything that changes the encoded views
        (rotations, crop, heading, altitude crop) so centering, window and fusion
        variants re-score instead of re-encoding."""
        key = (
            f"{queries.parent.name}-{queries.stem}__{recipe.model_spec}"
            f"__r{recipe.effective_rotations}_{recipe.query_crop}"
            f"_s{self.stride.value()}_m{self.max_queries.value()}"
        )
        if recipe.heading != "off":
            key += f"_h{recipe.heading}"
        if recipe.agl_scale and camera_k is not None:
            key += f"_agl{camera_k:g}x{recipe.agl_gate or 0:g}"
        return self.out_dir / "qcache" / f"{key}.npy"

    def run_inspector(self) -> None:
        if self._busy():
            return
        refs, queries = self._current_paths()
        if refs is None or queries is None:
            self.append_log("Select both a map and a flight.")
            return
        recipe = self._primary_recipe()
        tag = self._tag_for(refs, queries, recipe)
        per_query = self.out_dir / f"{tag}_per_query.csv"
        summary = self.out_dir / f"{tag}_summary.json"

        if per_query.exists() and summary.exists() and not self.inspector_force.isChecked():
            self.append_log(f"[inspector] loading cached {tag}")
            self.reload_results()
            index = self.run_combo.findText(tag)
            if index >= 0:
                self.run_combo.setCurrentIndex(index)
            self._show_set_mode()
            return

        self._queue_runs([recipe], "[inspector]")
        self._pending_inspect_tag = tag if self.queue or self.process else None

    def _ref_cache_path(self, refs: Path, model: str) -> Path:
        """Per-map reference descriptors; an ensemble spec maps to one file per member."""
        return self.out_dir / f"cache_{refs.parent.name}__{model}.npy"

    def _start_next(self) -> None:
        if not self.queue:
            self.set_status("READY")
            self.run_button.setEnabled(True)
            self.inspector_run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reload_results()
            if self._pending_inspect_tag:
                index = self.run_combo.findText(self._pending_inspect_tag)
                if index >= 0:
                    self.run_combo.setCurrentIndex(index)
                    self._show_set_mode()
                self._pending_inspect_tag = None
            self._refresh_queue_line()
            self._progress_done()
            self._pipeline_result_from_summary(self.current_tag)
            self._set_pipe_phase("done")
            self.append_log("All runs finished.")
            return

        tag, args = self.queue.pop(0)
        self.current_tag = tag
        self._run_index += 1
        self._run_t0 = time.monotonic()
        self._phase = None
        self.set_status(f"RUNNING {tag}", busy=True)
        self._refresh_queue_line()
        self._progress_start(self._run_prefix() + " preparing…")
        self.process = self._make_process()
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._run_finished)
        self._spawn(self.process, args)

    def _make_process(self) -> QProcess:
        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.project_root))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        return proc

    def _spawn(self, proc: QProcess, args: list[str]) -> None:
        proc.start(sys.executable, args)

    def _terminate_process(self, proc: QProcess | None) -> None:
        """Stop ``proc`` now, synchronously, and stop listening to it.

        Signals are disconnected first so the kill does not re-enter the run
        state machine, then SIGTERM, then SIGKILL, each with a bounded wait.
        """
        if proc is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for signal_name in ("readyReadStandardOutput", "finished"):
                try:
                    getattr(proc, signal_name).disconnect()
                except (RuntimeError, TypeError):
                    pass
        if proc.state() == QProcess.ProcessState.NotRunning:
            proc.deleteLater()
            return

        pid = int(proc.processId() or 0)
        proc.terminate()
        if not proc.waitForFinished(1500):
            proc.kill()
            if not proc.waitForFinished(1500) and pid:
                # last resort: the OS-level signal, in case the Qt handle is wedged
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                proc.waitForFinished(1000)
        proc.deleteLater()

    def _run_prefix(self) -> str:
        span = f"{self._run_index}/{self._queue_total}" if self._queue_total > 1 else ""
        return f"run {span} · {self.current_tag} ·".replace("run  ·", "run ·").strip()

    _TQDM_RE = re.compile(r"Encoding images:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
    _QUERY_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+queries")

    def _read_output(self) -> None:
        if self.process is None or self._stopping:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        for chunk in text.replace("\r", "\n").splitlines():
            stripped = chunk.strip()
            if not stripped:
                continue

            tqdm_m = self._TQDM_RE.search(stripped)
            if tqdm_m:
                done, whole = int(tqdm_m.group(1)), int(tqdm_m.group(2))
                self._progress_phase("encode-ref", done, whole, "encoding reference tiles")
                self._set_pipe_phase("encode")
                continue  # do not spill tqdm redraws into the log

            if "it/s]" in stripped or "batch/s]" in stripped:
                continue

            self.append_log(chunk.rstrip())

            step = self._QUERY_RE.search(stripped)
            if step:
                done, whole = int(step.group(1)), int(step.group(2))
                self._progress_phase("queries", done, whole, "queries")
                self._set_pipe_phase("search")
            elif re.search(r"queries=\d+", stripped):
                self._progress_update(label=f"{self._run_prefix()} encoding reference tiles…")
            elif "encoded" in stripped and "refs" in stripped:
                self._progress_update(label=f"{self._run_prefix()} encoding queries…")
                self._set_pipe_phase("search")
            elif "_summary.json" in stripped:
                self._set_pipe_phase("fuse")

    def _run_finished(self, code: int, _status: Any) -> None:
        if self._stopping:
            return
        if self._run_t0 is not None and code == 0:
            self._run_walltimes.append(time.monotonic() - self._run_t0)
        self.append_log(f"[{self.current_tag}] exited with code {code}")
        self.process = None
        self.reload_results()
        self._start_next()

    def stop_benchmark(self) -> None:
        self._stopping = True
        self.queue = []
        self._pending_inspect_tag = None
        self.stop_button.setEnabled(False)
        self.set_status("STOPPING…", busy=True)
        self.append_log("Stopping — killing the running process…")
        self._terminate_process(self.process)
        self.process = None
        self._terminate_process(self.single_process)
        self.single_process = None
        self._terminate_process(self.traj_process)
        self.traj_process = None
        if hasattr(self, "traj_run_button"):
            self.traj_run_button.setEnabled(True)
        if hasattr(self, "studio_run_button"):
            self.studio_run_button.setEnabled(True)
        self._stopping = False
        self.current_tag = None
        self.set_status("STOPPED")
        self.append_log("Stopped.")
        self.run_button.setEnabled(True)
        self.inspector_run_button.setEnabled(True)
        self.single_button.setEnabled(True)
        self._refresh_queue_line()
        self._progress_done()
        self._refresh_findings()

    def closeEvent(self, event: Any) -> None:
        self._stopping = True
        self._terminate_process(self.process)
        self.process = None
        self._terminate_process(self.single_process)
        self.single_process = None
        self._terminate_process(self.traj_process)
        self.traj_process = None
        super().closeEvent(event)

    # ------------------------------------------------------- single image --
    def _browse_single_image(self) -> None:
        _, queries = self._current_paths()
        start = str(queries.parent if queries else self.project_root / "data")
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose a query image", start,
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.JPG *.JPEG *.PNG);;All files (*)",
        )
        if selected:
            self.single_image_edit.setText(selected)

    def _single_image_changed(self) -> None:
        if not hasattr(self, "single_hint"):
            return
        refs, _ = self._current_paths()
        recipe = self._primary_recipe() if hasattr(self, "card_fusion") else Recipe()
        chips = [c for c in recipe.chips() if not c.startswith("ensemble")]
        self.single_encoder_label.setText(
            " + ".join(recipe.encoders) + ("  ·  " + "  ·  ".join(chips) if chips else "")
        )
        image = self.single_image_edit.text().strip()
        if not image:
            self.single_hint.setText(
                f"Searches {refs.parent.name if refs else 'no'} map. "
                "Pick an image to localize."
            )
            return
        row = self._query_row_for(Path(image))
        if row is not None:
            if row.get("yaw_deg") is not None:
                self.single_heading.setText(f"{row['yaw_deg']:.1f}")
            if row.get("height_m") is not None:
                self.single_altitude.setText(f"{row['height_m']:.1f}")
            known = f"ground truth {row['latitude']:.6f}, {row['longitude']:.6f} from the flight"
        else:
            known = "not in the selected flight — coordinates will be reported without error"
        self.single_hint.setText(
            f"Searches {refs.parent.name if refs else 'no'} map · {known}"
        )

    def _ground_truth_for(self, image: Path) -> tuple[float, float] | None:
        row = self._query_row_for(image)
        return (row["latitude"], row["longitude"]) if row is not None else None

    def _query_row_for(self, image: Path) -> dict | None:
        """The image's row in the selected flight CSV: position, heading, altitude."""
        _, queries = self._current_paths()
        if queries is None or not queries.exists():
            return None
        try:
            frame = pd.read_csv(queries)
        except (OSError, pd.errors.ParserError):
            return None
        if not {"image_path", "latitude", "longitude"} <= set(frame.columns):
            return None
        target = image.expanduser().resolve()
        for _, row in frame.iterrows():
            candidate = resolve_path(row["image_path"], queries.parent)
            if candidate is None:
                continue
            try:
                if candidate.expanduser().resolve() == target:
                    def opt(name: str) -> float | None:
                        value = row.get(name)
                        return float(value) if value is not None and pd.notna(value) else None

                    return {
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "yaw_deg": opt("yaw_deg"),
                        "height_m": opt("height_m"),
                    }
            except OSError:
                continue
        return None

    def run_single_query(self) -> None:
        if self._busy():
            return
        refs, _ = self._current_paths()
        image = Path(self.single_image_edit.text().strip()).expanduser()
        if refs is None:
            self.append_log("Select a map first.")
            return
        if not image.is_file():
            self.append_log(f"Not an image file: {image}")
            self.single_hint.setText(f"Not an image file: {image}")
            return

        def number(edit: QLineEdit) -> float | None:
            try:
                return float(edit.text().strip())
            except ValueError:
                return None

        recipe = self._primary_recipe()
        spec = recipe.model_spec
        heading = number(self.single_heading)
        altitude = number(self.single_altitude)
        rotations = recipe.effective_rotations
        if recipe.heading != "off" and heading is None:
            rotations = 4  # the recipe wants a heading this frame does not have
        self.single_result_path = Path(tempfile.gettempdir()) / "avl_single_result.json"
        args = [
            str(self.project_root / "scripts" / "visloc_query.py"),
            "--refs", str(refs),
            "--image", str(image),
            "--model", spec,
            "--rotations", str(rotations),
            "--query-crop", recipe.query_crop,
            "--fusion", recipe.fusion,
            "--center", recipe.center,
            "--ref-cache", str(self._ref_cache_path(refs, spec)),
            "--vocab", str(self.out_dir / f"vocab_{refs.parent.name}__{spec}.npz"),
            "--out", str(self.single_result_path),
        ]
        if recipe.heading != "off" and heading is not None:
            args += ["--heading-deg", f"{heading:g}", "--heading-mode", recipe.heading]
        if recipe.agl_scale and altitude is not None:
            args += [
                "--altitude-asl", f"{altitude:g}",
                "--camera-k", f"{self.agl_k.value():g}",
                "--agl-gate", f"{recipe.agl_gate:g}",
            ]
        ground_truth = self._ground_truth_for(image)
        if ground_truth:
            args += ["--gt-lat", f"{ground_truth[0]:.8f}", "--gt-lon", f"{ground_truth[1]:.8f}"]

        self.single_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_status(f"LOCALIZING {image.name}", busy=True)
        self._progress_start(f"localizing {image.name} · loading {spec}…")
        self.append_log(f"[single] {image.name} vs {refs.parent.name} with {spec}")
        self._show_image_mode()
        self.retrieval_panel.clear()
        self.retrieval_panel.query_preview.set_path(image)

        self.single_process = self._make_process()
        self.single_process.readyReadStandardOutput.connect(self._read_single_output)
        self.single_process.finished.connect(self._single_finished)
        self._spawn(self.single_process, args)

    def _read_single_output(self) -> None:
        if self.single_process is None or self._stopping:
            return
        text = bytes(self.single_process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or "it/s]" in line or stripped.startswith("RESULT_JSON"):
                continue
            self.append_log(line.rstrip())
            if stripped.startswith("[query]"):
                self._progress_update(label=stripped[: 90])
            if "reference descriptors" in stripped:
                self._set_pipe_phase("search")
            elif stripped.startswith("[query] fused"):
                self._set_pipe_phase("fuse")

    def _single_finished(self, code: int, _status: Any) -> None:
        if self._stopping:
            return
        self.single_process = None
        self.single_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_status("READY")
        self._progress_done()
        if code != 0:
            self.append_log(f"[single] failed with code {code}")
            self.single_hint.setText("Localization failed — open the LOG drawer at the bottom.")
            self.log_drawer.open()
            return
        try:
            payload = json.loads(self.single_result_path.read_text())
        except (OSError, json.JSONDecodeError, AttributeError):
            self.append_log("[single] could not read the result JSON")
            return
        self._render_single(payload)

    def _render_single(self, payload: dict) -> None:
        matches = [
            Match(
                image_id=str(entry.get("image_id", "")),
                latitude=entry.get("latitude"),
                longitude=entry.get("longitude"),
                score=entry.get("score"),
                error_m=entry.get("error_m"),
                image_path=resolve_path(entry.get("image_path"), self.project_root),
            )
            for entry in payload.get("matches", [])
        ]
        fused = payload.get("fused") or {}
        ground_truth = payload.get("ground_truth")

        parts = []
        if matches and matches[0].error_m is not None:
            parts.append(f"top-1 {matches[0].error_m:.0f} m")
        if payload.get("fused_error_m") is not None:
            parts.append(f"fused {payload['fused_error_m']:.0f} m")
        errors = [m.error_m for m in matches if m.error_m is not None]
        if errors:
            parts.append(f"best of top-5 {min(errors):.0f} m")
        parts.append(f"confidence {payload.get('confidence', float('nan')):.3f}")
        parts.append(f"spread {payload.get('spread_m', float('nan')):.0f} m")

        plan = payload.get("plan") or {}
        center = plan.get("center")
        self.retrieval_panel.set_result(
            resolve_path(payload.get("query_image"), self.project_root),
            matches,
            ground_truth=(
                (ground_truth["latitude"], ground_truth["longitude"]) if ground_truth else None
            ),
            fused=(fused.get("latitude"), fused.get("longitude")) if fused else None,
            error_parts=parts,
            plan_text=plan_text(
                plan.get("rotate_deg"), plan.get("crop_fraction"), plan.get("agl_m"),
                f"centred on the {center} mean" if center and center != "off" else "",
            ),
        )
        timing = (payload.get("timings") or {}).get("query_encode_ms")
        self.single_hint.setText(
            f"{payload.get('model')} · {payload.get('n_refs')} tiles · "
            f"{payload.get('rotations')} rotations · query encoded in "
            f"{timing:.0f} ms" if timing else f"{payload.get('model')} · {payload.get('n_refs')} tiles"
        )

        if fused.get("latitude") is not None:
            err = payload.get("fused_error_m")
            self._set_node(
                "result",
                f"{fused['latitude']:.6f}, {fused['longitude']:.6f}",
                f"fused estimate" + (f"  ·  {err:.0f} m from truth" if err is not None else ""),
            )
        self._set_pipe_phase("done")

    def _set_run_details(self, summary: dict, summary_path: Path) -> None:
        if not hasattr(self, "run_details"):
            return
        recall = summary.get("recall_top1_within_m", {}).get("100")
        fused = (summary.get("fused_within_m") or {}).get("100")
        ms = summary.get("query_ms_per_image")
        result = "result —"
        if recall is not None:
            result = f"result:  top-1 <100 m {100 * recall:.1f}%"
            if fused is not None:
                result += f"  ·  fused {100 * fused:.1f}%"
        if ms is not None and not summary.get("query_cache_hit"):
            result += f"  ·  {ms:.0f} ms / frame"
        try:
            recipe = Recipe.from_summary(summary)
            steps = "  ·  ".join(c for c in recipe.chips() if not c.startswith("ensemble")) or "no steps"
            encoders = " + ".join(recipe.encoders)
        except ValueError:
            steps, encoders = "?", summary.get("model", "?")
        self.run_details.setText(
            f"{encoders}  ·  map {Path(summary.get('refs_csv', '')).parent.name}  ·  "
            f"{summary.get('n_refs', '?')} tiles / {summary.get('n_queries', '?')} frames\n"
            f"steps: {steps}\n"
            f"{self._run_jalali_text(summary, summary_path)}  ·  stride {summary.get('query_stride', 1)}\n"
            f"{result}"
        )

    @staticmethod
    def _run_when(data: dict, path: Path) -> datetime | None:
        """When the run finished — from the summary's finished_at, else the summary
        file's modification time for runs written before that field existed."""
        stamp = data.get("finished_at") or data.get("started_at")
        if stamp:
            try:
                return datetime.fromisoformat(str(stamp))
            except ValueError:
                pass
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None

    @classmethod
    def _run_jalali_text(cls, data: dict, path: Path) -> str:
        when = cls._run_when(data, path)
        return jalali_datetime_text(when) if when is not None else "-"

    # ------------------------------------------------------------ results --
    def reload_results(self) -> None:
        def mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        summaries = sorted(self.out_dir.glob("*_summary.json"), key=mtime, reverse=True)
        header = self.table.horizontalHeader()
        sort_col, sort_ord = header.sortIndicatorSection(), header.sortIndicatorOrder()
        self.table.setSortingEnabled(False)  # keep insertion order while (re)filling
        self.table.setRowCount(0)
        current_run = self.run_combo.currentText()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self._summaries = []
        maps: set[str] = set()

        for path in summaries:
            try:
                data = json.loads(path.read_text())
                data["top1_error_m"]["median"], data["recall_top1_within_m"]["100"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            self._summaries.append((path, data))
            try:
                recipe = Recipe.from_summary(data)
                chips = recipe.chips()
                encoders = recipe.encoders
            except ValueError:
                recipe, chips, encoders = None, [], (data.get("model", "?"),)
            map_name = Path(data.get("refs_csv", "")).parent.name or "-"
            flight = Path(data.get("queries_csv", "")).parent.name or "-"
            maps.add(map_name)
            chance = (data.get("chance_baseline") or {}).get("random_tile_within_m", {})
            fused_within = (data.get("fused_within_m") or {}).get("100")
            cached = bool(data.get("query_cache_hit"))
            ms = float("nan") if cached else data.get("query_ms_per_image", float("nan"))
            gm = gmac_estimate(encoders, GMAC)
            variants = int(data.get("rotations", 4)) * max(1, len(data.get("query_scales") or [1.0]))
            if (data.get("scale_from_agl") or {}).get("enabled"):
                variants = int(data.get("rotations", 4))
            gm_frame = gm * variants if gm is not None else None
            top1 = data["recall_top1_within_m"]["100"]
            when = self._run_when(data, path)

            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                " + ".join(encoders),
                "",
                map_name if map_name == flight else f"{map_name} · {flight}",
                str(data["n_queries"]),
                f"{100 * top1:.1f}%",
                f"{100 * fused_within:.1f}%" if fused_within is not None else "—",
                f"{100 * data['recall_topk_within_m']['100']:.1f}%",
                f"{data['top1_error_m']['median']:.0f} m",
                f"{100 * chance.get('100', float('nan')):.1f}%",
                f"{gm_frame:.0f}" if gm_frame is not None else "—",
                "cached" if cached else f"{ms:.0f}",
                self._run_jalali_text(data, path),
            ]
            sort_keys: list[float | None] = [
                None,
                float(len(chips)),
                None,
                data["n_queries"],
                top1,
                fused_within if fused_within is not None else float("nan"),
                data["recall_topk_within_m"]["100"],
                data["top1_error_m"]["median"],
                chance.get("100", float("nan")),
                gm_frame if gm_frame is not None else float("nan"),
                ms,
                when.timestamp() if when else 0.0,
            ]
            detail_tip = (
                f"{data['tag']}\n"
                f"encoders: {' + '.join(encoders)}\n"
                f"steps:    {', '.join(chips) or 'none'}\n"
                f"map:      {data.get('refs_csv', '')}\n"
                f"flight:   {data.get('queries_csv', '')}\n"
                f"frames {data['n_queries']} · stride {data.get('query_stride', 1)} · "
                f"{data.get('rotations', '?')} rotation(s) · fusion {data.get('fusion', '?')}"
                + ("\nframes re-scored from cache — ms/frame is not a latency" if cached else "")
            )
            for column, value in enumerate(values):
                key = sort_keys[column]
                item = QTableWidgetItem(value) if key is None else SortableItem(value, key)
                item.setToolTip(detail_tip if column < 3 else value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, data["tag"])
                    item.setData(Qt.ItemDataRole.UserRole + 1, map_name)
                    item.setData(
                        Qt.ItemDataRole.UserRole + 2,
                        " ".join([data["tag"], *encoders, *chips, map_name, flight]).lower(),
                    )
                    if recipe is not None and recipe.is_ensemble:
                        item.setForeground(QColor(ACCENT_300))
                elif column == COL["steps"]:
                    item.setData(CHIPS_ROLE, chips)
                elif column == COL["within100"]:
                    item.setData(FRACTION_ROLE, top1)
                elif column == COL["fused100"] and fused_within is not None:
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, column, item)
            self.run_combo.addItem(data["tag"], str(path).replace("_summary.json", "_per_query.csv"))
            self.run_combo.setItemData(
                self.run_combo.count() - 1, data["tag"], Qt.ItemDataRole.ToolTipRole
            )

        self.table.setSortingEnabled(True)
        if self.table.rowCount():
            self.table.sortItems(sort_col if sort_col >= 0 else COL["date"], sort_ord)

        wanted = self.filter_map.currentData()
        self.filter_map.blockSignals(True)
        self.filter_map.clear()
        self.filter_map.addItem("All maps", None)
        for name in sorted(maps):
            self.filter_map.addItem(name, name)
        self.filter_map.setCurrentIndex(max(0, self.filter_map.findData(wanted)) if wanted else 0)
        self.filter_map.blockSignals(False)
        self._apply_comparison_filter()

        index = self.run_combo.findText(current_run)
        self.run_combo.setCurrentIndex(max(0, index))
        self.run_combo.blockSignals(False)
        self._load_run_queries()
        self._refresh_queue_line()
        self._refresh_findings()
        self._refresh_overview()

    def _load_run_queries(self) -> None:
        path = self.run_combo.currentData()
        self.query_table.setSortingEnabled(False)
        self.query_table.setRowCount(0)
        self._per_query = None
        self._tile_dir = None
        self._queries_dir = None
        self._ref_lookup = None
        self._queries_lookup = None
        self._queries_frame = None
        self._query_stride = 1
        self.retrieval_panel.clear()
        if hasattr(self, "run_details"):
            self.run_details.setText("no run selected")
        if not path or not Path(path).exists():
            return
        frame = pd.read_csv(path)
        self._per_query = frame

        summary_path = Path(str(path).replace("_per_query.csv", "_summary.json"))
        try:
            summary = json.loads(summary_path.read_text())
            self._set_run_details(summary, summary_path)
            refs_csv = resolve_path(summary["refs_csv"], self.project_root)
            queries_csv = resolve_path(summary["queries_csv"], self.project_root)
            self._tile_dir = refs_csv.parent
            self._queries_dir = queries_csv.parent
            # ids arrive from the per-query CSV as strings; index both sides as
            # strings so an integer image_id column still joins.
            self._ref_lookup = self._index_by_image_id(pd.read_csv(refs_csv))
            queries = pd.read_csv(queries_csv)
            self._queries_frame = queries
            self._query_stride = max(1, int(summary.get("query_stride", 1)))
            # Some query CSVs (DenseUAV) carry no image_id, in which case
            # visloc_eval falls back to the row position as the query id.
            if "image_id" in queries.columns:
                self._queries_lookup = self._index_by_image_id(queries)
        except (OSError, json.JSONDecodeError, KeyError, AttributeError):
            pass

        for row, record in frame.iterrows():
            self.query_table.insertRow(row)
            cells = [
                (str(record["query_id"]), None),
                (f"{record['top1_error_m']:.0f} m", record["top1_error_m"]),
                (f"{record['best_topk_error_m']:.0f} m", record["best_topk_error_m"]),
                (f"{record['top1_score']:.3f}", record["top1_score"]),
            ]
            for column, (value, key) in enumerate(cells):
                item = QTableWidgetItem(value) if key is None else SortableItem(value, key)
                item.setToolTip(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, int(row))  # survives re-sorting
                if column == 1:
                    item.setForeground(
                        QColor(GOOD) if record["top1_error_m"] <= 100 else QColor(BAD)
                    )
                self.query_table.setItem(row, column, item)
        self.query_table.setSortingEnabled(True)
        if self.query_table.rowCount():
            self.query_table.selectRow(0)

    def _show_query(self) -> None:
        rows = self.query_table.selectionModel().selectedRows()
        if not rows or self._per_query is None:
            return
        id_item = self.query_table.item(rows[0].row(), 0)
        df_row = id_item.data(Qt.ItemDataRole.UserRole) if id_item is not None else None
        df_row = int(df_row) if df_row is not None else rows[0].row()
        record = self._per_query.iloc[df_row]

        ids = [value for value in str(record.get("topk_ids", "")).split("|") if value]
        if not ids and id_text(record.get("top1_id")):
            ids = [id_text(record.get("top1_id"))]  # runs predating the topk_ids column
        errors = split_floats(record.get("topk_errors_m"))
        scores = split_floats(record.get("topk_scores"))

        matches: list[Match] = []
        for rank, image_id in enumerate(ids):
            latitude, longitude, tile_path = self._ref_entry(image_id)
            matches.append(
                Match(
                    image_id=image_id,
                    latitude=latitude,
                    longitude=longitude,
                    score=scores[rank] if rank < len(scores) else None,
                    error_m=errors[rank] if rank < len(errors) else None,
                    image_path=tile_path,
                )
            )

        parts = []
        for label, key in [
            ("top-1", "top1_error_m"),
            ("fused", "fused_error_m"),
            ("best of top-5", "best_topk_error_m"),
        ]:
            value = record.get(key)
            if value is not None and pd.notna(value):
                parts.append(f"{label} {float(value):.0f} m")

        self.retrieval_panel.set_result(
            self._query_image_path(str(record["query_id"]), df_row, record),
            matches,
            ground_truth=(record.get("gt_lat"), record.get("gt_lon")),
            fused=(record.get("fused_lat"), record.get("fused_lon")),
            error_parts=parts,
            plan_text=plan_text(record.get("rotate_deg"), record.get("crop_fraction"), record.get("agl_m")),
        )

    # ------------------------------------------------------------ lookups --
    @staticmethod
    def _index_by_image_id(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["image_id"] = frame["image_id"].astype(str)
        return frame.drop_duplicates(subset="image_id").set_index("image_id")

    def _lookup(self, table: pd.DataFrame | None, image_id: str) -> pd.Series | None:
        if table is None or not image_id or image_id not in table.index:
            return None
        row = table.loc[image_id]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    def _ref_entry(self, image_id: str) -> tuple[float | None, float | None, Path | None]:
        """Latitude, longitude and on-disk path of a reference tile."""
        row = self._lookup(self._ref_lookup, image_id)
        if row is None:
            return None, None, None
        latitude = float(row["latitude"]) if pd.notna(row.get("latitude")) else None
        longitude = float(row["longitude"]) if pd.notna(row.get("longitude")) else None
        return latitude, longitude, resolve_path(row.get("image_path"), self._tile_dir)

    def _query_image_path(self, query_id: str, position: int, record: pd.Series) -> Path | None:
        """Locate the query frame, by image_id where the CSV has one and by row
        position (the id visloc_eval falls back to) where it does not."""
        row = self._lookup(self._queries_lookup, query_id)
        if row is None:
            row = self._query_row_by_position(position, record)
        if row is None:
            return None
        return resolve_path(row.get("image_path"), self._queries_dir)

    def _query_row_by_position(self, position: int, record: pd.Series) -> pd.Series | None:
        frame = self._queries_frame
        if frame is None or frame.empty:
            return None
        latitude, longitude = record.get("gt_lat"), record.get("gt_lon")

        index = position * self._query_stride
        if 0 <= index < len(frame):
            candidate = frame.iloc[index]
            if self._same_place(candidate, latitude, longitude):
                return candidate

        # Stride is unknown for runs written before it was recorded: fall back to
        # the first query sitting on the same ground-truth coordinate.
        if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
            return None
        matches = frame[
            (frame["latitude"] - float(latitude)).abs().le(1e-6)
            & (frame["longitude"] - float(longitude)).abs().le(1e-6)
        ]
        return None if matches.empty else matches.iloc[0]

    @staticmethod
    def _same_place(row: pd.Series, latitude: Any, longitude: Any) -> bool:
        if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
            return True  # nothing to check against; trust the position
        return (
            abs(float(row["latitude"]) - float(latitude)) <= 1e-6
            and abs(float(row["longitude"]) - float(longitude)) <= 1e-6
        )

    # ------------------------------------------------------------ helpers --
    def append_log(self, message: str) -> None:
        self.log_drawer.append(message)

    def _refresh_queue_line(self) -> None:
        if not hasattr(self, "queue_line"):
            return
        pending = len(self.queue) + (1 if self.process is not None else 0)
        if pending:
            self.queue_line.setText(f"{pending} run(s) queued")
        else:
            self.queue_line.setText(
                f"{self.table.rowCount()} saved result(s) in artifacts/visloc/"
            )

    def set_status(self, text: str, busy: bool = False) -> None:
        upper = text.upper()
        self.status.setText(upper if len(upper) < 60 else upper[:57] + "…")
        if busy:
            fg, bg = WARN, "rgba(242,193,78,0.12)"
        elif upper.startswith("STOPPED"):
            fg, bg = ACCENT_300, "rgba(139,127,240,0.14)"
        else:
            fg, bg = GOOD, "rgba(95,211,154,0.10)"
        self.status_pill.setStyleSheet(
            f"#statusPill {{ border: 0; border-radius: 9px; background: {bg}; }}"
        )
        self.status_dot.setStyleSheet(f"#statusDot {{ background: {fg}; border-radius: 3px; }}")
        self.status.setStyleSheet(
            f"#statusText {{ color: {fg}; font-size: 11px; font-weight: 600; letter-spacing: 0.8px; }}"
        )



def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AVL Console")
    app.setFont(ui_font(13))
    window = ModelBenchConsole()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
