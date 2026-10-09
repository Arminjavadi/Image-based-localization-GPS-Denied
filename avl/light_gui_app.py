from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

from avl.rerank import DEFAULT_BACKEND, OPTIONAL_BACKEND_HINT, RERANK_BACKENDS


def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "scripts" / "benchmark_kpis.py").exists():
            return candidate
    return Path.cwd()


def fmt(value: Any, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


class MetricBox(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("metricBox")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self.title = QLabel(title.upper())
        self.title.setObjectName("metricTitle")
        self.value = QLabel("-")
        self.value.setObjectName("metricValue")
        self.detail = QLabel("")
        self.detail.setObjectName("metricDetail")
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def set_value(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class ImagePreviewLabel(QLabel):
    def __init__(self, empty_text: str, width: int, height: int) -> None:
        super().__init__(empty_text)
        self._empty_text = empty_text
        self._source = QPixmap()
        self.setObjectName("imagePreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(width, height)
        self.setMaximumHeight(height)

    def set_path(self, path_value: str | Path | None) -> None:
        self._source = QPixmap()
        path = Path(path_value).expanduser() if path_value else None
        if path is None or not path.is_file():
            self.setText(self._empty_text)
            self.setToolTip("")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            pixmap = self._load_with_pillow(path)
        if pixmap.isNull():
            self.setText(f"Preview unavailable\n{path.name}")
            self.setToolTip(str(path))
            return

        self._source = pixmap
        self.setToolTip(str(path))
        self._render()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._source.isNull():
            return
        target = self.contentsRect().size()
        self.setPixmap(
            self._source.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _load_with_pillow(self, path: Path) -> QPixmap:
        if Image is None or ImageOps is None:
            return QPixmap()
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                data = image.tobytes("raw", "RGBA")
                qimage = QImage(
                    data,
                    image.width,
                    image.height,
                    image.width * 4,
                    QImage.Format.Format_RGBA8888,
                ).copy()
                return QPixmap.fromImage(qimage)
        except Exception:
            return QPixmap()


class MatchCard(QWidget):
    def __init__(self, match: dict[str, Any]) -> None:
        super().__init__()
        self.setObjectName("matchCard")
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        rank = match.get("rank", "-")
        score = match.get("score")
        heading = QLabel(f"#{rank}   similarity {fmt(score, '', 4)}")
        heading.setObjectName("matchRank")

        preview = ImagePreviewLabel("Image unavailable", 198, 142)
        preview.set_path(match.get("image_path"))

        # Only present when the geometric stage ran on this query.
        rerank = match.get("rerank") or {}
        verification: QLabel | None = None
        if rerank:
            inliers = rerank.get("inliers", 0)
            matches_found = rerank.get("matches", 0)
            verified = bool(rerank.get("verified"))
            previous = match.get("retrieval_rank")
            movement = ""
            if isinstance(previous, int) and isinstance(rank, int) and previous != rank:
                movement = f" · {'↑' if previous > rank else '↓'} was #{previous}"
            verification = QLabel(
                f"{'✓' if verified else '✗'} {inliers}/{matches_found} inliers{movement}"
            )
            verification.setObjectName("matchVerified" if verified else "matchRejected")
            verification.setWordWrap(True)
            verification.setToolTip(
                "RANSAC inliers between this reference tile and the query. "
                "A verified candidate shares a consistent homography with the query."
            )

        image_path_text = str(match.get("image_path") or "")
        image_path = Path(image_path_text) if image_path_text else None
        image_name = QLabel(image_path.name if image_path is not None else "Unknown image")
        image_name.setObjectName("matchName")
        image_name.setToolTip(image_path_text)

        path_label = QLabel("REFERENCE IMAGE PATH")
        path_label.setObjectName("matchPathLabel")
        path_field = QLineEdit(image_path_text)
        path_field.setObjectName("matchPath")
        path_field.setReadOnly(True)
        path_field.setToolTip(image_path_text)
        path_field.setCursorPosition(0)

        coordinates = QLabel(
            f"Lat  {fmt(match.get('latitude'), '', 7)}\n"
            f"Lon  {fmt(match.get('longitude'), '', 7)}"
        )
        coordinates.setObjectName("coordinates")

        image_id = match.get("image_id")
        identifier = QLabel(f"ID  {image_id}" if image_id else "")
        identifier.setObjectName("matchMeta")

        layout.addWidget(heading)
        if verification is not None:
            layout.addWidget(verification)
        layout.addWidget(preview)
        layout.addWidget(image_name)
        layout.addWidget(path_label)
        layout.addWidget(path_field)
        layout.addWidget(coordinates)
        layout.addWidget(identifier)


class LightAVLConsole(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.process: QProcess | None = None
        self.process_kind: str | None = None
        self.online_worker: QProcess | None = None
        self.online_worker_key: tuple[str, str, int] | None = None
        self.online_worker_ready = False
        self.online_worker_stopping = False
        self.online_stdout_buffer = ""
        self.online_engine_started_at: float | None = None
        self.online_request_started_at: float | None = None
        self.pending_online_request: dict[str, Any] | None = None
        self.project_root = find_project_root()
        self.output_json = self.project_root / "artifacts" / "gui_kpis.json"
        self.output_csv: Path | None = None
        self.localizations: list[dict[str, Any]] = []

        self.setWindowTitle("AVL Localization & Benchmark Console")
        self.resize(1240, 820)
        self.setMinimumSize(1040, 680)

        self._build_ui()
        self._apply_style()
        self._prefill_examples()
        QTimer.singleShot(300, self._preload_online_worker)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("AVL Localization Console")
        title.setObjectName("title")
        subtitle = QLabel("Build the map once, test many queries, or benchmark a query CSV")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.status = QLabel("READY")
        self.status.setObjectName("status")
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(self.status)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_controls(), 0)
        body.addWidget(self._build_results(), 1)
        layout.addLayout(body, 1)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.root_edit = self._path_row(layout, "Project root", "folder")
        self.metadata_edit = self._path_row(layout, "Satellite reference CSV", "csv")
        self.base_edit = self._path_row(layout, "Reference base folder", "folder")

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.model = QComboBox()
        self.model.addItems(["denseuav-vit", "mixvpr", "cosplace"])
        self.model.currentTextChanged.connect(self._sync_dimensions)
        self.dimension = QComboBox()
        self.backbone = QComboBox()
        self.backbone.addItems(["ResNet101", "ResNet50", "ResNet18"])
        self.index_type = QComboBox()
        self.index_type.addItems(["flat", "hnsw", "ivfpq"])
        self.device = QComboBox()
        self.device.addItems(["cuda", "cpu"])
        self.device.currentTextChanged.connect(self._online_config_changed)

        self.top_k = self._spin(1, 50, 5)
        self.batch = self._spin(1, 128, 16)
        self.repeats = self._spin(1, 20, 3)
        self.rotations = QComboBox()
        self.rotations.addItem("4 orientations", 4)
        self.rotations.addItem("Off", 1)
        self.rotations.currentIndexChanged.connect(self._online_config_changed)

        controls = [
            ("Model", self.model),
            ("Dimension", self.dimension),
            ("Backbone", self.backbone),
            ("Search", self.index_type),
            ("Device", self.device),
            ("Top K", self.top_k),
            ("Rotation search", self.rotations),
            ("Reference batch size", self.batch),
            ("Benchmark repeats", self.repeats),
        ]
        for row, (label, widget) in enumerate(controls):
            grid.addWidget(self._label(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)

        layout.addWidget(self._build_rerank_controls())

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setObjectName("modeTabs")
        self.mode_tabs.addTab(self._build_interactive_controls(), "Test & Visualize")
        self.mode_tabs.addTab(self._build_batch_controls(), "Batch KPI")
        layout.addWidget(self.mode_tabs)

        self.load_button = QPushButton("Load KPI JSON")
        self.load_button.clicked.connect(self.load_json_dialog)
        layout.addWidget(self.load_button)

        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("Offline build, online query, and benchmark output appears here.")
        layout.addWidget(self.log, 1)

        self._sync_dimensions()

        # The column holds more rows than a short window can show, so it scrolls
        # rather than compressing every control into an unreadable stack.
        scroller = QScrollArea()
        scroller.setObjectName("controlsScroll")
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroller.setFixedWidth(448)
        scroller.setWidget(panel)
        return scroller

    def _build_rerank_controls(self) -> QWidget:
        """Second-stage geometric verification of the retrieval short-list."""
        panel = QFrame()
        panel.setObjectName("rerankPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.rerank_enabled = QCheckBox("Geometric re-ranking")
        self.rerank_enabled.setObjectName("rerankToggle")
        self.rerank_enabled.setToolTip(
            "Verify the top candidates with local features + RANSAC and re-order "
            "them by inlier count before fusing a position."
        )
        self.rerank_enabled.toggled.connect(self._rerank_toggled)
        layout.addWidget(self.rerank_enabled)

        self.rerank_backend = QComboBox()
        for backend in RERANK_BACKENDS:
            label = backend
            if backend in OPTIONAL_BACKEND_HINT:
                label = f"{backend}  (needs {OPTIONAL_BACKEND_HINT[backend].split()[-1]})"
            self.rerank_backend.addItem(label, backend)
        self.rerank_backend.setCurrentIndex(RERANK_BACKENDS.index(DEFAULT_BACKEND))

        self.rerank_candidates = self._spin(1, 200, 10)
        self.rerank_candidates.setToolTip(
            "How deep the short-list handed to verification is. Deeper finds more "
            "true matches that retrieval ranked low, and costs one match per tile."
        )
        self.rerank_min_inliers = self._spin(0, 500, 12)
        self.rerank_min_inliers.setToolTip(
            "RANSAC inliers a candidate needs before it counts as verified."
        )
        self.rerank_max_features = self._spin(256, 8192, 2048)
        self.rerank_max_features.setToolTip("Keypoint budget per image.")

        self.rerank_blend = QDoubleSpinBox()
        self.rerank_blend.setRange(0.0, 1.0)
        self.rerank_blend.setSingleStep(0.1)
        self.rerank_blend.setDecimals(2)
        self.rerank_blend.setValue(0.5)
        self.rerank_blend.setToolTip(
            "0 keeps the descriptor order, 1 ranks purely on geometry, 0.5 blends both."
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        rerank_controls = [
            ("Matcher", self.rerank_backend),
            ("Candidates", self.rerank_candidates),
            ("Min inliers", self.rerank_min_inliers),
            ("Geometry weight", self.rerank_blend),
            ("Max features", self.rerank_max_features),
        ]
        for row, (label, widget) in enumerate(rerank_controls):
            grid.addWidget(self._label(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)

        self.rerank_hint = QLabel(
            "Off — matches are ranked by descriptor similarity alone."
        )
        self.rerank_hint.setObjectName("rerankHint")
        self.rerank_hint.setWordWrap(True)
        layout.addWidget(self.rerank_hint)

        self._rerank_widgets = [widget for _, widget in rerank_controls]
        self._rerank_toggled(False)
        return panel

    def _rerank_toggled(self, enabled: bool) -> None:
        for widget in getattr(self, "_rerank_widgets", []):
            widget.setEnabled(enabled)
        if enabled:
            self.rerank_hint.setText(
                "On — the top candidates are verified with local features; the "
                "stage adds latency but usually fixes Recall@1."
            )
        else:
            self.rerank_hint.setText(
                "Off — matches are ranked by descriptor similarity alone."
            )

    def _rerank_settings(self) -> dict[str, Any]:
        """Query-time re-ranking payload; the worker applies it without reloading."""
        return {
            "enabled": self.rerank_enabled.isChecked(),
            "backend": self.rerank_backend.currentData(),
            "candidates": self.rerank_candidates.value(),
            "min_inliers": self.rerank_min_inliers.value(),
            "max_features": self.rerank_max_features.value(),
            "blend": self.rerank_blend.value(),
        }

    def _build_interactive_controls(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        note = QLabel(
            "1. Build the offline index once.\n"
            "2. Select any query image and run only the online stage."
        )
        note.setObjectName("modeNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.query_edit = self._path_row(layout, "Query aerial image", "image")
        self.index_dir_edit = self._path_row(layout, "Reusable offline index folder", "folder")
        self.index_dir_edit.textChanged.connect(self._update_index_status)
        self.index_dir_edit.editingFinished.connect(self._online_config_changed)
        self.index_status = QLabel("Offline index has not been checked.")
        self.index_status.setObjectName("indexStatus")
        self.index_status.setWordWrap(True)
        layout.addWidget(self.index_status)

        row = QHBoxLayout()
        self.build_button = QPushButton("1. Build offline index")
        self.build_button.clicked.connect(self.build_offline_index)
        self.run_query_button = QPushButton("2. Run online query")
        self.run_query_button.setObjectName("primaryButton")
        self.run_query_button.clicked.connect(self.run_online_query)
        row.addWidget(self.build_button)
        row.addWidget(self.run_query_button)
        layout.addLayout(row)
        return page

    def _build_batch_controls(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        note = QLabel(
            "Select a query CSV with image_path and expected_image_id. "
            "The batch run reports feature/search speed and Recall@1/5/10."
        )
        note.setObjectName("modeNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.batch_query_edit = self._path_row(layout, "Multiple-query CSV", "csv")
        self.batch_query_base_edit = self._path_row(layout, "Query image base folder", "folder")
        self.batch_run_button = QPushButton("Run batch KPI benchmark")
        self.batch_run_button.setObjectName("primaryButton")
        self.batch_run_button.clicked.connect(self.run_batch_benchmark)
        layout.addWidget(self.batch_run_button)
        return page

    def _build_results(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(10)
        self.metrics = {
            "ref": MetricBox("Offline features"),
            "build": MetricBox("FAISS build"),
            "query": MetricBox("Query feature"),
            "search": MetricBox("Search"),
            "rerank": MetricBox("Re-rank"),
            "online": MetricBox("Online total"),
            "memory": MetricBox("Memory"),
            "quality": MetricBox("Quality"),
            "status": MetricBox("Run status"),
        }
        for index, metric in enumerate(self.metrics.values()):
            metrics_layout.addWidget(metric, index // 3, index % 3)
        layout.addLayout(metrics_layout)

        self.result_tabs = QTabWidget()
        self.result_tabs.setObjectName("resultTabs")

        localization_tab = QWidget()
        localization_layout = QVBoxLayout(localization_tab)
        localization_layout.setContentsMargins(2, 10, 2, 2)
        localization_layout.setSpacing(10)

        query_header = QHBoxLayout()
        query_title = QLabel("QUERY IMAGE")
        query_title.setObjectName("sectionTitle")
        self.query_selector = QComboBox()
        self.query_selector.setMinimumWidth(240)
        self.query_selector.setVisible(False)
        self.query_selector.currentIndexChanged.connect(self._render_localization)
        query_header.addWidget(query_title)
        query_header.addStretch(1)
        query_header.addWidget(self.query_selector)
        localization_layout.addLayout(query_header)

        query_row = QHBoxLayout()
        query_row.setSpacing(14)
        self.query_preview = ImagePreviewLabel("Choose a query aerial image", 300, 205)
        query_row.addWidget(self.query_preview, 1)

        position_panel = QWidget()
        position_panel.setObjectName("positionPanel")
        position_layout = QVBoxLayout(position_panel)
        position_layout.setContentsMargins(16, 14, 16, 14)
        position_layout.setSpacing(7)
        position_title = QLabel("ESTIMATED POSITION")
        position_title.setObjectName("sectionTitle")
        self.estimated_latitude = QLabel("-")
        self.estimated_latitude.setObjectName("positionValue")
        self.estimated_longitude = QLabel("-")
        self.estimated_longitude.setObjectName("positionValue")
        self.position_detail = QLabel("Run the pipeline to calculate location.")
        self.position_detail.setObjectName("positionDetail")
        self.position_detail.setWordWrap(True)
        position_layout.addWidget(position_title)
        position_layout.addWidget(self._label("LATITUDE"))
        position_layout.addWidget(self.estimated_latitude)
        position_layout.addWidget(self._label("LONGITUDE"))
        position_layout.addWidget(self.estimated_longitude)
        position_layout.addWidget(self.position_detail)
        position_layout.addStretch(1)
        query_row.addWidget(position_panel, 1)
        localization_layout.addLayout(query_row)

        matches_title = QLabel("TOP MATCHES")
        matches_title.setObjectName("sectionTitle")
        localization_layout.addWidget(matches_title)

        self.matches_scroll = QScrollArea()
        self.matches_scroll.setObjectName("matchesScroll")
        self.matches_scroll.setWidgetResizable(True)
        self.matches_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.matches_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.matches_scroll.setMinimumHeight(260)
        self.matches_content = QWidget()
        self.matches_layout = QHBoxLayout(self.matches_content)
        self.matches_layout.setContentsMargins(0, 0, 0, 0)
        self.matches_layout.setSpacing(10)
        self.matches_scroll.setWidget(self.matches_content)
        localization_layout.addWidget(self.matches_scroll, 1)
        self._show_empty_matches("Run the pipeline to view the ranked reference images.")
        self.result_tabs.addTab(localization_tab, "Localization")

        kpi_tab = QWidget()
        kpi_layout = QVBoxLayout(kpi_tab)
        kpi_layout.setContentsMargins(2, 10, 2, 2)
        self.table = QTableWidget(0, 13)
        self.table.setObjectName("runsTable")
        self.table.setHorizontalHeaderLabels(
            [
                "Model",
                "Index",
                "Refs",
                "Queries",
                "Ref ms/img",
                "Build s",
                "Query ms",
                "Search ms",
                "Online ms",
                "R@1",
                "R@5",
                "R@10",
                "Error",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        kpi_layout.addWidget(self.table)
        self.result_tabs.addTab(kpi_tab, "KPI Runs")

        layout.addWidget(self.result_tabs, 1)
        return panel

    def _path_row(self, parent: QVBoxLayout, title: str, mode: str) -> QLineEdit:
        parent.addWidget(self._label(title))
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setMinimumHeight(30)
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self._browse_path(edit, mode))
        row.addWidget(edit, 1)
        row.addWidget(button)
        parent.addLayout(row)
        return edit

    def _browse_path(self, edit: QLineEdit, mode: str) -> None:
        start = edit.text() or str(self.project_root)
        if mode == "folder":
            selected = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif mode == "csv":
            selected, _ = QFileDialog.getOpenFileName(self, "Choose CSV", start, "CSV files (*.csv);;All files (*)")
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Choose aerial image",
                start,
                "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All files (*)",
            )
        if selected:
            edit.setText(selected)
            if edit is self.query_edit:
                self.query_preview.set_path(selected)

    def _prefill_examples(self) -> None:
        self.root_edit.setText(str(self.project_root))
        dense_query = (
            self.project_root
            / "data"
            / "DenseUAV"
            / "DenseUAV"
            / "test"
            / "query_drone"
            / "002256"
            / "H100.JPG"
        )
        dense_metadata = (
            self.project_root
            / "data"
            / "denseuav_avl"
            / "references_gallery_satellite.csv"
        )
        if dense_query.exists() and dense_metadata.exists():
            self.query_edit.setText(str(dense_query))
            self.metadata_edit.setText(str(dense_metadata))
            self.base_edit.setText("/")
            dense_query_csv = self.project_root / "data" / "denseuav_avl" / "queries_test_drone.csv"
            self.batch_query_edit.setText(str(dense_query_csv))
            self.batch_query_base_edit.setText("/")
            self.index_dir_edit.setText(str(self.project_root / "artifacts" / "interactive_denseuav_index"))
        else:
            self.query_edit.setText(str(self.project_root / "examples" / "queries" / "query.jpg"))
            self.metadata_edit.setText(str(self.project_root / "examples" / "references.csv"))
            self.base_edit.setText(str(self.project_root))
            self.batch_query_edit.clear()
            self.batch_query_base_edit.setText(str(self.project_root))
            self.index_dir_edit.setText(str(self.project_root / "artifacts" / "interactive_demo_index"))
        self.query_preview.set_path(self.query_edit.text())
        self._update_index_status()

    def _sync_dimensions(self) -> None:
        current = self.model.currentText()
        self.dimension.clear()
        if current == "denseuav-vit":
            self.dimension.addItem("512")
            self.backbone.setEnabled(False)
        elif current == "mixvpr":
            self.dimension.addItems(["4096", "512", "128"])
            self.backbone.setEnabled(False)
        else:
            self.dimension.addItems(["2048", "4096", "512"])
            self.backbone.setEnabled(True)

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _python_and_script(self, script_name: str) -> tuple[Path, Path] | None:
        project_root = Path(self.root_edit.text()).expanduser()
        python_exec = project_root / ".venv" / "bin" / "python"
        script = project_root / "scripts" / script_name
        if not python_exec.exists():
            self.set_status("Missing .venv Python", error=True)
            self.append_log(f"Could not find {python_exec}")
            return None
        if not script.exists():
            self.set_status("Missing script", error=True)
            self.append_log(f"Could not find {script}")
            return None
        return python_exec, script

    def _model_config(self) -> str:
        value = f"{self.model.currentText()}:{self.dimension.currentText()}"
        if self.model.currentText() == "cosplace":
            value = f"{value}:{self.backbone.currentText()}"
        return value

    def build_offline_index(self) -> None:
        if self.process is not None:
            self.append_log("Another operation is already running.")
            return
        self._stop_online_worker()

        command = self._python_and_script("build_index.py")
        if command is None:
            return
        validation_error = self._validate_reference_csv()
        if validation_error is not None:
            self.set_status("Invalid reference CSV", error=True)
            self.append_log(validation_error)
            return

        index_dir_text = self.index_dir_edit.text().strip()
        if not index_dir_text:
            self.set_status("Missing index folder", error=True)
            return
        index_dir = Path(index_dir_text).expanduser()
        python_exec, script = command
        args = [
            str(script),
            "--metadata",
            self.metadata_edit.text(),
            "--output",
            str(index_dir),
            # this console picks the encoder and rotations itself; the recipe presets
            # (heading, centering, altitude crop) need flight telemetry it does not have
            "--preset",
            "none",
            "--model",
            self.model.currentText(),
            "--descriptor-dim",
            self.dimension.currentText(),
            "--cosplace-backbone",
            self.backbone.currentText(),
            "--index-type",
            self.index_type.currentText(),
            "--device",
            self.device.currentText(),
            "--batch-size",
            str(self.batch.value()),
            "--query-rotations",
            str(self.rotations.currentData()),
            "--quiet",
        ]
        if self.base_edit.text().strip():
            args[3:3] = ["--base-dir", self.base_edit.text()]
        self.output_json = index_dir / "build_report.json"
        self._start_process("offline", python_exec, args, self.output_json)

    def run_online_query(self) -> None:
        if self.process is not None:
            self.append_log("Another operation is already running.")
            return
        if self.pending_online_request is not None:
            self.append_log("An online query is already running.")
            return
        validation_error = self._validate_query_image()
        if validation_error is not None:
            self.set_status("Cannot run query", error=True)
            self.append_log(validation_error)
            return

        project_root = Path(self.root_edit.text()).expanduser()
        request_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time_ns() % 1_000_000_000):09d}"
        self.output_json = project_root / "artifacts" / f"online_query_{request_id}.json"
        self.pending_online_request = {
            "command": "query",
            "request_id": request_id,
            "query": str(Path(self.query_edit.text()).expanduser().resolve()),
            "top_k": self.top_k.value(),
            "output_json": str(self.output_json),
            "rerank": self._rerank_settings(),
        }
        self._set_action_buttons_enabled(False)
        if self._online_worker_matches_current_config():
            if self.online_worker_ready:
                self._dispatch_pending_online_query()
            else:
                self.set_status("Online engine loading")
                self.append_log("Query queued; it will run as soon as the engine is ready.")
            return

        self.set_status("Loading online engine")
        self.append_log("Loading the persistent online engine once; later queries reuse it.")
        self._start_online_worker()

    def run_batch_benchmark(self) -> None:
        if self.process is not None:
            self.append_log("Another operation is already running.")
            return
        self._stop_online_worker()

        command = self._python_and_script("benchmark_kpis.py")
        if command is None:
            return
        validation_error = self._validate_reference_csv(require_image_id=True) or self._validate_query_csv()
        if validation_error is not None:
            self.set_status("Invalid batch input", error=True)
            self.append_log(validation_error)
            return

        project_root = Path(self.root_edit.text()).expanduser()
        python_exec, script = command
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_json = project_root / "artifacts" / f"batch_kpis_{timestamp}.json"
        self.output_csv = project_root / "artifacts" / f"batch_kpis_{timestamp}.csv"
        args = [
            str(script),
            "--metadata",
            self.metadata_edit.text(),
            "--query-metadata",
            self.batch_query_edit.text(),
            "--model-config",
            self._model_config(),
            "--index-types",
            self.index_type.currentText(),
            "--top-k",
            str(self.top_k.value()),
            "--recall-k",
            "1",
            "5",
            "10",
            "--query-rotations",
            str(self.rotations.currentData()),
            "--device",
            self.device.currentText(),
            "--batch-size",
            str(self.batch.value()),
            "--repeats",
            str(self.repeats.value()),
            "--no-localizations",
            "--quiet",
            "--output-json",
            str(self.output_json),
            "--output-csv",
            str(self.output_csv),
        ]
        rerank = self._rerank_settings()
        if rerank["enabled"]:
            args.extend(
                [
                    "--rerank",
                    "--rerank-backend",
                    str(rerank["backend"]),
                    "--rerank-candidates",
                    str(rerank["candidates"]),
                    "--rerank-min-inliers",
                    str(rerank["min_inliers"]),
                    "--rerank-max-features",
                    str(rerank["max_features"]),
                    "--rerank-blend",
                    str(rerank["blend"]),
                ]
            )
        insertion = 3
        if self.base_edit.text().strip():
            args[insertion:insertion] = ["--base-dir", self.base_edit.text()]
            insertion += 2
        if self.batch_query_base_edit.text().strip():
            args[insertion:insertion] = ["--query-base-dir", self.batch_query_base_edit.text()]
        self._start_process("batch", python_exec, args, self.output_json)

    def _current_online_worker_key(self) -> tuple[str, str, int]:
        index_dir = str(Path(self.index_dir_edit.text()).expanduser().resolve())
        return index_dir, self.device.currentText(), int(self.rotations.currentData())

    def _online_worker_matches_current_config(self) -> bool:
        return (
            self.online_worker is not None
            and self.online_worker.state() != QProcess.ProcessState.NotRunning
            and self.online_worker_key == self._current_online_worker_key()
        )

    def _preload_online_worker(self) -> None:
        if self.process is not None or self.pending_online_request is not None:
            return
        if self._validate_query_image() is not None:
            return
        if self._online_worker_matches_current_config():
            return
        self._start_online_worker()

    def _start_online_worker(self) -> None:
        command = self._python_and_script("online_worker.py")
        if command is None:
            self.pending_online_request = None
            self._set_action_buttons_enabled(True)
            return

        self._stop_online_worker()
        python_exec, script = command
        key = self._current_online_worker_key()
        args = [
            str(script),
            "--index",
            key[0],
            "--device",
            key[1],
            "--query-rotations",
            str(key[2]),
        ]
        warmup_path = Path(self.query_edit.text()).expanduser()
        if warmup_path.is_file():
            args.extend(["--warmup-image", str(warmup_path.resolve())])

        self.online_worker = QProcess(self)
        self.online_worker_key = key
        self.online_worker_ready = False
        self.online_worker_stopping = False
        self.online_stdout_buffer = ""
        self.online_engine_started_at = time.perf_counter()
        self.online_worker.setWorkingDirectory(str(Path(self.root_edit.text()).expanduser()))
        self.online_worker.readyReadStandardOutput.connect(self._read_online_worker_stdout)
        self.online_worker.readyReadStandardError.connect(self._read_online_worker_stderr)
        self.online_worker.finished.connect(self._online_worker_finished)
        self.online_worker.start(str(python_exec), args)
        self._update_index_status()

    def _dispatch_pending_online_query(self) -> None:
        if (
            self.pending_online_request is None
            or self.online_worker is None
            or not self.online_worker_ready
        ):
            return
        self.online_request_started_at = time.perf_counter()
        self.set_status("Localizing")
        payload = json.dumps(self.pending_online_request, separators=(",", ":")) + "\n"
        self.online_worker.write(payload.encode("utf-8"))

    def _read_online_worker_stdout(self) -> None:
        if self.online_worker is None:
            return
        self.online_stdout_buffer += bytes(
            self.online_worker.readAllStandardOutput()
        ).decode(errors="replace")
        while "\n" in self.online_stdout_buffer:
            line, self.online_stdout_buffer = self.online_stdout_buffer.split("\n", 1)
            if line.startswith("AVL_EVENT "):
                try:
                    self._handle_online_worker_event(json.loads(line[len("AVL_EVENT "):]))
                except json.JSONDecodeError as exc:
                    self.append_log(f"Invalid online worker event: {exc}")
            elif line.strip():
                self.append_log(line)

    def _read_online_worker_stderr(self) -> None:
        if self.online_worker is not None:
            text = bytes(self.online_worker.readAllStandardError()).decode(errors="replace")
            if text:
                self.append_log(text)

    def _handle_online_worker_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event")
        if event_type == "ready":
            self.online_worker_ready = True
            startup_ms = None
            if self.online_engine_started_at is not None:
                startup_ms = (time.perf_counter() - self.online_engine_started_at) * 1000.0
            self.append_log(
                "Online engine ready"
                + (f" in {startup_ms:.0f} ms" if startup_ms is not None else "")
                + f" · {event.get('model_label')} · {event.get('reference_count')} references"
            )
            self.set_status("Online engine ready")
            self._update_index_status()
            if self.pending_online_request is not None:
                self._dispatch_pending_online_query()
            else:
                self._set_action_buttons_enabled(True)
            return

        if event_type == "result":
            output_path = Path(str(event["output_json"]))
            self.load_report(output_path)
            visible_ms = None
            if self.online_request_started_at is not None:
                visible_ms = (time.perf_counter() - self.online_request_started_at) * 1000.0
            if visible_ms is not None:
                self.metrics["online"].set_value(
                    f"{visible_ms:.2f} ms",
                    (
                        f"visible result · worker "
                        f"{float(event.get('end_to_end_ms', 0.0)):.2f} ms · "
                        f"pipeline {float(event.get('pipeline_ms', 0.0)):.2f} ms"
                    ),
                )
                self.append_log(
                    f"Visible result {visible_ms:.1f} ms · "
                    f"worker {float(event.get('end_to_end_ms', 0.0)):.1f} ms · "
                    f"feature + search {float(event.get('pipeline_ms', 0.0)):.1f} ms"
                )
            rerank_summary = event.get("rerank")
            if rerank_summary:
                if rerank_summary.get("error"):
                    self.append_log(
                        f"Re-ranking fell back to descriptor order: {rerank_summary['error']}"
                    )
                else:
                    self.append_log(
                        f"Re-ranked {rerank_summary.get('candidates')} candidates with "
                        f"{rerank_summary.get('backend')} · "
                        f"{rerank_summary.get('verified')} verified · "
                        f"{float(event.get('rerank_ms') or 0.0):.1f} ms"
                    )
            self.pending_online_request = None
            self.online_request_started_at = None
            self._set_action_buttons_enabled(True)
            self.set_status("Online query complete")
            self.result_tabs.setCurrentIndex(0)
            return

        if event_type == "error":
            self.append_log(str(event.get("message", "Unknown online worker error")))
            self.pending_online_request = None
            self.online_request_started_at = None
            self._set_action_buttons_enabled(True)
            self.set_status("Online query failed", error=True)

    def _online_worker_finished(self, exit_code: int, _status) -> None:
        was_stopping = self.online_worker_stopping
        self.online_worker = None
        self.online_worker_key = None
        self.online_worker_ready = False
        self.online_worker_stopping = False
        self._update_index_status()
        if was_stopping:
            return
        self.append_log(f"Online engine stopped unexpectedly ({exit_code}).")
        if self.pending_online_request is not None:
            self.pending_online_request = None
            self._set_action_buttons_enabled(True)
            self.set_status("Online engine failed", error=True)
        else:
            self.set_status("Online engine unavailable", error=True)

    def _stop_online_worker(self) -> None:
        worker = self.online_worker
        if worker is None:
            return
        self.online_worker_stopping = True
        if worker.state() != QProcess.ProcessState.NotRunning:
            worker.write(b'{"command":"shutdown"}\n')
            worker.waitForFinished(1000)
        if worker.state() != QProcess.ProcessState.NotRunning:
            worker.kill()
            worker.waitForFinished(1000)
        self.online_worker = None
        self.online_worker_key = None
        self.online_worker_ready = False
        self.online_worker_stopping = False

    def _online_config_changed(self, *_args) -> None:
        if not hasattr(self, "index_dir_edit"):
            return
        if self.online_worker is not None and not self._online_worker_matches_current_config():
            self._stop_online_worker()
        self._update_index_status()
        QTimer.singleShot(150, self._preload_online_worker)

    def _start_process(
        self,
        kind: str,
        python_exec: Path,
        args: list[str],
        output_json: Path,
    ) -> None:
        project_root = Path(self.root_edit.text()).expanduser()
        self.process_kind = kind
        self.output_json = output_json
        self.log.clear()
        self.append_log("$ " + " ".join([str(python_exec), *args]))
        self.set_status(
            {
                "offline": "Building offline index",
                "online": "Running online query",
                "batch": "Running batch KPI",
            }.get(kind, "Running")
        )
        self._set_action_buttons_enabled(False)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(project_root))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.start(str(python_exec), args)

    def _validate_reference_csv(self, require_image_id: bool = False) -> str | None:
        metadata_path = Path(self.metadata_edit.text()).expanduser()
        if not metadata_path.is_file():
            return f"Reference CSV does not exist: {metadata_path}"

        try:
            with metadata_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                fields = set(reader.fieldnames or [])
                if "expected_image_id" in fields and "image_id" not in fields:
                    return (
                        f"{metadata_path.name} is a query CSV. Choose "
                        "references_gallery_satellite.csv as the reference database."
                    )
                required = {"image_path", "latitude", "longitude"}
                if require_image_id:
                    required.add("image_id")
                missing = sorted(required - fields)
                if missing:
                    return f"Reference CSV is missing columns: {', '.join(missing)}"
        except (OSError, csv.Error) as exc:
            return f"Could not read reference CSV: {exc}"
        return None

    def _validate_query_image(self) -> str | None:
        query_path = Path(self.query_edit.text()).expanduser()
        if not query_path.is_file():
            return f"Query image does not exist: {query_path}"
        if not self.index_dir_edit.text().strip():
            return "Choose an offline index folder and build it once before running queries."
        index_dir = Path(self.index_dir_edit.text()).expanduser()
        if not (index_dir / "index.faiss").is_file() or not (index_dir / "metadata.json").is_file():
            return (
                f"No reusable offline index found in {index_dir}. "
                "Click 'Build offline index' once before running queries."
            )
        return None

    def _validate_query_csv(self) -> str | None:
        query_csv = Path(self.batch_query_edit.text()).expanduser()
        if not query_csv.is_file():
            return f"Multiple-query CSV does not exist: {query_csv}"
        try:
            with query_csv.open(newline="", encoding="utf-8") as file:
                fields = set(csv.DictReader(file).fieldnames or [])
        except (OSError, csv.Error) as exc:
            return f"Could not read query CSV: {exc}"
        missing = sorted({"image_path", "expected_image_id"} - fields)
        if missing:
            return (
                "Query CSV must contain image_path and expected_image_id "
                f"to calculate Recall@1/5/10. Missing: {', '.join(missing)}"
            )
        return None

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        self.build_button.setEnabled(enabled)
        self.run_query_button.setEnabled(enabled)
        self.batch_run_button.setEnabled(enabled)

    def _update_index_status(self) -> None:
        if not self.index_dir_edit.text().strip():
            self.index_status.setText("NOT BUILT: choose an index folder first.")
            return
        index_dir = Path(self.index_dir_edit.text()).expanduser()
        if (index_dir / "index.faiss").is_file() and (index_dir / "metadata.json").is_file():
            saved_label = ""
            try:
                metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
                config = metadata.get("config", {})
                saved_label = (
                    f" ({config.get('model', '?')}:{config.get('descriptor_dim', '?')}, "
                    f"{config.get('index_type', '?')})"
                )
            except (OSError, json.JSONDecodeError):
                pass
            if self._online_worker_matches_current_config() and self.online_worker_ready:
                engine_state = "ENGINE READY"
            elif self._online_worker_matches_current_config():
                engine_state = "ENGINE LOADING"
            else:
                engine_state = "INDEX READY · ENGINE COLD"
            self.index_status.setText(
                f"{engine_state}{saved_label}: {index_dir}. "
                "The model and index stay loaded between query images."
            )
        else:
            self.index_status.setText("NOT BUILT: run the offline stage once before online queries.")

    def _read_stdout(self) -> None:
        if self.process is not None:
            self.append_log(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

    def _read_stderr(self) -> None:
        if self.process is not None:
            self.append_log(bytes(self.process.readAllStandardError()).decode(errors="replace"))

    def _process_finished(self, exit_code: int, _status) -> None:
        kind = self.process_kind
        self._set_action_buttons_enabled(True)
        self.process = None
        self.process_kind = None
        if exit_code != 0:
            self.set_status(f"Failed ({exit_code})", error=True)
            return
        if kind == "offline":
            self._update_index_status()
            self.set_status("Offline index ready")
        elif kind == "online":
            self.set_status("Online query complete")
            self.result_tabs.setCurrentIndex(0)
        elif kind == "batch":
            self.set_status("Batch KPI complete")
            self.result_tabs.setCurrentIndex(1)
            if self.output_csv is not None:
                self.append_log(f"CSV KPI summary: {self.output_csv}")
        else:
            self.set_status("Complete")
        self.load_report(self.output_json)
        if kind == "offline":
            QTimer.singleShot(100, self._preload_online_worker)

    def load_json_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load KPI JSON",
            str(Path(self.root_edit.text()) / "artifacts"),
            "JSON files (*.json);;All files (*)",
        )
        if selected:
            self.load_report(Path(selected))

    def load_report(self, path: Path) -> None:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.set_status("JSON error", error=True)
            self.append_log(f"Could not load {path}: {exc}")
            return

        runs = report.get("runs", [])
        self.render_table(runs)
        first_ok = next((run for run in runs if not run.get("error")), runs[0] if runs else None)
        if first_ok is not None:
            self.render_metrics(first_ok)
            self.render_localizations(first_ok)
        else:
            self.render_localizations({})
        if report.get("mode") == "interactive_offline":
            self.position_detail.setText(
                "Offline index is ready. Select a query image and click 'Run online query'."
            )
            self._show_empty_matches("The reference descriptors and FAISS index are saved for reuse.")
        elif report.get("query_metadata"):
            self.result_tabs.setCurrentIndex(1)
        self.append_log(f"Loaded KPI report: {path}")

    def render_metrics(self, run: dict[str, Any]) -> None:
        offline = run.get("offline", {})
        online = run.get("online", {})
        quality = run.get("quality", {})
        query = online.get("query_encode_ms", {})
        search = online.get("search_ms", {})

        self.metrics["ref"].set_value(
            fmt(offline.get("reference_encode_s"), " s"),
            (
                f"{fmt(offline.get('reference_encode_ms_per_image'), ' ms/img')} · "
                f"{fmt(offline.get('reference_images_per_s'), ' img/s')} · "
                f"{offline.get('reference_count', '-')} refs"
            ),
        )
        self.metrics["build"].set_value(
            fmt(offline.get("faiss_build_s"), " s", 3),
            fmt(offline.get("faiss_build_ms_per_image"), " ms/img", 3),
        )
        self.metrics["query"].set_value(
            fmt(query.get("mean"), " ms"),
            (
                f"p95 {fmt(query.get('p95'), ' ms')} · "
                f"{fmt(online.get('query_feature_images_per_s'), ' img/s')}"
            ),
        )
        self.metrics["search"].set_value(
            fmt(search.get("mean"), " ms", 3),
            (
                f"p95 {fmt(search.get('p95'), ' ms', 3)} · "
                f"{fmt(online.get('search_queries_per_s'), ' q/s')}"
            ),
        )
        rerank_ms = online.get("rerank_ms") or {}
        rerank = online.get("rerank") or {}
        rerank_run = self._first_rerank_summary(run)
        if rerank_ms.get("mean") is None:
            self.metrics["rerank"].set_value("off", "descriptor order only")
        else:
            verified = (rerank_run or {}).get("verified")
            candidates = (rerank_run or {}).get("candidates", rerank.get("candidates"))
            detail = f"{rerank.get('backend', '-')}"
            if verified is not None:
                detail += f" · {verified}/{candidates} verified"
            elif candidates is not None:
                detail += f" · {candidates} candidates"
            self.metrics["rerank"].set_value(fmt(rerank_ms.get("mean"), " ms"), detail)

        self.metrics["online"].set_value(
            fmt(
                online.get(
                    "end_to_end_ms",
                    online.get("encode_plus_search_ms_mean"),
                ),
                " ms",
            ),
            (
                "request end-to-end · feature + search"
                + (" + re-rank" if rerank_ms.get("mean") is not None else "")
                + f" {fmt(online.get('encode_plus_search_ms_mean'), ' ms')}"
            ),
        )
        self.metrics["memory"].set_value(
            fmt(offline.get("faiss_index_size_mb"), " MB"),
            f"descriptors {fmt(offline.get('descriptor_memory_mb'), ' MB')}",
        )
        recall_at = quality.get("recall_at", {})
        recall_1 = recall_at.get("1", quality.get("top1_hit_rate"))
        recall_5 = recall_at.get("5", quality.get("recall_at_k"))
        recall_10 = recall_at.get("10")
        self.metrics["quality"].set_value(
            f"R@1 {fmt(recall_1, '', 3)}",
            f"R@5 {fmt(recall_5, '', 3)} · R@10 {fmt(recall_10, '', 3)}",
        )
        self.metrics["status"].set_value("OK" if not run.get("error") else "ERROR", run.get("error") or "")

    @staticmethod
    def _first_rerank_summary(run: dict[str, Any]) -> dict[str, Any] | None:
        for localization in run.get("localizations", []) or []:
            summary = localization.get("rerank")
            if summary:
                return summary
        return None

    def render_localizations(self, run: dict[str, Any]) -> None:
        self.localizations = run.get("localizations", [])
        self.query_selector.blockSignals(True)
        self.query_selector.clear()
        for index, result in enumerate(self.localizations, start=1):
            path = Path(str(result.get("query_image_path", "")))
            self.query_selector.addItem(f"{index}. {path.name or 'Query'}")
        self.query_selector.setVisible(len(self.localizations) > 1)
        self.query_selector.blockSignals(False)

        if self.localizations:
            self.query_selector.setCurrentIndex(0)
            self._render_localization(0)
            return

        self.query_preview.set_path(self.query_edit.text())
        self.estimated_latitude.setText("-")
        self.estimated_longitude.setText("-")
        self.position_detail.setText("This report does not contain localization matches.")
        self._show_empty_matches("Run a new pipeline to generate Top-K visual matches.")

    def _render_localization(self, index: int) -> None:
        if index < 0 or index >= len(self.localizations):
            return

        result = self.localizations[index]
        self.query_preview.set_path(result.get("query_image_path"))
        estimated = result.get("estimated_position") or {}
        self.estimated_latitude.setText(fmt(estimated.get("latitude"), "", 7))
        self.estimated_longitude.setText(fmt(estimated.get("longitude"), "", 7))

        detail_parts: list[str] = []
        altitude = estimated.get("altitude_m")
        if altitude is not None:
            detail_parts.append(f"Altitude {fmt(altitude, ' m')}")
        error_m = result.get("localization_error_m")
        if error_m is not None:
            detail_parts.append(f"Ground-truth error {fmt(error_m, ' m')}")
        ground_truth = result.get("query_ground_truth") or {}
        expected_id = ground_truth.get("expected_image_id")
        if expected_id:
            detail_parts.append(f"Expected ID {expected_id}")
        fusion_count = result.get("fusion_match_count")
        if fusion_count:
            detail_parts.append(f"Fused from {fusion_count} strongest locations")
        self.position_detail.setText("  |  ".join(detail_parts) or "Similarity-weighted position")

        self._clear_matches()
        matches = result.get("matches", [])
        if not matches:
            self._show_empty_matches("No valid reference matches were returned.")
            return
        for match in matches:
            self.matches_layout.addWidget(MatchCard(match))
        self.matches_layout.addStretch(1)

    def _clear_matches(self) -> None:
        while self.matches_layout.count():
            item = self.matches_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _show_empty_matches(self, text: str) -> None:
        self._clear_matches()
        label = QLabel(text)
        label.setObjectName("emptyMatches")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matches_layout.addWidget(label, 1)

    def render_table(self, runs: list[dict[str, Any]]) -> None:
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            offline = run.get("offline", {})
            online = run.get("online", {})
            quality = run.get("quality", {})
            recall_at = quality.get("recall_at", {})
            recall_1 = recall_at.get("1", quality.get("top1_hit_rate"))
            recall_5 = recall_at.get("5", quality.get("recall_at_k"))
            query = online.get("query_encode_ms", {})
            search = online.get("search_ms", {})
            values = [
                run.get("model_label"),
                run.get("index_type"),
                offline.get("reference_count"),
                online.get("query_count"),
                fmt(offline.get("reference_encode_ms_per_image")),
                fmt(offline.get("faiss_build_s"), "", 3),
                fmt(query.get("mean")),
                fmt(search.get("mean"), "", 3),
                fmt(online.get("encode_plus_search_ms_mean")),
                fmt(recall_1, "", 3),
                fmt(recall_5, "", 3),
                fmt(recall_at.get("10"), "", 3),
                run.get("error") or "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col not in {0, 12}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        if not text.endswith("\n"):
            self.log.insertPlainText("\n")
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(text.upper())
        self.status.setProperty("error", error)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_online_worker()
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            self.process.waitForFinished(1000)
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
        super().closeEvent(event)

    def _apply_style(self) -> None:
        QApplication.instance().setFont(QFont("Arial", 10))
        self.setStyleSheet(
            """
            QWidget#root {
                background: #091114;
                color: #eef8f7;
            }
            QLabel#title {
                font-size: 28px;
                font-weight: 800;
                color: #ffffff;
            }
            QLabel#subtitle,
            QLabel#fieldLabel,
            QLabel#modeNote,
            QLabel#indexStatus {
                color: #9fb8b5;
            }
            QLabel#fieldLabel {
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#modeNote {
                background: #0b171a;
                border: 1px solid #203a3e;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
            }
            QLabel#indexStatus {
                font-size: 10px;
            }
            QScrollArea#controlsScroll {
                background: transparent;
                border: none;
            }
            QFrame#rerankPanel {
                background: #0b171a;
                border: 1px solid #203a3e;
                border-radius: 6px;
            }
            QCheckBox#rerankToggle {
                color: #eef8f7;
                font-weight: 800;
            }
            QLabel#rerankHint {
                color: #8fb0ac;
                font-size: 10px;
            }
            QLabel#status {
                background: #123a3d;
                border: 1px solid #2dd4bf;
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 800;
                color: #dffffb;
            }
            QLabel#status[error="true"] {
                background: #451a1a;
                border-color: #fb7185;
                color: #ffe4e6;
            }
            QWidget#panel,
            QWidget#metricBox {
                background: #101c20;
                border: 1px solid #20363a;
                border-radius: 6px;
            }
            QLabel#metricTitle {
                color: #8ea5a2;
                font-size: 10px;
                font-weight: 800;
            }
            QLabel#metricValue {
                color: #ffffff;
                font-size: 21px;
                font-weight: 800;
            }
            QLabel#metricDetail {
                color: #8fb6b1;
                font-size: 11px;
            }
            QLabel#sectionTitle {
                color: #a8c8c4;
                font-size: 11px;
                font-weight: 800;
            }
            QLabel#imagePreview {
                background: #050b0d;
                border: 1px solid #28454a;
                border-radius: 5px;
                color: #78918e;
            }
            QWidget#positionPanel {
                background: #0b171a;
                border: 1px solid #203a3e;
                border-radius: 6px;
            }
            QLabel#positionValue {
                color: #ffffff;
                font-size: 23px;
                font-weight: 800;
            }
            QLabel#positionDetail,
            QLabel#matchMeta,
            QLabel#emptyMatches {
                color: #8fb0ac;
                font-size: 11px;
            }
            QWidget#matchCard {
                background: #0b171a;
                border: 1px solid #203a3e;
                border-radius: 6px;
            }
            QLabel#matchRank {
                color: #5eead4;
                font-size: 12px;
                font-weight: 800;
            }
            QLabel#matchVerified,
            QLabel#matchRejected {
                font-size: 11px;
                font-weight: 700;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QLabel#matchVerified {
                color: #062b21;
                background: #34d399;
            }
            QLabel#matchRejected {
                color: #f3d0d0;
                background: #4a2020;
            }
            QLabel#matchName {
                color: #f3fffd;
                font-weight: 700;
            }
            QLabel#matchPathLabel {
                color: #8ea5a2;
                font-size: 9px;
                font-weight: 800;
            }
            QLineEdit#matchPath {
                min-height: 24px;
                padding: 0 6px;
                color: #bce7e1;
                font-family: monospace;
                font-size: 10px;
            }
            QLabel#coordinates {
                color: #d9f4f0;
                font-family: monospace;
                font-size: 12px;
            }
            QLineEdit,
            QComboBox,
            QSpinBox,
            QTextEdit,
            QTableWidget {
                background: #071013;
                border: 1px solid #28454a;
                border-radius: 5px;
                color: #f3fffd;
                selection-background-color: #0f766e;
            }
            QPushButton {
                min-height: 30px;
                border: 1px solid #2b5559;
                border-radius: 5px;
                padding: 0 12px;
                color: #eef8f7;
                background: #172b30;
                font-weight: 700;
            }
            QPushButton#primaryButton {
                background: #2dd4bf;
                color: #061014;
                border: none;
            }
            QPushButton:disabled {
                color: #647876;
                background: #10191c;
            }
            QHeaderView::section {
                background: #183238;
                color: #e6fffb;
                border: none;
                padding: 7px;
                font-weight: 800;
            }
            QTabWidget::pane {
                border: 1px solid #20363a;
                border-radius: 5px;
                top: -1px;
            }
            QTabBar::tab {
                min-width: 110px;
                min-height: 30px;
                padding: 0 12px;
                color: #9fb8b5;
                background: #0a1518;
                border: 1px solid #20363a;
            }
            QTabBar::tab:selected {
                color: #f3fffd;
                background: #183238;
                border-bottom-color: #183238;
            }
            QScrollArea#matchesScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#matchesScroll > QWidget > QWidget {
                background: transparent;
            }
            QTextEdit#log {
                font-family: monospace;
                font-size: 11px;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AVL Localization Console")
    window = LightAVLConsole()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
