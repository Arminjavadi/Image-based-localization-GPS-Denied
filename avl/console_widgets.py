"""Components of the AVL console (avl/model_bench_gui.py).

Kept apart from the console so that file stays about behaviour, not painting. The
look is "Nocturne Console": a deep blue-black workspace, panels one step lighter,
hairline borders, Ubuntu Sans, one blurple accent for selection and the primary
action, status colours only where they mean good / warning / bad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QEvent,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from avl.findings import STATUS_LABEL, Finding

# ── tokens ──────────────────────────────────────────────────────────────────────
FONT_UI = "Ubuntu Sans"
FONT_MONO = "Ubuntu Sans Mono"
MONO = f"'{FONT_MONO}', 'DejaVu Sans Mono', monospace"

BG = "#0d0f15"          # workspace
CHROME = "#11141c"      # top bar, rail, status bar, inspector
PANEL = "#161a24"       # cards
RAISED = "#1d2230"      # inputs, hovered rows
LINE = "#232838"        # hairlines
LINE_STRONG = "#30364a"
TEXT = "#e8eaf2"
TEXT_2 = "#b9bfd0"
MUTED = "#858ca3"
FAINT = "#5d6479"
ACCENT = "#8b7ff0"
ACCENT_300 = "#d2cefd"
ACCENT_400 = "#b0a7fb"
ACCENT_800 = "#3b3470"
ACCENT_900 = "#211d3d"
GOOD = "#5fd39a"
WARN = "#f2c14e"
BAD = "#f2727a"

# kept for older imports
SURFACE = PANEL
GROUND = BG

# Chart series — validated as a pair on PANEL with the dataviz palette checker
# (lightness band, chroma, CVD ΔE 23.8, normal-vision ΔE 24.9, ≥ 3:1 contrast).
SERIES_TOP1 = "#9184d9"
SERIES_FUSED = "#d95926"

STATUS_GLYPH = {"adopted": "✓", "gated": "✓", "accuracy": "✓", "not adopted": "✕", "prototype": "◐"}


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def ui_font(px: float, weight: QFont.Weight = QFont.Weight.Normal, mono: bool = False) -> QFont:
    font = QFont(FONT_MONO if mono else FONT_UI)
    font.setPixelSize(int(round(px)))
    font.setWeight(weight)
    return font


# ── icons ───────────────────────────────────────────────────────────────────────
def draw_icon(p: QPainter, name: str, rect: QRectF, colour: QColor) -> None:
    """Minimal line icons on a 24-unit grid, drawn into ``rect``."""
    s = rect.width() / 24.0
    p.save()
    p.translate(rect.topLeft())
    p.scale(s, s)
    pen = QPen(colour, 1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    if name == "overview":
        for x, y, w, h in ((3, 3, 8, 10), (13, 3, 8, 6), (13, 11, 8, 10), (3, 15, 8, 6)):
            p.drawRoundedRect(QRectF(x, y, w, h), 2, 2)
    elif name == "runs":
        for y in (6, 12, 18):
            p.drawLine(QPointF(9, y), QPointF(21, y))
            p.drawEllipse(QPointF(4.5, y), 1.2, 1.2)
    elif name == "findings":
        p.drawLine(QPointF(3, 21), QPointF(21, 21))
        for x, top in ((5, 14), (11, 9), (17, 4)):
            p.drawRoundedRect(QRectF(x, top, 3.5, 21 - top - 2.5), 1, 1)
    elif name == "inspect":
        p.drawEllipse(QPointF(12, 12), 7, 7)
        p.drawEllipse(QPointF(12, 12), 2, 2)
        for a, b in (((12, 1.5), (12, 5)), ((12, 19), (12, 22.5)), ((1.5, 12), (5, 12)), ((19, 12), (22.5, 12))):
            p.drawLine(QPointF(*a), QPointF(*b))
    elif name == "models":
        p.drawPolygon([QPointF(12, 3), QPointF(21, 7.5), QPointF(12, 12), QPointF(3, 7.5)])
        p.drawPolyline([QPointF(3, 12), QPointF(12, 16.5), QPointF(21, 12)])
        p.drawPolyline([QPointF(3, 16.5), QPointF(12, 21), QPointF(21, 16.5)])
    elif name == "trajectory":
        path = QPainterPath(QPointF(4, 19))
        path.cubicTo(QPointF(9, 19), QPointF(7, 6), QPointF(13, 8))
        path.cubicTo(QPointF(18, 10), QPointF(15, 4), QPointF(20, 4))
        p.drawPath(path)
        p.setBrush(colour)
        p.drawEllipse(QPointF(4, 19), 1.8, 1.8)
        p.drawEllipse(QPointF(20, 4), 1.8, 1.8)
    elif name == "studio":
        p.drawPolygon([QPointF(3, 6), QPointF(9, 3.5), QPointF(15, 6), QPointF(21, 3.5),
                       QPointF(21, 18), QPointF(15, 20.5), QPointF(9, 18), QPointF(3, 20.5)])
        p.drawLine(QPointF(9, 3.5), QPointF(9, 18))
        p.drawLine(QPointF(15, 6), QPointF(15, 20.5))
    elif name == "play":
        p.setBrush(colour)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon([QPointF(7, 4.5), QPointF(19.5, 12), QPointF(7, 19.5)])
    elif name == "stop":
        p.setBrush(colour)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    elif name == "panel":
        p.drawRoundedRect(QRectF(3, 4, 18, 16), 2.5, 2.5)
        p.drawLine(QPointF(14, 4), QPointF(14, 20))
    elif name == "terminal":
        p.drawRoundedRect(QRectF(2.5, 4, 19, 16), 2.5, 2.5)
        p.drawPolyline([QPointF(6.5, 9), QPointF(9.5, 12), QPointF(6.5, 15)])
        p.drawLine(QPointF(11.5, 15), QPointF(16.5, 15))
    elif name == "refresh":
        path = QPainterPath()
        path.arcMoveTo(QRectF(4, 4, 16, 16), 60)
        path.arcTo(QRectF(4, 4, 16, 16), 60, 280)
        p.drawPath(path)
        p.drawPolyline([QPointF(16, 2.8), QPointF(16.2, 6.8), QPointF(12.3, 7)])
    p.restore()


class BrandMark(QWidget):
    """The console's mark: a rounded tile with an accent gradient and a crosshair."""

    def __init__(self, size: int = 26) -> None:
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0.0, QColor("#a99ff7"))
        grad.setColorAt(1.0, QColor("#5b4fd0"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(r, 7, 7)
        c = r.center()
        pen = QPen(QColor("#ffffff"), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        rad = r.width() * 0.2
        p.drawEllipse(c, rad, rad)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            p.drawLine(QPointF(c.x() + dx * rad * 1.5, c.y() + dy * rad * 1.5),
                       QPointF(c.x() + dx * rad * 2.1, c.y() + dy * rad * 2.1))


class IconButton(QPushButton):
    """A square tool button with a painted icon (top bar, status bar)."""

    def __init__(self, icon: str, tooltip: str, checkable: bool = False, size: int = 30) -> None:
        super().__init__()
        self.icon_name = icon
        self.setObjectName("iconBtn")
        self.setToolTip(tooltip)
        self.setCheckable(checkable)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(ACCENT_300 if self.isChecked() else (TEXT if self.underMouse() else MUTED))
        if not self.isEnabled():
            colour = QColor(FAINT)
        side = min(self.width(), self.height()) * 0.56
        draw_icon(p, self.icon_name, QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side), colour)


# ── navigation rail ────────────────────────────────────────────────────────────
class NavButton(QAbstractButton):
    """One destination in the left rail: icon over a short label."""

    def __init__(self, icon: str, label: str, tooltip: str = "") -> None:
        super().__init__()
        self.icon_name = icon
        self.label = label
        self.setCheckable(True)
        self.setToolTip(tooltip or label)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(72, 60)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def sizeHint(self) -> QSize:
        return QSize(72, 60)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        hover = self.underMouse()
        if self.isChecked():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(139, 127, 240, 34))
            p.drawRoundedRect(r.adjusted(8, 3, -8, -3), 9, 9)
            p.setBrush(QColor(ACCENT))
            p.drawRoundedRect(QRectF(0, r.center().y() - 11, 3, 22), 1.5, 1.5)
        elif hover:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 12))
            p.drawRoundedRect(r.adjusted(8, 3, -8, -3), 9, 9)
        colour = QColor(ACCENT_300 if self.isChecked() else (TEXT if hover else MUTED))
        draw_icon(p, self.icon_name, QRectF(r.center().x() - 11, 10, 22, 22), colour)
        p.setPen(colour)
        p.setFont(ui_font(10.5, QFont.Weight.Medium if self.isChecked() else QFont.Weight.Normal))
        p.drawText(QRectF(0, 36, r.width(), 16), Qt.AlignmentFlag.AlignCenter, self.label)


class NavRail(QFrame):
    """Vertical navigation; emits the index of the chosen page."""

    changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("navRail")
        self.setFixedWidth(76)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 10, 2, 10)
        self._layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[NavButton] = []
        self._layout.addStretch(1)

    def add(self, icon: str, label: str, tooltip: str = "") -> NavButton:
        button = NavButton(icon, label, tooltip)
        index = len(self._buttons)
        button.toggled.connect(lambda on, i=index: on and self.changed.emit(i))
        self._group.addButton(button)
        self._buttons.append(button)
        self._layout.insertWidget(self._layout.count() - 1, button, 0, Qt.AlignmentFlag.AlignHCenter)
        if index == 0:
            button.setChecked(True)
        return button

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)


# ── switch ─────────────────────────────────────────────────────────────────────
class Switch(QAbstractButton):
    """A toggle switch with the checkable-button API (setChecked / toggled / isChecked)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(34, 19)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _get_knob(self) -> float:
        return self._pos

    def _set_knob(self, value: float) -> None:
        self._pos = value
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool) -> None:  # noqa: N802 - Qt API
        super().setChecked(on)
        if self.signalsBlocked() or not self.isVisible():
            self._anim.stop()
            self._pos = 1.0 if on else 0.0
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(34, 19)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        off_c, on_c = QColor(LINE_STRONG), QColor(ACCENT)
        t = self._pos
        track = QColor(
            int(off_c.red() + (on_c.red() - off_c.red()) * t),
            int(off_c.green() + (on_c.green() - off_c.green()) * t),
            int(off_c.blue() + (on_c.blue() - off_c.blue()) * t),
        )
        if not self.isEnabled():
            track = QColor(RAISED)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.left() + 2 + (r.width() - d - 4) * t
        p.setBrush(QColor("#ffffff" if self.isEnabled() else MUTED))
        p.drawEllipse(QRectF(x, r.top() + 2, d, d))


# ── segmented control ──────────────────────────────────────────────────────────
class Segmented(QFrame):
    """A row of mutually exclusive buttons; emits the chosen option's value."""

    changed = Signal(object)

    def __init__(self, options: list[tuple[str, Any]], tooltips: dict[Any, str] | None = None) -> None:
        super().__init__()
        self.setObjectName("seg")
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[Any, QPushButton] = {}
        for label, value in options:
            button = QPushButton(label)
            button.setObjectName("segBtn")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            if tooltips and value in tooltips:
                button.setToolTip(tooltips[value])
            button.toggled.connect(lambda on, v=value: on and self.changed.emit(v))
            self._group.addButton(button)
            self._buttons[value] = button
            row.addWidget(button, 1)
        if options:
            self._buttons[options[0][1]].setChecked(True)

    def value(self) -> Any:
        for value, button in self._buttons.items():
            if button.isChecked():
                return value
        return None

    def set_value(self, value: Any, emit: bool = False) -> None:
        button = self._buttons.get(value)
        if button is None:
            # no option matches (a "custom" state): clear every button
            self._group.setExclusive(False)
            for b in self._buttons.values():
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)
            self._group.setExclusive(True)
            return
        button.blockSignals(not emit)
        button.setChecked(True)
        button.blockSignals(False)


# ── recipe step ────────────────────────────────────────────────────────────────
class MethodCard(QFrame):
    """One recipe step in the inspector: switch, name, measured effect, settings.

    The settings (``body``) show only while the step is on; the badge and the one-line
    summary come from avl.findings; "i" opens the full finding.
    """

    toggled = Signal(bool)
    info_requested = Signal(str)

    def __init__(self, finding: Finding, checkable: bool = True, title: str | None = None) -> None:
        super().__init__()
        self.finding = finding
        self.setObjectName("stepRow")
        self.setProperty("on", "1")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 11, 12, 11)
        outer.setSpacing(11)

        self.check: Switch | None = None
        lead = QVBoxLayout()
        lead.setContentsMargins(0, 1, 0, 0)
        if checkable:
            self.check = Switch()
            self.check.toggled.connect(self._on_toggled)
            lead.addWidget(self.check)
        else:
            dot = QLabel("●")
            dot.setObjectName("stepDot")
            dot.setFixedWidth(34)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lead.addWidget(dot)
        lead.addStretch(1)
        outer.addLayout(lead)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel(title or finding.title)
        self.title.setObjectName("stepTitle")
        head.addWidget(self.title)
        head.addStretch(1)
        self.badge = QLabel(finding.badge)
        self.badge.setObjectName("impactBadge")
        self.badge.setProperty("status", finding.status)
        self.badge.setToolTip(f"Measured: {finding.badge}  ·  {STATUS_LABEL.get(finding.status, finding.status)}")
        head.addWidget(self.badge)
        info = QPushButton("i")
        info.setObjectName("infoBtn")
        info.setFixedSize(18, 18)
        info.setToolTip("What this step does and what it measured — opens Findings")
        info.setCursor(Qt.CursorShape.PointingHandCursor)
        info.clicked.connect(lambda: self.info_requested.emit(finding.key))
        head.addWidget(info)
        col.addLayout(head)

        self.summary = QLabel(finding.summary)
        self.summary.setObjectName("stepSummary")
        self.summary.setWordWrap(True)
        col.addWidget(self.summary)

        self._body_host = QWidget()
        self.body = QVBoxLayout(self._body_host)
        self.body.setContentsMargins(0, 5, 0, 0)
        self.body.setSpacing(6)
        col.addWidget(self._body_host)

        self.note = QLabel("")
        self.note.setObjectName("methodNote")
        self.note.setWordWrap(True)
        self.note.hide()
        col.addWidget(self.note)
        outer.addLayout(col, 1)
        self._available = True

    def _on_toggled(self, on: bool) -> None:
        self._sync()
        self.toggled.emit(on)

    def _sync(self) -> None:
        on = self.is_on()
        self.setProperty("on", "1" if on else "0")
        for widget in (self, self.title, self.summary):
            repolish(widget)
        self._body_host.setVisible(on)

    def is_on(self) -> bool:
        """Switched on and fed by the selected data — what a run will use."""
        if self.check is None:
            return self._available
        return self._available and self.check.isChecked()

    def is_checked(self) -> bool:
        """Switched on, whether or not the selected data can feed it."""
        return True if self.check is None else self.check.isChecked()

    @property
    def available(self) -> bool:
        return self._available

    def set_checked(self, on: bool) -> None:
        if self.check is None:
            return
        self.check.blockSignals(True)
        self.check.setChecked(on)
        self.check.blockSignals(False)
        self._sync()

    def set_available(self, ok: bool, reason: str = "") -> None:
        """Grey the step out when the selected flight lacks what it needs."""
        self._available = ok
        if self.check is not None:
            self.check.setEnabled(ok)
        self.note.setText(reason)
        self.note.setVisible(bool(reason))
        self._sync()

    def set_note(self, text: str) -> None:
        self.note.setText(text)
        self.note.setVisible(bool(text))


# ── stat tile ──────────────────────────────────────────────────────────────────
class StatTile(QFrame):
    """label · big value · delta vs a named baseline · footnote (dataviz stat tile)."""

    clicked = Signal()

    def __init__(self, label: str, hero: bool = False) -> None:
        super().__init__()
        self.setObjectName("statTile")
        self.setProperty("hero", "1" if hero else "0")
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 14)
        v.setSpacing(3)
        self.label = QLabel(label)
        self.label.setObjectName("tileLabel")
        v.addWidget(self.label)
        row = QHBoxLayout()
        row.setSpacing(5)
        self.value = QLabel("—")
        self.value.setObjectName("tileValueHero" if hero else "tileValue")
        self.unit = QLabel("")
        self.unit.setObjectName("tileUnit")
        row.addWidget(self.value, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addStretch(1)
        v.addLayout(row)
        self.delta = QLabel("")
        self.delta.setObjectName("tileDelta")
        self.delta.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.delta)
        self.foot = QLabel("")
        self.foot.setObjectName("tileFoot")
        self.foot.setWordWrap(True)
        v.addWidget(self.foot)
        v.addStretch(1)

    def set(self, value: str, unit: str = "", delta: str = "", foot: str = "") -> None:
        self.value.setText(value)
        self.unit.setText(unit)
        self.delta.setText(delta)
        self.delta.setVisible(bool(delta))
        self.foot.setText(foot)
        self.foot.setVisible(bool(foot))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)


def delta_html(value: float | None, unit: str, baseline: str, higher_is_better: bool = True) -> str:
    """'▲ +17.4 pts  vs baseline' — direction glyph + sign, status colour by goodness."""
    if value is None or value != value:
        return ""
    if abs(value) < 0.05:
        return f"<span style='color:{MUTED}'>= {baseline}</span>"
    good = (value > 0) == higher_is_better
    glyph = "▲" if value > 0 else "▼"
    colour = GOOD if good else BAD
    return (f"<span style='color:{colour}'>{glyph} {value:+.1f}{unit}</span>"
            f"<span style='color:{MUTED}'>&nbsp; vs {baseline}</span>")


# ── pipeline flow ──────────────────────────────────────────────────────────────
@dataclass
class FlowNode:
    key: str
    title: str
    value: str = "-"
    detail: str = ""
    off: bool = False
    phase: str = "idle"  # idle | pending | active | done


class FlowStrip(QWidget):
    """The pipeline as a row of stages that always fits its width: text is elided,
    never clipped; switched-off stages are hollow; the running stage glows."""

    clicked = Signal(str)

    def __init__(self, nodes: list[FlowNode], height: int = 96) -> None:
        super().__init__()
        self.nodes = nodes
        self._by_key = {n.key: n for n in nodes}
        self.setFixedHeight(height)
        self.setMouseTracking(True)
        self.setMinimumWidth(560)
        self._hover: str | None = None
        self.clickable: set[str] = set()

    def node(self, key: str) -> FlowNode | None:
        return self._by_key.get(key)

    def update_node(self, key: str, **fields) -> None:
        node = self._by_key.get(key)
        if node is None:
            return
        for name, value in fields.items():
            if value is not None:
                setattr(node, name, value)
        self.update()

    def _rects(self) -> list[QRectF]:
        n = max(1, len(self.nodes))
        gap = 18.0
        w = (self.width() - gap * (n - 1)) / n
        return [QRectF(i * (w + gap), 4, w, self.height() - 8) for i in range(n)]

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rects = self._rects()
        title_f = ui_font(10.5, QFont.Weight.DemiBold)
        value_f = ui_font(13.5, QFont.Weight.Medium)
        detail_f = ui_font(11)
        dfm = QFontMetrics(detail_f)
        for i, (node, r) in enumerate(zip(self.nodes, rects)):
            if i:
                prev = rects[i - 1]
                cx = (prev.right() + r.left()) / 2
                cy = r.center().y()
                pen = QPen(QColor(LINE_STRONG), 1.6)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawPolyline([QPointF(cx - 3, cy - 5), QPointF(cx + 2, cy), QPointF(cx - 3, cy + 5)])

            active = node.phase == "active"
            if active:
                for k, alpha in ((8, 16), (4, 36)):
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(139, 127, 240, alpha))
                    p.drawRoundedRect(r.adjusted(-k / 2, -k / 2, k / 2, k / 2), 13, 13)
            if node.off:
                p.setBrush(Qt.BrushStyle.NoBrush)
                border = QColor(LINE)
            else:
                p.setBrush(QColor("#1b1937" if active else PANEL))
                border = QColor(ACCENT if active else (LINE_STRONG if self._hover == node.key else LINE))
            p.setPen(QPen(border, 1))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 11, 11)
            if not node.off and not active:
                # a short tick on the top edge marks the stage as on (green once done)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(GOOD if node.phase == "done" else ACCENT))
                p.drawRoundedRect(QRectF(r.left() + 14, r.top() + 0.5, 20, 2.5), 1.2, 1.2)

            inner = r.adjusted(14, 12, -12, -8)
            p.setFont(title_f)
            p.setPen(QColor(FAINT if node.off else MUTED))
            title = ("✓  " if node.phase == "done" else "") + node.title.upper()
            off_w = dfm.horizontalAdvance("off") + 14 if node.off else 0
            p.drawText(QRectF(inner.left(), inner.top(), inner.width() - off_w, 15), Qt.AlignmentFlag.AlignLeft,
                       QFontMetrics(title_f).elidedText(title, Qt.TextElideMode.ElideRight, int(inner.width() - off_w)))
            p.setFont(value_f)
            p.setPen(QColor(FAINT if node.off else TEXT))
            fm = QFontMetrics(value_f)
            p.drawText(QRectF(inner.left(), inner.top() + 19, inner.width(), 20), Qt.AlignmentFlag.AlignLeft,
                       fm.elidedText(node.value, Qt.TextElideMode.ElideRight, int(inner.width())))
            p.setFont(detail_f)
            p.setPen(QColor(FAINT if node.off else MUTED))
            for j, line in enumerate(_wrap(node.detail, dfm, inner.width(), 2)):
                p.drawText(QRectF(inner.left(), inner.top() + 43 + j * 15, inner.width(), 15),
                           Qt.AlignmentFlag.AlignLeft, line)
            if node.off:
                tr = QRectF(r.right() - off_w - 8, r.top() + 10, off_w, 16)
                p.setPen(QPen(QColor(LINE_STRONG), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(tr, 8, 8)
                p.setPen(QColor(FAINT))
                p.drawText(tr, Qt.AlignmentFlag.AlignCenter, "off")

    def _key_at(self, pos: QPointF) -> str | None:
        for node, r in zip(self.nodes, self._rects()):
            if r.contains(pos):
                return node.key
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        key = self._key_at(event.position())
        if key != self._hover:
            self._hover = key
            self.setCursor(Qt.CursorShape.PointingHandCursor if key in self.clickable else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = None
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        key = self._key_at(event.position())
        if key in self.clickable:
            self.clicked.emit(key)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            key = self._key_at(QPointF(event.pos()))
            node = self._by_key.get(key) if key else None
            if node is not None:
                QToolTip.showText(event.globalPos(), f"{node.title}\n{node.value}\n{node.detail}", self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)


def _wrap(text: str, fm: QFontMetrics, width: float, max_lines: int) -> list[str]:
    words = (text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if fm.horizontalAdvance(trial) <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                current = ""
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if lines and " ".join(lines) != " ".join(words):
        lines[-1] = fm.elidedText(lines[-1] + " …", Qt.TextElideMode.ElideRight, int(width))
    return [fm.elidedText(line, Qt.TextElideMode.ElideRight, int(width)) for line in lines]


# ── finding card (Findings page) ───────────────────────────────────────────────
class FindingCard(QFrame):
    """The full story of one method: idea, what we did, numbers, cost, needs."""

    use_requested = Signal(str)

    def __init__(self, finding: Finding, can_use: bool) -> None:
        super().__init__()
        self.setObjectName("findingCard")
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 16)
        v.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel(finding.title)
        title.setObjectName("findingTitle")
        head.addWidget(title)
        head.addStretch(1)
        status = QLabel(f"{STATUS_GLYPH.get(finding.status, '')}  {STATUS_LABEL.get(finding.status, finding.status)}")
        status.setObjectName("statusTag")
        status.setProperty("status", finding.status)
        head.addWidget(status)
        v.addLayout(head)

        badge = QLabel(finding.badge)
        badge.setObjectName("findingBadge")
        v.addWidget(badge)

        for label, text in (("IDEA", finding.idea), ("WHAT WE DO", finding.did)):
            cap = QLabel(label)
            cap.setObjectName("overline")
            body = QLabel(text)
            body.setObjectName("findingText")
            body.setWordWrap(True)
            v.addWidget(cap)
            v.addWidget(body)

        if finding.numbers:
            grid = QFrame()
            grid.setObjectName("findingNumbers")
            g = QVBoxLayout(grid)
            g.setContentsMargins(12, 9, 12, 9)
            g.setSpacing(6)
            for setting, result in finding.numbers:
                line = QHBoxLayout()
                a = QLabel(setting)
                a.setObjectName("numSetting")
                a.setWordWrap(True)
                b = QLabel(result)
                b.setObjectName("numResult")
                b.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                b.setWordWrap(True)
                line.addWidget(a, 3)
                line.addWidget(b, 4)
                g.addLayout(line)
            v.addWidget(grid)

        foot = QHBoxLayout()
        meta = QLabel(
            f"<span style='color:{FAINT}'>COST</span>&nbsp; {finding.cost}"
            f"<br><span style='color:{FAINT}'>NEEDS</span>&nbsp; {finding.needs}"
        )
        meta.setObjectName("findingMeta")
        meta.setWordWrap(True)
        meta.setTextFormat(Qt.TextFormat.RichText)
        foot.addWidget(meta, 1)
        if can_use:
            use = QPushButton("Use in recipe")
            use.setObjectName("secondary")
            use.setCursor(Qt.CursorShape.PointingHandCursor)
            use.clicked.connect(lambda: self.use_requested.emit(finding.key))
            foot.addWidget(use, 0, Qt.AlignmentFlag.AlignBottom)
        v.addLayout(foot)
        section = QLabel(f"report §{finding.section}")
        section.setObjectName("caption")
        v.addWidget(section)


# ── output panel ───────────────────────────────────────────────────────────────
class LogDrawer(QFrame):
    """The subprocess output, shared by every page; opened from the status bar."""

    toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("logDrawer")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(16, 6, 10, 6)
        bar.setSpacing(10)
        title = QLabel("OUTPUT")
        title.setObjectName("panelTitle")
        self.last = QLabel("")
        self.last.setObjectName("logLast")
        self.last.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        clear = QPushButton("Clear")
        clear.setObjectName("ghost")
        close = QPushButton("Hide")
        close.setObjectName("ghost")
        bar.addWidget(title)
        bar.addWidget(self.last, 1)
        bar.addWidget(clear)
        bar.addWidget(close)
        v.addLayout(bar)
        self.text = QPlainTextEdit()
        self.text.setObjectName("log")
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)
        self.text.setFixedHeight(200)
        v.addWidget(self.text)
        clear.clicked.connect(self.text.clear)
        close.clicked.connect(lambda: self.set_open(False))
        self.hide()

    def set_open(self, on: bool) -> None:
        self.setVisible(on)
        self.toggled.emit(on)

    def append(self, message: str) -> None:
        self.text.appendPlainText(message)
        self.last.setText(message.strip().splitlines()[-1] if message.strip() else "")

    def open(self) -> None:
        self.set_open(True)


# ── table delegates ────────────────────────────────────────────────────────────
CHIPS_ROLE = Qt.ItemDataRole.UserRole + 11
FRACTION_ROLE = Qt.ItemDataRole.UserRole + 12


class ChipsDelegate(QStyledItemDelegate):
    """Paints the recipe steps of a run as small pills."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        self.initStyleOption(option, index)
        chips = index.data(CHIPS_ROLE) or []
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(139, 127, 240, 38))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = ui_font(11)
        painter.setFont(font)
        fm = QFontMetrics(font)
        x = option.rect.left() + 8
        h = fm.height() + 5
        y = option.rect.center().y() - h / 2
        right = option.rect.right() - 6
        for n, chip in enumerate(chips):
            w = fm.horizontalAdvance(chip) + 14
            if x + w > right:
                more = f"+{len(chips) - n}"
                painter.setPen(QColor(MUTED))
                painter.drawText(QRectF(x, y, fm.horizontalAdvance(more) + 6, h), Qt.AlignmentFlag.AlignVCenter, more)
                break
            accent = chip.startswith("ensemble")
            rect = QRectF(x, y, w, h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(139, 127, 240, 46) if accent else QColor(RAISED))
            painter.drawRoundedRect(rect, h / 2, h / 2)
            painter.setPen(QColor(ACCENT_300 if accent else TEXT_2))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, chip)
            x += w + 5
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        chips = index.data(CHIPS_ROLE) or []
        fm = QFontMetrics(ui_font(11))
        width = sum(fm.horizontalAdvance(c) + 14 for c in chips) + 5 * len(chips) + 16
        return QSize(min(max(width, 60), 360), fm.height() + 18)


class BarDelegate(QStyledItemDelegate):
    """A percentage as text plus a thin in-cell bar on a 0–100 % track."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        self.initStyleOption(option, index)
        fraction = index.data(FRACTION_ROLE)
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(139, 127, 240, 38))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(10, 0, -10, 0)
        font = ui_font(13, QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance("100.0%") + 8
        painter.setPen(QColor(TEXT))
        painter.drawText(QRectF(rect.left(), rect.top(), text_w, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, index.data() or "")
        if fraction is not None and fraction == fraction:
            track = QRectF(rect.left() + text_w, rect.center().y() - 2.5, max(20.0, rect.width() - text_w), 5)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(LINE))
            painter.drawRoundedRect(track, 2.5, 2.5)
            filled = QRectF(track.left(), track.top(),
                            track.width() * max(0.0, min(1.0, float(fraction))), track.height())
            painter.setBrush(QColor(SERIES_TOP1))
            painter.drawRoundedRect(filled, 2.5, 2.5)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        fm = QFontMetrics(ui_font(13, QFont.Weight.DemiBold))
        return QSize(fm.horizontalAdvance("100.0%") + 120, fm.height() + 18)


# ── progress ladder ────────────────────────────────────────────────────────────
@dataclass
class LadderStep:
    label: str
    top1: float | None = None      # fraction, top-1 within 100 m
    fused: float | None = None     # fraction, fused top-5 within 100 m
    detail: str = ""               # tooltip: which run, frames, errors
    note: str = ""                 # text when the step has no run
    ensemble: bool = False


class LadderChart(QWidget):
    """The report's "progress, step by step" as horizontal bars, from saved runs.

    One row per recipe step: the top-1 bar (series 1) with its value past the tip, the
    fused result as a dot (series 2), and — in the full version — a right-hand column
    with both numbers and the change from the previous step, the chart's table view.
    """

    def __init__(self, compact: bool = False) -> None:
        super().__init__()
        self.steps: list[LadderStep] = []
        self.empty_text = "no runs on this map yet"
        self.compact = compact
        self.LABEL_W = 178 if compact else 230
        self.RIGHT_W = 0 if compact else 200
        self.ROW_H = 30 if compact else 40
        self.BAR_H = 12 if compact else 16
        self.TOP = 30
        self.AXIS_H = 24
        self.setMouseTracking(True)
        self.setMinimumHeight(self.TOP + self.AXIS_H + self.ROW_H)

    def set_steps(self, steps: list[LadderStep]) -> None:
        self.steps = steps
        self.setMinimumHeight(self.TOP + self.AXIS_H + self.ROW_H * max(1, len(steps)) + 6)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(700, self.minimumHeight())

    def _plot_rect(self) -> QRectF:
        right_pad = (self.RIGHT_W + 54) if self.RIGHT_W else 52
        return QRectF(
            self.LABEL_W, self.TOP, max(120.0, self.width() - self.LABEL_W - right_pad),
            self.ROW_H * max(1, len(self.steps)),
        )

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = self._plot_rect()
        font = ui_font(12.5)
        small = ui_font(11)
        fm = QFontMetrics(font)
        sfm = QFontMetrics(small)

        # legend (two series -> always a legend)
        p.setFont(small)
        lx = plot.left()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(SERIES_TOP1))
        p.drawRoundedRect(QRectF(lx, 9, 16, 8), 2, 2)
        p.setPen(QColor(TEXT_2))
        text = "top-1 within 100 m"
        p.drawText(QPointF(lx + 22, 17), text)
        lx += 22 + sfm.horizontalAdvance(text) + 20
        self._dot(p, QPointF(lx + 5, 13), SERIES_FUSED)
        p.setPen(QColor(TEXT_2))
        p.drawText(QPointF(lx + 16, 17), "fused top-5 within 100 m")

        # gridlines 0..100 %, hairline, solid
        for pct in (0, 25, 50, 75, 100):
            x = plot.left() + plot.width() * pct / 100
            p.setPen(QPen(QColor(LINE), 1))
            p.drawLine(QPointF(x, plot.top() - 4), QPointF(x, plot.bottom()))
            p.setPen(QColor(FAINT))
            label = f"{pct}%"
            p.drawText(QPointF(x - sfm.horizontalAdvance(label) / 2, plot.bottom() + 16), label)

        if not self.steps:
            p.setPen(QColor(FAINT))
            p.setFont(font)
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return

        previous: float | None = None
        for i, step in enumerate(self.steps):
            top = plot.top() + i * self.ROW_H
            cy = top + self.ROW_H / 2
            p.setFont(font)
            p.setPen(QColor(TEXT if step.top1 is not None else MUTED))
            label_rect = QRectF(0, top, self.LABEL_W - 14, self.ROW_H)
            p.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       fm.elidedText(step.label, Qt.TextElideMode.ElideRight, int(label_rect.width())))

            if step.top1 is None:
                p.setFont(small)
                p.setPen(QColor(FAINT))
                p.drawText(QRectF(plot.left() + 6, top, plot.width(), self.ROW_H),
                           Qt.AlignmentFlag.AlignVCenter, step.note or "not run")
                continue

            # bar: square at the baseline, 4 px rounded data end, <= 24 px thick
            width = plot.width() * max(0.0, min(1.0, step.top1))
            bar = QRectF(plot.left(), cy - self.BAR_H / 2, max(width, 2.0), self.BAR_H)
            path = QPainterPath()
            r = min(4.0, bar.width() / 2)
            path.moveTo(bar.left(), bar.top())
            path.lineTo(bar.right() - r, bar.top())
            path.quadTo(bar.right(), bar.top(), bar.right(), bar.top() + r)
            path.lineTo(bar.right(), bar.bottom() - r)
            path.quadTo(bar.right(), bar.bottom(), bar.right() - r, bar.bottom())
            path.lineTo(bar.left(), bar.bottom())
            path.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(SERIES_TOP1))
            p.drawPath(path)

            # fused dot, then the value past the bar tip — or past the dot when the
            # dot sits on the tip, so the label never collides with a mark
            label_x = bar.right() + 8
            if step.fused is not None:
                fx = plot.left() + plot.width() * max(0.0, min(1.0, step.fused))
                self._dot(p, QPointF(fx, cy), SERIES_FUSED)
                if fx + 7.5 > label_x - 2:
                    label_x = max(label_x, fx + 12)
            p.setFont(small)
            p.setPen(QColor(TEXT))
            value = f"{100 * step.top1:.1f}%"
            p.drawText(QPointF(label_x, cy + 4), value)

            if self.RIGHT_W:
                right = QRectF(plot.right() + 54, top, self.RIGHT_W, self.ROW_H)
                p.setPen(QColor(TEXT_2))
                fused = f"fused {100 * step.fused:.1f}%" if step.fused is not None else "fused —"
                p.drawText(right, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           f"{value}  ·  {fused}")
                if previous is not None:
                    delta = 100 * (step.top1 - previous)
                    p.setPen(QColor(GOOD if delta > 0.05 else BAD if delta < -0.05 else MUTED))
                    glyph = "▲" if delta > 0.05 else "▼" if delta < -0.05 else "="
                    p.drawText(right.adjusted(0, 0, -4, 0),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                               f"{glyph} {delta:+.1f}")
            previous = step.top1

    @staticmethod
    def _dot(p: QPainter, centre: QPointF, colour: str) -> None:
        p.setPen(QPen(QColor(PANEL), 2))  # 2 px surface ring
        p.setBrush(QColor(colour))
        p.drawEllipse(centre, 5.5, 5.5)

    def event(self, event) -> bool:  # tooltips per row, generous hit area
        if event.type() == QEvent.Type.ToolTip:
            plot = self._plot_rect()
            row = int((event.pos().y() - plot.top()) // self.ROW_H)
            if 0 <= row < len(self.steps) and event.pos().y() >= plot.top():
                step = self.steps[row]
                QToolTip.showText(event.globalPos(), step.detail or step.note or step.label, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)


class ActionButton(QPushButton):
    """A text button with a painted icon on its left (Run, Stop)."""

    def __init__(self, icon: str, text: str, object_name: str) -> None:
        super().__init__(text)
        self.icon_name = icon
        self.setObjectName(object_name)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            colour = QColor(FAINT)
        elif self.objectName() == "runBtn":
            colour = QColor("#ffffff")
        else:
            colour = QColor(TEXT_2)
        side = 13.0
        draw_icon(p, self.icon_name, QRectF(12, (self.height() - side) / 2, side, side), colour)


