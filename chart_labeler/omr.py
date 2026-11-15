"""Runs the homr optical-music-recognition models on one page image.

homr produces two independent readings of a page:
  * a transformer reading: the symbol sequence per staff (pitches with their
    effective accidentals, clefs, key signatures) with approximate positions
  * a segmentation reading: precise notehead blobs and staff line geometry

This module collects both, scaled into PDF points.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Token:
    rhythm: str  # e.g. note_4, chord, barline, clef_F4, keySignature_-5
    pitch: str  # e.g. D2 (or "." / "_" when not a note)
    lift: str  # effective accidental: "_" natural, "b", "#", "bb", "##", "N"
    position: str  # upper, upper2, lower, lower2
    xy: tuple[float, float] | None  # approximate, from model attention
    duration: float = 0.0  # in whole notes; 0 for grace notes and non-notes


@dataclass
class Notehead:
    x: float
    y: float
    w: float
    h: float
    step: float | None = None  # steps above the bottom staff line, measured in the image
    added: bool = False  # found by chart_labeler.heads, not by homr


@dataclass
class StaffRead:
    """One 5-line staff. `grid` is a list of (x, [y of 5 lines top to bottom])."""

    grid: list[tuple[float, list[float]]]
    is_lower_of_grand: bool
    is_upper_of_grand: bool
    tokens: list[Token] = field(default_factory=list)
    noteheads: list[Notehead] = field(default_factory=list)
    last_in_system: bool = True  # no other staff of the same system lies below it

    def _fit(self) -> tuple[np.ndarray, float, float]:
        """Least-squares model of the staff, robust to misdetected line points.

        Individual grid points are noisy by up to half a line spacing. The staff
        as a whole is a smooth curve: centre(x) quadratic, spacing(x) linear.
        """
        cached = getattr(self, "_model", None)
        if cached is not None:
            return cached
        xs = np.array([g[0] for g in self.grid], dtype=float)
        ys = np.array([g[1] for g in self.grid], dtype=float)
        x0, span = float(xs.mean()), float(max(np.ptp(xs), 1.0))
        xn = np.repeat((xs - x0) / span, 5)
        k = np.tile(np.arange(5) - 2.0, len(xs))
        y = ys.reshape(-1)
        cols = [np.ones_like(xn), xn, xn**2, k, k * xn] if len(xs) >= 4 else [np.ones_like(xn), k]
        design = np.stack(cols, axis=1)
        keep = np.ones(len(y), dtype=bool)
        coef = np.zeros(design.shape[1])
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
            residual = np.abs(design @ coef - y)
            spacing = abs(coef[3] if len(cols) > 2 else coef[1])
            keep = residual <= max(0.12 * spacing, 2.5 * float(np.median(residual[keep])))
            if keep.sum() < design.shape[1] + 2:
                break
        self._model = (coef, x0, span)
        return self._model

    def lines_at(self, x: float) -> list[float]:
        coef, x0, span = self._fit()
        xn = (x - x0) / span
        if len(coef) == 2:
            centre, spacing = coef[0], coef[1]
        else:
            centre = coef[0] + coef[1] * xn + coef[2] * xn**2
            spacing = coef[3] + coef[4] * xn
        return [float(centre + (i - 2) * spacing) for i in range(5)]

    def unit_at(self, x: float) -> float:
        lines = self.lines_at(x)
        return (lines[4] - lines[0]) / 4

    def step_at(self, x: float, y: float) -> float:
        """Diatonic steps above the bottom line (fractional)."""
        lines = self.lines_at(x)
        return (lines[4] - y) / ((lines[4] - lines[0]) / 8)

    def y_of_step(self, x: float, step: float) -> float:
        lines = self.lines_at(x)
        return lines[4] - step * (lines[4] - lines[0]) / 8

    @property
    def x_range(self) -> tuple[float, float]:
        return self.grid[0][0], self.grid[-1][0]


_initialised = False


def _init() -> tuple[bool, bool]:
    global _initialised
    from homr import main as hm

    seg_gpu = False  # CPU is fast enough and avoids CoreML shape warnings
    if not _initialised:
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        hm.download_weights(seg_gpu, False, False)
        _initialised = True
    return seg_gpu, False


def read_page(image_path: str, scale: float = 1.0, verbose: bool = False) -> list[StaffRead]:
    """Returns every staff on the page, top to bottom.

    Coordinates are image pixels multiplied by `scale`.
    """
    from chart_labeler.heads import locate_noteheads

    result = read_staffs(image_path, scale, verbose)
    locate_noteheads(image_path, result, scale)
    return result


def read_staffs(image_path: str, scale: float = 1.0, verbose: bool = False) -> list[StaffRead]:
    """homr's own reading of the page: the slow part of read_page."""
    from homr import main as hm
    from homr.staff_parsing import parse_staff_image
    from homr.staff_regions import StaffRegions
    from homr.system_repair import repair_systems
    from homr.transformer.configs import Config

    seg_gpu, tr_gpu = _init()
    config = hm.ProcessingConfig(False, False, False, False, -1, tr_gpu, seg_gpu, False, False)
    sink = io.StringIO()
    with contextlib.ExitStack() as stack:
        if not verbose:
            stack.enter_context(contextlib.redirect_stderr(sink))
            stack.enter_context(contextlib.redirect_stdout(sink))
        multi_staffs, image, debug, _, _, to_image = hm.detect_staffs_in_image(image_path, config)

        def to_input(point: tuple[float, float]) -> tuple[float, float]:
            x, y = to_image(point)
            return x * scale, y * scale

        multi_staffs = repair_systems(debug, multi_staffs)
        tconf = Config()
        tconf.use_gpu_inference = tr_gpu
        regions = StaffRegions(multi_staffs)
        result: list[StaffRead] = []
        index = 0
        for multi in multi_staffs:
            for staff in multi.staffs:
                symbols = parse_staff_image(debug, index, staff, image, regions, tconf, to_input)
                index += 1
                reads = _convert(staff, symbols, to_input)
                for read in reads:
                    read.last_in_system = False
                result.extend(reads)
            if multi.staffs:
                result[-1].last_in_system = True
    return result


def _convert(staff, symbols, to_input) -> list[StaffRead]:
    def tok(s) -> Token:
        xy = s.image_coordinates
        duration = float(s.get_duration().fraction) if s.rhythm.startswith(("note", "rest")) else 0.0
        point = (float(xy[0]), float(xy[1])) if xy else None
        return Token(s.rhythm, s.pitch, s.lift, s.position, point, duration)

    def grid_of(points, sl) -> list[tuple[float, list[float]]]:
        out = []
        for p in points:
            pts = [to_input((p.x, y)) for y in p.y[sl]]
            out.append((float(np.mean([q[0] for q in pts])), [float(q[1]) for q in pts]))
        return sorted(out)

    def heads(st) -> list[Notehead]:
        out = []
        for n in st.get_notes():
            cx, cy = to_input(n.center)
            ex, ey = to_input((n.center[0] + n.box.size[0], n.center[1] + n.box.size[1]))
            out.append(Notehead(float(cx), float(cy), float(abs(ex - cx)), float(abs(ey - cy))))
        return sorted(out, key=lambda h: h.x)

    tokens = [tok(s) for s in symbols]
    if staff.is_grandstaff:
        split = staff.split()
        if split is not None:
            upper, lower = split
            pts = [p for p in staff.grid if len(p.y) == 10]
            up = StaffRead(grid_of(pts, slice(0, 5)), False, True, tokens, heads(upper))
            lo = StaffRead(grid_of(pts, slice(5, 10)), True, False, tokens, heads(lower))
            return [up, lo]
    return [StaffRead(grid_of(staff.grid, slice(0, 5)), False, False, tokens, heads(staff))]
