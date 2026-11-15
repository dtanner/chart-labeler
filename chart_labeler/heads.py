"""Finds noteheads directly in the page image and measures their staff position.

homr's own notehead search stops a few ledger lines away from the staff, so
the low notes of bass octaves (C1, D1, ...) are not found, and those are also
the notes its transformer misreads most. This module looks for every notehead
around a staff by matching the page's own notehead shapes (filled and hollow), and
measures each one against the staff lines as they are actually printed next to
it, which stays accurate many ledger lines away.
"""

from __future__ import annotations

import cv2
import numpy as np

from chart_labeler.omr import Notehead, StaffRead

MATCH_SCORE = 0.72  # template correlation that counts as a notehead
MIN_EXAMPLES = 6  # known noteheads needed to learn the page's notehead shape
MIN_HOLLOW = 3  # the same for half and whole notes, which are rarer
REACH_ABOVE = 5.0  # search range around a staff, in line spacings
REACH_BELOW = 10.0


def locate_noteheads(image_path: str, staffs: list[StaffRead], scale: float) -> None:
    """Adds the noteheads homr missed to every staff and measures `Notehead.step` of all.

    `scale` converts image pixels to the staff coordinates (as in read_page).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or not staffs:
        return
    dark = (255.0 - image.astype(np.float32)) / 255.0
    k = 1.0 / scale  # staff coordinates -> pixels
    templates = [t for t in (_template(dark, staffs, k, True), _template(dark, staffs, k, False)) if t is not None]
    ordered = sorted(staffs, key=lambda s: s.lines_at(s.x_range[0])[0])
    for index, staff in enumerate(ordered):
        x0, x1 = staff.x_range
        middle = (x0 + x1) / 2
        lines, unit = staff.lines_at(middle), staff.unit_at(middle)
        if templates and not staff.is_upper_of_grand:
            top, bottom = lines[0] - REACH_ABOVE * unit, lines[4] + REACH_BELOW * unit
            if index > 0:
                top = max(top, (ordered[index - 1].lines_at(middle)[4] + lines[0]) / 2)
            if index + 1 < len(ordered):
                bottom = min(bottom, ordered[index + 1].lines_at(middle)[0] - unit)
            for template in templates:
                _add_matches(dark, template, staff, (x0, top, x1, bottom), k)
        for head in staff.noteheads:
            head.step = _precise_step(dark, staff, head.x, head.y, k)


def _template(dark: np.ndarray, staffs: list[StaffRead], k: float, filled: bool) -> np.ndarray | None:
    """The typical filled or hollow notehead of this page, learned from the ones homr found."""
    examples = [(s, h) for s in staffs for h in s.noteheads if not h.added]
    if len(examples) < MIN_EXAMPLES:
        return None
    unit = float(np.median([s.unit_at(h.x) for s, h in examples])) * k
    half_w, half_h = int(round(0.85 * unit)), int(round(0.6 * unit))
    patches = []
    for staff, head in examples:
        if abs(staff.unit_at(head.x) * k - unit) > 0.12 * unit:
            continue  # a cue-size staff has smaller noteheads
        cx, cy = int(round(head.x * k)), int(round(head.y * k))
        patch = dark[cy - half_h : cy + half_h + 1, cx - half_w : cx + half_w + 1]
        if patch.shape != (2 * half_h + 1, 2 * half_w + 1):
            continue
        centre = patch[half_h - 1 : half_h + 2, half_w - 1 : half_w + 2].mean()
        if (centre > 0.7) if filled else (centre < 0.35 and patch.mean() > 0.2):
            patches.append(patch)
    if len(patches) < (MIN_EXAMPLES if filled else MIN_HOLLOW):
        return None
    return np.median(np.stack(patches), axis=0).astype(np.float32)


def _add_matches(dark: np.ndarray, template: np.ndarray, staff: StaffRead,
                 box: tuple[float, float, float, float], k: float) -> None:
    th, tw = template.shape
    xa, ya = max(int(box[0] * k) - tw, 0), max(int(box[1] * k), 0)
    xb, yb = min(int(box[2] * k) + tw, dark.shape[1]), min(int(box[3] * k), dark.shape[0])
    if yb - ya <= th or xb - xa <= tw:
        return
    score = cv2.matchTemplate(dark[ya:yb, xa:xb], template, cv2.TM_CCOEFF_NORMED)
    while True:
        _, best, _, (px, py) = cv2.minMaxLoc(score)
        if best < MATCH_SCORE:
            break
        score[max(py - th // 2, 0) : py + th // 2 + 1, max(px - tw // 2, 0) : px + tw // 2 + 1] = 0
        x, y = (px + xa + tw // 2) / k, (py + ya + th // 2) / k
        unit = staff.unit_at(x)
        # Noteheads homr found itself are left exactly as they are.
        if any(abs(h.x - x) < 0.8 * unit and abs(h.y - y) < 1.0 * unit for h in staff.noteheads):
            continue
        staff.noteheads.append(Notehead(x, y, tw / k / 2, th / k / 2, added=True))
    staff.noteheads.sort(key=lambda h: h.x)


def _precise_step(dark: np.ndarray, staff: StaffRead, x: float, y: float, k: float) -> float:
    """Diatonic steps of (x, y) above the bottom staff line, from the printed lines.

    homr's staff model is good to a fraction of a line spacing, which is not
    enough ten steps below the staff. Here the five lines are found in the
    image right beside the note: the top line position and the line spacing
    that put the most ink under all five lines.
    """
    model = staff.lines_at(x)
    unit = (model[4] - model[0]) / 4
    x0, x1 = staff.x_range
    xa, xb = int(max(x - 10 * unit, x0) * k), int(min(x + 10 * unit, x1) * k)
    ya, yb = int((model[0] - 1.5 * unit) * k), int((model[4] + 1.5 * unit) * k)
    if xb - xa < 4 or ya < 0 or yb >= dark.shape[0]:
        return staff.step_at(x, y)
    # A staff line is dark along the whole window; beams, stems and noteheads
    # are not, so a low quantile across the columns keeps only the lines.
    profile = np.quantile(dark[ya:yb, xa:xb], 0.3, axis=1)
    rows = np.arange(len(profile), dtype=float)
    best = (0.0, model[0] * k - ya, unit * k)
    for spacing in np.linspace(0.92 * unit * k, 1.08 * unit * k, 33):
        for top in np.arange((model[0] - 0.6 * unit) * k - ya, (model[0] + 0.6 * unit) * k - ya, 0.25):
            ink = float(np.interp(top + spacing * np.arange(5), rows, profile).sum())
            if ink > best[0]:
                best = (ink, top, spacing)
    ink, top, spacing = best
    if ink < 5 * 0.25:  # no clear lines here (a faint or tilted scan): keep the model
        return staff.step_at(x, y)
    # Sharpen each line to its ink centre, then fit top and spacing to all five.
    centres = []
    for i in range(5):
        near = np.abs(rows - (top + i * spacing)) <= 0.3 * spacing
        weight = profile[near]
        if weight.sum() <= 0:
            return staff.step_at(x, y)
        centres.append(float((rows[near] * weight).sum() / weight.sum()))
    spacing, top = np.polyfit(np.arange(5), centres, 1)
    bottom = ya + top + 4 * spacing
    return (bottom - y * k) / (spacing / 2)
