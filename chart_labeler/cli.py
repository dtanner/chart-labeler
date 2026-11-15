"""chart-labeler: writes the name of each bass note onto a PDF song chart."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymupdf

from chart_labeler.label import (
    Label,
    StaffLabels,
    apply_key_signatures,
    group_by_staff,
    labels_for_staff,
    merge_readings,
)
from chart_labeler.omr import prepare, read_page

# Each page is read once per resolution; the readings then vote.
RESOLUTIONS = (200, 150, 260)
# --fast reads once; 150 dpi was the most accurate single reading on the benchmark.
FAST_RESOLUTIONS = (150,)
VERIFIED = (0.85, 0.05, 0.05)  # red
UNVERIFIED = (0.95, 0.5, 0.0)  # orange
FONT = "hebo"  # Helvetica Bold, built into every PDF viewer
CAP_HEIGHT = 0.72
SHRINKS = 3  # times crowded labels may be made smaller


INK_DPI = 100


def _ink_map(page: pymupdf.Page) -> np.ndarray:
    """True where the page, as displayed, has dark print."""
    pix = page.get_pixmap(dpi=INK_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width) < 150


def _find_spot(ink: np.ndarray, label: Label, width: float, height: float, unit: float) -> tuple[float, float]:
    """Picks the label's centre x and top y so that it covers no print.

    Tries, in order: just under the note, beside a note that hangs below the
    staff, then further down. Falls back to the least covered of those.
    """
    k = INK_DPI / 72
    below = max(label.y, label.staff_bottom) + 1.3 * unit
    spots = [(label.x, below + n * 0.3 * unit) for n in range(4)]
    if label.y > label.staff_bottom + 0.9 * unit:
        beside = label.y - height / 2
        spots += [(label.x + 1.2 * unit + width / 2, beside), (label.x - 1.2 * unit - width / 2, beside)]
    spots += [(label.x, below + n * 0.3 * unit) for n in range(4, 13)]
    best, best_ink = spots[0], None
    for x, top in spots:
        y0, y1 = int(top * k), int((top + height) * k) + 1
        x0, x1 = int((x - width / 2) * k), int((x + width / 2) * k) + 1
        covered = int(ink[max(y0, 0) : y1, max(x0, 0) : x1].sum())
        if covered == 0:
            return x, top
        if best_ink is None or covered < best_ink:
            best, best_ink = (x, top), covered
    return best


def _place(labels: list[Label], ink: np.ndarray) -> list[tuple[Label, pymupdf.Point, float]]:
    """Chooses a font size and baseline for every label of one staff.

    A label sits under its note, below the staff, or wherever nearby is free
    of print (stems, accents, pedal marks). Labels of closely spaced
    notes are nudged apart sideways, and shrunk where nudging is not enough,
    so they always stay in reading order.
    """
    unit = sum(l.unit for l in labels) / len(labels)
    base = 2.3 * unit
    pad = 0.25 * unit
    spots = [
        _find_spot(ink, l, pymupdf.get_text_length(l.text, FONT, base), CAP_HEIGHT * base, unit)
        for l in labels
    ]
    wanted = [x for x, _ in spots]
    tops = [top for _, top in spots]
    sizes = [base] * len(labels)

    def same_height(i: int, j: int) -> bool:
        return tops[j] < tops[i] + CAP_HEIGHT * sizes[i] and tops[i] < tops[j] + CAP_HEIGHT * sizes[j]

    for attempt in range(SHRINKS + 1):
        widths = [pymupdf.get_text_length(l.text, FONT, fs) for l, fs in zip(labels, sizes)]
        centres = list(wanted)
        for _ in range(200):
            moved = False
            for i in range(len(labels) - 1):
                # The next label at the same height, which may not be the next label:
                # one beside a low note can sit between two that are under the staff.
                j = next((j for j in range(i + 1, len(labels)) if same_height(i, j)), None)
                if j is None:
                    continue
                overlap = (widths[i] + widths[j]) / 2 + pad - (centres[j] - centres[i])
                if overlap > 0.01:
                    centres[i] -= overlap / 2
                    centres[j] += overlap / 2
                    moved = True
            if not moved:
                break
        crowded = [i for i in range(len(labels)) if abs(centres[i] - wanted[i]) > 0.9 * unit]
        if not crowded or attempt == SHRINKS:
            break  # the layout just computed matches the current sizes
        for i in crowded:
            sizes[i] = max(sizes[i] * 0.82, 0.55 * base)
    return [
        (l, pymupdf.Point(c - w / 2, top + CAP_HEIGHT * fs), fs)
        for l, c, w, top, fs in zip(labels, centres, widths, tops, sizes)
    ]


def _draw(page: pymupdf.Page, labels: list[Label]) -> None:
    to_page = page.derotation_matrix  # labels are in the page as displayed
    ink = _ink_map(page)
    for group in group_by_staff(labels):
        for label, point, fs in _place(group, ink):
            color = VERIFIED if label.verified else UNVERIFIED
            common = dict(fontname=FONT, fontsize=fs, rotate=page.rotation)
            origin = point * to_page
            # White halo first so the name stays readable over slurs and pedal marks.
            page.insert_text(origin, label.text, color=(1, 1, 1), fill=(1, 1, 1), render_mode=2,
                             border_width=0.2, **common)
            page.insert_text(origin, label.text, color=color, **common)


def _warn(text: str) -> None:
    print(f"\n  {text}", file=sys.stderr)


def read_labels(doc: pymupdf.Document, resolutions=RESOLUTIONS, progress=None,
                all_voices: bool = False, read=read_page, warn=_warn) -> list[list[Label]]:
    """Reads the whole document once per resolution and returns voted labels per page.

    A page that fails to read is left unlabelled in that reading, with a warning.
    """
    runs: list[list[list[Label]]] = []  # run -> page -> labels
    with tempfile.TemporaryDirectory() as tmp:
        for run, dpi in enumerate(resolutions):
            staffs_in_order: list[StaffLabels] = []
            pages: list[list[Label]] = []
            for number, page in enumerate(doc, start=1):
                if progress:
                    progress(run, number)
                image_path = str(Path(tmp) / f"page-{dpi}-{number}.png")
                page.get_pixmap(dpi=dpi, alpha=False).save(image_path)
                try:
                    staffs = read(image_path, scale=72 / dpi)
                except Exception as error:
                    warn(f"page {number} could not be read at {dpi} dpi: {type(error).__name__}: {error}")
                    staffs = []
                page_staffs = [labels_for_staff(s, all_voices) for s in staffs]
                staffs_in_order.extend(page_staffs)
                pages.append([l for s in page_staffs for l in s.labels])
            apply_key_signatures(staffs_in_order)
            runs.append(pages)
    return [merge_readings([run[p] for run in runs]) for p in range(doc.page_count)]


def label_pdf(src: Path, dst: Path, fast: bool = False, all_voices: bool = False) -> list[list[Label]]:
    doc = pymupdf.open(src)
    resolutions = FAST_RESOLUTIONS if fast else RESOLUTIONS

    def progress(run: int, number: int) -> None:
        print(f"\r  reading {run + 1} of {len(resolutions)}: page {number}/{doc.page_count}   ",
              end="", file=sys.stderr, flush=True)

    pages = read_labels(doc, resolutions, progress, all_voices)
    print(file=sys.stderr)
    for page, labels in zip(doc, pages):
        _draw(page, labels)
    doc.save(dst, garbage=3, deflate=True)
    return pages


def _jobs(sources: list[Path], output: Path | None) -> list[tuple[Path, Path]]:
    """Pairs every input with its output file, or exits before any work is done."""
    jobs = []
    for src in sources:
        if not src.is_file():
            sys.exit(f"Not a file: {src}")
        if output is None:
            dst = src.with_name(f"{src.stem} (bass notes).pdf")
        elif output.is_dir() or len(sources) > 1:
            dst = output / f"{src.stem} (bass notes).pdf"
        else:
            dst = output
        jobs.append((src, dst))
    inputs = {src.resolve() for src, _ in jobs}
    seen: dict[Path, Path] = {}
    for src, dst in jobs:
        if dst.resolve() in inputs:
            sys.exit(f"The output {dst} would overwrite an input file; choose another -o")
        if dst.resolve() in seen:
            sys.exit(f"{seen[dst.resolve()]} and {src} would both be written to {dst}; "
                     "label them into different folders")
        seen[dst.resolve()] = src
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(prog="chart-labeler", description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path, help="chart PDF(s) to label")
    parser.add_argument("-o", "--output", type=Path,
                        help="output file (single input) or directory; "
                             "default: '<name> (bass notes).pdf' beside the input")
    parser.add_argument("--fast", action="store_true",
                        help="read each page once instead of three times (quicker, fewer checks)")
    parser.add_argument("--all-voices", action="store_true",
                        help="also label notes played above a held bass note "
                             "(by default only the held note is labelled)")
    args = parser.parse_args()
    jobs = _jobs(args.pdf, args.output)
    try:
        prepare()
    except Exception as error:
        sys.exit(f"Could not load the recognition models: {type(error).__name__}: {error}")
    for src, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Reading {src.name}", file=sys.stderr)
        pages = label_pdf(src, dst, fast=args.fast, all_voices=args.all_voices)
        total = sum(len(p) for p in pages)
        print(f"Wrote {dst}", file=sys.stderr)
        if total == 0:
            print("  No bass-clef notes were found in this chart.", file=sys.stderr)
            continue
        doubtful = [(n, l) for n, p in enumerate(pages, start=1) for l in p if not l.verified]
        print(f"  {total} bass notes labelled: {total - len(doubtful)} in red (checks agreed), "
              f"{len(doubtful)} in orange (less certain)", file=sys.stderr)
        if doubtful:
            for number, label in doubtful:
                print(f"    page {number}: {label.text} - {label.why}", file=sys.stderr)


if __name__ == "__main__":
    main()
