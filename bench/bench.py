"""Measures chart-labeler against scores whose notes are known.

Each score of the corpus is a MuseScore file. MuseScore renders it to PDF and
also exports the notes (MusicXML) and where every beat sits on the page, which
together give the bass note that belongs at each position: the answer key.

    python -I bench/bench.py build     render the sample and write the answer keys
    python -I bench/bench.py scans     make scan-like copies of every PDF
    python -I bench/bench.py score     label every PDF and compare with the key
    python -I bench/bench.py show      crop the page around every error

Data lives in bench-data/, which is not part of the program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench-data"
SONGS = DATA / "songs"
CACHE = DATA / "omr-cache"
MSCORE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
UNITS_PER_POINT = 60.0  # MuseScore position files
LETTERS = "CDEFGAB"
ALTER_TEXT = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "x"}
HEAD_OFFSET = 0.6  # notehead centre right of its beat position, in staff spaces
REACH = 2.0  # how far a label may sit from its note, in staff spaces
FAR = 7.0  # a right name within this distance counts as misplaced, not missing
MAX_PAGES = 6


# ---------------------------------------------------------------- answer key


def _positions(path: Path) -> list[dict]:
    out = []
    for e in ET.parse(path).getroot().find("elements"):
        out.append({
            "page": int(e.get("page")),
            "x": float(e.get("x")) / UNITS_PER_POINT,
            "y": float(e.get("y")) / UNITS_PER_POINT,
            "w": float(e.get("sx")) / UNITS_PER_POINT,
            "h": float(e.get("sy")) / UNITS_PER_POINT,
        })
    return out


def _part_measures(part: ET.Element) -> list[list[dict]]:
    """Every note and rest of a part, per measure, with its start and end time."""
    divisions = 1
    measures = []
    for measure in part.findall("measure"):
        now = Fraction(0)
        last_start = Fraction(0)
        events = []
        for el in measure:
            if el.tag == "attributes":
                d = el.findtext("divisions")
                if d:
                    divisions = int(d)
            elif el.tag == "backup":
                now -= Fraction(int(el.findtext("duration")), divisions)
            elif el.tag == "forward":
                now += Fraction(int(el.findtext("duration")), divisions)
            elif el.tag == "note":
                if el.find("grace") is not None:
                    continue
                length = Fraction(int(el.findtext("duration") or 0), divisions)
                chord = el.find("chord") is not None
                start = last_start if chord else now
                pitch = el.find("pitch")
                event = {
                    "start": start,
                    "end": start + length,
                    "staff": int(el.findtext("staff") or 1),
                    "voice": el.findtext("voice") or "1",
                    "printed": el.get("print-object") != "no" and el.find("cue") is None,
                    "pitch": None,
                }
                if pitch is not None:
                    step = pitch.findtext("step")
                    octave = int(pitch.findtext("octave"))
                    alter = int(float(pitch.findtext("alter") or 0))
                    event["pitch"] = octave * 7 + LETTERS.index(step)
                    event["name"] = step + ALTER_TEXT[alter]
                events.append(event)
                if not chord:
                    last_start = now
                    now += length
        measures.append(events)
    return measures


def answer_key(song: Path) -> dict | str:
    """The bass note expected at each position, or the reason the score is unusable."""
    root = ET.parse(song / "score.musicxml").getroot()
    parts = root.findall("part")
    pianos = [p for p in parts if any(s.text == "2" for s in p.iter("staves"))]
    if len(pianos) != 1 or len(parts) > 3:
        return "not a voice and piano score"
    for part in parts:
        if part is not pianos[0] and any(c.findtext("sign") == "F" for c in part.iter("clef")):
            return "a part other than the piano uses the bass clef"
    scaling = root.find("defaults/scaling")
    space = float(scaling.findtext("millimeters")) / float(scaling.findtext("tenths")) * 10 * 72 / 25.4
    boxes = _positions(song / "score.mpos")
    beats = _positions(song / "score.spos")
    per_part = [_part_measures(p) for p in parts]
    if any(len(m) != len(boxes) for m in per_part):
        return "measure count differs between the notes and the layout"
    if max(b["page"] for b in boxes) + 1 > MAX_PAGES:
        return "too long"
    piano = per_part[parts.index(pianos[0])]

    measures = []
    for index, box in enumerate(boxes):
        starts = sorted({e["start"] for m in per_part for e in m[index]})
        here = sorted(
            (b for b in beats
             if b["page"] == box["page"] and abs(b["y"] - box["y"]) < 1
             and box["x"] - 0.5 <= b["x"] < box["x"] + box["w"] - 0.5),
            key=lambda b: b["x"],
        )
        entry = {"page": box["page"], "x0": box["x"], "x1": box["x"] + box["w"],
                 "y0": box["y"], "y1": box["y"] + box["h"], "scored": len(here) == len(starts), "notes": []}
        measures.append(entry)
        if not entry["scored"]:
            continue
        x_of = {start: beat["x"] + HEAD_OFFSET * space for start, beat in zip(starts, here)}
        low = [e for e in piano[index] if e["staff"] == 2 and e["pitch"] is not None and e["printed"]]
        for start in sorted({e["start"] for e in low}):
            note = min((e for e in low if e["start"] == start), key=lambda e: e["pitch"])
            held = any(
                o["pitch"] < note["pitch"] and o["voice"] != note["voice"] and o["start"] < start < o["end"]
                for o in low
            )
            entry["notes"].append({"x": x_of[start], "name": note["name"], "optional": held})
    return {"space": space, "pages": max(b["page"] for b in boxes) + 1, "measures": measures}


def build(args) -> None:
    sources = [DATA / "lieder" / line for line in (DATA / "sample.txt").read_text().splitlines() if line]
    sources = [s for s in sources if s.is_file()]
    jobs = []
    for source in sources:
        song = SONGS / source.stem
        song.mkdir(parents=True, exist_ok=True)
        (song / "source.txt").write_text(str(source.relative_to(DATA)) + "\n")
        if not (song / "score.spos").exists():
            jobs.append({"in": str(source),
                         "out": [str(song / f"score.{ext}") for ext in ("pdf", "musicxml", "mpos", "spos")]})
    if jobs:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(jobs, f)
        print(f"rendering {len(jobs)} scores with MuseScore", file=sys.stderr)
        subprocess.run([MSCORE, "-j", f.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    kept = 0
    for source in sources:
        song = SONGS / source.stem
        (song / "key.json").unlink(missing_ok=True)
        if not (song / "score.spos").exists():
            print(f"skip {song.name}: MuseScore could not render it")
            continue
        key = answer_key(song)
        if isinstance(key, str):
            print(f"skip {song.name}: {key}")
            continue
        scored = [m for m in key["measures"] if m["scored"]]
        if len(scored) < 0.9 * len(key["measures"]) or kept >= args.limit:
            print(f"skip {song.name}: " + ("sample is full" if kept >= args.limit else "layout could not be aligned"))
            continue
        kept += 1
        (song / "key.json").write_text(json.dumps(key))
        notes = sum(len(m["notes"]) for m in scored)
        print(f"keep {song.name}: {key['pages']} pages, {len(scored)}/{len(key['measures'])} measures, {notes} notes")
    print(f"{kept} scores in the benchmark")


# ---------------------------------------------------------------- scan copies


def scans(args) -> None:
    """A copy of each PDF as a slightly crooked, blurred, noisy 150 dpi scan."""
    import cv2
    import numpy as np
    import pymupdf

    for song in _songs():
        target = song / "scan.pdf"
        if target.exists():
            continue
        rng = np.random.default_rng(int(hashlib.sha1(song.name.encode()).hexdigest()[:8], 16))
        doc, out, turns = pymupdf.open(song / "score.pdf"), pymupdf.open(), []
        for page in doc:
            pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(np.float32)
            angle = float(rng.uniform(-0.8, 0.8))
            turn = cv2.getRotationMatrix2D((pix.width / 2, pix.height / 2), angle, 1.0)
            image = cv2.warpAffine(image, turn, (pix.width, pix.height), borderValue=255)
            image = cv2.GaussianBlur(image, (0, 0), 0.9)
            shade = np.linspace(0, float(rng.uniform(10, 30)), pix.width, dtype=np.float32)[None, :]
            image = 235 - (255 - image) * 0.85 - shade + rng.normal(0, 6, image.shape)
            ok, jpeg = cv2.imencode(".jpg", np.clip(image, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 60])
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=jpeg.tobytes())
            turns.append(angle)
        out.save(target)
        (song / "scan.json").write_text(json.dumps(turns))
        print(f"wrote {target.relative_to(DATA)}", file=sys.stderr)


# ---------------------------------------------------------------- scoring


def _songs() -> list[Path]:
    return sorted(s for s in SONGS.iterdir() if (s / "key.json").exists())


def _cached_read(image_path: str, scale: float):
    """read_page, with homr's slow part kept on disk between benchmark runs."""
    from chart_labeler.heads import locate_noteheads
    from chart_labeler.omr import read_staffs

    digest = hashlib.sha1(Path(image_path).read_bytes() + repr(scale).encode() + b"v2").hexdigest()
    stored = CACHE / f"{digest}.pkl"
    if stored.exists():
        staffs = pickle.loads(stored.read_bytes())
    else:
        try:
            staffs = read_staffs(image_path, scale)
        except Exception:
            staffs = None
        CACHE.mkdir(exist_ok=True)
        stored.write_bytes(pickle.dumps(staffs))
    if staffs is None:
        raise ValueError("no music found on the page")
    locate_noteheads(image_path, staffs, scale)
    return staffs


def _label(job: tuple[str, str, tuple[int, ...]]) -> list[list[dict]]:
    import pymupdf

    from chart_labeler.cli import read_labels

    song, variant, resolutions = job
    doc = pymupdf.open(SONGS / song / f"{variant}.pdf")
    return [[asdict(l) for l in page] for page in read_labels(doc, resolutions, read=_cached_read)]


def _unturn(label: dict, angle: float, width: float, height: float) -> tuple[float, float]:
    """Where a point of the crooked scan lies on the straight page."""
    a = math.radians(-angle)
    dx, dy = label["x"] - width / 2, label["staff_bottom"] - height / 2
    return (width / 2 + dx * math.cos(a) + dy * math.sin(a), height / 2 - dx * math.sin(a) + dy * math.cos(a))


def compare(key: dict, pages: list[list[dict]], turns: list[float] | None, sizes) -> list[dict]:
    """One row per label and per expected note: how they correspond."""
    space = key["space"]
    rows = []
    for number, labels in enumerate(pages):
        measures = [m for m in key["measures"] if m["page"] == number]
        slots: dict[int, list[dict]] = {}
        for label in labels:
            x, bottom = label["x"], label["staff_bottom"]
            if turns:
                x, bottom = _unturn(label, turns[number], *sizes[number])
            row = {"page": number, "x": label["x"], "y": label["y"], "got": label["text"],
                   "red": label["verified"], "why": label["why"], "want": None}
            near = [m for m in measures if m["x0"] - space <= x < m["x1"] + space * 0.2
                    and abs(bottom - m["y1"]) < 3 * space]
            if not near:
                rows.append({**row, "kind": "extra", "note": "not on a piano bass staff"})
                continue
            measure = min(near, key=lambda m: abs((m["x0"] + m["x1"]) / 2 - x))
            if measure["scored"]:
                slots.setdefault(id(measure), []).append({**row, "sx": x})
        for measure in measures:
            if not measure["scored"]:
                continue
            mine = slots.get(id(measure), [])
            pairs = sorted(
                (abs(row["sx"] - note["x"]), i, j)
                for i, row in enumerate(mine) for j, note in enumerate(measure["notes"])
                if abs(row["sx"] - note["x"]) <= REACH * space
            )
            row_of: dict[int, int] = {}
            used: set[int] = set()
            for _, i, j in pairs:
                if i not in used and j not in row_of:
                    row_of[j] = i
                    used.add(i)
            # A label with the right name that stands too far from its note.
            for j, note in enumerate(measure["notes"]):
                if j in row_of or note["optional"]:
                    continue
                stray = [i for i, row in enumerate(mine) if i not in used and row["got"] == note["name"]
                         and abs(row["sx"] - note["x"]) <= FAR * space]
                if stray:
                    i = min(stray, key=lambda i: abs(mine[i]["sx"] - note["x"]))
                    used.add(i)
                    row_of[j] = i
                    mine[i]["far"] = True
            for j, note in enumerate(measure["notes"]):
                if j in row_of:
                    row = mine[row_of[j]]
                    kind = "misplaced" if row.pop("far", False) else "correct" if row["got"] == note["name"] else "wrong"
                    rows.append({**row, "want": note["name"], "kind": kind})
                elif not note["optional"]:
                    rows.append({"page": number, "x": note["x"], "y": measure["y1"], "got": None, "red": None,
                                 "why": "", "want": note["name"], "kind": "missed"})
            for i, row in enumerate(mine):
                if i not in used:
                    rows.append({**row, "kind": "extra"})
    for row in rows:
        row.pop("sx", None)
    return rows


COLUMNS = ("notes", "red ok", "RED WRONG", "orange ok", "orange wrong", "misplaced", "missed", "extra")


def _counts(rows: list[dict]) -> list[int]:
    def n(kind, red=None):
        return sum(1 for r in rows if r["kind"] == kind and (red is None or r["red"] == red))

    return [n("correct") + n("wrong") + n("misplaced") + n("missed"), n("correct", True), n("wrong", True),
            n("correct", False), n("wrong", False), n("misplaced"), n("missed"), n("extra")]


def score(args) -> None:
    import pymupdf

    from chart_labeler.cli import RESOLUTIONS

    variants = ["score", "scan"] if args.scans else ["score"]
    resolutions = tuple(args.resolutions or RESOLUTIONS)
    jobs = [(s.name, v, resolutions) for s in _songs() for v in variants if (s / f"{v}.pdf").exists()]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_label, jobs))
    report = {}
    for variant in variants:
        print(f"\n{variant}.pdf\n{'song':<12}" + "".join(f"{c:>14}" for c in COLUMNS))
        total = [0] * len(COLUMNS)
        for (song, v, _), pages in zip(jobs, results):
            if v != variant:
                continue
            key = json.loads((SONGS / song / "key.json").read_text())
            turns = json.loads((SONGS / song / "scan.json").read_text()) if variant == "scan" else None
            sizes = [(p.rect.width, p.rect.height) for p in pymupdf.open(SONGS / song / "score.pdf")]
            rows = compare(key, pages, turns, sizes)
            report[f"{song}/{variant}"] = rows
            counts = _counts(rows)
            total = [a + b for a, b in zip(total, counts)]
            print(f"{song:<12}" + "".join(f"{c:>14}" for c in counts))
        print(f"{'TOTAL':<12}" + "".join(f"{c:>14}" for c in total))
        if total[0]:
            labelled = sum(total[1:6])
            print(f"labels with the wrong name: {total[2] + total[4]} of {labelled} "
                  f"({100 * (total[2] + total[4]) / max(labelled, 1):.2f}%), of which in red: {total[2]}; "
                  f"notes without a label: {100 * total[6] / total[0]:.2f}%")
    (DATA / "report.json").write_text(json.dumps(report, indent=1))


def show(args) -> None:
    """Crops of the page around each error, for looking at what went wrong."""
    import pymupdf

    report = json.loads((DATA / "report.json").read_text())
    out = DATA / "errors"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    shown = 0
    for name, rows in report.items():
        song, variant = name.split("/")
        doc = pymupdf.open(SONGS / song / f"{variant}.pdf")
        for row in rows:
            if row["kind"] not in args.kinds or (args.red and row["kind"] != "missed" and not row["red"]):
                continue
            if shown >= args.limit:
                return
            page = doc[row["page"]]
            clip = pymupdf.Rect(row["x"] - 90, row["y"] - 90, row["x"] + 90, row["y"] + 60) & page.rect
            shape = page.new_shape()
            shape.draw_line((row["x"], clip.y0), (row["x"], clip.y0 + 12))
            shape.finish(color=(0, 0.4, 1), width=1.2)
            shape.commit()
            file = out / f"{shown:03d}-{song}-{variant}-p{row['page'] + 1}-{row['kind']}.png"
            page.get_pixmap(dpi=200, clip=clip).save(file)
            colour = "" if row["red"] is None else ("red" if row["red"] else "orange")
            print(f"{file.name}: {row['kind']} {colour} got {row['got']} want {row['want']} {row['why']}"
                  f"{row.get('note', '')}")
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(run=build)
    sub.add_parser("scans").set_defaults(run=scans)
    p = sub.add_parser("score")
    p.add_argument("--scans", action="store_true", help="also score the scan-like copies")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--resolutions", type=int, nargs="*", help="dpi of each reading (default: the program's own)")
    p.add_argument("--only", nargs="*", help="song names to score")
    p.set_defaults(run=score)
    p = sub.add_parser("show")
    p.add_argument("--kinds", nargs="*", default=["wrong", "misplaced", "missed", "extra"])
    p.add_argument("--red", action="store_true", help="only labels printed in red")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(run=show)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
