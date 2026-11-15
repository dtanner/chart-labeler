"""Turns OMR readings of a chart into positioned note-name labels.

Accuracy comes from cross-checking independent evidence:
  * the transformer's pitch for a note must coincide with a segmented notehead
    at the matching height on the staff
  * every page is read several times at different resolutions and the readings
    must agree
  * accidentals are checked against the key signature, taking the key after a
    mid-line key change from the start of the following system, where the
    signature is printed in full and read reliably
Anything that fails a check is still labelled, but marked unverified.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from chart_labeler.omr import Notehead, StaffRead, Token

LETTERS = "CDEFGAB"
# Diatonic index (octave * 7 + letter) of the bottom staff line for each clef.
CLEF_BOTTOM_LINE = {
    "clef_F4": 2 * 7 + 4,  # G2, bass
    "clef_G2": 4 * 7 + 2,  # E4, treble
    "clef_C3": 3 * 7 + 3,  # F3, alto
    "clef_C4": 3 * 7 + 1,  # D3, tenor
    "clef_F3": 2 * 7 + 6,  # B2, baritone
    "clef_G1": 4 * 7 + 4,  # G4, french violin
    "clef_C1": 4 * 7 + 0,  # C4, soprano
    "clef_C2": 3 * 7 + 5,  # A3, mezzo-soprano
}
ACCIDENTAL_TEXT = {"b": "b", "#": "#", "bb": "bb", "##": "x"}
SHARP_ORDER = "FCGDAEB"
WIDE_REACH = 6.0  # staff spaces a note may lie from where the model points


@dataclass
class Label:
    text: str  # e.g. "Db"
    x: float  # notehead centre
    y: float
    unit: float  # staff line spacing at x
    verified: bool
    staff_bottom: float  # y of the bottom staff line at x
    staff_top: float
    step: int = 0  # diatonic steps above the bottom staff line
    key_text: str | None = None  # the name the key signature alone implies
    why: str = ""  # reason when unverified
    weak: bool = False  # read under a misread key signature; outvoted by any other reading


@dataclass
class StaffLabels:
    """The labels of one staff from one reading, with its key signature facts."""

    labels: list[Label] = field(default_factory=list)
    segments: list[int] = field(default_factory=list)  # per label: key changes seen before it
    letters: list[str] = field(default_factory=list)
    start_key: int | None = None  # signature printed at the start of the staff
    changes: list[int] = field(default_factory=list)  # mid-line signatures as read


def diatonic(pitch: str) -> int:
    return int(pitch[1:]) * 7 + LETTERS.index(pitch[0])


def note_name(token: Token) -> str:
    return token.pitch[0] + ACCIDENTAL_TEXT.get(token.lift, "")


def name_in_key(letter: str, key: int) -> str:
    """Note name when only the key signature applies (key = sharps, negative = flats)."""
    if key > 0 and SHARP_ORDER.index(letter) < key:
        return letter + "#"
    if key < 0 and SHARP_ORDER[::-1].index(letter) < -key:
        return letter + "b"
    return letter


def _is_note(t: Token) -> bool:
    if not t.rhythm.startswith("note") or len(t.pitch) < 2 or t.pitch[0] not in LETTERS:
        return False
    return "G" not in t.rhythm.split("_", 1)[1]  # grace notes are not played as bass notes


def _key_value(t: Token) -> int | None:
    try:
        return int(t.rhythm.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


@dataclass
class _Reading:
    token: Token
    step: float  # diatonic steps above the bottom staff line
    segment: int
    above: list[tuple[int, Token]] = field(default_factory=list)  # rest of its chord: step, token


def _lowest_notes(
    staff: StaffRead, all_voices: bool
) -> tuple[list[_Reading], bool, int | None, list[int]]:
    """The bass note of every beat position on this staff, in reading order.

    That is the lowest note of each chord. Unless `all_voices` is set, a note
    is left out when a lower note of the other voice on the staff is still
    sounding underneath it: the held note is the bass there, and the notes
    above it are accompaniment.
    """
    grand = staff.is_lower_of_grand or staff.is_upper_of_grand
    want_lower = staff.is_lower_of_grand

    def mine(t: Token) -> bool:
        return not grand or t.position.startswith("lower") == want_lower

    # Tokens joined by "chord" markers start at the same moment.
    groups: list[list[Token]] = []
    joined = False
    for t in staff.tokens:
        if t.rhythm == "chord":
            joined = True
            continue
        if joined and groups:
            groups[-1].append(t)
        else:
            groups.append([t])
        joined = False

    clef = "clef_F4" if want_lower else "clef_G2"
    saw_bass = False
    start_key: int | None = None
    changes: list[int] = []
    seen_note = False
    out: list[_Reading] = []
    now = 0.0
    voice_free_at: dict[str, float] = {}  # per voice: when its current note or rest ends
    held: list[tuple[int, float, str]] = []  # sounding notes on this staff: pitch, end, voice

    for group in groups:
        timed = [t for t in group if t.rhythm.startswith(("note", "rest")) and t.duration > 0]
        for t in group:
            if t.rhythm in CLEF_BOTTOM_LINE and mine(t):
                clef = t.rhythm
            elif t.rhythm.startswith("keySignature"):
                value = _key_value(t)
                if value is not None and seen_note:
                    changes.append(value)
                elif value is not None:
                    start_key = value
            elif "barline" in t.rhythm:
                now, voice_free_at, held = 0.0, {}, []
        if not timed:
            continue
        seen_note = True
        notes = [t for t in timed if _is_note(t) and mine(t)]
        if notes:
            token = min(notes, key=lambda t: diatonic(t.pitch))
            pitch = diatonic(token.pitch)
            covered = any(
                p < pitch and end > now + 1e-6 and voice != token.position for p, end, voice in held
            )
            if all_voices or not covered:
                saw_bass = saw_bass or clef == "clef_F4"
                bottom = CLEF_BOTTOM_LINE[clef]
                above = [(diatonic(t.pitch) - bottom, t) for t in notes if t is not token]
                out.append(_Reading(token, pitch - bottom, len(changes), above))
        for t in timed:
            voice_free_at[t.position] = now + t.duration
            if _is_note(t) and mine(t):
                held.append((diatonic(t.pitch), now + t.duration, t.position))
        # The next group starts when the first voice becomes free again.
        now = min(end for end in voice_free_at.values() if end > now + 1e-6)
        held = [h for h in held if h[1] > now + 1e-6]
    return out, saw_bass, start_key, changes


def _head_step(staff: StaffRead, head: Notehead) -> float:
    """Staff position used to pair a notehead with the model's note.

    homr's noteheads lie near the staff, where its own staff model is what the
    pairing was tuned on. Added noteheads lie far below, where only the
    measured position is accurate enough.
    """
    return head.step if head.added and head.step is not None else staff.step_at(head.x, head.y)


def _measured_step(staff: StaffRead, head: Notehead) -> float:
    return head.step if head.step is not None else staff.step_at(head.x, head.y)


def _octave_below(reading: _Reading, staff: StaffRead, taken: set[int]) -> tuple[Notehead, Token, int] | None:
    """Corrects a misread bottom note of an octave from where its notehead sits.

    Far below the staff the model often gets the low note of an octave wrong
    (C1 read as E1) while reading the upper note correctly. When the lowest
    notehead in the chord's column sits exactly one or two octaves under a
    chord note that is itself confirmed by a notehead, the low note has that
    note's name. Returns the low notehead, the chord note it doubles and the
    number of octaves between them.
    """
    if reading.token.xy is None:
        return None
    x = reading.token.xy[0]
    column = [h for h in staff.noteheads
              if id(h) not in taken and abs(h.x - x) <= 2.0 * staff.unit_at(h.x)]
    if not column:
        return None
    low = max(column, key=lambda h: h.y)
    low_step = _measured_step(staff, low)
    if low_step > -1.5:
        return None  # on the staff the model reads reliably; a tight chord there is not an octave
    for step, token in reading.above:
        gap = step - round(low_step)
        if gap <= 0 or gap % 7 or abs(low_step - round(low_step)) > 0.3:
            continue
        for h in column:
            if abs(h.x - low.x) <= 1.3 * staff.unit_at(h.x) and abs(_measured_step(staff, h) - step) <= 0.4:
                return low, token, gap // 7
    return None


def _match(readings: list[_Reading], staff: StaffRead) -> list[Notehead | None]:
    """Pairs each transformer note with the segmented notehead it describes."""
    pairs = []
    for i, r in enumerate(readings):
        if r.token.xy is None:
            continue
        for j, h in enumerate(staff.noteheads):
            unit = staff.unit_at(h.x)
            dx = abs(h.x - r.token.xy[0]) / unit
            dstep = abs(_head_step(staff, h) - r.step)
            if dx <= 2.5 and dstep <= 0.7:
                pairs.append((dx + dstep, i, j))
    result: list[Notehead | None] = [None] * len(readings)
    used: set[int] = set()
    for _, i, j in sorted(pairs):
        if result[i] is None and j not in used:
            result[i] = staff.noteheads[j]
            used.add(j)
    # The model's x for a note can be several spaces off, mostly for long
    # notes in a sparse measure. A free notehead at exactly the note's height
    # that lies between the noteheads of the neighbouring notes is that note.
    for i, r in enumerate(readings):
        if result[i] is not None or r.token.xy is None:
            continue
        before = next((result[k].x for k in range(i - 1, -1, -1) if result[k] is not None), float("-inf"))
        after = next((result[k].x for k in range(i + 1, len(readings)) if result[k] is not None), float("inf"))
        best = None
        for j, h in enumerate(staff.noteheads):
            unit = staff.unit_at(h.x)
            dx = abs(h.x - r.token.xy[0]) / unit
            if (j not in used and dx <= WIDE_REACH and abs(_head_step(staff, h) - r.step) <= 0.5
                    and before + 0.5 * unit < h.x < after - 0.5 * unit and (best is None or dx < best[0])):
                best = (dx, j)
        if best is not None:
            result[i] = staff.noteheads[best[1]]
            used.add(best[1])
    return result


def _is_bass_staff(staff: StaffRead, saw_bass: bool) -> bool:
    if staff.is_upper_of_grand:
        return False
    if not staff.is_lower_of_grand and not saw_bass:
        return False
    # A lower staff without a bass clef: two treble staffs paired by mistake
    # (two voices, or the voice with the piano's right hand), and the piano's
    # bass staff is further down.
    return saw_bass or staff.last_in_system


def may_get_labels(staff: StaffRead) -> bool:
    """False for a staff that labels_for_staff gives no labels, whatever the voices option."""
    return not staff.is_upper_of_grand and _is_bass_staff(staff, _lowest_notes(staff, True)[1])


def labels_for_staff(staff: StaffRead, all_voices: bool = False) -> StaffLabels:
    """Labels for one staff from one reading. Staffs without bass notes give none."""
    result = StaffLabels()
    if staff.is_upper_of_grand:
        return result
    readings, saw_bass, result.start_key, result.changes = _lowest_notes(staff, all_voices)
    if not _is_bass_staff(staff, saw_bass):
        return result
    heads = _match(readings, staff)
    taken = {id(h) for h in heads if h is not None}
    x_min, x_max = staff.x_range
    for r, head in zip(readings, heads):
        if head is None:
            fixed = _octave_below(r, staff, taken)
            if fixed is not None:
                head, doubled, octaves = fixed
                taken.add(id(head))
                pitch = doubled.pitch[0] + str(int(doubled.pitch[1:]) - octaves)
                r = _Reading(Token(r.token.rhythm, pitch, doubled.lift, r.token.position, r.token.xy),
                             round(_measured_step(staff, head)), r.segment)
        if head is not None:
            x = head.x
        elif r.token.xy is not None:
            x = min(max(r.token.xy[0], x_min), x_max)
        else:
            continue
        lines = staff.lines_at(x)
        result.labels.append(
            Label(
                text=note_name(r.token),
                x=x,
                y=staff.y_of_step(x, r.step),
                unit=staff.unit_at(x),
                verified=head is not None,
                staff_bottom=lines[4],
                staff_top=lines[0],
                step=int(r.step),
                why="" if head is not None else "no notehead found at this position",
            )
        )
        result.segments.append(r.segment)
        result.letters.append(r.token.pitch[0])
    return result


def apply_key_signatures(staffs: list[StaffLabels]) -> None:
    """Fills Label.key_text for the staffs of one reading, in document order."""
    current: int | None = None
    labelled = [s for s in staffs if s.labels or s.start_key is not None]
    for index, staff in enumerate(labelled):
        later = [s.start_key for s in labelled[index + 1 :] if s.start_key is not None]
        following = later[0] if later else None
        if staff.start_key is not None:
            # A signature never differs from both the line before and the line
            # after unless it changes within the line: such a one was misread,
            # and the notes of the line were read under the wrong signature.
            before = current if current is not None else (later[1] if len(later) > 1 else None)
            misread = (following is not None and staff.start_key != following and before == following
                       and not staff.changes)
            if misread:
                for label, letter in zip(staff.labels, staff.letters):
                    label.weak = True
                    if label.text == name_in_key(letter, staff.start_key):
                        label.text = name_in_key(letter, following)
            current = following if misread else staff.start_key
        keys = [current]
        for n, change in enumerate(staff.changes):
            last = n == len(staff.changes) - 1
            # The signature at a mid-line change is cramped among cancelling
            # naturals; the next system restates it cleanly.
            keys.append(following if last and following is not None else change)
        for label, segment, letter in zip(staff.labels, staff.segments, staff.letters):
            key = keys[min(segment, len(keys) - 1)]
            if key is not None:
                label.key_text = name_in_key(letter, key)
        current = keys[-1]


def _same_staff(a: Label, b: Label) -> bool:
    return abs(a.staff_bottom - b.staff_bottom) < 2 * a.unit


def merge_readings(readings: list[list[Label]]) -> list[Label]:
    """Combines the labels of several readings of the same page by vote."""
    total = len(readings)
    clusters: list[list[tuple[int, Label]]] = []
    for run, labels in enumerate(readings):
        for label in labels:
            best = None
            for cluster in clusters:
                first = cluster[0][1]
                if any(r == run for r, _ in cluster) or first.step != label.step:
                    continue
                distance = abs(label.x - statistics.median(l.x for _, l in cluster))
                if _same_staff(first, label) and distance < 1.5 * label.unit:
                    if best is None or distance < best[0]:
                        best = (distance, cluster)
            if best is None:
                clusters.append([(run, label)])
            else:
                best[1].append((run, label))

    def staff_support(label: Label) -> int:
        """How many readings labelled anything on this label's staff."""
        return sum(any(_same_staff(label, other) for other in labels) for labels in readings)

    merged: list[tuple[int, int, Label]] = []  # support, located, label
    for cluster in clusters:
        members = [l for _, l in cluster]
        if 2 * staff_support(members[0]) <= total:
            continue  # most readings took this for a staff that is not a bass part
        located = [l for l in members if l.verified]
        if 2 * len(members) <= total and not located:
            continue  # seen by a minority and no notehead there: a misreading
        texts = Counter(l.text for l in members if not l.weak) or Counter(l.text for l in members)
        key_texts = Counter(l.key_text for l in members if l.key_text)
        key_text = key_texts.most_common(1)[0][0] if key_texts else None
        why = ""
        if len(texts) == 1:
            text = next(iter(texts))
            if all(l.weak for l in members):
                why = "the key signature of this line was unclear"
        else:
            text = key_text if key_text in texts else texts.most_common(1)[0][0]
            why = "readings disagree: " + ", ".join(f"{t} x{n}" for t, n in texts.most_common())
        if not why and 2 * len(members) <= total:
            why = f"found in only {len(members)} of {total} readings"
        if not why and 2 * len(located) <= total:
            why = "no notehead found at this position"
        source = located or members
        label = Label(
            text=text,
            x=statistics.median(l.x for l in source),
            y=statistics.median(l.y for l in source),
            unit=source[0].unit,
            verified=not why,
            staff_bottom=statistics.median(l.staff_bottom for l in source),
            staff_top=statistics.median(l.staff_top for l in source),
            step=source[0].step,
            key_text=key_text,
            why=why,
        )
        merged.append((len(members), len(located), label))

    # One label per position: where readings put different notes at the same
    # spot, keep the better supported one and mark it.
    merged.sort(key=lambda m: (-m[0], -m[1]))
    result: list[Label] = []
    for support, _, label in merged:
        rival = next((r for r in result if _same_staff(r, label) and abs(r.x - label.x) < 0.8 * label.unit), None)
        if rival is None:
            result.append(label)
        elif rival.text != label.text:
            rival.verified = False
            rival.why = rival.why or f"some readings saw {label.text} here"
    return [label for staff in group_by_staff(result) for label in staff]


def group_by_staff(labels: list[Label]) -> list[list[Label]]:
    """Splits a page's labels into staffs, top to bottom, each left to right."""
    groups: list[list[Label]] = []
    for label in sorted(labels, key=lambda l: l.staff_bottom):
        if groups and abs(groups[-1][-1].staff_bottom - label.staff_bottom) < 4 * label.unit:
            groups[-1].append(label)
        else:
            groups.append([label])
    return [sorted(group, key=lambda l: l.x) for group in groups]
