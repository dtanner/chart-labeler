import numpy as np
import pymupdf
import pytest

from chart_labeler.cli import CAP_HEIGHT, FONT, _jobs, _place
from chart_labeler.label import Label

UNIT = 10.0
BOTTOM = 140.0
BLANK = np.zeros((1000, 1000), dtype=bool)  # a page without print


def label(x, y=BOTTOM - 2 * UNIT, text="C"):
    return Label(text=text, x=x, y=y, unit=UNIT, verified=True, staff_bottom=BOTTOM, staff_top=BOTTOM - 4 * UNIT)


def boxes(placed):
    out = []
    for l, point, fs in placed:
        width = pymupdf.get_text_length(l.text, FONT, fs)
        out.append((point.x, point.x + width, point.y - CAP_HEIGHT * fs, point.y))
    return out


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def test_crowded_labels_stay_centred_on_their_notes():
    # Five notes far too close for full-size labels: the outer ones get shrunk.
    xs = [100.0, 102.0, 104.0, 106.0, 108.0]
    placed = _place([label(x) for x in xs], BLANK)
    sizes = [fs for _, _, fs in placed]
    assert sizes[0] < sizes[2]
    centres = [(b[0] + b[1]) / 2 for b in boxes(placed)]
    # The arrangement is symmetric about the middle note, so the labels must be too.
    for k in range(5):
        assert centres[k] + centres[4 - k] == pytest.approx(2 * xs[2], abs=0.01)


def test_labels_at_the_same_height_do_not_overlap_across_a_lower_one():
    # The middle note hangs far below the staff, so its label sits lower down,
    # between the two labels under the staff.
    labels = [label(100.0), label(103.0, y=BOTTOM + 6 * UNIT), label(106.0)]
    placed = boxes(_place(labels, BLANK))
    assert not overlaps(placed[0], placed[2])


def test_inputs_are_checked_before_any_work(tmp_path):
    a = tmp_path / "a.pdf"
    a.write_bytes(b"")
    with pytest.raises(SystemExit, match="Not a file"):
        _jobs([a, tmp_path / "missing.pdf"], None)


def test_output_may_not_overwrite_an_input(tmp_path):
    a = tmp_path / "a.pdf"
    a.write_bytes(b"")
    with pytest.raises(SystemExit, match="overwrite an input"):
        _jobs([a], a)


def test_inputs_with_the_same_name_may_not_share_an_output(tmp_path):
    for folder in ("set1", "set2"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "song.pdf").write_bytes(b"")
    with pytest.raises(SystemExit, match="both be written"):
        _jobs([tmp_path / "set1/song.pdf", tmp_path / "set2/song.pdf"], tmp_path / "out")


def test_default_output_is_beside_the_input(tmp_path):
    a = tmp_path / "song.pdf"
    a.write_bytes(b"")
    assert _jobs([a], None) == [(a, tmp_path / "song (bass notes).pdf")]
