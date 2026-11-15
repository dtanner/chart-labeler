from chart_labeler.label import (
    Label,
    apply_key_signatures,
    labels_for_staff,
    merge_readings,
    name_in_key,
)
from chart_labeler.omr import Notehead, StaffRead, Token

UNIT = 10.0
BOTTOM = 140.0  # y of the bottom staff line


def staff(tokens, lower=True, heads=()):
    grid = [(x, [BOTTOM - 4 * UNIT + i * UNIT for i in range(5)]) for x in (0.0, 500.0, 1000.0)]
    return StaffRead(grid, lower, False, list(tokens), list(heads))


def note(pitch, lift="_", kern="4", position="lower", x=None, duration=None):
    durations = {"1": 1.0, "2": 0.5, "4": 0.25, "8": 0.125}
    xy = (x, 0.0) if x is not None else (100.0, 0.0)
    return Token(f"note_{kern}", pitch, lift, position, xy, duration or durations[kern])


CHORD = Token("chord", ".", ".", ".", None)
BAR = Token("barline", ".", ".", ".", None)


def key(n):
    return Token(f"keySignature_{n}", ".", ".", ".", None)


def texts(result):
    return [l.text for l in result.labels]


def test_name_in_key():
    assert name_in_key("B", -5) == "Bb"
    assert name_in_key("C", -5) == "C"
    assert name_in_key("F", 4) == "F#"
    assert name_in_key("A", 4) == "A"
    assert name_in_key("G", 0) == "G"


def test_lowest_note_of_a_chord_is_labelled():
    tokens = [note("A2", "b", "1"), CHORD, note("D2", "b", "1"), CHORD, note("F4", position="upper")]
    assert texts(labels_for_staff(staff(tokens))) == ["Db"]


def test_notes_above_a_held_bass_note_are_left_out():
    tokens = [
        note("D3", "b", "2", "lower2", x=100), CHORD, note("D3", "b", "8", x=100),
        note("A3", "b", "8", x=130), note("D4", "b", "8", x=160), note("A3", "b", "8", x=190),
        note("G2", "b", "2", x=220),
    ]
    assert texts(labels_for_staff(staff(tokens))) == ["Db", "Gb"]
    assert texts(labels_for_staff(staff(tokens), all_voices=True)) == ["Db", "Ab", "Db", "Ab", "Gb"]


def test_held_notes_end_at_the_barline():
    tokens = [note("D2", "b", "1", "lower2"), CHORD, note("G2", "b", "2"), BAR, note("A2", "b", "2")]
    assert texts(labels_for_staff(staff(tokens))) == ["Db", "Ab"]


def test_label_sits_on_the_matching_notehead():
    # D2 hangs under the first ledger line: 3 steps below the bottom line.
    head = Notehead(104.0, BOTTOM + 1.5 * UNIT, 12.0, 10.0)
    result = labels_for_staff(staff([note("D2", "b", x=100)], heads=[head]))
    assert result.labels[0].verified and result.labels[0].x == 104.0
    wrong_height = Notehead(104.0, BOTTOM, 12.0, 10.0)
    result = labels_for_staff(staff([note("D2", "b", x=100)], heads=[wrong_height]))
    assert not result.labels[0].verified


def test_treble_only_staff_gets_no_labels():
    tokens = [Token("clef_G2", ".", ".", "upper", None), note("C5", position="upper")]
    assert texts(labels_for_staff(staff(tokens, lower=False))) == []


def test_key_after_midline_change_comes_from_next_system():
    first = labels_for_staff(staff([key(-5), note("D2", "b"), key(7), note("B2", "#")]))
    second = labels_for_staff(staff([key(4), note("E2")]))
    apply_key_signatures([first, second])
    assert [l.key_text for l in first.labels] == ["Db", "B"]
    assert [l.key_text for l in second.labels] == ["E"]


def label(text, x=100.0, verified=True, key_text=None):
    return Label(text, x, BOTTOM, UNIT, verified, BOTTOM, BOTTOM - 4 * UNIT, step=0, key_text=key_text)


def test_unanimous_readings_are_verified():
    merged = merge_readings([[label("Gb")], [label("Gb", 101.0)], [label("Gb", 99.0)]])
    assert [(l.text, l.verified) for l in merged] == [("Gb", True)]


def test_disagreement_is_settled_by_the_key_signature_and_flagged():
    runs = [[label("B#", key_text="B")], [label("B", key_text="B")], [label("B#", key_text="B")]]
    merged = merge_readings(runs)
    assert [(l.text, l.verified) for l in merged] == [("B", False)]


def test_note_seen_once_without_a_notehead_is_dropped():
    merged = merge_readings([[label("Gb", verified=False)], [], []])
    assert merged == []


def head_at(x, step, added=False):
    return Notehead(x, BOTTOM - step * UNIT / 2, 6.0, 5.0, step=float(step), added=added)


def test_misread_low_note_of_an_octave_takes_the_name_of_the_note_it_doubles():
    # The model reads the octave C2 + C1 as C2 + E1; the low notehead sits at C1.
    tokens = [note("C2", x=100.0), CHORD, note("E1", x=100.0)]
    result = labels_for_staff(staff(tokens, heads=[head_at(100.0, -4), head_at(100.0, -11, added=True)]))
    assert texts(result) == ["C"]
    assert result.labels[0].verified
    assert result.labels[0].step == -11


def test_low_note_that_is_not_an_octave_keeps_the_reading_and_stays_unverified():
    tokens = [note("C2", x=100.0), CHORD, note("E1", x=100.0)]
    result = labels_for_staff(staff(tokens, heads=[head_at(100.0, -4), head_at(100.0, -10, added=True)]))
    assert texts(result) == ["E"]
    assert not result.labels[0].verified


def test_octave_correction_keeps_the_accidental_of_the_doubled_note():
    tokens = [note("C2", lift="#", x=100.0), CHORD, note("E1", x=100.0)]
    result = labels_for_staff(staff(tokens, heads=[head_at(100.0, -4), head_at(100.0, -11, added=True)]))
    assert texts(result) == ["C#"]


def test_misread_key_signature_is_taken_from_neighbouring_lines():
    def line(k, lift):
        return labels_for_staff(staff([key(k), note("B2", lift, x=100.0), note("A2", "#", x=200.0)],
                                      heads=[Notehead(100.0, BOTTOM - 2 * UNIT / 2, 5, 5),
                                             Notehead(200.0, BOTTOM - 1 * UNIT / 2, 5, 5)]))

    lines = [line(-2, "b"), line(0, "_"), line(-2, "b")]
    apply_key_signatures(lines)
    assert texts(lines[1]) == ["Bb", "A#"]  # the flat is restored, the printed sharp is kept
    assert lines[1].labels[0].key_text == "Bb"


def test_reading_under_misread_key_is_outvoted():
    def reading(text, weak):
        return [Label(text, 100.0, 130.0, UNIT, True, BOTTOM, BOTTOM - 4 * UNIT, step=2, weak=weak)]

    merged = merge_readings([reading("Bb", True), reading("B", False), reading("B", False)])
    assert [(l.text, l.verified) for l in merged] == [("B", True)]
    merged = merge_readings([reading("Bb", True), reading("Bb", True), reading("Bb", True)])
    assert [(l.text, l.verified) for l in merged] == [("Bb", False)]
