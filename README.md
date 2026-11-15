# chart-labeler

Writes the name of each bass note onto a PDF song chart, so the bass line can
be played without reading staff notation. Works on scans, phone photos saved
as PDF, and PDFs exported from notation software. Runs entirely on your own
computer; nothing is uploaded.

## Use

Needs [uv](https://docs.astral.sh/uv/). Clone the repository, then run the
`chart-labeler` file in it. It is a uv script: the first run fetches Python 3.13
and the dependencies listed at its top, at the versions in `chart-labeler.lock`.

```sh
git clone https://github.com/dtanner/chart-labeler.git
cd chart-labeler
./chart-labeler "/path/to/song.pdf"
```

It works from any directory, so it can be linked onto the PATH once and then
used anywhere:

```sh
ln -s "$PWD/chart-labeler" ~/.local/bin/chart-labeler
chart-labeler "/path/to/song.pdf"
```

This writes `song (bass notes).pdf` next to the original. The original is not
changed. Several PDFs can be given at once; they are all checked before any is
read. The first run also downloads the recognition models (about 250 MB). A 5-page song takes about 2 minutes.

Options:

- `-o PATH` output file, or a directory when labelling several PDFs
- `--all-voices` also label notes played above a held bass note
- `--fast` read each page once instead of three times

## What gets labelled

On piano scores, the bottom (bass-clef) staff of each system. On single-staff
parts, any staff in bass clef. For each beat position the lowest note is
named, with its sharp or flat from the key signature or an accidental applied:
`Db`, `F#`, `C`.

When a low note is held while other notes move above it on the same staff,
only the held note is labelled, because that is the bass. `--all-voices`
labels the moving notes too.

## Colours

- **Red**: the checks below agreed. That makes a wrong name unlikely, not
  impossible: the checks can agree on the same mistake.
- **Orange**: less certain. The command prints the reason for each one.
  Usually the note name is right and only its exact position could not be
  confirmed.

## How it checks itself

Recognition uses [homr](https://github.com/liebharc/homr). Each note must pass:

1. The pitch the model reads has to coincide with a notehead found separately
   at the matching height on the staff.
2. Each page is read three times at different resolutions; the readings vote.
3. Noteheads many ledger lines below the staff are searched for separately
   and measured against the printed staff lines. When the low note of an
   octave was read differently from the note it doubles (C over "E"), the
   notehead's position decides.
4. Accidentals that the readings disagree on are settled by the key signature.
   After a key change in the middle of a line, the key is taken from the start
   of the next line, where it is printed in full. A key signature that differs
   from the lines before and after it was misread, and is replaced by theirs.
5. A staff is labelled only when most readings agree it is a bass part, so a
   vocal line or the piano's right hand does not get labels.

Known limit: double flats and double sharps are read as plain notes (`Bbb`
comes out as `B`), and can be printed in red.

## Development

```sh
uv run pytest
```

The dependencies are listed twice: in `pyproject.toml` for development, and at
the top of `chart-labeler` for the launcher, with its own lock file. A test
checks that the two lists match. After changing them, run `uv lock` and
`uv lock --script chart-labeler`.

GitHub Actions runs the tests on every push.

### Measuring accuracy

`bench/bench.py` scores the labeller against scores whose notes are known:
songs for voice and piano from [OpenScore Lieder](https://github.com/OpenScore/Lieder),
rendered to PDF by MuseScore 4, which also exports each note and its place on
the page. Every label is compared with the note printed there.

```sh
.venv/bin/python -I bench/bench.py build     # render the sample, write the answer keys
.venv/bin/python -I bench/bench.py scans     # scan-like copies (crooked, blurred, noisy)
.venv/bin/python -I bench/bench.py score --scans
.venv/bin/python -I bench/bench.py show      # crops around each error, in bench-data/errors
```

`build` expects the Lieder repository in `bench-data/lieder` and the list of
scores to use in `bench-data/sample.txt`. The first `score` takes about half an
hour; later runs reuse the stored recognition results and take a few minutes.
The column to watch is `RED WRONG`: a wrong name printed as certain.

## License

GNU Affero General Public License v3.0 or later (see `LICENSE`), matching
[homr](https://github.com/liebharc/homr), which does the music reading.
