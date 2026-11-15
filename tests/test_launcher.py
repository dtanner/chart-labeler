import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_launcher_dependencies_match_pyproject():
    """The chart-labeler script declares its own dependencies; they must stay those of the project."""
    text = (ROOT / "chart-labeler").read_text()
    block = re.search(r"^# /// script$\n(.*?)^# ///$", text, re.S | re.M).group(1)
    script = tomllib.loads("".join(line.removeprefix("# ").removeprefix("#") + "\n" for line in block.splitlines()))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert script["dependencies"] == project["dependencies"]
    assert script["requires-python"] == project["requires-python"]
