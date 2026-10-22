"""Locate files referenced by historical records after workspace reorganization.

Original records keep their bytes and hashes. Only local readers translate paths;
this module does not modify records or restore deleted caches/archives.
"""

import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _moves():
    manifest = json.loads((ROOT / "docs/workspace-paths.json").read_text())
    return sorted(manifest["moves"].items(), key=lambda pair: len(pair[0]), reverse=True)


def workspace_path(value):
    """Return the current absolute path for a repository path in an old record."""
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path
    relative = path.as_posix()
    for old, new in _moves():
        if relative == old or relative.startswith(old + "/"):
            relative = new + relative[len(old):]
            break
    return ROOT / relative
