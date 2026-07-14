"""
Merges freshly-computed pipeline results into dashboard/index.html's
embedded `const DATA = {...}` blob, and (optionally) writes the merged
object to a standalone JSON artifact.

Only the keys passed in `new_data` are added/overwritten -- everything
else already embedded in the dashboard (hand-curated visualisations such
as match trajectories) is left untouched. The returned summary says
exactly which keys were updated and which were retained, so a run report
can log that nothing computed went stale (DEF-011).

DEF-011: the end of the embedded object is found by actually parsing the
JSON (json.JSONDecoder.raw_decode) rather than searching for a ';\\n'
sentinel, which could also appear inside a JSON string value and silently
truncate the blob.
"""
from __future__ import annotations

import json

import numpy as np

_MARKER = "const DATA = "


def _jsonable(o):
    """json.dumps default hook for numpy scalars/arrays."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def update_dashboard_data(new_data: dict, html_path: str,
                          json_artifact_path: str | None = None) -> dict:
    """Read html_path's embedded DATA object, merge new_data into it
    (shallow update: each top-level key in new_data replaces that same
    key in DATA wholesale), and write the result back to html_path.

    If json_artifact_path is given, the merged object is also written
    there (pretty-printed) as a standalone, diffable artifact.

    Returns {"updated": [...], "retained": [...]} -- the top-level keys
    that were overwritten/added vs. left as previously embedded.
    """
    with open(html_path) as f:
        html = f.read()

    start = html.index(_MARKER) + len(_MARKER)
    data, end = json.JSONDecoder().raw_decode(html, start)
    if not isinstance(data, dict):
        raise ValueError(f"embedded DATA in {html_path} is not a JSON object")

    updated = sorted(new_data.keys())
    retained = sorted(set(data.keys()) - set(new_data.keys()))
    data.update(new_data)

    blob = json.dumps(data, default=_jsonable)
    html = html[:start] + blob + html[end:]
    with open(html_path, "w") as f:
        f.write(html)

    if json_artifact_path is not None:
        with open(json_artifact_path, "w") as f:
            json.dump(data, f, indent=2, default=_jsonable)

    return {"updated": updated, "retained": retained}
