"""
Merges freshly-computed pipeline results into dashboard/index.html's
embedded `const DATA = {...}` blob.

Only the keys passed in `new_data` are added/overwritten -- everything
else already embedded in the dashboard (e.g. match trajectories, the
tournament table) is left untouched. This keeps every dashboard number
traceable to an actual run_all.py computation rather than a hand-typed
value, without requiring the whole dashboard's data model to be rebuilt
in one pass.
"""
from __future__ import annotations

import json

_MARKER = "const DATA = "


def update_dashboard_data(new_data: dict, html_path: str) -> None:
    """Read html_path's embedded DATA object, merge new_data into it
    (shallow update: each top-level key in new_data replaces that same
    key in DATA wholesale), and write the result back to html_path."""
    with open(html_path) as f:
        html = f.read()

    start = html.index(_MARKER) + len(_MARKER)
    end = html.index(";\n", start)

    data = json.loads(html[start:end])
    data.update(new_data)

    html = html[:start] + json.dumps(data) + html[end:]
    with open(html_path, "w") as f:
        f.write(html)
