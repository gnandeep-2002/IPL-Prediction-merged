from __future__ import annotations

import json

import numpy as np

_MARKER = "const DATA = "


def _jsonable(o):
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
