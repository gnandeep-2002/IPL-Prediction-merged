"""
Tests for src/dashboard_export.py's DATA-blob merge logic.
"""
import json

from src.dashboard_export import update_dashboard_data


def _write_fixture(path, data):
    path.write_text(
        "<html><body><script>\n"
        f"const DATA = {json.dumps(data)};\n"
        "console.log(DATA);\n"
        "</script></body></html>"
    )


def test_new_top_level_key_is_added(tmp_path):
    html_path = tmp_path / "dashboard.html"
    _write_fixture(html_path, {"existing": {"a": 1}})

    update_dashboard_data({"new_section": {"b": 2}}, str(html_path))

    text = html_path.read_text()
    start = text.index("const DATA = ") + len("const DATA = ")
    end = text.index(";\n", start)
    data = json.loads(text[start:end])
    assert data == {"existing": {"a": 1}, "new_section": {"b": 2}}


def test_existing_key_is_overwritten_wholesale(tmp_path):
    html_path = tmp_path / "dashboard.html"
    _write_fixture(html_path, {"pre_match": {"old": True}})

    update_dashboard_data({"pre_match": {"new": True}}, str(html_path))

    text = html_path.read_text()
    start = text.index("const DATA = ") + len("const DATA = ")
    end = text.index(";\n", start)
    data = json.loads(text[start:end])
    assert data["pre_match"] == {"new": True}


def test_untouched_keys_are_preserved(tmp_path):
    html_path = tmp_path / "dashboard.html"
    _write_fixture(html_path, {"keep_me": [1, 2, 3], "also_keep": "hello"})

    update_dashboard_data({"new_key": 42}, str(html_path))

    text = html_path.read_text()
    start = text.index("const DATA = ") + len("const DATA = ")
    end = text.index(";\n", start)
    data = json.loads(text[start:end])
    assert data["keep_me"] == [1, 2, 3]
    assert data["also_keep"] == "hello"
    assert data["new_key"] == 42


def test_html_outside_the_data_blob_is_unchanged(tmp_path):
    html_path = tmp_path / "dashboard.html"
    _write_fixture(html_path, {"a": 1})
    before = html_path.read_text()
    before_prefix = before[:before.index("const DATA = ")]
    before_suffix = before[before.index(";\n", before.index("const DATA = ")):]

    update_dashboard_data({"b": 2}, str(html_path))

    after = html_path.read_text()
    assert after.startswith(before_prefix)
    assert after.endswith(before_suffix)
