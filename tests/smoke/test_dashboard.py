"""The results dashboard is built from the committed result files."""

import json
import re
from pathlib import Path

from neurodrive.benchmark.dashboard import build_dashboard, collect

RESULTS = Path("results")


def test_collect_reads_the_committed_results():
    data = collect(RESULTS)
    names = [b["name"] for b in data["benchmarks"]]
    assert "ppo_vs_neat_1m" in names and "ppo_vs_neat_5m" in names
    b5 = next(b for b in data["benchmarks"] if b["name"] == "ppo_vs_neat_5m")
    assert len(b5["runs"]) == 6 and len(b5["tracks"]) == 26
    assert b5["reselected"] is not None and "ppo/1" in b5["reselected"]
    assert (
        data["grand_prix"]["standings"][0]["points"]
        >= data["grand_prix"]["standings"][-1]["points"]
    )


def test_dashboard_is_one_self_contained_page(tmp_path):
    out = build_dashboard(RESULTS, tmp_path / "d.html")
    html = out.read_text(encoding="utf-8")
    assert "<script src" not in html and "<link" not in html  # works offline
    payload = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert payload is not None
    assert json.loads(payload.group(1))["benchmarks"]
    assert "__DATA__" not in html
