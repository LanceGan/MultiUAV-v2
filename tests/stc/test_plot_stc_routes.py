"""Unit coverage for the sealed STC route-figure source selection."""

from pathlib import Path

import pytest

from plots.plot_stc_routes import _figure_stem, _resolve_source, _source_paths


def test_route_sources_reuse_main_run_without_gatsac_supplement():
    sources = _source_paths(Path("/runs/u3"))

    assert set(sources) == {"main"}
    assert _resolve_source("main", sources) == "main"
    assert _resolve_source("gatsac", sources) == "main"


def test_route_sources_prefer_separate_gatsac_supplement():
    sources = _source_paths(Path("/runs/u2"), Path("/runs/gatsac-u2"))

    assert set(sources) == {"main", "gatsac"}
    assert _resolve_source("gatsac", sources) == "gatsac"
    assert sources["gatsac"]["raw"] == Path("/runs/gatsac-u2/raw_records.json")


def test_route_source_rejects_unknown_key():
    with pytest.raises(KeyError, match="unavailable"):
        _resolve_source("unknown", {"main": {}})


def test_route_figure_stem_is_scenario_specific():
    assert _figure_stem(3, 42, 0) == "stc_u3_s42_seed0_routes"
    assert _figure_stem(4, 43, 2) == "stc_u4_s43_seed2_routes"
