"""LRU cache behaviour for services.market.charts.

WHY: chart cache eviction and hit/miss accounting were uncovered by existing
tests. These behaviours are load-bearing: a leak here grows matplotlib memory
unboundedly in long-running processes.
"""

from __future__ import annotations

import pytest

from services.market import charts as cs


@pytest.fixture(autouse=True)
def _reset_cache():
    cs.ChartService.cache_clear()
    yield
    cs.ChartService.cache_clear()


def test_cache_put_and_get_round_trip():
    cs.ChartService.cache_put(("k", 1), "PAYLOAD")
    assert cs.ChartService.cache_get(("k", 1)) == "PAYLOAD"


def test_cache_miss_returns_none_and_increments_miss_counter():
    assert cs.ChartService.cache_get(("missing",)) is None
    stats = cs.ChartService.cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0


def test_cache_hit_increments_hit_counter_and_promotes_recency():
    cs.ChartService.cache_put(("a",), "A")
    cs.ChartService.cache_put(("b",), "B")
    # Access "a" so it becomes most-recent.
    assert cs.ChartService.cache_get(("a",)) == "A"
    stats = cs.ChartService.cache_stats()
    assert stats["hits"] == 1
    assert stats["size"] == 2


def test_cache_evicts_oldest_when_over_max(monkeypatch):
    monkeypatch.setattr(cs, "_CACHE_MAX_ENTRIES", 3)
    for i in range(5):
        cs.ChartService.cache_put((i,), f"v{i}")
    stats = cs.ChartService.cache_stats()
    assert stats["size"] == 3
    # Oldest two (0, 1) should have been evicted.
    assert cs.ChartService.cache_get((0,)) is None
    assert cs.ChartService.cache_get((1,)) is None
    assert cs.ChartService.cache_get((4,)) == "v4"


def test_cache_lru_promotion_keeps_recently_used(monkeypatch):
    monkeypatch.setattr(cs, "_CACHE_MAX_ENTRIES", 2)
    cs.ChartService.cache_put(("x",), "X")
    cs.ChartService.cache_put(("y",), "Y")
    # Touch x so it becomes most-recent; inserting z should evict y.
    cs.ChartService.cache_get(("x",))
    cs.ChartService.cache_put(("z",), "Z")
    assert cs.ChartService.cache_get(("x",)) == "X"
    assert cs.ChartService.cache_get(("y",)) is None
    assert cs.ChartService.cache_get(("z",)) == "Z"


def test_cached_or_build_memoises_calls():
    """The chart-level memo entry point (replaces the removed cached_chart
    decorator / generate_cached — single cache pattern per 2026-09 review)."""
    from services.market.analysis.statistical import _cached_or_build

    calls = {"n": 0}

    def builder() -> str:
        calls["n"] += 1
        return f"png-{calls['n']}"

    assert _cached_or_build(("t", "AAPL"), builder) == "png-1"
    assert _cached_or_build(("t", "AAPL"), builder) == "png-1"  # served from cache
    assert calls["n"] == 1
    assert _cached_or_build(("t", "MSFT"), builder) == "png-2"
    assert calls["n"] == 2


def test_cached_or_build_does_not_cache_empty_result():
    from services.market.analysis.statistical import _cached_or_build

    calls = {"n": 0}

    def builder() -> str:
        calls["n"] += 1
        return ""  # falsy: must not be cached

    _cached_or_build(("k",), builder)
    _cached_or_build(("k",), builder)
    assert calls["n"] == 2


def test_cached_or_build_does_not_cache_none_or_non_string():
    from services.market.analysis.statistical import _cached_or_build

    assert _cached_or_build(("none",), lambda: None) is None
    assert cs.ChartService.cache_get(("none",)) is None

    # A builder returning a non-string (e.g. a raw Figure) must not be cached —
    # only base64 strings belong in the chart cache.
    sentinel = object()
    assert _cached_or_build(("obj",), lambda: sentinel) is sentinel
    assert cs.ChartService.cache_get(("obj",)) is None


def test_features_hash_stable_for_same_input():
    h1 = cs.features_hash({"a": 1, "b": [1, 2, 3]})
    h2 = cs.features_hash({"b": [1, 2, 3], "a": 1})  # different key order
    assert h1 == h2
    assert len(h1) == 12


def test_features_hash_handles_unserialisable_input():
    # Should not raise; falls back to repr.
    h = cs.features_hash({"obj": object()})
    assert isinstance(h, str) and len(h) == 12
