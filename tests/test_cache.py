"""Behavioural tests for the GPU shader variant cache stubs.

Every test is red on the shipped stubs: each one reaches a public function that
contains only ``raise NotImplementedError``.  Do not edit, skip, xfail, or
weaken these tests.
"""

import pytest

from variant.cache import (
    VariantError,
    compile_budget,
    insert,
    invalidate,
    lookup,
    merge_features,
    validate_cache,
    variant_batch,
    variant_key,
)


def make_cache(capacity, *entries):
    """Build a cache from ``(key, program, last_used, deps)`` tuples."""
    slots = []
    clock = 0
    for key, program, last_used, deps in entries:
        slots.append(
            {
                "key": key,
                "program": program,
                "last_used": last_used,
                "deps": list(deps),
            }
        )
        clock = max(clock, last_used + 1)
    return {"capacity": capacity, "clock": clock, "slots": slots}


def slot_keys(cache):
    """Return cache keys in slot order."""
    return [slot["key"] for slot in cache["slots"]]


def request(base, program, deps=()):
    """Build one valid batch request."""
    return {
        "base": base,
        "features": {"features": {"quality": "high", "pass": "forward"}},
        "program": program,
        "deps": list(deps),
    }


def test_defect_01_key_is_independent_of_feature_order():
    first = variant_key(
        "mat/standard",
        {"features": {"quality": "high", "pass": "forward", "fog": 1}},
    )
    second = variant_key(
        "mat/standard",
        {"features": {"fog": 1, "quality": "high", "pass": "forward"}},
    )
    assert first == second


def test_defect_02_key_uses_stable_sha256_digest():
    key = variant_key(
        "mat/standard",
        {"features": {"quality": "high", "pass": "forward"}},
    )
    assert isinstance(key, str)
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")
    assert key == variant_key(
        "mat/standard",
        {"features": {"pass": "forward", "quality": "high"}},
    )


def test_defect_03_lru_eviction_keeps_hot_variant():
    cache = make_cache(
        2,
        ("mat/hot", "program-hot", 5, ("shaders/hot.hlsl",)),
        ("mat/cold", "program-cold", 1, ("shaders/cold.hlsl",)),
    )
    cache = insert(cache, "mat/new", "program-new", 2)
    assert len(cache["slots"]) == 2
    assert set(slot_keys(cache)) == {"mat/hot", "mat/new"}


def test_defect_04_compile_budget_uses_milliseconds():
    pending = [
        {"key": "heavy", "ms": 9},
        {"key": "small", "ms": 4},
        {"key": "tiny", "ms": 1},
    ]
    assert compile_budget(pending, 10) == ["heavy", "tiny"]
    assert compile_budget([{"key": "exact", "ms": 10}], 10) == ["exact"]
    assert compile_budget([{"key": "over", "ms": 11}], 10) == []


def test_defect_05_invalidate_follows_dependencies():
    cache = make_cache(
        4,
        (
            "mat/fog",
            "program-fog",
            0,
            ("shaders/fog.hlsl", "shaders/common.hlsl"),
        ),
        ("mat/shadow", "program-shadow", 1, ("shaders/shadow.hlsl",)),
        ("mat/common", "program-common", 2, ("shaders/common.hlsl",)),
    )
    removed = invalidate(cache, ["shaders/common.hlsl"])
    assert removed == ["mat/fog", "mat/common"]
    assert slot_keys(cache) == ["mat/shadow"]


def test_defect_06_merge_conflict_prefers_b_and_marks():
    base = {"features": {"quality": "high", "fog": 1}}
    override = {"features": {"quality": "low", "shadow": 1}}
    merged = merge_features(base, override)
    assert merged["features"] == {"quality": "low", "fog": 1, "shadow": 1}
    assert merged["conflicts"] == ["quality"]
    assert base["features"]["quality"] == "high"


def test_defect_07_lookup_refreshes_lru_recency():
    cache = make_cache(
        2,
        ("mat/a", "program-a", 0, ()),
        ("mat/b", "program-b", 1, ()),
    )
    assert lookup(cache, "mat/a") == 0
    cache = insert(cache, "mat/c", "program-c", 2)
    assert set(slot_keys(cache)) == {"mat/a", "mat/c"}


def test_defect_08_insert_enforces_capacity():
    cache = {"capacity": 2, "clock": 0, "slots": []}
    for key in ("key-a", "key-b", "key-c"):
        cache = insert(cache, key, f"program-{key}", 2)
    assert cache["capacity"] == 2
    assert len(cache["slots"]) == 2
    assert set(slot_keys(cache)) == {"key-b", "key-c"}
    assert validate_cache(cache, 2) == []

    broken = make_cache(
        1,
        ("duplicate", "", 0, ()),
        ("duplicate", "program-ok", 1, ()),
    )
    problems = validate_cache(broken, 2)
    assert any(problem.startswith("duplicate key:") for problem in problems)
    assert any(problem.startswith("capacity mismatch:") for problem in problems)
    assert any(problem.startswith("empty program:") for problem in problems)


def test_defect_09_batch_is_isolated_and_atomic():
    first = variant_batch(
        [request("mat/a", "program-a", ("shaders/a.hlsl",))],
        {"capacity": 2},
    )
    second = variant_batch(
        [request("mat/b", "program-b", ("shaders/b.hlsl",))],
        {"capacity": 2},
    )
    assert [item["program"] for item in first["results"]] == ["program-a"]
    assert [item["program"] for item in second["results"]] == ["program-b"]
    assert len(first["cache"]["slots"]) == 1
    assert len(second["cache"]["slots"]) == 1

    seed = make_cache(
        2,
        ("mat/seed", "program-seed", 0, ("shaders/seed.hlsl",)),
    )
    seed_before = make_cache(
        2,
        ("mat/seed", "program-seed", 0, ("shaders/seed.hlsl",)),
    )
    invalid = request("mat/b", "", ("shaders/b.hlsl",))
    with pytest.raises(VariantError):
        variant_batch(
            [request("mat/a", "program-a", ()), invalid],
            {"capacity": 2, "cache": seed},
        )
    assert seed == seed_before


def test_defect_10_capacity_zero_insert_raises():
    cache = {"capacity": 0, "clock": 0, "slots": []}
    with pytest.raises(VariantError):
        insert(cache, "mat/a", "program-a", 0)


def test_combo_feature_order_same_key():
    left = variant_key(
        "mat/standard",
        {"features": {"quality": "high", "fog": 1, "pass": "forward"}},
    )
    right = variant_key(
        "mat/standard",
        {"features": {"pass": "forward", "quality": "high", "fog": 1}},
    )
    assert left == right


def test_combo_single_variant_lookup_hit():
    cache = make_cache(1, ("mat/only", "program-only", 0, ()))
    assert lookup(cache, "mat/only") == 0
    assert lookup(cache, "mat/missing") == -1


def test_combo_capacity_zero_batch_raises():
    with pytest.raises(VariantError):
        variant_batch(
            [request("mat/a", "program-a")],
            {"capacity": 0},
        )


def test_combo_empty_batch_returns_empty():
    assert variant_batch([], {"capacity": 2}) == {}


def test_combo_batch_preserves_order():
    requests = [
        request("mat/a", "program-a", ("shaders/a.hlsl",)),
        request("mat/b", "program-b", ("shaders/b.hlsl",)),
        request("mat/c", "program-c", ("shaders/c.hlsl",)),
    ]
    result = variant_batch(requests, {"capacity": 3})
    assert [item["program"] for item in result["results"]] == [
        "program-a",
        "program-b",
        "program-c",
    ]
    assert len({item["key"] for item in result["results"]}) == 3
