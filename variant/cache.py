"""GPU shader variant cache primitives.

A variant is represented as a small dict::

    {"features": {"quality": "high", "pass": "forward"}}

A cache is a mutable dict with the following shape::

    {
        "capacity": 4,              # maximum number of slots
        "clock": 3,                 # next logical access tick
        "slots": [
            {
                "key": "...",       # result of variant_key()
                "program": "...",   # compiled program id; never empty
                "last_used": 2,     # logical tick of the last hit or insert
                "deps": ["..."],    # source files that feed this variant
            },
        ],
    }

A slot's index is its position in ``slots``.  All helpers are deterministic,
offline, and use only the Python standard library.  Stable keys use
``hashlib``; ordered bookkeeping may use ``collections``.  Nothing reads the
wall clock, the network, or random state, and nothing sleeps.
"""

import copy
import hashlib
import json
import math
from collections.abc import Mapping

_MISSING = object()


class VariantError(Exception):
    """Raised for invalid variant-cache input or an impossible request."""


def _is_int(value) -> bool:
    """Return ``True`` for a real integer, excluding ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value) -> bool:
    """Return ``True`` for a strictly positive real integer."""
    return _is_int(value) and value > 0


def _canonical(value, _seen=None):
    """Return a deterministic JSON-encodable form of one feature value.

    Mappings are rebuilt with their keys validated and left for
    :func:`json.dumps`'s ``sort_keys`` to order; lists keep their order.
    Values that cannot be encoded deterministically raise :class:`VariantError`.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VariantError("feature value is not a finite number")
        return value
    if isinstance(value, (list, Mapping)):
        if _seen is None:
            _seen = set()
        identity = id(value)
        if identity in _seen:
            raise VariantError("feature value contains a cycle")
        _seen.add(identity)
        try:
            if isinstance(value, list):
                return [_canonical(item, _seen) for item in value]
            result = {}
            for name, item in value.items():
                if not isinstance(name, str):
                    raise VariantError("feature names must be strings")
                result[name] = _canonical(item, _seen)
            return result
        finally:
            _seen.discard(identity)
    raise VariantError(
        "feature value of type "
        f"{type(value).__name__} cannot be encoded deterministically"
    )


def _feature_mapping(value, label: str) -> Mapping:
    """Validate a variant dict and return its ``"features"`` mapping."""
    if not isinstance(value, Mapping):
        raise VariantError(f"{label} must be a variant mapping")
    features = value.get("features")
    if not isinstance(features, Mapping):
        raise VariantError(f"{label} must contain a mapping under 'features'")
    return features


def _require_cache(cache) -> list:
    """Validate the shared cache envelope and return its slot list."""
    if not isinstance(cache, dict):
        raise VariantError("cache must be a mutable mapping")
    slots = cache.get("slots")
    if not isinstance(slots, list):
        raise VariantError("cache['slots'] must be a list")
    clock = cache.get("clock")
    if not _is_int(clock) or clock < 0:
        raise VariantError("cache['clock'] must be a non-negative integer")
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            raise VariantError(f"cache slot {index} must be a mapping")
        if not isinstance(slot.get("key"), str):
            raise VariantError(f"cache slot {index} has no string key")
        if not _is_int(slot.get("last_used")):
            raise VariantError(f"cache slot {index} has no integer last_used")
    return slots


def _evict(cache: dict) -> None:
    """Remove the least recently used slot (ties break by lowest index)."""
    slots = cache["slots"]
    victim = min(
        range(len(slots)),
        key=lambda index: (slots[index]["last_used"], index),
    )
    del slots[victim]


def _trim_to_capacity(cache: dict, capacity: int) -> None:
    """Evict LRU slots until the cache holds at most ``capacity`` slots."""
    while len(cache["slots"]) > capacity:
        _evict(cache)


def variant_key(base: str, features: dict) -> str:
    """Return a stable cache key for one shader variant.

    ``features`` is a variant dict shaped like ``{"features": {...}}``.  Only
    its ``"features"`` mapping contributes to the key; sibling metadata is
    ignored.  Feature-name order must never matter: names are sorted
    lexicographically at every nesting level, and a nested mapping is
    canonicalised recursively.  A list value keeps its order because it is
    data, not a set of alternatives.  The canonical representation is hashed
    with SHA-256, never with Python's per-process ``hash()``.

    Args:
        base: non-empty shader or material base name, included verbatim.
        features: variant dict ``{"features": {...}}`` whose feature names are
            strings and whose values are deterministic scalars, lists, or
            mappings.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest.  Equal ``base``
        and equal feature content produce the same digest regardless of dict
        insertion order; a different base or different content is designed to
        produce a different digest.

    Raises:
        VariantError: ``base`` is not a non-empty string, ``features`` is not a
            mapping with a mapping under ``"features"``, or a value cannot be
            encoded deterministically.
    """
    if not isinstance(base, str) or not base:
        raise VariantError("base must be a non-empty string")
    feature_map = _feature_mapping(features, "features")
    payload = json.dumps(
        {"base": base, "features": _canonical(feature_map)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lookup(cache: dict, key: str) -> int:
    """Find a variant and record the hit for LRU purposes.

    On a hit, set that slot's ``last_used`` to the current ``cache["clock"]``
    and then advance ``cache["clock"]`` by one, so a later :func:`insert`
    observes the hit as the most recently used slot.  On a miss, change
    neither recency nor the clock.  Key comparison is exact string equality;
    there is no prefix, case-folding, or fuzzy matching, and the tolerance is
    zero.

    Args:
        cache: cache mapping in the module-level shape.
        key: exact cache key returned by :func:`variant_key`.

    Returns:
        The slot index (position in ``cache["slots"]``) on a hit, otherwise
        ``-1``.

    Raises:
        VariantError: ``cache`` or ``key`` is malformed.
    """
    slots = _require_cache(cache)
    if not isinstance(key, str) or not key:
        raise VariantError("key must be a non-empty string")
    for index, slot in enumerate(slots):
        if slot["key"] == key:
            slot["last_used"] = cache["clock"]
            cache["clock"] += 1
            return index
    return -1


def insert(cache: dict, key: str, program: str, capacity: int) -> dict:
    """Insert or refresh one compiled variant under a hard capacity limit.

    ``capacity`` is authoritative and must be a positive integer; it is
    recorded in ``cache["capacity"]``.  If ``key`` already exists, replace its
    program and refresh ``last_used`` without creating a slot.  If it is new
    and there is room, append a slot.  If the cache is full, evict exactly the
    slot with the smallest ``last_used`` value (ties break by lowest slot
    index), then append the new slot.  The number of slots must never exceed
    capacity, so an over-capacity input is trimmed by LRU until it fits.
    Surviving slots keep their relative order and the new slot is appended.

    A refreshed or newly inserted slot receives the current logical tick, and
    the clock advances by one.  An existing slot keeps its dependency list; a
    new slot starts with ``"deps": []``.  An empty program is not a valid
    compiled variant.

    Args:
        cache: cache mapping in the module-level shape.
        key: exact cache key; must be a non-empty string.
        program: non-empty compiled program identifier or blob.
        capacity: positive maximum number of slots.

    Returns:
        The same ``cache`` object after mutation.

    Raises:
        VariantError: ``capacity`` is not a positive integer, ``program`` or
            ``key`` is empty or malformed, or ``cache`` has an invalid shape.
    """
    if not _is_positive_int(capacity):
        raise VariantError("capacity must be a positive integer")
    if not isinstance(key, str) or not key:
        raise VariantError("key must be a non-empty string")
    if not isinstance(program, str) or not program:
        raise VariantError("program must be a non-empty string")
    slots = _require_cache(cache)

    cache["capacity"] = capacity
    _trim_to_capacity(cache, capacity)

    for slot in slots:
        if slot["key"] == key:
            slot["program"] = program
            slot["last_used"] = cache["clock"]
            cache["clock"] += 1
            return cache

    if len(slots) >= capacity:
        _evict(cache)

    slots.append(
        {
            "key": key,
            "program": program,
            "last_used": cache["clock"],
            "deps": [],
        }
    )
    cache["clock"] += 1
    return cache


def compile_budget(pending: list[dict], max_ms: int) -> list[str]:
    """Select the variants that fit in one frame's compile-time budget.

    ``pending`` is processed in input order.  Each item is a dict with at least
    ``{"key": str, "ms": non-negative int}``, where ``ms`` is the full compile
    cost of that variant.  A candidate is selected when
    ``spent_ms + item["ms"] <= max_ms``; a candidate that does not fit is
    skipped and later, smaller candidates are still considered.  The returned
    keys preserve input order.  The arithmetic is exact integer arithmetic
    with tolerance zero: using exactly all of the budget is allowed, while
    exceeding it by even one millisecond is not.  There is no rounding and no
    floating-point epsilon.

    Args:
        pending: compile candidates in submission order.
        max_ms: non-negative integer millisecond budget for this frame.

    Returns:
        The selected keys, in the same relative order as ``pending``.

    Raises:
        VariantError: ``max_ms`` is not a non-negative integer, or an item is
            malformed or has a negative/non-integer ``ms``.
    """
    if not _is_int(max_ms) or max_ms < 0:
        raise VariantError("max_ms must be a non-negative integer")
    if not isinstance(pending, list):
        raise VariantError("pending must be a list")

    spent_ms = 0
    selected = []
    for index, item in enumerate(pending):
        if not isinstance(item, Mapping):
            raise VariantError(f"pending item {index} must be a mapping")
        key = item.get("key")
        cost = item.get("ms")
        if not isinstance(key, str) or not key:
            raise VariantError(f"pending item {index} has no non-empty string key")
        if not _is_int(cost) or cost < 0:
            raise VariantError(
                f"pending item {index} ms must be a non-negative integer"
            )
        if spent_ms + cost <= max_ms:
            selected.append(key)
            spent_ms += cost
    return selected


def invalidate(cache: dict, changed_files: list[str]) -> list[str]:
    """Remove cached variants whose dependencies changed.

    Each slot's ``deps`` list names every source file that feeds that variant,
    including transitive headers.  A slot is invalid when any dependency is
    exactly equal to any entry in ``changed_files``.  Matching is full-string
    and exact, with tolerance zero: it is not enough to compare only the
    variant key, the program name, or a basename.  Matching slots are removed
    in place, survivors keep their order, and the removed keys are returned in
    their original slot order.

    Args:
        cache: cache mapping whose slots contain ``deps`` lists.
        changed_files: exact file paths changed by this hot-reload event.

    Returns:
        Keys removed from the cache, in original slot order.

    Raises:
        VariantError: a slot is missing a well-formed dependency list or
            ``changed_files`` is malformed.
    """
    slots = _require_cache(cache)
    if not isinstance(changed_files, list):
        raise VariantError("changed_files must be a list of strings")
    for name in changed_files:
        if not isinstance(name, str):
            raise VariantError("changed_files must contain only strings")

    for index, slot in enumerate(slots):
        deps = slot.get("deps")
        if not isinstance(deps, list) or any(
            not isinstance(dep, str) for dep in deps
        ):
            raise VariantError(
                f"cache slot {index} has no well-formed dependency list"
            )

    changed = set(changed_files)
    removed = []
    survivors = []
    for slot in slots:
        if any(dep in changed for dep in slot["deps"]):
            removed.append(slot["key"])
        else:
            survivors.append(slot)
    slots[:] = survivors
    return removed


def merge_features(a: dict, b: dict) -> dict:
    """Merge two variants' feature maps, preferring ``b`` on a conflict.

    Both arguments are variant dicts shaped like ``{"features": {...}}``.
    Features present on only one side are copied.  When both sides contain the
    same feature name, equal values are not a conflict and are kept once; a
    different value is a conflict, in which case ``b``'s value wins and the
    feature name is recorded.  The input dicts are not mutated.

    Args:
        a: base variant.
        b: overriding variant.

    Returns:
        A new dict with this shape::

            {"features": {...}, "conflicts": ["sorted", "names"]}

        ``conflicts`` is sorted ascending and is empty when there is no
        conflict.

    Raises:
        VariantError: either argument is not a variant dict with a mapping
            under ``"features"``.
    """
    features_a = _feature_mapping(a, "a")
    features_b = _feature_mapping(b, "b")
    for name in list(features_a) + list(features_b):
        if not isinstance(name, str):
            raise VariantError("feature names must be strings")

    merged = dict(features_a)
    conflicts = []
    for name, value in features_b.items():
        if name in merged and merged[name] != value:
            conflicts.append(name)
        merged[name] = value
    conflicts.sort()
    return {"features": merged, "conflicts": conflicts}


def validate_cache(cache: dict, capacity: int) -> list[str]:
    """Return all structural problems found in a cache.

    Independent checks are all performed; the result is not short-circuited.
    Issue strings start with one of these stable prefixes:

    * ``"duplicate key:"`` -- two or more slots have the same exact key;
    * ``"capacity mismatch:"`` -- ``cache["capacity"]`` differs from the
      expected ``capacity``, or the number of slots exceeds ``capacity``, or
      the expected capacity is not positive;
    * ``"empty program:"`` -- a slot's program is missing, ``None``, or the
      empty string.

    Per-slot findings follow slot order.  The function does not raise for
    malformed slot contents and does not mutate the cache.

    Args:
        cache: cache mapping to inspect.
        capacity: expected positive maximum number of slots.

    Returns:
        A deterministic list of human-readable issue strings; it is empty only
        for a structurally valid cache.

    Raises:
        VariantError: ``cache`` itself is not a mapping.
    """
    if not isinstance(cache, Mapping):
        raise VariantError("cache must be a mapping")

    issues = []
    slots = cache.get("slots")
    if not isinstance(slots, list):
        issues.append("capacity mismatch: cache has no slot list")
        slots = []

    if not _is_positive_int(capacity):
        issues.append(
            "capacity mismatch: expected capacity must be a positive integer, "
            f"got {capacity!r}"
        )
    else:
        stored = cache.get("capacity")
        if not _is_int(stored) or stored != capacity:
            issues.append(
                f"capacity mismatch: cache capacity {stored!r} does not match "
                f"expected {capacity!r}"
            )
        if len(slots) > capacity:
            issues.append(
                f"capacity mismatch: {len(slots)} slots exceed capacity "
                f"{capacity!r}"
            )

    positions = {}
    for index, slot in enumerate(slots):
        if isinstance(slot, Mapping):
            slot_key = slot.get("key")
            if isinstance(slot_key, str):
                positions.setdefault(slot_key, []).append(index)
    for slot_key, indexes in positions.items():
        if len(indexes) > 1:
            issues.append(
                f"duplicate key: {slot_key!r} appears at slots {indexes}"
            )

    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            issues.append(f"empty program: slot {index} is not a mapping")
            continue
        program = slot.get("program", _MISSING)
        if program is _MISSING or program is None or (
            isinstance(program, str) and program == ""
        ):
            issues.append(f"empty program: slot {index} has no program")
    return issues


def _validate_request(request: dict) -> tuple:
    """Validate one batch request and return ``(key, program, deps)``."""
    if not isinstance(request, Mapping):
        raise VariantError("each request must be a mapping")
    base = request.get("base")
    if not isinstance(base, str) or not base:
        raise VariantError("request base must be a non-empty string")
    features = request.get("features")
    _feature_mapping(features, "request features")
    program = request.get("program")
    if not isinstance(program, str) or not program:
        raise VariantError("request program must be a non-empty string")
    deps = request.get("deps", _MISSING)
    if deps is _MISSING:
        deps = []
    if not isinstance(deps, list) or any(
        not isinstance(dep, str) for dep in deps
    ):
        raise VariantError("request deps must be a list of strings")
    return variant_key(base, features), program, list(deps)


def _copy_seed(seed, capacity: int) -> dict:
    """Return a private working cache built from an optional seed cache."""
    if seed is _MISSING or seed is None:
        return {"capacity": capacity, "clock": 0, "slots": []}
    if not isinstance(seed, Mapping):
        raise VariantError("seed cache must be a mapping")
    slots = seed.get("slots")
    clock = seed.get("clock", 0)
    if not isinstance(slots, list):
        raise VariantError("seed cache has no slot list")
    if not _is_int(clock) or clock < 0:
        raise VariantError("seed cache clock must be a non-negative integer")
    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            raise VariantError(f"seed slot {index} must be a mapping")
        if not isinstance(slot.get("key"), str) or not slot["key"]:
            raise VariantError(f"seed slot {index} has no non-empty string key")
        program = slot.get("program")
        if not isinstance(program, str) or not program:
            raise VariantError(f"seed slot {index} has no program")
        if not _is_int(slot.get("last_used")):
            raise VariantError(f"seed slot {index} has no integer last_used")
        deps = slot.get("deps", [])
        if not isinstance(deps, list) or any(
            not isinstance(dep, str) for dep in deps
        ):
            raise VariantError(f"seed slot {index} has no dependency list")

    working = {
        "capacity": capacity,
        "clock": clock,
        "slots": copy.deepcopy(list(slots)),
    }
    _trim_to_capacity(working, capacity)
    return working


def variant_batch(requests: list[dict], config: dict) -> dict:
    """Compile a batch of variant requests atomically and in order.

    ``config`` is a mapping with:

    * ``"capacity"``: required positive integer slot limit;
    * ``"cache"``: optional seed cache mapping.  It is copied and never
      mutated, so a failed batch cannot leave partial state behind and two
      batches never share a mutable cache handle.

    Each request is a dict with ``"base"``, ``"features"`` (the variant shape
    ``{"features": {...}}``), a non-empty ``"program"``, and an optional
    ``"deps"`` list.  Every request is validated before any compile work; one
    invalid request raises and rolls back the whole batch.  Keys are computed
    per request, so equal feature maps under different bases are distinct and
    one request cannot overwrite another request's program.  A request whose
    key is already present in the working cache reuses the cached program;
    otherwise its program is inserted subject to the capacity limit.

    An empty request list returns ``{}``.

    Args:
        requests: batch requests in submission order.
        config: capacity and optional seed cache.

    Returns:
        For a non-empty batch, a dict::

            {
                "results": [
                    {"key": str, "program": str, "slot": int},
                    ...,
                ],
                "cache": {...},
            }

        ``results`` preserves request order, and ``cache`` is the resulting
        private working cache.

    Raises:
        VariantError: capacity is invalid, any request is malformed or has an
            empty program, or the seed cache is malformed.
    """
    if not isinstance(config, Mapping):
        raise VariantError("config must be a mapping")
    capacity = config.get("capacity")
    if not _is_positive_int(capacity):
        raise VariantError("config capacity must be a positive integer")
    if not isinstance(requests, list):
        raise VariantError("requests must be a list")

    if not requests:
        return {}

    # Validate every request before doing any compile work, so one bad request
    # rolls back the whole batch.
    validated = [_validate_request(request) for request in requests]

    working = _copy_seed(config.get("cache", _MISSING), capacity)

    results = []
    for key, program, deps in validated:
        index = lookup(working, key)
        if index != -1:
            results.append(
                {
                    "key": key,
                    "program": working["slots"][index]["program"],
                    "slot": index,
                }
            )
            continue
        insert(working, key, program, capacity)
        index = len(working["slots"]) - 1
        working["slots"][index]["deps"] = list(deps)
        results.append({"key": key, "program": program, "slot": index})

    return {"results": results, "cache": working}
