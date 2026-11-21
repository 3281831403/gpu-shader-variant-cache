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


class VariantError(Exception):
    """Raised for invalid variant-cache input or an impossible request."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
