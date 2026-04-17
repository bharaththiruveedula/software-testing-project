"""
Property-based tests (Hypothesis).

These test *invariants* rather than specific cases, covering the classes of
input we can't enumerate by hand:

    - SSE parser: event sequence is independent of how the raw byte stream
      is sliced at I/O boundaries (the exact failure mode of the old
      pre-refactor parser).
    - Statistical harness: percentile is bounded by min/max, bootstrap
      CI bounds lie within sample range, Mann-Whitney p is in [0, 1] and
      symmetric under group swap.
"""

from __future__ import annotations

import json
import random
from typing import List

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from load_runner import parse_sse_stream
from stat_regression import bootstrap_ci, mann_whitney_u, percentile


# ── SSE parser invariants ────────────────────────────────────────────────────


def _build_sse_body(payloads: List[str]) -> bytes:
    """Build a syntactic SSE body containing the given data payloads."""
    parts = [f"data: {p}\n\n" for p in payloads]
    return "".join(parts).encode("utf-8")


@st.composite
def sse_bodies_and_splits(draw):
    # Restrict to printable ASCII minus control chars: real SSE servers
    # never emit U+001E/U+001C etc., but str.splitlines() treats those as
    # line boundaries, so they'd trip the assertion on synthetic inputs.
    payloads = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    min_codepoint=0x20,
                    max_codepoint=0x7E,
                    blacklist_characters="\n\r",
                ),
                min_size=1,
                max_size=30,
            ),
            min_size=1,
            max_size=8,
        )
    )
    body = _build_sse_body(payloads)
    if len(body) <= 1:
        split_points: List[int] = []
    else:
        n_splits = draw(st.integers(min_value=0, max_value=min(6, len(body) - 1)))
        split_points = sorted(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=len(body) - 1),
                    min_size=n_splits,
                    max_size=n_splits,
                    unique=True,
                )
            )
        )
    # Build chunked slicing
    prev = 0
    chunks: List[bytes] = []
    for sp in split_points:
        chunks.append(body[prev:sp])
        prev = sp
    chunks.append(body[prev:])
    return payloads, chunks


class TestSSEParserProperties:

    @given(sse_bodies_and_splits())
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_event_count_matches_payload_count(self, case):
        payloads, chunks = case
        events = list(parse_sse_stream(chunks))
        assert len(events) == len(payloads)

    @given(sse_bodies_and_splits())
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_event_data_roundtrips_through_arbitrary_chunking(self, case):
        payloads, chunks = case
        events = list(parse_sse_stream(chunks))
        assert [e.data for e in events] == [p.strip() for p in payloads]

    @given(st.binary(min_size=0, max_size=64))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_parser_never_crashes_on_garbage(self, blob: bytes):
        # The parser must tolerate anything coming down the wire.
        list(parse_sse_stream([blob]))


# ── Stat harness invariants ──────────────────────────────────────────────────


@st.composite
def sample_and_p(draw):
    xs = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    p = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    return xs, p


class TestStatHarnessProperties:

    @given(sample_and_p())
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_percentile_is_bounded_by_min_max(self, case):
        xs, p = case
        out = percentile(xs, p)
        assert min(xs) <= out <= max(xs)

    @given(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_percentile_monotone_in_p(self, xs):
        vals = [percentile(xs, p) for p in (0, 10, 25, 50, 75, 90, 100)]
        for a, b in zip(vals, vals[1:]):
            assert a <= b + 1e-9

    @given(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_bootstrap_ci_within_sample_range(self, xs):
        lower, upper = bootstrap_ci(
            xs, resamples=100, rng=random.Random(1)
        )
        assert lower <= upper
        assert min(xs) - 1e-9 <= lower
        assert upper <= max(xs) + 1e-9

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=20,
        ),
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=20,
        ),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_mann_whitney_p_in_unit_interval(self, a, b):
        _, p = mann_whitney_u(a, b, alternative="two-sided")
        assert 0.0 <= p <= 1.0

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=15,
        ),
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=15,
        ),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_mann_whitney_two_sided_symmetric_under_swap(self, a, b):
        _, p_ab = mann_whitney_u(a, b, alternative="two-sided")
        _, p_ba = mann_whitney_u(b, a, alternative="two-sided")
        assert abs(p_ab - p_ba) < 1e-9
