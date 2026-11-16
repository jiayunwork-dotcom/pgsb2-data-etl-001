# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

from collections import Counter
from contextlib import contextmanager
from io import BytesIO

import pytest

from petl import MISSING
from petl.transform.hashjoins import (
    HashJoinView, HashLeftJoinView, HashRightJoinView, HashAntiJoinView,
    HashLookupJoinView,
)
from petl.transform.joins import (
    JoinView, AntiJoinView, LookupJoinView,
)
from petl.transform.sorts import SortView


def wrap(rows):
    """Wrap raw rows (MISSING included) as an ordinary petl table."""
    from petl import wrap as _wrap
    return _wrap(rows)


def multiset(view):
    return Counter(tuple(r) for r in view)


# --------------------------------------------------------------------------
# small canonical datasets with hand-written expected output
# --------------------------------------------------------------------------

LEFT = [
    ('id', 'v'),
    (1, 'a'),
    (2, None),       # explicit None key
    (MISSING, 'c'),  # absent key
    (4, MISSING),    # absent non-key cell
    (1, 'a2'),       # duplicated complete key
]
RIGHT = [
    ('id', 'd'),
    (1, 'x'),
    (3, 'y'),
    (MISSING, 'z'),  # absent key on right
    (1, 'x2'),       # duplicated complete key
]

# inner join, match_missing=False: MISSING keys excluded, 1-vs-2 cartesian
INNER_FALSE = [
    ('id', 'v', 'd'),
    (1, 'a', 'x'),
    (1, 'a', 'x2'),
    (1, 'a2', 'x'),
    (1, 'a2', 'x2'),
]
# inner join, match_missing=True: the two all-MISSING key rows also pair
INNER_TRUE = INNER_FALSE + [(MISSING, 'c', 'z')]

LEFT_FALSE = INNER_FALSE + [
    (2, None, None),
    (MISSING, 'c', None),
    (4, MISSING, None),
]
LEFT_TRUE = INNER_TRUE + [
    (2, None, None),
    (4, MISSING, None),
]

# full outer, match_missing=False
OUTER_FALSE_HEADER = ('id', 'v', 'd')
OUTER_FALSE = [
    OUTER_FALSE_HEADER,
    (MISSING, 'c', None),   # left-only missing key (left outer)
    (MISSING, None, 'z'),   # right-only missing key (right outer backfill)
    (1, 'a', 'x'),
    (1, 'a', 'x2'),
    (1, 'a2', 'x'),
    (1, 'a2', 'x2'),
    (2, None, None),
    (3, None, 'y'),
    (4, MISSING, None),
]
# full outer, match_missing=True: the MISSING pair is matched, no padding
OUTER_TRUE = [
    OUTER_FALSE_HEADER,
    (MISSING, 'c', 'z'),
    (1, 'a', 'x'),
    (1, 'a', 'x2'),
    (1, 'a2', 'x'),
    (1, 'a2', 'x2'),
    (2, None, None),
    (3, None, 'y'),
    (4, MISSING, None),
]

RIGHT_FALSE = [
    ('id', 'v', 'd'),
    (MISSING, None, 'z'),
    (1, 'a', 'x'),
    (1, 'a', 'x2'),
    (1, 'a2', 'x'),
    (1, 'a2', 'x2'),
    (3, None, 'y'),
]
RIGHT_TRUE = [
    ('id', 'v', 'd'),
    (MISSING, 'c', 'z'),
    (1, 'a', 'x'),
    (1, 'a', 'x2'),
    (1, 'a2', 'x'),
    (1, 'a2', 'x2'),
    (3, None, 'y'),
]


JOIN_KWARGS_MATRIX = [
    dict(),  # default sorted path, in-memory
    dict(buffersize=2),  # force external merge-sort spill files
    dict(cache=False, buffersize=2),  # no sort cache, still spill
    dict(cache=False),
]


@pytest.mark.parametrize('extra', JOIN_KWARGS_MATRIX)
def test_join_match_missing_false(extra):
    table = wrap(LEFT).join(wrap(RIGHT), key='id', match_missing=False,
                            **extra)
    assert isinstance(table, JoinView)
    assert multiset(table) == multiset(INNER_FALSE)


@pytest.mark.parametrize('extra', JOIN_KWARGS_MATRIX)
def test_join_match_missing_true(extra):
    table = wrap(LEFT).join(wrap(RIGHT), key='id', match_missing=True,
                            **extra)
    assert multiset(table) == multiset(INNER_TRUE)


def test_presorted_matches_sorted_and_hash_paths():
    for match_missing in (False, True):
        expected_inner = INNER_FALSE if not match_missing else INNER_TRUE
        expected_left = LEFT_FALSE if not match_missing else LEFT_TRUE
        expected_outer = OUTER_FALSE if not match_missing else OUTER_TRUE
        expected_right = RIGHT_FALSE if not match_missing else RIGHT_TRUE

        ls = wrap(LEFT).sort('id')
        rs = wrap(RIGHT).sort('id')

        presorted = wrap(ls).join(wrap(rs), key='id', presorted=True,
                                  match_missing=match_missing)
        sorted_join = wrap(LEFT).join(wrap(RIGHT), key='id',
                                      match_missing=match_missing)
        hashed = wrap(LEFT).hashjoin(wrap(RIGHT), key='id',
                                     match_missing=match_missing)
        assert isinstance(presorted, JoinView)
        assert isinstance(hashed, HashJoinView)
        assert multiset(presorted) == multiset(sorted_join)
        assert multiset(hashed) == multiset(expected_inner)

        for kind, expected in (
            ('leftjoin', expected_left),
            ('outerjoin', expected_outer),
        ):
            a = multiset(getattr(wrap(LEFT), kind)(
                wrap(RIGHT), key='id', match_missing=match_missing))
            b = multiset(getattr(wrap(ls), kind)(
                wrap(rs), key='id', presorted=True,
                match_missing=match_missing))
            assert a == b == multiset(expected), (kind, match_missing)

        hr = multiset(wrap(LEFT).hashrightjoin(
            wrap(RIGHT), key='id', match_missing=match_missing))
        assert hr == multiset(expected_right)


def test_default_behaviour_matches_legacy_match_missing_false():
    # omitting the flag must be exactly match_missing=False
    assert multiset(wrap(LEFT).join(wrap(RIGHT), key='id')) \
        == multiset(wrap(LEFT).join(wrap(RIGHT), key='id',
                                    match_missing=False))
    assert multiset(wrap(LEFT).outerjoin(wrap(RIGHT), key='id')) \
        == multiset(OUTER_FALSE)


def test_none_keys_still_match():
    # None keeps the library's historical equi-key semantics regardless of
    # the new flag
    left = [('k', 'x'), (None, 'l1'), (None, 'l2')]
    right = [('k', 'y'), (None, 'r1')]
    for flag in (False, True):
        result = multiset(wrap(left).join(wrap(right), key='k',
                                          match_missing=flag))
        assert result == Counter({
            ('k', 'x', 'y'): 1,
            (None, 'l1', 'r1'): 1,
            (None, 'l2', 'r1'): 1,
        })


def test_missing_never_matches_none():
    left = [('k', 'x'), (MISSING, 'lm')]
    right = [('k', 'y'), (None, 'rn')]
    for flag in (False, True):
        assert multiset(wrap(left).join(wrap(right), key='k',
                                        match_missing=flag)) \
            == Counter({('k', 'x', 'y'): 1})


# --------------------------------------------------------------------------
# outer fill value: None, MISSING and a custom ordinary value
# --------------------------------------------------------------------------

@pytest.mark.parametrize('fill', [None, MISSING, 'NA'])
def test_outerjoin_fill_values_keep_absent_cells(fill):
    table = wrap(LEFT).outerjoin(wrap(RIGHT), key='id', missing=fill,
                                 match_missing=False)
    rows = [tuple(r) for r in table]
    header = rows[0]
    assert header == ('id', 'v', 'd')
    by_id = {}
    for row in rows[1:]:
        by_id.setdefault(row[0], []).append(row)

    # right-only row (id 3): left side cells are filled with `fill`
    assert by_id[3] == [(3, fill, 'y')]
    # left-only rows with absent key: right non-key cell filled with `fill`,
    # while the genuinely absent left cell stays MISSING
    missing_left = [r for r in by_id[MISSING] if r[1] == 'c']
    assert missing_left == [(MISSING, 'c', fill)]
    # right-only missing-key row back-fills key position as MISSING (not
    # erased by the default missing=None)
    missing_right = [r for r in by_id[MISSING] if r[2] == 'z']
    assert missing_right == [(MISSING, fill, 'z')]
    # row 4 carried a MISSING non-key cell on the left; it must survive
    row4 = by_id[4]
    assert row4 == [(4, MISSING, fill)]


def test_outerjoin_missing_marker_fill_hash_path():
    result = multiset(wrap(LEFT).hashrightjoin(
        wrap(RIGHT), key='id', missing=MISSING, match_missing=False))
    # the right-only id=3 row gets MISSING for the left value cell
    assert result[(3, MISSING, 'y')] == 1
    # the right-only MISSING key keeps its key identity
    assert result[(MISSING, MISSING, 'z')] == 1


# --------------------------------------------------------------------------
# compound keys with partial absence
# --------------------------------------------------------------------------

COMPOUND_LEFT = [
    ('a', 'b', 'lv'),
    (1, 'x', 1),
    (1, 'x', 2),       # duplicated complete compound key
    (MISSING, 'x', 3),  # first component missing
    (1, MISSING, 4),   # second component missing
    (None, 'x', 5),    # explicit None component
]
COMPOUND_RIGHT = [
    ('c', 'd', 'rv'),
    (1, 'x', 10),
    (1, 'x', 11),      # duplicated complete compound key
    (MISSING, 'x', 12),
    (1, MISSING, 13),
    (None, 'x', 14),
]
COMPOUND_EXPECTED_FALSE = [
    ('a', 'b', 'lv', 'rv'),
    (1, 'x', 1, 10),
    (1, 'x', 1, 11),
    (1, 'x', 2, 10),
    (1, 'x', 2, 11),
    (None, 'x', 5, 14),  # None==None still matches
]
COMPOUND_EXPECTED_TRUE = COMPOUND_EXPECTED_FALSE + [
    (MISSING, 'x', 3, 12),
    (1, MISSING, 4, 13),
]


def test_compound_partial_missing_all_paths():
    lkey, rkey = ('a', 'b'), ('c', 'd')
    for flag, expected in ((False, COMPOUND_EXPECTED_FALSE),
                           (True, COMPOUND_EXPECTED_TRUE)):
        sorted_t = wrap(COMPOUND_LEFT).join(
            wrap(COMPOUND_RIGHT), lkey=lkey, rkey=rkey, match_missing=flag)
        spilled = wrap(COMPOUND_LEFT).join(
            wrap(COMPOUND_RIGHT), lkey=lkey, rkey=rkey,
            match_missing=flag, buffersize=2)
        hashed = wrap(COMPOUND_LEFT).hashjoin(
            wrap(COMPOUND_RIGHT), lkey=lkey, rkey=rkey, match_missing=flag)
        ls = wrap(COMPOUND_LEFT).sort(lkey)
        rs = wrap(COMPOUND_RIGHT).sort(rkey)
        pre = wrap(ls).join(wrap(rs), lkey=lkey, rkey=rkey,
                            presorted=True, match_missing=flag)
        expected_ms = multiset(expected)
        assert multiset(sorted_t) == expected_ms
        assert multiset(spilled) == expected_ms
        assert multiset(hashed) == expected_ms
        assert multiset(pre) == expected_ms
        # duplicate complete keys retained their cartesian multiplicity
        assert expected_ms[(1, 'x', 1, 10)] == 1
        assert expected_ms[(1, 'x', 2, 11)] == 1


def test_compound_partial_missing_outer_and_left():
    lkey, rkey = ('a', 'b'), ('c', 'd')
    left_false = multiset(wrap(COMPOUND_LEFT).leftjoin(
        wrap(COMPOUND_RIGHT), lkey=lkey, rkey=rkey, match_missing=False))
    # the two partial-missing left rows appear once, right-padded
    assert left_false[(MISSING, 'x', 3, None)] == 1
    assert left_false[(1, MISSING, 4, None)] == 1
    outer_true = multiset(wrap(COMPOUND_LEFT).outerjoin(
        wrap(COMPOUND_RIGHT), lkey=lkey, rkey=rkey, match_missing=True))
    assert outer_true[(MISSING, 'x', 3, 12)] == 1
    assert outer_true[(1, MISSING, 4, 13)] == 1
    hash_left = multiset(wrap(COMPOUND_LEFT).hashleftjoin(
        wrap(COMPOUND_RIGHT), lkey=lkey, rkey=rkey, match_missing=False))
    assert hash_left == left_false


# --------------------------------------------------------------------------
# different key names and prefixes
# --------------------------------------------------------------------------

def test_different_key_names_and_prefixes():
    for flag in (False, True):
        view = wrap(LEFT).join(
            wrap(RIGHT), lkey='id', rkey='id',
            lprefix='L_', rprefix='R_', match_missing=flag)
        header = tuple(view.header())
        assert header == ('L_id', 'L_v', 'R_d')
        # prefixing changes only header names; row values are identical to
        # the non-prefixed join
        expected = INNER_FALSE if not flag else INNER_TRUE
        assert multiset(view.data()) == Counter(
            tuple(r) for r in expected[1:]
        )


# --------------------------------------------------------------------------
# both sides entirely missing keys, no complete keys at all
# --------------------------------------------------------------------------

def test_all_missing_keys_no_cartesian():
    n, m = 4, 3
    left = [('k', 'v')] + [(MISSING, i) for i in range(n)]
    right = [('k', 'd')] + [(MISSING, i) for i in range(m)]

    # inner: empty either way
    assert multiset(wrap(left).join(wrap(right), key='k',
                                    match_missing=False).data()) == Counter()
    # with match_missing, markers pair positionally -> n*m rows
    true_ms = multiset(wrap(left).join(wrap(right), key='k',
                                       match_missing=True).data())
    assert sum(true_ms.values()) == n * m
    assert all(k == MISSING for k, _, _ in true_ms)

    # left outer keeps every left row, no cross pairing when disabled
    lj = multiset(wrap(left).leftjoin(wrap(right), key='k',
                                      match_missing=False).data())
    assert sum(lj.values()) == n
    assert all(r[0] is MISSING and r[2] is None for r in lj)

    # outer join emits every row from both sides when disabled
    oj = multiset(wrap(left).outerjoin(wrap(right), key='k',
                                       match_missing=False).data())
    assert sum(oj.values()) == n + m
    hash_lj = multiset(wrap(left).hashleftjoin(wrap(right), key='k',
                                               match_missing=False).data())
    assert hash_lj == lj
    hash_j = multiset(wrap(left).hashjoin(wrap(right), key='k',
                                          match_missing=False).data())
    assert sum(hash_j.values()) == 0
    hash_j_true = multiset(wrap(left).hashjoin(wrap(right), key='k',
                                               match_missing=True).data())
    assert sum(hash_j_true.values()) == n * m


def test_empty_tables():
    empty = [('k', 'v')]
    other = [('k', 'd'), (1, 'x')]
    for flag in (False, True):
        assert multiset(wrap(empty).join(wrap(other), key='k',
                                         match_missing=flag).data()) \
            == Counter()
        assert multiset(wrap(other).join(wrap(empty), key='k',
                                         match_missing=flag).data()) \
            == Counter()
        outer = list(wrap(empty).outerjoin(wrap(other), key='k',
                                           match_missing=flag))
        assert outer == [('k', 'v', 'd'), (1, None, 'x')]


# --------------------------------------------------------------------------
# anti and lookup joins
# --------------------------------------------------------------------------

def test_antijoin_missing_rows_retained():
    # default semantics: MISSING-keyed left rows cannot match, so they survive
    result = multiset(wrap(LEFT).antijoin(
        wrap(RIGHT), key='id', match_missing=False).data())
    rows = set(result)
    # id 4 and the MISSING key row have no right counterpart; id 2 (None)
    # also has no right key equal to None
    assert (MISSING, 'c') in rows
    assert (4, MISSING) in rows
    assert (2, None) in rows
    assert (1, 'a') not in rows
    hresult = multiset(wrap(LEFT).hashantijoin(
        wrap(RIGHT), key='id', match_missing=False).data())
    assert hresult == result
    # with match_missing=True the two MISSING-key rows pair up, so the left
    # one is anti-joined away instead
    result_true = multiset(wrap(LEFT).antijoin(
        wrap(RIGHT), key='id', match_missing=True).data())
    assert (MISSING, 'c') not in set(result_true)
    assert (4, MISSING) in set(result_true)


def test_lookupjoin_missing_handling():
    # lookup join keeps one arbitrary right row per key
    lj_false = multiset(wrap(LEFT).lookupjoin(
        wrap(RIGHT), key='id', match_missing=False))
    assert lj_false[(MISSING, 'c', None)] == 1
    assert lj_false[(4, MISSING, None)] == 1
    # complete key still joins, but only one right row is chosen
    complete = [r for r in lj_false if r[0] == 1]
    assert len(complete) == 2  # two left rows, each matched once
    lj_true = multiset(wrap(LEFT).lookupjoin(
        wrap(RIGHT), key='id', match_missing=True))
    assert lj_true[(MISSING, 'c', 'z')] == 1
    hl = multiset(wrap(LEFT).hashlookupjoin(
        wrap(RIGHT), key='id', match_missing=False))
    assert hl[(MISSING, 'c', None)] == 1


# --------------------------------------------------------------------------
# evaluation timing, caching and resource cleanup
# --------------------------------------------------------------------------

def test_single_pass_source_consumed_once_per_iteration():
    # each pass over the join view must make exactly one pass over the
    # source rows; existence checks must not trigger an extra mid-pass scan.
    # A re-iterable source is used because the sorted join architecture
    # legitimately opens each table once per output pass (the documented
    # evaluation timing), whereas a bare generator cannot be reopened.
    class CountingTable(object):
        def __init__(self):
            self.opens = 0
            self.rows_seen = 0

        def __iter__(self):
            self.opens += 1
            yield ('id', 'v')
            for rec in SPARSE_SRC:
                self.rows_seen += 1
                if 'id' not in rec:
                    yield (MISSING, rec['v'])
                elif 'v' not in rec:
                    yield (rec['id'], MISSING)
                else:
                    yield (rec['id'], rec['v'])

    source = CountingTable()
    view = wrap(source).leftjoin(wrap(RIGHT), key='id',
                                 match_missing=False, cache=False)
    rows1 = list(view)
    opens_after_first = source.opens
    # every open consumes the whole source exactly once (no partial or
    # duplicated scans from the MISSING bookkeeping)
    assert source.rows_seen == opens_after_first * len(SPARSE_SRC)
    rows2 = list(view)
    # a second output pass reopens the same fixed number of times and again
    # reads each source row exactly once per open
    assert source.opens == 2 * opens_after_first
    assert source.rows_seen == source.opens * len(SPARSE_SRC)
    assert len(rows1) == len(rows2)


def json_bytes(records):
    import json as _json
    return _json.dumps(records).encode('utf-8')


SPARSE_SRC = [
    {'id': 1, 'v': 'a'}, {'v': 'c'}, {'id': 4},
]


class _MutableBytesSource(object):
    """Minimal petl source whose contents can be swapped between passes."""

    def __init__(self, payload):
        self.payload = payload

    @contextmanager
    def open(self, mode='rb'):
        stream = BytesIO(self.payload)
        try:
            yield stream
        finally:
            stream.close()


def test_cache_false_reflects_current_input_each_pass():
    source = _MutableBytesSource(json_bytes(SPARSE_SRC))
    from petl import fromjson
    left = fromjson(source, header=['id', 'v'], preserve_missing=True)
    right = wrap(RIGHT)
    view = left.hashleftjoin(right, key='id', cache=False,
                             match_missing=False)
    first = multiset(view)
    assert first[(MISSING, 'c', None)] == 1
    # replace underlying bytes; cache=False must rebuild from current input
    source.payload = json_bytes([{'id': 9, 'v': 'new'}])
    second = multiset(view)
    assert (9, 'new', None) in second
    assert (MISSING, 'c', None) not in second


def test_cache_true_reuses_right_lookup_across_iterations():
    view = wrap(SPARSE_LEFT_TABLE).hashleftjoin(
        wrap(RIGHT), key='id', cache=True, match_missing=False)
    assert view.rlookup is None
    list(view)
    assert view.rlookup is not None
    cached = view.rlookup
    list(view)
    assert view.rlookup is cached  # same lookup object reused


SPARSE_LEFT_TABLE = [
    ('id', 'v'), (1, 'a'), (MISSING, 'c'), (4, MISSING),
]


def test_independent_iterators_have_independent_state():
    view = wrap(SPARSE_LEFT_TABLE).hashjoin(
        wrap(RIGHT), key='id', match_missing=True)
    total = sum(multiset(view.data()).values())
    it1 = iter(view)
    next(it1)  # header
    first = next(it1)
    it2 = iter(view)
    next(it2)  # header of an independent iterator
    second = next(it2)
    assert first == second
    # advancing one iterator does not exhaust the other
    assert len(list(it1)) == total - 1
    assert len(list(it2)) == total - 1


class FailingSource(object):
    def __init__(self, fail_after_header):
        self.fail_after = fail_after_header
        self.opened = 0
        self.closed = 0

    def __iter__(self):
        self.opened += 1
        try:
            yield ('id', 'v')
            for i in range(self.fail_after):
                yield (i, 'ok')
            raise RuntimeError('boom from source')
        finally:
            self.closed += 1


def test_source_exception_propagates_and_sort_resources_released(tmpdir):
    import os as _os
    import gc
    failing = FailingSource(fail_after_header=3)
    view = wrap(failing).sort('id', buffersize=2, tempdir=str(tmpdir))
    with pytest.raises(RuntimeError, match='boom from source'):
        list(view)
    # chunk files may have been written before the source failed; the
    # SortView retains them via its documented cache lifecycle, and clearing
    # the cache must release all of them
    view.clearcache()
    gc.collect()
    leftovers = [n for n in _os.listdir(str(tmpdir)) if n.startswith('tmp')]
    assert leftovers == []
    assert failing.opened == failing.closed


def test_early_close_releases_new_resources(tmpdir):
    left = wrap([('id', 'v')] + [(MISSING, i) for i in range(100)])
    right = wrap([('id', 'd')] + [(MISSING, i) for i in range(50)])
    view = left.outerjoin(right, key='id', buffersize=10,
                          tempdir=str(tmpdir), match_missing=False)
    it = iter(view)
    next(it)  # header
    for _ in range(5):
        next(it)
    del it
    import gc
    gc.collect()
    # no exception and the SortView still owns its documented cache;
    # clearing it must leave no temp files behind
    view.left.clearcache()
    view.right.clearcache()
    gc.collect()
    assert [n for n in __import__('os').listdir(str(tmpdir))
            if n.startswith('tmp')] == []


def test_view_types_unchanged_and_rows_are_plain_tuples():
    sorted_view = wrap(LEFT).join(wrap(RIGHT), key='id')
    assert isinstance(sorted_view, JoinView)
    assert isinstance(sorted_view.left, SortView)
    assert isinstance(wrap(LEFT).antijoin(wrap(RIGHT), key='id'),
                      AntiJoinView)
    assert isinstance(wrap(LEFT).lookupjoin(wrap(RIGHT), key='id'),
                      LookupJoinView)
    assert isinstance(wrap(LEFT).hashjoin(wrap(RIGHT), key='id'),
                      HashJoinView)
    assert isinstance(wrap(LEFT).hashleftjoin(wrap(RIGHT), key='id'),
                      HashLeftJoinView)
    assert isinstance(wrap(LEFT).hashrightjoin(wrap(RIGHT), key='id'),
                      HashRightJoinView)
    assert isinstance(wrap(LEFT).hashantijoin(wrap(RIGHT), key='id'),
                      HashAntiJoinView)
    assert isinstance(wrap(LEFT).hashlookupjoin(wrap(RIGHT), key='id'),
                      HashLookupJoinView)
    for row in sorted_view:
        assert type(row) is tuple


if __name__ == '__main__':
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__]))
