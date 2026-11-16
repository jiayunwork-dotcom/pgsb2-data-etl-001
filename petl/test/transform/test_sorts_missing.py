# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import os
import pickle

import pytest

from petl import MISSING, sort, issorted, wrap


TABLE = [
    ('k', 'other'),
    ('z', 1),
    (None, 2),
    (1, 3),
    (MISSING, 4),
    ('a', 5),
    (MISSING, 6),
    (None, 7),
]

# ascending: MISSING first, then None, then numbers, then strings
ASC_KEYS = [MISSING, MISSING, None, None, 1, 'a', 'z']
# stability: MISSING rows keep input order (4 before 6), None rows too (2, 7)
ASC_ROWS = [
    (MISSING, 4), (MISSING, 6),
    (None, 2), (None, 7),
    (1, 3), ('a', 5), ('z', 1),
]


def test_sort_missing_before_none_ascending():
    result = list(sort(TABLE, 'k'))
    assert result[0] == ('k', 'other')
    assert result[1:] == ASC_ROWS
    assert [r[0] for r in result[1:]] == ASC_KEYS
    assert issorted(sort(TABLE, 'k'), key='k')


def test_sort_reverse_inverts_order():
    result = list(sort(TABLE, 'k', reverse=True))
    keys = [r[0] for r in result[1:]]
    assert keys == list(reversed(ASC_KEYS))
    # MISSING is last, None just before it
    assert result[-1][0] is MISSING
    assert result[-3][0] is None
    assert result[1] == ('z', 1)
    assert issorted(sort(TABLE, 'k', reverse=True), key='k', reverse=True)


def test_sort_stability_among_missing_and_none():
    # rows sharing MISSING and None keys retain original relative order
    asc = sort(TABLE, 'k')
    others = [r[1] for r in asc if r[0] is MISSING]
    assert others == [4, 6]
    none_others = [r[1] for r in asc if r[0] is None]
    assert none_others == [2, 7]


COMPOUND = [
    ('a', 'b'),
    (MISSING, 'z'),
    (1, MISSING),
    (None, 'm'),
    (1, None),
    (1, 'a'),
    (MISSING, 'a'),
]


def test_compound_key_componentwise():
    result = [tuple(r) for r in sort(COMPOUND, ['a', 'b'])][1:]
    # first component ordering: MISSING < None < 1
    assert result == [
        (MISSING, 'a'),
        (MISSING, 'z'),
        (None, 'm'),
        (1, MISSING),   # within a=1, MISSING < None < 'a'
        (1, None),
        (1, 'a'),
    ]
    assert issorted(sort(COMPOUND, ['a', 'b']), key=['a', 'b'])


def test_lexical_sort_with_missing_cells():
    result = [tuple(r) for r in sort(COMPOUND)][1:]
    # lexical sort compares whole rows; MISSING in the first cell dominates
    assert result[0] == (MISSING, 'a')
    assert result[1] == (MISSING, 'z')
    assert result[2] == (None, 'm')


@pytest.mark.parametrize('buffersize', [1, 2, 3])
def test_external_sort_spill_keeps_order_and_identity(tmpdir, buffersize):
    view = sort(TABLE, 'k', buffersize=buffersize, tempdir=str(tmpdir))
    result = list(view)
    assert result[1:] == ASC_ROWS
    # markers unpickled from chunk files stay the same singleton
    assert all(r[0] is MISSING for r in result[1:3])
    # merge-sorted re-read from the file cache on a second pass
    again = list(view)
    assert again == result
    assert again[1][0] is MISSING
    view.clearcache()


def test_pickle_roundtrip_preserves_marker_in_sorted_output(tmpdir):
    path = os.path.join(str(tmpdir), 'sorted.p')
    sorted_table = sort(TABLE, 'k', buffersize=2, tempdir=str(tmpdir))
    import petl as etl
    etl.topickle(sorted_table, path)
    with open(path, 'rb') as f:
        hdr = pickle.load(f)
        first = pickle.load(f)
    assert hdr == ('k', 'other')
    assert first[0] is MISSING


def test_sort_cache_false_recomputes(tmpdir):
    view = sort(TABLE, 'k', cache=False, buffersize=2, tempdir=str(tmpdir))
    first = list(view)
    second = list(view)
    assert first == second
    assert first[1:] == ASC_ROWS


def test_none_ordering_unchanged_without_missing():
    table = [('k',), ('b',), (None,), ('a',), (1,)]
    result = [r[0] for r in sort(table, 'k')][1:]
    assert result == [None, 1, 'a', 'b']


def test_presorted_join_inputs_follow_missing_ordering():
    # join with presorted=True trusts this exact ordering
    left = wrap(sort([('k', 'v'), (MISSING, 'x'), (1, 'y')], 'k'))
    right = wrap(sort([('k', 'd'), (MISSING, 'p'), (1, 'q')], 'k'))
    from petl import join
    matched = list(join(left, right, key='k', presorted=True,
                        match_missing=True))
    assert matched[1] == (MISSING, 'x', 'p')
    assert matched[2] == (1, 'y', 'q')


if __name__ == '__main__':
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__]))
