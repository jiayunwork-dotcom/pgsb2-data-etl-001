"""Tests for join matching with NaN key values.

Matching contract: a key containing a NaN component (float NaN or
Decimal quiet NaN, any sign or payload) never matches another row - not
even a row carrying the identical NaN object - regardless of
``match_missing``. Under the sort order NaN keys are co-equal and sort
after None and before all other values. All join entry points (sorted
merge joins over the memory/spill/cache/presorted paths, and the hash
joins) must produce the same complete result multiset for the same
input. All expectations below are hand-written.
"""

from __future__ import print_function, division, absolute_import

import math
from decimal import Decimal, localcontext, InvalidOperation

import pytest

import petl as etl
from petl import MISSING
from petl.comparison import _isnan
from petl.test.helpers import eq_, ieq_nan, row_multiset


# ---------------------------------------------------------------------
# helpers


def presort_rows(table, keycols):
    """Independently sort a raw table into the documented petl order:
    MISSING < None < NaN < everything else (natural), stable. Used only
    to prepare presorted=True input; expectations are hand-written."""
    hdr, rows = table[0], list(table[1:])

    def rank(v):
        if v is MISSING:
            return (0, 0)
        if v is None:
            return (1, 0)
        if _isnan(v):
            return (2, 0)
        return (3, v)

    def keyfn(row):
        return tuple(rank(row[i]) for i in keycols)

    return [hdr] + sorted(rows, key=keyfn)


def run_merge(joinfn, left, right, key, keycols=(0,), **kw):
    """Run a sorted-merge join over the memory, spill, cache and
    presorted paths."""
    results = {}
    results['mem'] = list(joinfn(left, right, key=key, buffersize=None,
                                 cache=False, **kw))
    results['spill'] = list(joinfn(left, right, key=key, buffersize=2,
                                   cache=False, **kw))
    spill_cached = joinfn(left, right, key=key, buffersize=2, cache=True,
                          **kw)
    results['spill_cache1'] = list(spill_cached)
    results['spill_cache2'] = list(spill_cached)
    mem_cached = joinfn(left, right, key=key, buffersize=None, cache=True,
                        **kw)
    results['mem_cache1'] = list(mem_cached)
    results['mem_cache2'] = list(mem_cached)
    results['presorted'] = list(joinfn(presort_rows(left, keycols),
                                       presort_rows(right, keycols),
                                       key=key, presorted=True, **kw))
    return results


def run_hash(joinfn, left, right, key, **kw):
    """Run a hash join over the no-cache and cached paths (second pass
    reuses the cached lookup)."""
    results = {}
    results['hash'] = list(joinfn(left, right, key=key, cache=False, **kw))
    cached = joinfn(left, right, key=key, cache=True, **kw)
    results['hash_cache1'] = list(cached)
    results['hash_cache2'] = list(cached)
    return results


def assert_multiset(expect, actual):
    eq_(row_multiset(expect), row_multiset(actual))


def assert_all_paths_equal(expect, paths):
    for name, actual in paths.items():
        assert_multiset(expect, actual)


# ---------------------------------------------------------------------
# the original reproduction


def _repro():
    n = float('nan')
    left = [('k', 'l'), (1, 'L1'), (n, 'LN'), (2, 'L2'), (3, 'L3')]
    right = [('k', 'r'), (1, 'R1'), (2, 'R2'), (3, 'R3'), (n, 'RN')]
    return n, left, right


def test_repro_join():
    n, left, right = _repro()
    expect = [('k', 'l', 'r'),
              (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')]
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # merge path emits groups in key order; check one exactly
    ieq_nan(expect, paths['mem'])
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # hash join follows left row order; same sequence here
    ieq_nan(expect, paths['hash'])


def test_repro_leftjoin():
    n, left, right = _repro()
    expect = [('k', 'l', 'r'),
              (n, 'LN', None),
              (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')]
    paths = run_merge(etl.leftjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    ieq_nan(expect, paths['mem'])
    paths = run_hash(etl.hashleftjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # hash left join follows left row order: LN stays in position 2
    ieq_nan([('k', 'l', 'r'),
             (1, 'L1', 'R1'), (n, 'LN', None), (2, 'L2', 'R2'),
             (3, 'L3', 'R3')],
            paths['hash'])


def test_repro_rightjoin():
    n, left, right = _repro()
    expect = [('k', 'l', 'r'),
              (n, None, 'RN'),
              (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')]
    paths = run_merge(etl.rightjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    ieq_nan(expect, paths['mem'])
    paths = run_hash(etl.hashrightjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)


def test_repro_outerjoin():
    n, left, right = _repro()
    expect = [('k', 'l', 'r'),
              (n, 'LN', None), (n, None, 'RN'),
              (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')]
    paths = run_merge(etl.outerjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # exact merge-path order: left NaN group, right NaN group, matches
    ieq_nan(expect, paths['mem'])


def test_repro_antijoin_lookupjoin():
    n, left, right = _repro()
    # anti join keeps left rows whose key does not occur on the right;
    # the NaN row never matches, so it is always kept
    expect_anti = [('k', 'l'), (n, 'LN')]
    for kwargs in (dict(buffersize=None, cache=False),
                   dict(buffersize=2, cache=False),
                   dict(presorted=True)):
        if 'presorted' in kwargs:
            actual = etl.antijoin(presort_rows(left, (0,)),
                                  presort_rows(right, (0,)), 'k', **kwargs)
        else:
            actual = etl.antijoin(left, right, 'k', **kwargs)
        assert_multiset(expect_anti, list(actual))
    assert_multiset(expect_anti, list(etl.hashantijoin(left, right, 'k')))
    # lookup join: NaN left row is padded, others pick the first right row
    expect_lookup = [('k', 'l', 'r'),
                     (n, 'LN', None),
                     (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')]
    paths = run_merge(etl.lookupjoin, left, right, 'k')
    assert_all_paths_equal(expect_lookup, paths)
    assert_multiset(expect_lookup,
                    list(etl.hashlookupjoin(left, right, 'k')))


def test_repro_table_chaining():
    # Table method chaining picks up the same contract
    n, left, right = _repro()
    actual = (etl.wrap(left)
              .join(etl.wrap(right), key='k', buffersize=2, cache=False))
    ieq_nan([('k', 'l', 'r'),
             (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')],
            list(actual))
    actual = (etl.wrap(left)
              .hashleftjoin(etl.wrap(right), key='k', cache=False))
    ieq_nan([('k', 'l', 'r'),
             (1, 'L1', 'R1'), (n, 'LN', None), (2, 'L2', 'R2'),
             (3, 'L3', 'R3')],
            list(actual))
    actual = (etl.wrap(left)
              .outerjoin(etl.wrap(right), key='k', buffersize=2,
                         cache=False))
    assert_multiset([('k', 'l', 'r'),
                     (n, 'LN', None), (n, None, 'RN'),
                     (1, 'L1', 'R1'), (2, 'L2', 'R2'), (3, 'L3', 'R3')],
                    list(actual))


# ---------------------------------------------------------------------
# NaN on one side only


def test_nan_left_side_only():
    n = float('nan')
    left = [('k', 'l'), (n, 'LN'), (1, 'L1')]
    right = [('k', 'r'), (1, 'R1'), (2, 'R2')]
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1')],
                    list(etl.join(left, right, 'k', buffersize=2,
                                  cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, 'LN', None), (1, 'L1', 'R1')],
                    list(etl.leftjoin(left, right, 'k', buffersize=2,
                                      cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1'), (2, None, 'R2')],
                    list(etl.rightjoin(left, right, 'k', buffersize=2,
                                       cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, 'LN', None), (1, 'L1', 'R1'),
                     (2, None, 'R2')],
                    list(etl.outerjoin(left, right, 'k', buffersize=2,
                                       cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1')],
                    list(etl.hashjoin(left, right, 'k', cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, 'LN', None), (1, 'L1', 'R1')],
                    list(etl.hashleftjoin(left, right, 'k', cache=False)))


def test_nan_right_side_only():
    n = float('nan')
    left = [('k', 'l'), (1, 'L1'), (2, 'L2')]
    right = [('k', 'r'), (n, 'RN'), (1, 'R1')]
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1')],
                    list(etl.join(left, right, 'k', buffersize=2,
                                  cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1'), (2, 'L2', None)],
                    list(etl.leftjoin(left, right, 'k', buffersize=2,
                                      cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, None, 'RN'), (1, 'L1', 'R1')],
                    list(etl.rightjoin(left, right, 'k', buffersize=2,
                                       cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1'), (2, 'L2', None),
                     (n, None, 'RN')],
                    list(etl.outerjoin(left, right, 'k', buffersize=2,
                                       cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1')],
                    list(etl.hashjoin(left, right, 'k', cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1'), (2, 'L2', None)],
                    list(etl.hashleftjoin(left, right, 'k', cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, None, 'RN'), (1, 'L1', 'R1')],
                    list(etl.hashrightjoin(left, right, 'k', cache=False)))


# ---------------------------------------------------------------------
# same object, different objects, signs, payloads


def test_nan_same_object_reused_both_sides():
    # the identical NaN object appears on both sides: still no match
    n = float('nan')
    left = [('k', 'l'), (n, 'LN'), (1, 'L1')]
    right = [('k', 'r'), (n, 'RN'), (1, 'R1')]
    expect_outer = [('k', 'l', 'r'),
                    (n, 'LN', None), (n, None, 'RN'), (1, 'L1', 'R1')]
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1')],
                    list(etl.join(left, right, 'k', buffersize=2,
                                  cache=False)))
    assert_multiset([('k', 'l', 'r'), (1, 'L1', 'R1')],
                    list(etl.hashjoin(left, right, 'k', cache=False)))
    assert_multiset(expect_outer,
                    list(etl.outerjoin(left, right, 'k', buffersize=2,
                                       cache=False)))
    # and the output cells carry the original object (memory path)
    actual = list(etl.outerjoin(left, right, 'k', buffersize=None,
                                cache=False))
    assert actual[1][0] is n
    assert actual[2][0] is n


def test_nan_different_objects_and_categories():
    # every distinct NaN value on both sides: none of them match
    nl1, nl2 = float('nan'), Decimal('NaN9')
    nr1, nr2 = Decimal('NaN9'), -float('nan')
    left = [('k', 'l'), (nl1, 'L1'), (nl2, 'L2'), (7, 'L7')]
    right = [('k', 'r'), (nr1, 'R1'), (nr2, 'R2'), (7, 'R7')]
    expect_outer = [('k', 'l', 'r'),
                    (nl1, 'L1', None), (nl2, 'L2', None),
                    (nr1, None, 'R1'), (nr2, None, 'R2'),
                    (7, 'L7', 'R7')]
    assert_multiset([('k', 'l', 'r'), (7, 'L7', 'R7')],
                    list(etl.join(left, right, 'k', buffersize=2,
                                  cache=False)))
    paths = run_merge(etl.outerjoin, left, right, 'k')
    assert_all_paths_equal(expect_outer, paths)
    paths = run_hash(etl.hashleftjoin, left, right, 'k')
    assert_all_paths_equal([('k', 'l', 'r'),
                            (nl1, 'L1', None), (nl2, 'L2', None),
                            (7, 'L7', 'R7')],
                           paths)
    # output NaN cells keep category, sign and payload
    actual = list(etl.outerjoin(left, right, 'k', buffersize=2,
                                cache=False))
    cells = {}
    for row in actual[1:]:
        key = row[0] if row[1] is not None else row[0]
        cells[row[1] if row[1] is not None else row[2]] = row[0]
    assert type(cells['L1']) is float
    assert isinstance(cells['L2'], Decimal) and cells['L2'].is_qnan()
    eq_((9,), cells['L2'].as_tuple().digits)
    assert isinstance(cells['R1'], Decimal) and cells['R1'].is_qnan()
    eq_((9,), cells['R1'].as_tuple().digits)
    assert type(cells['R2']) is float
    assert math.copysign(1.0, cells['R2']) < 0


def test_nan_decimal_signs_and_payloads_no_match():
    # same payload, different signs; same sign, different payloads:
    # NaN never matches NaN
    left = [('k', 'l'),
            (Decimal('-NaN7'), 'L1'),
            (Decimal('NaN7'), 'L2'),
            (Decimal('NaN70'), 'L3')]
    right = [('k', 'r'),
             (Decimal('NaN7'), 'R1'),
             (Decimal('-NaN7'), 'R2'),
             (Decimal('-NaN70'), 'R3')]
    assert_multiset([('k', 'l', 'r')],
                    list(etl.join(left, right, 'k', buffersize=2,
                                  cache=False)))
    assert_multiset([('k', 'l', 'r')],
                    list(etl.hashjoin(left, right, 'k', cache=False)))
    outer = list(etl.outerjoin(left, right, 'k', buffersize=2,
                               cache=False))
    eq_(7, len(outer))  # header + 3 left-only + 3 right-only


# ---------------------------------------------------------------------
# compound keys


def test_nan_compound_key_positions():
    n1, n2 = float('nan'), Decimal('NaN')
    left = [('a', 'b', 'l'),
            (1, n1, 'L1'),      # NaN in second component
            (n2, 1, 'L2'),      # NaN in first component
            (1, 2, 'L3'),
            (n1, n2, 'L4')]     # NaN in both components
    right = [('a', 'b', 'r'),
             (1, n1, 'R1'),     # identical NaN object, same position
             (n2, 1, 'R2'),
             (1, 2, 'R3'),
             (2, n1, 'R4')]
    # only the fully non-NaN key (1, 2) may match
    expect_join = [('a', 'b', 'l', 'r'), (1, 2, 'L3', 'R3')]
    paths = run_merge(etl.join, left, right, ('a', 'b'), keycols=(0, 1))
    assert_all_paths_equal(expect_join, paths)
    paths = run_hash(etl.hashjoin, left, right, ('a', 'b'))
    assert_all_paths_equal(expect_join, paths)
    # outer join keeps every NaN-bearing row on its own side
    expect_outer = [('a', 'b', 'l', 'r'),
                    (1, n1, 'L1', None),
                    (n2, 1, 'L2', None),
                    (n1, n2, 'L4', None),
                    (1, 2, 'L3', 'R3'),
                    (1, n1, None, 'R1'),
                    (n2, 1, None, 'R2'),
                    (2, n1, None, 'R4')]
    paths = run_merge(etl.outerjoin, left, right, ('a', 'b'),
                      keycols=(0, 1))
    assert_all_paths_equal(expect_outer, paths)
    # hash left join: all NaN-bearing left rows padded
    expect_hleft = [('a', 'b', 'l', 'r'),
                    (1, n1, 'L1', None),
                    (n2, 1, 'L2', None),
                    (1, 2, 'L3', 'R3'),
                    (n1, n2, 'L4', None)]
    paths = run_hash(etl.hashleftjoin, left, right, ('a', 'b'))
    assert_all_paths_equal(expect_hleft, paths)


def test_nan_compound_with_none_and_missing():
    # NaN, None and MISSING in compound keys keep their distinct rules
    n = float('nan')
    left = [('a', 'b', 'l'),
            (1, n, 'L-nan'),
            (1, None, 'L-none'),
            (1, MISSING, 'L-miss'),
            (1, 5, 'L-5')]
    right = [('a', 'b', 'r'),
             (1, n, 'R-nan'),
             (1, None, 'R-none'),
             (1, MISSING, 'R-miss'),
             (1, 5, 'R-5')]
    # default: None components match, MISSING and NaN do not
    expect = [('a', 'b', 'l', 'r'),
              (1, None, 'L-none', 'R-none'),
              (1, 5, 'L-5', 'R-5')]
    paths = run_merge(etl.join, left, right, ('a', 'b'), keycols=(0, 1))
    assert_all_paths_equal(expect, paths)
    paths = run_hash(etl.hashjoin, left, right, ('a', 'b'))
    assert_all_paths_equal(expect, paths)
    # match_missing=True: MISSING matches MISSING; NaN still does not
    expect_mm = [('a', 'b', 'l', 'r'),
                 (1, None, 'L-none', 'R-none'),
                 (1, MISSING, 'L-miss', 'R-miss'),
                 (1, 5, 'L-5', 'R-5')]
    paths = run_merge(etl.join, left, right, ('a', 'b'), keycols=(0, 1),
                      match_missing=True)
    assert_all_paths_equal(expect_mm, paths)
    paths = run_hash(etl.hashjoin, left, right, ('a', 'b'),
                     match_missing=True)
    assert_all_paths_equal(expect_mm, paths)


# ---------------------------------------------------------------------
# empty and all-NaN tables


def test_empty_tables():
    n = float('nan')
    lempty = [('k', 'l')]
    rempty = [('k', 'r')]
    left = [('k', 'l'), (n, 'LN'), (1, 'L1')]
    right = [('k', 'r'), (n, 'RN'), (1, 'R1')]
    # left empty
    assert_multiset([('k', 'l', 'r')],
                    list(etl.join(lempty, right, 'k', buffersize=2,
                                  cache=False)))
    assert_multiset([('k', 'l', 'r')],
                    list(etl.leftjoin(lempty, right, 'k', buffersize=2,
                                      cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, None, 'RN'), (1, None, 'R1')],
                    list(etl.outerjoin(lempty, right, 'k', buffersize=2,
                                       cache=False)))
    assert_multiset([('k', 'l', 'r')],
                    list(etl.hashjoin(lempty, right, 'k', cache=False)))
    assert_multiset([('k', 'l', 'r')],
                    list(etl.hashleftjoin(lempty, right, 'k',
                                          cache=False)))
    # right empty
    assert_multiset([('k', 'l', 'r')],
                    list(etl.join(left, rempty, 'k', buffersize=2,
                                  cache=False)))
    assert_multiset([('k', 'l', 'r'), (n, 'LN', None), (1, 'L1', None)],
                    list(etl.leftjoin(left, rempty, 'k', buffersize=2,
                                      cache=False)))
    assert_multiset([('k', 'l', 'r')],
                    list(etl.hashjoin(left, rempty, 'k', cache=False)))
    # both empty
    assert_multiset([('k', 'l', 'r')],
                    list(etl.outerjoin(lempty, rempty, 'k', buffersize=2,
                                       cache=False)))
    assert_multiset([('k', 'l', 'r')],
                    list(etl.hashjoin(lempty, rempty, 'k', cache=False)))


def test_all_nan_tables():
    # both sides entirely NaN keys: no pairs at all, outer joins keep
    # every row exactly once
    nl = [float('nan'), Decimal('NaN')]
    nr = [float('nan'), Decimal('-NaN3'), float('nan')]
    left = [('k', 'l')] + [(k, 'L%d' % i) for i, k in enumerate(nl)]
    right = [('k', 'r')] + [(k, 'R%d' % i) for i, k in enumerate(nr)]
    hdr = [('k', 'l', 'r')]
    expect_left = hdr + [(k, 'L%d' % i, None) for i, k in enumerate(nl)]
    expect_right = hdr + [(k, None, 'R%d' % i) for i, k in enumerate(nr)]
    expect_outer = expect_left + expect_right[1:]
    for joinfn, expect in ((etl.join, hdr),
                           (etl.leftjoin, expect_left),
                           (etl.rightjoin, expect_right),
                           (etl.outerjoin, expect_outer)):
        paths = run_merge(joinfn, left, right, 'k')
        assert_all_paths_equal(expect, paths)
    assert_all_paths_equal(hdr, run_hash(etl.hashjoin, left, right, 'k'))
    assert_all_paths_equal(expect_left,
                           run_hash(etl.hashleftjoin, left, right, 'k'))
    assert_all_paths_equal(expect_right,
                           run_hash(etl.hashrightjoin, left, right, 'k'))
    # anti joins keep everything
    assert_multiset(left, list(etl.antijoin(left, right, 'k',
                                            buffersize=2, cache=False)))
    assert_multiset(left, list(etl.hashantijoin(left, right, 'k')))


# ---------------------------------------------------------------------
# duplicate finite keys: cartesian product and multiplicity


def test_duplicate_finite_keys_with_nan():
    n = float('nan')
    left = [('k', 'l'),
            (1, 'L1a'), (1, 'L1b'), (2, 'L2'), (n, 'LN')]
    right = [('k', 'r'),
             (1, 'R1a'), (1, 'R1b'), (1, 'R1c'), (2, 'R2'),
             (n, 'RNa'), (n, 'RNb')]
    # inner join: 2x3 = 6 rows for key 1, 1 row for key 2, none for NaN
    expect_join = [('k', 'l', 'r')]
    for ltag in ('L1a', 'L1b'):
        for rtag in ('R1a', 'R1b', 'R1c'):
            expect_join.append((1, ltag, rtag))
    expect_join.append((2, 'L2', 'R2'))
    eq_(8, len(expect_join))  # header + 7
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect_join, paths)
    # exact merge-path order: key 1 group (L1a x R1a..c, L1b x R1a..c),
    # then key 2
    ieq_nan(expect_join, paths['mem'])
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect_join, paths)
    # left join adds the padded NaN row; outer adds the two right NaN rows
    expect_left = expect_join + [(n, 'LN', None)]
    expect_outer = expect_left + [(n, None, 'RNa'), (n, None, 'RNb')]
    assert_all_paths_equal(expect_left,
                           run_merge(etl.leftjoin, left, right, 'k'))
    assert_all_paths_equal(expect_left,
                           run_hash(etl.hashleftjoin, left, right, 'k'))
    assert_all_paths_equal(expect_outer,
                           run_merge(etl.outerjoin, left, right, 'k'))
    # duplicate NaN keys on both sides still produce zero pairs
    eq_(0, len([r for r in paths['hash']
                if r[1] == 'LN' and r[2] is not None]))


# ---------------------------------------------------------------------
# mixed numeric types, big integers, infinities, signed zero


def test_mixed_numeric_keys_match():
    # 1, 1.0, Decimal('1') and True are equal and must match; the output
    # key cell is the left row's original object
    left = [('k', 'l'),
            (1, 'L-int'),
            (1.0, 'L-float'),
            (Decimal('1'), 'L-dec'),
            (True, 'L-bool')]
    right = [('k', 'r'),
             (Decimal('1'), 'R-dec'),
             (1, 'R-int'),
             (True, 'R-bool')]
    expect = [('k', 'l', 'r')]
    for lk, ltag in ((1, 'L-int'), (1.0, 'L-float'),
                     (Decimal('1'), 'L-dec'), (True, 'L-bool')):
        for rtag in ('R-dec', 'R-int', 'R-bool'):
            expect.append((lk, ltag, rtag))
    eq_(13, len(expect))  # header + 4x3
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)


def test_big_int_keys_not_float_merged():
    big1 = 2**53 + 1        # not exactly representable as float
    big2 = 10**100
    left = [('k', 'l'), (big1, 'L-big1'), (big2, 'L-big2')]
    right = [('k', 'r'), (2**53, 'R-2^53'), (float(2**53), 'R-float'),
             (big1, 'R-big1'), (big2, 'R-big2')]
    # big1 matches only big1; 2**53 and float(2**53) are equal to each
    # other but not to big1; big2 matches big2 exactly
    expect = [('k', 'l', 'r'),
              (big1, 'L-big1', 'R-big1'),
              (big2, 'L-big2', 'R-big2')]
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # and the right side's 2**53 / 2**53.0 rows do match each other
    expect_r = [('k', 'r', 'l2')]
    right2 = [('k', 'r'), (2**53, 'R-int'), (2.0**53, 'R-float')]
    assert_multiset([('k', 'r', 'l'), (2**53, 'R-int', 'L'),
                     (2.0**53, 'R-float', 'L')],
                    list(etl.join(right2, [('k', 'l'), (2**53, 'L')],
                                  'k', buffersize=2, cache=False)))


def test_infinity_keys():
    inf, ninf = float('inf'), float('-inf')
    left = [('k', 'l'), (inf, 'L-inf'), (ninf, 'L-ninf'),
            (Decimal('Infinity'), 'L-decinf')]
    right = [('k', 'r'), (inf, 'R-inf'), (Decimal('-Infinity'), 'R-ndec'),
             (ninf, 'R-ninf')]
    # +inf keys are all mutually equal (incl. Decimal Infinity);
    # -inf keys likewise; +inf never matches -inf
    expect = [('k', 'l', 'r'),
              (inf, 'L-inf', 'R-inf'),
              (Decimal('Infinity'), 'L-decinf', 'R-inf'),
              (ninf, 'L-ninf', 'R-ndec'),
              (ninf, 'L-ninf', 'R-ninf')]
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)


def test_signed_zero_keys():
    left = [('k', 'l'),
            (0.0, 'L-pos'),
            (-0.0, 'L-neg'),
            (Decimal('-0'), 'L-decneg'),
            (0, 'L-int')]
    right = [('k', 'r'),
             (0, 'R-int'),
             (Decimal('0'), 'R-dec'),
             (-0.0, 'R-neg')]
    # all zero keys are equal: 4 x 3 = 12 pairs
    expect = [('k', 'l', 'r')]
    for lk, ltag in ((0.0, 'L-pos'), (-0.0, 'L-neg'),
                     (Decimal('-0'), 'L-decneg'), (0, 'L-int')):
        for rtag in ('R-int', 'R-dec', 'R-neg'):
            expect.append((lk, ltag, rtag))
    eq_(13, len(expect))
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # signed zero values are carried through unchanged
    actual = list(etl.join(left, right, 'k', buffersize=None,
                           cache=False))
    neg_rows = [r for r in actual[1:] if r[1] == 'L-neg']
    eq_(3, len(neg_rows))
    for r in neg_rows:
        assert type(r[0]) is float and math.copysign(1.0, r[0]) < 0
    dec_rows = [r for r in actual[1:] if r[1] == 'L-decneg']
    eq_(3, len(dec_rows))
    for r in dec_rows:
        assert isinstance(r[0], Decimal) and r[0].is_signed()


# ---------------------------------------------------------------------
# NaN in non-key fields


def test_nan_in_nonkey_field():
    nl, nr = float('nan'), Decimal('-NaN5')
    left = [('k', 'v'), (1, nl), (2, 'x')]
    right = [('k', 'w'), (1, nr), (2, 'y')]
    expect = [('k', 'v', 'w'), (1, nl, nr), (2, 'x', 'y')]
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect, paths)
    # memory path keeps the identical objects
    actual = list(etl.join(left, right, 'k', buffersize=None,
                           cache=False))
    assert actual[1][1] is nl
    assert actual[1][2] is nr


# ---------------------------------------------------------------------
# None and MISSING interplay, both match_missing modes


def _none_missing_tables():
    n = float('nan')
    left = [('k', 'l'),
            (None, 'LNone'),
            (MISSING, 'LMiss'),
            (n, 'LNan'),
            (1, 'L1')]
    right = [('k', 'r'),
             (None, 'RNone'),
             (MISSING, 'RMiss'),
             (n, 'RNan'),
             (1, 'R1')]
    return n, left, right


def test_none_missing_nan_default():
    n, left, right = _none_missing_tables()
    # None keys match None; MISSING and NaN keys never match
    expect_join = [('k', 'l', 'r'),
                   (None, 'LNone', 'RNone'),
                   (1, 'L1', 'R1')]
    paths = run_merge(etl.join, left, right, 'k')
    assert_all_paths_equal(expect_join, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k')
    assert_all_paths_equal(expect_join, paths)
    expect_outer = [('k', 'l', 'r'),
                    (MISSING, 'LMiss', None),
                    (MISSING, None, 'RMiss'),
                    (None, 'LNone', 'RNone'),
                    (n, 'LNan', None),
                    (n, None, 'RNan'),
                    (1, 'L1', 'R1')]
    paths = run_merge(etl.outerjoin, left, right, 'k')
    assert_all_paths_equal(expect_outer, paths)
    ieq_nan(expect_outer, paths['mem'])


def test_none_missing_nan_match_missing_true():
    n, left, right = _none_missing_tables()
    # match_missing=True lets MISSING match MISSING only; NaN still
    # never matches
    expect_join = [('k', 'l', 'r'),
                   (None, 'LNone', 'RNone'),
                   (MISSING, 'LMiss', 'RMiss'),
                   (1, 'L1', 'R1')]
    paths = run_merge(etl.join, left, right, 'k', match_missing=True)
    assert_all_paths_equal(expect_join, paths)
    paths = run_hash(etl.hashjoin, left, right, 'k', match_missing=True)
    assert_all_paths_equal(expect_join, paths)
    expect_outer = [('k', 'l', 'r'),
                    (MISSING, 'LMiss', 'RMiss'),
                    (None, 'LNone', 'RNone'),
                    (n, 'LNan', None),
                    (n, None, 'RNan'),
                    (1, 'L1', 'R1')]
    paths = run_merge(etl.outerjoin, left, right, 'k', match_missing=True)
    assert_all_paths_equal(expect_outer, paths)
    ieq_nan(expect_outer, paths['mem'])
    # hash left join with match_missing: only the NaN row is padded
    expect_hleft = [('k', 'l', 'r'),
                    (None, 'LNone', 'RNone'),
                    (MISSING, 'LMiss', 'RMiss'),
                    (n, 'LNan', None),
                    (1, 'L1', 'R1')]
    paths = run_hash(etl.hashleftjoin, left, right, 'k',
                     match_missing=True)
    assert_all_paths_equal(expect_hleft, paths)


# ---------------------------------------------------------------------
# decimal context safety end to end


def test_joins_decimal_context_not_disturbed():
    left = [('k', 'l'),
            (Decimal('NaN'), 'LN'),
            (Decimal('1'), 'L1'),
            (Decimal('2'), 'L2')]
    right = [('k', 'r'),
             (Decimal('-NaN7'), 'RN'),
             (Decimal('1'), 'R1'),
             (Decimal('2'), 'R2')]
    expect = [('k', 'l', 'r'),
              (Decimal('1'), 'L1', 'R1'), (Decimal('2'), 'L2', 'R2')]
    for trap in (True, False):
        with localcontext() as ctx:
            ctx.traps[InvalidOperation] = trap
            ctx.clear_flags()
            # exercise every NaN-aware path under this context
            list(etl.join(left, right, 'k', buffersize=2, cache=False))
            list(etl.join(left, right, 'k', buffersize=None, cache=False))
            list(etl.hashjoin(left, right, 'k', cache=False))
            list(etl.outerjoin(left, right, 'k', buffersize=2,
                               cache=False))
            list(etl.hashleftjoin(left, right, 'k', cache=False))
            list(etl.sort(left, 'k', buffersize=2, cache=False))
            flags = [k for k, v in ctx.flags.items() if v]
            eq_([], flags)
            eq_(trap, ctx.traps[InvalidOperation])
    # results identical under both trap regimes
    with localcontext() as ctx:
        ctx.traps[InvalidOperation] = True
        join_trap = list(etl.join(left, right, 'k', buffersize=2,
                                  cache=False))
        hash_trap = list(etl.hashjoin(left, right, 'k', cache=False))
    with localcontext() as ctx:
        ctx.traps[InvalidOperation] = False
        join_notrap = list(etl.join(left, right, 'k', buffersize=2,
                                    cache=False))
        hash_notrap = list(etl.hashjoin(left, right, 'k', cache=False))
    ieq_nan(expect, join_trap)
    ieq_nan(expect, join_notrap)
    ieq_nan(expect, hash_trap)
    ieq_nan(expect, hash_notrap)


# ---------------------------------------------------------------------
# cross-path consistency on a mixed table


def test_cross_path_consistency():
    n1, n2 = float('nan'), Decimal('NaN2')
    left = [('k', 'l'),
            (3, 'L3'), (n1, 'LN1'), (None, 'LNone'), (1, 'L1'),
            (2, 'L2a'), (2, 'L2b'), (n2, 'LN2')]
    right = [('k', 'r'),
             (2, 'R2'), (n2, 'RN2'), (1, 'R1'), (4, 'R4'),
             (None, 'RNone'), (n1, 'RN1')]
    expect_outer = [('k', 'l', 'r'),
                    (None, 'LNone', 'RNone'),
                    (n1, 'LN1', None), (n2, 'LN2', None),
                    (n2, None, 'RN2'), (n1, None, 'RN1'),
                    (1, 'L1', 'R1'),
                    (2, 'L2a', 'R2'), (2, 'L2b', 'R2'),
                    (3, 'L3', None),
                    (4, None, 'R4')]
    paths = run_merge(etl.outerjoin, left, right, 'k')
    assert_all_paths_equal(expect_outer, paths)
    ieq_nan(expect_outer, paths['mem'])
    # every merge variant agrees with every other
    multisets = {name: row_multiset(rows) for name, rows in paths.items()}
    first = multisets['mem']
    for name, ms in multisets.items():
        eq_(first, ms)


# ---------------------------------------------------------------------
# laziness, independence and lifecycle


def test_join_iterators_independent():
    n = float('nan')
    left = [('k', 'l'), (n, 'LN'), (1, 'L1'), (2, 'L2')]
    right = [('k', 'r'), (n, 'RN'), (1, 'R1'), (3, 'R3')]
    view = etl.outerjoin(left, right, 'k', buffersize=2, cache=False)
    it1 = iter(view)
    it2 = iter(view)
    # interleave consumption: each iterator keeps its own merge state
    next(it1)
    next(it1)
    rows1 = list(it1)
    rows2 = list(it2)
    # header + 5 data rows: 2 left-only NaN/right NaN, 1 match, 2 outer
    eq_(6, len(rows2))
    eq_(6, len(rows1) + 2)
    # the partially consumed iterator saw exactly the same stream
    ieq_nan(rows2[2:], rows1)
    # a fresh full pass equals the second iterator's rows
    ieq_nan(rows2, list(view))


def test_join_source_exception_propagates():
    n = float('nan')

    def failing():
        yield ('k', 'l')
        yield (1, 'L1')
        yield (n, 'LN')
        raise RuntimeError('boom')

    view = etl.outerjoin(failing(), [('k', 'r'), (1, 'R1')], 'k',
                         buffersize=2, cache=False)
    with pytest.raises(RuntimeError):
        list(view)


def test_join_iterator_close():
    n = float('nan')
    left = [('k', 'l')] + [(i, 'L%d' % i) for i in range(10)]
    left.insert(3, (n, 'LN'))
    right = [('k', 'r')] + [(i, 'R%d' % i) for i in range(10)]
    view = etl.leftjoin(left, right, 'k', buffersize=2, cache=False)
    it = iter(view)
    next(it)
    next(it)
    it.close()  # must not raise
    # view is still re-iterable afterwards
    eq_(12, len(list(view)))
