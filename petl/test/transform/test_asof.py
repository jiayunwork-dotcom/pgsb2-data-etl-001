from __future__ import absolute_import, print_function, division


import gc
import math
import os
import random
from datetime import datetime, timedelta

import pytest

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.transform.asof import asofjoin, AsofJoinView


# ---------------------------------------------------------------------------
# naive reference implementation (list based, deliberately independent of the
# streaming algorithm) used by the randomised comparison tests
# ---------------------------------------------------------------------------

def _naive_asof(left, right, ltime_idx, rtime_idx, lgroup_idx, rgroup_idx,
                direction, tolerance, allow_exact_matches, missing,
                rprefix='right_'):
    lhdr = left[0]
    rhdr = right[0]
    ldata = left[1:]
    rdata = right[1:]
    rout_idx = [i for i in range(len(rhdr)) if i not in set(rgroup_idx)]
    out = [tuple(lhdr) +
           tuple((rprefix + str(rhdr[i])) if rprefix is not None
                 else rhdr[i] for i in rout_idx)]

    def gvals(row, idxs):
        return tuple(row[i] for i in idxs)

    # stable order: groups ascending, then left time ascending, then the
    # original left row order. Group values use petl's relaxed (None-safe)
    # comparison semantics.
    from petl.comparison import Comparable
    order = sorted(
        range(len(ldata)),
        key=lambda i: (Comparable(tuple(ldata[i][j] for j in lgroup_idx)),
                       ldata[i][ltime_idx], i))
    for i in order:
        lrow = ldata[i]
        lt = lrow[ltime_idx]
        lg = gvals(lrow, lgroup_idx)
        candidates = []
        for rn, rrow in enumerate(rdata):
            if rgroup_idx and gvals(rrow, rgroup_idx) != lg:
                continue
            rt = rrow[rtime_idx]
            if not allow_exact_matches and rt == lt:
                continue
            if direction == 'backward' and rt <= lt:
                pass
            elif direction == 'forward' and rt >= lt:
                pass
            elif direction == 'nearest':
                pass
            else:
                continue
            # tolerance is inclusive; out-of-range rows are not candidates
            if tolerance is not None and abs(rt - lt) > tolerance:
                continue
            candidates.append((rn, rrow))
        if not candidates:
            chosen = None
        elif direction == 'backward':
            # latest time; last row in original order on a time tie
            chosen = max(candidates,
                         key=lambda c: (c[1][rtime_idx], c[0]))
        elif direction == 'forward':
            # earliest time; first row in original order on a time tie
            chosen = min(candidates,
                         key=lambda c: (c[1][rtime_idx], c[0]))
        else:
            def nkey(c):
                rn, rrow = c
                dist = abs(rrow[rtime_idx] - lt)
                # ties go to the earlier time; on equal time take the last
                # right row in original order
                earlier = 0 if rrow[rtime_idx] <= lt else 1
                return (dist, earlier, -rn if earlier == 0 else rn)
            chosen = min(candidates, key=nkey)
        if chosen is None:
            out.append(tuple(lrow) + (missing,) * len(rout_idx))
        else:
            rrow = chosen[1]
            out.append(tuple(lrow) + tuple(rrow[i] for i in rout_idx))
    return out


# ---------------------------------------------------------------------------
# the worked example from the specification
# ---------------------------------------------------------------------------

def _example_tables():
    left = [('asset', 't', 'event'),
            ('B', 7, 'eB'),
            ('A', 4, 'eA'),
            ('A', 6, 'eC')]
    right = [('asset', 't', 'value'),
             ('A', 5, 'later'),
             ('A', 3, 'old'),
             ('A', 3, 'revised'),
             ('B', 10, 'future')]
    return left, right


def test_asofjoin_example_backward():
    left, right = _example_tables()
    expect = [('asset', 't', 'event', 'right_t', 'right_value'),
              ('A', 4, 'eA', 3, 'revised'),
              ('A', 6, 'eC', 5, 'later'),
              ('B', 7, 'eB', None, None)]
    ieq(expect, asofjoin(left, right, key='t', by='asset'))
    # fluent/chained API
    ieq(expect, etl.wrap(left).asofjoin(right, key='t', by='asset'))


def test_asofjoin_example_all_directions():
    left, right = _example_tables()
    backward = [('asset', 't', 'event', 'right_t', 'right_value'),
                ('A', 4, 'eA', 3, 'revised'),
                ('A', 6, 'eC', 5, 'later'),
                ('B', 7, 'eB', None, None)]
    forward = [('asset', 't', 'event', 'right_t', 'right_value'),
               ('A', 4, 'eA', 5, 'later'),
               ('A', 6, 'eC', None, None),
               ('B', 7, 'eB', 10, 'future')]
    nearest = [('asset', 't', 'event', 'right_t', 'right_value'),
               ('A', 4, 'eA', 3, 'revised'),
               ('A', 6, 'eC', 5, 'later'),
               ('B', 7, 'eB', 10, 'future')]
    ieq(backward, asofjoin(left, right, key='t', by='asset'))
    ieq(forward, asofjoin(left, right, key='t', by='asset',
                          direction='forward'))
    ieq(nearest, asofjoin(left, right, key='t', by='asset',
                          direction='nearest'))


def test_asofjoin_presorted_matches_sorted():
    left, right = _example_tables()
    ls = list(etl.sort(left, key=('asset', 't')))
    rs = list(etl.sort(right, key=('asset', 't')))
    for direction in ('backward', 'forward', 'nearest'):
        got_sort = list(asofjoin(left, right, key='t', by='asset',
                                 direction=direction))
        got_pre = list(asofjoin(ls, rs, key='t', by='asset',
                                direction=direction, presorted=True))
        ieq(got_sort, got_pre)


# ---------------------------------------------------------------------------
# headers, prefixes and field selection
# ---------------------------------------------------------------------------

def test_asofjoin_output_header_and_prefixes():
    left = [('g', 't', 'e'), ('A', 1, 'x')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    tbl = asofjoin(left, right, key='t', by='g')
    eq_(('g', 't', 'e', 'right_t', 'right_v'), next(iter(tbl)))
    tbl = asofjoin(left, right, key='t', by='g', lprefix='l_',
                   rprefix='r_')
    eq_(('l_g', 'l_t', 'l_e', 'r_t', 'r_v'), next(iter(tbl)))
    # rprefix None keeps right non-group names verbatim; the right time
    # field 't' then clashes with the left 't', which is rejected
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g', rprefix=None))
    # a non-clashing right table with rprefix None is fine
    right2 = [('g', 'obs', 'w'), ('A', 1, 'q')]
    tbl = asofjoin(left, right2, lkey='t', rkey='obs', by='g',
                   rprefix=None)
    eq_(('g', 't', 'e', 'obs', 'w'), next(iter(tbl)))


def test_asofjoin_duplicate_output_header_rejected():
    left = [('g', 't', 'e'), ('A', 1, 'x')]
    right = [('g', 't', 'e'), ('A', 1, 'z')]  # 'e' clashes after output
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g', rprefix=None))


def test_asofjoin_duplicate_input_header_rejected():
    left = [('g', 't', 't'), ('A', 1, 2)]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g'))
    dup_right = [('g', 't', 't'), ('A', 1, 2)]
    clean_left = [('g', 't', 'e'), ('A', 1, 'x')]
    with pytest.raises(ArgumentError):
        list(asofjoin(clean_left, dup_right, key='t', by='g'))


def test_asofjoin_integer_and_negative_time_indices():
    left = [('g', 't', 'e'),
            ('B', 7, 'eB'), ('A', 4, 'eA'), ('A', 6, 'eC')]
    right = [('g', 'obs_t', 'value'),
             ('A', 5, 'later'), ('A', 3, 'old'), ('A', 3, 'revised'),
             ('B', 10, 'future')]
    # negative indices: t is position -2 on both tables
    got = list(asofjoin(left, right, lkey=-2, rkey=1, by='g'))
    expect = list(asofjoin(left, right, lkey='t', rkey='obs_t', by='g'))
    ieq(expect, got)
    # plain integer positions for the time fields; by stays a name
    got2 = list(asofjoin(left, right, lkey=1, rkey=1, by='g'))
    ieq(expect, got2)


def test_asofjoin_missing_field_and_bad_index():
    left = [('g', 't', 'e'), ('A', 1, 'x')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(FieldSelectionError):
        list(asofjoin(left, right, key='nope', by='g'))
    with pytest.raises(FieldSelectionError):
        list(asofjoin(left, right, key=99, by='g'))
    with pytest.raises(FieldSelectionError):
        list(asofjoin(left, right, key=-99, by='g'))
    with pytest.raises(FieldSelectionError):
        list(asofjoin(left, right, key='t', by='nope'))


def test_asofjoin_key_argument_conflicts():
    left = [('g', 't', 'e'), ('A', 1, 'x')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        asofjoin(left, right)  # no time field at all
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', lkey='t')
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', rkey='t')
    with pytest.raises(ArgumentError):
        asofjoin(left, right, lkey='t')  # only one side
    with pytest.raises(ArgumentError):
        asofjoin(left, right, rkey='t')  # only one side
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='t'))  # time doubles as group


def test_asofjoin_invalid_direction_and_options():
    left = [('g', 't', 'e'), ('A', 1, 'x')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by='g', direction='sideways')
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by='g', tolerance=-1)
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by='g', tolerance=float('nan'))
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by='g', tolerance=float('inf'))
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by='g', tolerance='1')
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by='g',
                 tolerance=timedelta(days=-1))
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by=('g', 'g'))
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by=42)
    with pytest.raises(ArgumentError):
        asofjoin(left, right, key='t', by=[])


def test_asofjoin_tolerance_domain_mismatch():
    left = [('g', 't', 'e'), ('A', 1, 'x')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g',
                      tolerance=timedelta(seconds=1)))
    dleft = [('g', 't', 'e'),
             ('A', datetime(2020, 1, 1, 12), 'x')]
    dright = [('g', 't', 'v'),
              ('A', datetime(2020, 1, 1, 11), 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(dleft, dright, key='t', by='g', tolerance=1))


# ---------------------------------------------------------------------------
# matching semantics
# ---------------------------------------------------------------------------

def test_asofjoin_no_grouping():
    left = [('t', 'e'), (7, 'a'), (4, 'b'), (6, 'c')]
    right = [('t', 'v'), (5, 'x'), (3, 'y'), (3, 'z')]
    expect = [('t', 'e', 'right_t', 'right_v'),
              (4, 'b', 3, 'z'),
              (6, 'c', 5, 'x'),
              (7, 'a', 5, 'x')]
    ieq(expect, asofjoin(left, right, key='t'))
    ls = list(etl.sort(left, key='t'))
    rs = list(etl.sort(right, key='t'))
    ieq(expect, asofjoin(ls, rs, key='t', presorted=True))


def test_asofjoin_compound_group():
    left = [('a', 'b', 't', 'e'),
            ('X', 'P', 2, 'l1'),
            ('X', 'Q', 2, 'l2'),
            ('X', 'P', 5, 'l3')]
    right = [('a', 'b', 't', 'v'),
             ('X', 'P', 1, 'r1'),
             ('X', 'P', 4, 'r2'),
             ('X', 'Q', 9, 'r3')]
    got = list(asofjoin(left, right, key='t', by=('a', 'b')))
    expect = [
        ('a', 'b', 't', 'e', 'right_t', 'right_v'),
        ('X', 'P', 2, 'l1', 1, 'r1'),
        ('X', 'P', 5, 'l3', 4, 'r2'),
        ('X', 'Q', 2, 'l2', None, None),
    ]
    ieq(expect, got)


def test_asofjoin_different_time_field_names():
    left = [('g', 'when', 'e'), ('A', 4, 'a'), ('A', 6, 'b')]
    right = [('g', 'observed_at', 'v'), ('A', 5, 'x'), ('A', 3, 'y')]
    got = list(asofjoin(left, right, lkey='when', rkey='observed_at',
                        by='g'))
    expect = [('g', 'when', 'e', 'right_observed_at', 'right_v'),
              ('A', 4, 'a', 3, 'y'),
              ('A', 6, 'b', 5, 'x')]
    ieq(expect, got)


def test_asofjoin_allow_exact_matches_false():
    left = [('g', 't', 'e'), ('A', 5, 'a'), ('A', 6, 'b')]
    right = [('g', 't', 'v'),
             ('A', 5, 'exact'), ('A', 5, 'exact2'),
             ('A', 4, 'before'), ('A', 7, 'after')]
    # at event 5 the equal block is excluded, leaving the strict predecessor
    # at 4; at event 6 the former equal block is now a strict predecessor
    got = list(asofjoin(left, right, key='t', by='g',
                        allow_exact_matches=False))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
              ('A', 5, 'a', 4, 'before'),
              ('A', 6, 'b', 5, 'exact2')]
    ieq(expect, got)
    got_fwd = list(asofjoin(left, right, key='t', by='g',
                            direction='forward',
                            allow_exact_matches=False))
    expect_fwd = [('g', 't', 'e', 'right_t', 'right_v'),
                  ('A', 5, 'a', 7, 'after'),
                  ('A', 6, 'b', 7, 'after')]
    ieq(expect_fwd, got_fwd)
    got_near = list(asofjoin(left, right, key='t', by='g',
                             direction='nearest',
                             allow_exact_matches=False))
    expect_near = [('g', 't', 'e', 'right_t', 'right_v'),
                   # dist 1 to 4 vs dist 2 to 7
                   ('A', 5, 'a', 4, 'before'),
                   # dist 1 to 5 vs dist 1 to 7: earlier time wins
                   ('A', 6, 'b', 5, 'exact2')]
    ieq(expect_near, got_near)
    # an event with only equal-time right rows really has no candidate
    only_eq_left = [('g', 't', 'e'), ('A', 5, 'z')]
    only_eq_right = [('g', 't', 'v'), ('A', 5, 'x'), ('A', 5, 'y')]
    got2 = list(asofjoin(only_eq_left, only_eq_right, key='t', by='g',
                         allow_exact_matches=False))
    ieq([('g', 't', 'e', 'right_t', 'right_v'),
         ('A', 5, 'z', None, None)], got2)


def test_asofjoin_repeated_right_time_tie_rules():
    left = [('g', 't', 'e'), ('A', 5, 'a')]
    right = [('g', 't', 'v'),
             ('A', 5, 'first'), ('A', 5, 'second'), ('A', 5, 'third')]
    ieq([('g', 't', 'e', 'right_t', 'right_v'),
         ('A', 5, 'a', 5, 'third')],
        asofjoin(left, right, key='t', by='g'))
    ieq([('g', 't', 'e', 'right_t', 'right_v'),
         ('A', 5, 'a', 5, 'first')],
        asofjoin(left, right, key='t', by='g', direction='forward'))
    ieq([('g', 't', 'e', 'right_t', 'right_v'),
         ('A', 5, 'a', 5, 'third')],
        asofjoin(left, right, key='t', by='g', direction='nearest'))


def test_asofjoin_nearest_equal_distance_prefers_earlier():
    left = [('g', 't', 'e'), ('A', 4, 'a')]
    right = [('g', 't', 'v'), ('A', 3, 'old'), ('A', 5, 'new')]
    ieq([('g', 't', 'e', 'right_t', 'right_v'),
         ('A', 4, 'a', 3, 'old')],
        asofjoin(left, right, key='t', by='g', direction='nearest'))


def test_asofjoin_tolerance_boundaries():
    left = [('g', 't', 'e'), ('A', 4, 'a'), ('A', 10, 'b')]
    right = [('g', 't', 'v'), ('A', 3, 'x'), ('A', 8, 'y')]
    # inclusive boundary: distance exactly 1 matches
    got = list(asofjoin(left, right, key='t', by='g', tolerance=1))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
              ('A', 4, 'a', 3, 'x'),
              ('A', 10, 'b', None, None)]
    ieq(expect, got)
    # float tolerance with int times
    got = list(asofjoin(left, right, key='t', by='g', tolerance=2.0))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
              ('A', 4, 'a', 3, 'x'),
              ('A', 10, 'b', 8, 'y')]
    ieq(expect, got)


def test_asofjoin_datetime_domain():
    base = datetime(2021, 6, 1)
    left = [('g', 't', 'e'),
            ('A', base + timedelta(hours=4), 'a'),
            ('A', base + timedelta(hours=6), 'b')]
    right = [('g', 't', 'v'),
             ('A', base + timedelta(hours=5), 'x'),
             ('A', base + timedelta(hours=3), 'y')]
    got = list(asofjoin(left, right, key='t', by='g'))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
              ('A', base + timedelta(hours=4), 'a',
               base + timedelta(hours=3), 'y'),
              ('A', base + timedelta(hours=6), 'b',
               base + timedelta(hours=5), 'x')]
    ieq(expect, got)
    # timedelta tolerance at an inclusive boundary
    got = list(asofjoin(left, right, key='t', by='g',
                        tolerance=timedelta(hours=1)))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
              ('A', base + timedelta(hours=4), 'a',
               base + timedelta(hours=3), 'y'),
              ('A', base + timedelta(hours=6), 'b',
               base + timedelta(hours=5), 'x')]
    ieq(expect, got)
    got = list(asofjoin(left, right, key='t', by='g',
                        tolerance=timedelta(minutes=30)))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
              ('A', base + timedelta(hours=4), 'a', None, None),
              ('A', base + timedelta(hours=6), 'b', None, None)]
    ieq(expect, got)


def test_asofjoin_repeated_left_rows_preserved():
    left = [('g', 't', 'e'),
            ('A', 5, 'one'), ('A', 5, 'two'), ('A', 5, 'three')]
    right = [('g', 't', 'v'), ('A', 4, 'r')]
    got = list(asofjoin(left, right, key='t', by='g'))
    eq_(4, len(got))
    for suffix, row in zip(('one', 'two', 'three'), got[1:]):
        eq_(suffix, row[2])
        eq_(4, row[3])


def test_asofjoin_none_group_value():
    left = [('g', 't', 'e'),
            (None, 2, 'a'), ('A', 2, 'b'), (None, 3, 'c')]
    right = [('g', 't', 'v'),
             (None, 1, 'r1'), (None, 1, 'r2'), ('A', 1, 'r3')]
    got = list(asofjoin(left, right, key='t', by='g'))
    expect = [('g', 't', 'e', 'right_t', 'right_v'),
            (None, 2, 'a', 1, 'r2'),
            (None, 3, 'c', 1, 'r2'),
            ('A', 2, 'b', 1, 'r3')]
    ieq(expect, got)


def test_asofjoin_missing_fill_value():
    left = [('g', 't', 'e'), ('A', 1, 'a')]
    right = [('g', 't', 'v'), ('B', 1, 'z')]
    got = list(asofjoin(left, right, key='t', by='g', missing='NA'))
    eq_(('A', 1, 'a', 'NA', 'NA'), got[1])


# ---------------------------------------------------------------------------
# empty tables
# ---------------------------------------------------------------------------

def test_asofjoin_header_only_left():
    left = [('g', 't', 'e')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    got = list(asofjoin(left, right, key='t', by='g'))
    eq_([('g', 't', 'e', 'right_t', 'right_v')], got)


def test_asofjoin_header_only_right():
    left = [('g', 't', 'e'), ('A', 1, 'a')]
    right = [('g', 't', 'v')]
    got = list(asofjoin(left, right, key='t', by='g'))
    eq_([('g', 't', 'e', 'right_t', 'right_v'),
         ('A', 1, 'a', None, None)], got)


def test_asofjoin_no_header_rejected():
    with pytest.raises(ArgumentError):
        list(asofjoin([], [('g', 't')], key='t'))
    with pytest.raises(ArgumentError):
        list(asofjoin([('g', 't')], [], key='t'))


# ---------------------------------------------------------------------------
# invalid data values (lazy, raised while consuming)
# ---------------------------------------------------------------------------

def test_asofjoin_bad_time_values():
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    bad_values = (None, float('nan'), float('inf'), -float('inf'),
                  '2020-01-01', True)
    for bad in bad_values:
        left = [('g', 't', 'e'), ('A', bad, 'a')]
        with pytest.raises(ArgumentError):
            list(asofjoin(left, right, key='t', by='g'))
    # same checks on the right side
    for bad in bad_values:
        r = [('g', 't', 'v'), ('A', bad, 'z')]
        l = [('g', 't', 'e'), ('A', 1, 'a')]
        with pytest.raises(ArgumentError):
            list(asofjoin(l, r, key='t', by='g'))


def test_asofjoin_mixed_time_domains_rejected():
    left = [('g', 't', 'e'), ('A', 1, 'a'),
            ('A', datetime(2020, 1, 1), 'b')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g'))
    # cross-table domain mismatch
    l2 = [('g', 't', 'e'), ('A', datetime(2020, 1, 1), 'a')]
    r2 = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(l2, r2, key='t', by='g'))


def test_asofjoin_timezone_aware_datetime_rejected():
    from datetime import timezone
    left = [('g', 't', 'e'),
            ('A', datetime(2020, 1, 1, tzinfo=timezone.utc), 'a')]
    right = [('g', 't', 'v'),
             ('A', datetime(2019, 1, 1), 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g'))


def test_asofjoin_wrong_row_width():
    left = [('g', 't', 'e'), ('A', 1)]  # short row
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    with pytest.raises(ArgumentError):
        list(asofjoin(left, right, key='t', by='g'))
    left2 = [('g', 't', 'e'), ('A', 1, 'a', 'extra')]  # long row
    with pytest.raises(ArgumentError):
        list(asofjoin(left2, right, key='t', by='g'))


def test_asofjoin_presorted_out_of_order_reports_side_and_rows():
    left = [('g', 't', 'e'), ('A', 2, 'a'), ('A', 1, 'b')]
    right = [('g', 't', 'v'), ('A', 1, 'z')]
    try:
        list(asofjoin(left, right, key='t', by='g', presorted=True))
    except ArgumentError as e:
        msg = str(e)
        assert 'left' in msg
        assert 'row 1' in msg and 'row 2' in msg
    else:
        pytest.fail('expected ArgumentError')
    # right side regression: the out-of-order row is consumed as a candidate
    l2 = [('g', 't', 'e'), ('A', 3, 'a')]
    r2 = [('g', 't', 'v'), ('A', 2, 'z'), ('A', 1, 'y')]
    try:
        list(asofjoin(l2, r2, key='t', by='g', presorted=True))
    except ArgumentError as e:
        assert 'right' in str(e)
    else:
        pytest.fail('expected ArgumentError')
    # an unconsumed right tail is not force-drained, so its disorder is not
    # required to be reported
    l3 = [('g', 't', 'e'), ('A', 1, 'a')]
    r3 = [('g', 't', 'v'), ('A', 2, 'z'), ('A', 1, 'y')]
    list(asofjoin(l3, r3, key='t', by='g', presorted=True))


# ---------------------------------------------------------------------------
# laziness and single-iteration contracts
# ---------------------------------------------------------------------------

class _CountingTable(object):
    """Counts iterators created and close() calls.

    The backing rows are held as a list. The join contract is that exactly
    one iterator is requested from the table per result iteration.
    """

    def __init__(self, rows):
        self.rows = list(rows)
        self.iter_count = 0
        self.closed = 0

    def __iter__(self):
        self.iter_count += 1
        outer = self
        source = iter(self.rows)

        def gen():
            try:
                for row in source:
                    yield row
            finally:
                outer.closed += 1

        return gen()


def _drain(view):
    """Consume a view through exactly one generator, bypassing __len__."""
    gen = iter(view)
    rows = []
    while True:
        try:
            rows.append(next(gen))
        except StopIteration:
            return rows


def test_asofjoin_view_creation_does_not_read_inputs():
    left = _CountingTable([('g', 't', 'e'), ('A', 1, 'a')])
    right = _CountingTable([('g', 't', 'v'), ('A', 1, 'z')])
    view = asofjoin(left, right, key='t', by='g')
    assert isinstance(view, AsofJoinView)
    eq_(0, left.iter_count)
    eq_(0, right.iter_count)
    # even with sorting configured, nothing happens until iteration
    view2 = asofjoin(left, right, key='t', by='g', buffersize=2)
    eq_(0, left.iter_count)
    eq_(0, right.iter_count)
    _drain(view2)
    eq_(1, left.iter_count)
    eq_(1, right.iter_count)


def test_asofjoin_single_iterator_per_input_presorted():
    left = _CountingTable([('g', 't', 'e'),
                           ('A', 1, 'a'), ('A', 2, 'b')])
    right = _CountingTable([('g', 't', 'v'),
                            ('A', 1, 'x'), ('A', 2, 'y')])
    _drain(asofjoin(left, right, key='t', by='g', presorted=True))
    eq_(1, left.iter_count)
    eq_(1, right.iter_count)


def test_asofjoin_single_iterator_per_input_sorted():
    left = _CountingTable([('g', 't', 'e'),
                           ('A', 2, 'b'), ('A', 1, 'a')])
    right = _CountingTable([('g', 't', 'v'),
                            ('A', 2, 'y'), ('A', 1, 'x')])
    _drain(asofjoin(left, right, key='t', by='g', buffersize=1))
    eq_(1, left.iter_count)
    eq_(1, right.iter_count)


def test_asofjoin_reiterable_sources_give_same_result():
    left, right = _example_tables()
    view = asofjoin(left, right, key='t', by='asset')
    first = list(view)
    second = list(view)
    ieq(first, second)


def test_asofjoin_does_not_modify_inputs():
    left, right = _example_tables()
    left_copy = [tuple(r) for r in left]
    right_copy = [tuple(r) for r in right]
    list(asofjoin(left, right, key='t', by='asset', buffersize=2))
    eq_(left_copy, left)
    eq_(right_copy, right)


# ---------------------------------------------------------------------------
# streaming / bounded-memory behaviour of the presorted path
# ---------------------------------------------------------------------------

class _StreamingSource(object):
    """Generator-backed table with a closeable iterator and read probe."""

    def __init__(self, hdr, row_fn):
        self.hdr = hdr
        self.row_fn = row_fn
        self.last_read = 0

    def __iter__(self):
        outer = self
        yield self.hdr
        for i, row in enumerate(self.row_fn(), start=1):
            outer.last_read = i
            yield row


def test_asofjoin_presorted_streams_without_buffering_huge_group():
    # A huge group must not be retained in memory: the join emits its result
    # after walking right rows one at a time and never holds a growing
    # collection of them. Forward direction only needs the first row of each
    # equal-time block plus its boundary, so consumption stays close to the
    # event time rather than draining the whole table.
    n = 5000

    def right_rows():
        for i in range(n):
            yield ('A', 1, 'r%05d' % i)
        for i in range(n):
            yield ('A', 2, 's%05d' % i)
        for i in range(n):
            yield ('A', 9, 'late%05d' % i)

    def left_rows():
        yield ('A', 3, 'event')

    right = _StreamingSource(('g', 't', 'v'), right_rows)
    left = _StreamingSource(('g', 't', 'e'), left_rows)
    view = asofjoin(left, right, key='t', by='g', presorted=True,
                    direction='forward')
    it = iter(view)
    next(it)  # header
    out = next(it)
    eq_(('A', 3, 'event', 9, 'late00000'), out)
    # only the boundary between t=2 and t=9 had to be reached; the giant
    # t=9 block itself and anything beyond it is not force-drained
    assert right.last_read < 3 * n, \
        'right table drained beyond the candidate boundary (read %d of %d)' \
        % (right.last_read, 3 * n)

    # backward variant: the last row of the t=2 block is the answer and the
    # later t=9 block must still not be entered
    right2 = _StreamingSource(('g', 't', 'v'), right_rows)
    left2 = _StreamingSource(('g', 't', 'e'), left_rows)
    view2 = asofjoin(left2, right2, key='t', by='g', presorted=True,
                     direction='backward')
    it2 = iter(view2)
    next(it2)
    out2 = next(it2)
    eq_(('A', 3, 'event', 2, 's%05d' % (n - 1)), out2)
    assert right2.last_read < 3 * n


def test_asofjoin_presorted_duplicate_time_block_not_materialised():
    n = 2000

    def right_rows():
        for i in range(n):
            yield ('A', 5, i)

    def left_rows():
        yield ('A', 5, 'event')

    right = _StreamingSource(('g', 't', 'v'), right_rows)
    left = _StreamingSource(('g', 't', 'e'), left_rows)
    view = asofjoin(left, right, key='t', by='g', presorted=True,
                    direction='forward')
    it = iter(view)
    next(it)
    out = next(it)
    # forward selects the first row of the block; walking the block retains
    # only its boundaries, never the block as a whole
    eq_(('A', 5, 'event', 5, 0), out)
    gc.collect()


def test_asofjoin_presorted_retained_state_is_constant_sized():
    # Track live right-row objects while a giant repeated block is streamed:
    # a correct implementation discards each row after advancing, so the
    # number of live rows stays bounded even as tens of thousands pass by.
    import weakref

    class Row(object):
        def __init__(self, t, i):
            self.t = t
            self.i = i

    n = 20000
    live = []

    def right_rows():
        for i in range(n):
            marker = Row(5, i)
            row = ('A', 5, marker)
            live.append(weakref.ref(marker))
            del marker
            yield row

    def left_rows():
        # _StreamingSource prepends the header itself
        yield ('A', 6, 'event')

    right = _StreamingSource(('g', 't', 'v'), right_rows)
    left = _StreamingSource(('g', 't', 'e'), left_rows)
    view = asofjoin(left, right, key='t', by='g', presorted=True,
                    direction='backward')
    it = iter(view)
    next(it)
    next(it)  # forces the whole t=5 block to be walked
    gc.collect()
    survivors = [r for r in live if r() is not None]
    # at most a constant handful of row objects may remain referenced
    assert len(survivors) < 10, \
        'join retained %d rows of a %d-row equal-time block' % \
        (len(survivors), n)


# ---------------------------------------------------------------------------
# resource release
# ---------------------------------------------------------------------------

class _CloseableCountingTable(object):

    def __init__(self, rows, fail_at=None):
        self.rows = rows
        self.fail_at = fail_at
        self.closed = 0
        self.iters = 0

    def __iter__(self):
        self.iters += 1
        outer = self
        rows = self.rows

        class _It(object):
            def __init__(self):
                self._it = iter(rows)
                self._n = 0
                self._closed = False

            def __iter__(self):
                return self

            def __next__(self):
                row = next(self._it)
                self._n += 1
                if outer.fail_at is not None and self._n == outer.fail_at:
                    raise RuntimeError('boom at row %d' % self._n)
                return row

            def close(self):
                if not self._closed:
                    self._closed = True
                    outer.closed += 1

        return _It()


def test_asofjoin_close_on_exhaustion():
    left = _CloseableCountingTable([('g', 't', 'e'), ('A', 1, 'a')])
    right = _CloseableCountingTable([('g', 't', 'v'), ('A', 1, 'z')])
    it = iter(asofjoin(left, right, key='t', by='g', presorted=True))
    list(it)
    it.close()
    eq_(1, left.closed)
    eq_(1, right.closed)


def test_asofjoin_close_on_early_abandonment():
    left = _CloseableCountingTable(
        [('g', 't', 'e')] + [('A', i, 'e%d' % i) for i in range(100)])
    right = _CloseableCountingTable(
        [('g', 't', 'v')] + [('A', i, 'v%d' % i) for i in range(100)])
    it = iter(asofjoin(left, right, key='t', by='g', presorted=True))
    next(it)
    next(it)
    it.close()
    eq_(1, left.closed)
    eq_(1, right.closed)
    # further consumption must not pull more source rows
    with pytest.raises(StopIteration):
        next(it)


def test_asofjoin_source_exception_propagates_and_closes():
    left = _CloseableCountingTable(
        [('g', 't', 'e'), ('A', 1, 'a'), ('A', 2, 'b')])
    right = _CloseableCountingTable(
        [('g', 't', 'v'), ('A', 1, 'z')], fail_at=2)
    it = iter(asofjoin(left, right, key='t', by='g', presorted=True))
    with pytest.raises(RuntimeError):
        list(it)
    eq_(1, right.closed)
    eq_(1, left.closed)


def test_asofjoin_temp_files_removed(tmp_path):
    tmpdir = str(tmp_path)
    left, right = _example_tables()
    # enlarge to force several external chunks
    big_left = [left[0]] + left[1:] * 50
    big_right = [right[0]] + right[1:] * 50
    view = asofjoin(big_left, big_right, key='t', by='asset',
                    buffersize=3, tempdir=tmpdir)
    list(view)
    gc.collect()
    eq_([], os.listdir(tmpdir))
    # same guarantee when the consumer stops early
    view2 = asofjoin(big_left, big_right, key='t', by='asset',
                     buffersize=3, tempdir=tmpdir)
    it = iter(view2)
    next(it)
    next(it)
    it.close()
    gc.collect()
    eq_([], os.listdir(tmpdir))


# ---------------------------------------------------------------------------
# randomised comparison against the naive reference
# ---------------------------------------------------------------------------

def _random_tables(rng, datetime_domain=False):
    groups = ('A', 'B', None, 'C')
    nleft = rng.randint(1, 40)
    nright = rng.randint(0, 40)
    if datetime_domain:
        epoch = datetime(2020, 1, 1)

        def t():
            return epoch + timedelta(seconds=rng.randint(0, 600))
    else:
        def t():
            return rng.choice((rng.randint(0, 30),
                               rng.randint(0, 30) + 0.5))
    lhdr = ('g', 't', 'e')
    rhdr = ('g', 't', 'v')
    left = [lhdr]
    for i in range(nleft):
        left.append((rng.choice(groups), t(), 'l%d' % i))
    right = [rhdr]
    for i in range(nright):
        right.append((rng.choice(groups), t(), 'r%d' % i))
    return left, right


@pytest.mark.parametrize('direction', ('backward', 'forward', 'nearest'))
@pytest.mark.parametrize('datetime_domain', (False, True))
def test_asofjoin_random_vs_naive(direction, datetime_domain):
    rng = random.Random(20220714)
    for trial in range(40):
        left, right = _random_tables(rng, datetime_domain)
        tolerance = None
        if rng.random() < 0.5:
            if datetime_domain:
                tolerance = timedelta(seconds=rng.randint(0, 200))
            else:
                tolerance = rng.randint(0, 15)
        exact = rng.random() < 0.8
        expect = _naive_asof(left, right, 1, 1, [0], [0], direction,
                             tolerance, exact, None)
        # unsorted path, sometimes forcing external merge
        bs = rng.choice((None, 1, 2, 5))
        got = list(asofjoin(left, right, key='t', by='g',
                            direction=direction, tolerance=tolerance,
                            allow_exact_matches=exact, buffersize=bs))
        eq_(expect, got)
        # presorted path on pre-sorted copies
        ls = list(etl.sort(left, key=('g', 't')))
        rs = list(etl.sort(right, key=('g', 't')))
        got_pre = list(asofjoin(ls, rs, key='t', by='g',
                                direction=direction, tolerance=tolerance,
                                allow_exact_matches=exact, presorted=True))
        eq_(expect, got_pre)


def test_asofjoin_random_no_group_vs_naive():
    rng = random.Random(424242)
    for trial in range(30):
        left = [('t', 'e')]
        nleft = rng.randint(1, 30)
        for i in range(nleft):
            left.append((rng.randint(0, 25), 'l%d' % i))
        right = [('t', 'v')]
        for i in range(rng.randint(0, 30)):
            right.append((rng.randint(0, 25), 'r%d' % i))
        tol = rng.randint(0, 8)
        expect = _naive_asof(left, right, 0, 0, [], [], 'nearest',
                             tol, True, None)
        got = list(asofjoin(left, right, key='t', direction='nearest',
                            tolerance=tol))
        eq_(expect, got)
        # presorted variant
        ls = list(etl.sort(left, key='t'))
        rs = list(etl.sort(right, key='t'))
        got_pre = list(asofjoin(ls, rs, key='t', direction='nearest',
                                tolerance=tol, presorted=True))
        eq_(expect, got_pre)


def test_asofjoin_external_sort_matches_presorted_stable_order():
    rng = random.Random(99)
    left, right = _random_tables(rng)
    # duplicate event times within groups so stability is exercised
    for row in left[1:]:
        pass
    pre_l = list(etl.sort(left, key=('g', 't')))
    pre_r = list(etl.sort(right, key=('g', 't')))
    for direction in ('backward', 'forward', 'nearest'):
        expect = list(asofjoin(pre_l, pre_r, key='t', by='g',
                               direction=direction, presorted=True))
        got = list(asofjoin(left, right, key='t', by='g',
                            direction=direction, buffersize=2))
        eq_(expect, got)


def test_asofjoin_naive_tie_break_sanity():
    # direct check of the reference helper itself
    left = [('g', 't', 'e'), ('A', 4, 'a')]
    right = [('g', 't', 'v'),
             ('A', 3, 'x'), ('A', 3, 'y'), ('A', 5, 'p'), ('A', 5, 'q')]
    nb = _naive_asof(left, right, 1, 1, [0], [0], 'backward', None,
                     True, None)
    eq_(3, nb[1][3])
    eq_('y', nb[1][4])
    nf = _naive_asof(left, right, 1, 1, [0], [0], 'forward', None,
                     True, None)
    eq_(5, nf[1][3])
    eq_('p', nf[1][4])
