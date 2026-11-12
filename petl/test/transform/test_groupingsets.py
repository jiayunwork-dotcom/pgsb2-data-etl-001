from __future__ import absolute_import, print_function, division


import itertools
import json
import os
import random
import struct
import subprocess
import sys
from fractions import Fraction

import pytest

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError
from petl.transform.groupingsets import groupstate, groupingsets, \
    loadgroupstate, GroupState


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def normval(v):
    # -0.0 is treated as 0.0
    if isinstance(v, float) and v == 0.0:
        return 0.0
    return v


def assert_tables_equal(expect, actual):
    """Compare two tables row by row, requiring exact types and
    bit-identical floats."""
    er = [tuple(r) for r in expect]
    ar = [tuple(r) for r in actual]
    assert len(er) == len(ar), \
        'row count differs: %d != %d\nexpect: %r\nactual: %r' \
        % (len(er), len(ar), er, ar)
    for i, (erow, arow) in enumerate(zip(er, ar)):
        assert len(erow) == len(arow), 'row %d width differs' % i
        for j, (ev, av) in enumerate(zip(erow, arow)):
            if ev is None or av is None:
                assert ev is None and av is None, \
                    'row %d col %d: %r != %r' % (i, j, ev, av)
            elif isinstance(ev, float) or isinstance(av, float):
                assert isinstance(ev, float) and isinstance(av, float), \
                    'row %d col %d: %r (%s) != %r (%s)' \
                    % (i, j, ev, type(ev).__name__, av, type(av).__name__)
                # repr of a float round-trips, so equal repr means
                # bit-identical (and distinguishes -0.0 from 0.0)
                assert repr(ev) == repr(av), \
                    'row %d col %d: %r != %r' % (i, j, ev, av)
            else:
                assert type(ev) is type(av) and ev == av, \
                    'row %d col %d: %r (%s) != %r (%s)' \
                    % (i, j, ev, type(ev).__name__, av, type(av).__name__)


# ---------------------------------------------------------------------
# independent exact reference implementation
# ---------------------------------------------------------------------

def ref_sort_key(gkey):
    # petl sort comparison semantics: None smallest, then numbers,
    # then strings
    def key(v):
        if v is None:
            return (0, 0)
        if isinstance(v, (int, float)):
            return (1, v)
        return (2, v)
    return tuple(key(v) for v in gkey)


def ref_measure(recs, mtype, field):
    if mtype == 'count':
        if field is None:
            return len(recs)
        return sum(1 for r in recs if r[field] is not None)
    vals = [normval(r[field]) for r in recs if r[field] is not None]
    if mtype == 'sum':
        if not vals:
            return None
        total = sum((Fraction(v) for v in vals), Fraction(0))
        if all(isinstance(v, int) for v in vals):
            return int(total)
        return float(total)
    if mtype == 'mean':
        if not vals:
            return None
        return float(sum((Fraction(v) for v in vals), Fraction(0))
                     / len(vals))
    if mtype == 'min' or mtype == 'max':
        if not vals:
            return None
        m = min(vals) if mtype == 'min' else max(vals)
        if isinstance(m, float):
            # an int/float tie resolves to the int
            for v in vals:
                if isinstance(v, int) and v == m:
                    return v
        return m
    if mtype == 'countdistinct':
        return len(set(vals))
    raise AssertionError(mtype)


def reference(table, key, sets, measures):
    """Compute the expected output rows (without header) directly from
    the detail rows."""
    header = list(table[0])
    rows = [list(r) for r in table[1:]]
    dims = [key] if isinstance(key, str) else list(key)
    n = len(dims)
    out = []
    for gset in sets:
        members = frozenset([gset] if isinstance(gset, str) else gset)
        positions = [i for i, d in enumerate(dims) if d in members]
        grouping = 0
        for i in range(n):
            if dims[i] not in members:
                grouping |= 2 ** (n - 1 - i)
        groups = {}
        for row in rows:
            rec = dict(zip(header, row))
            gkey = tuple(rec[dims[p]] for p in positions)
            groups.setdefault(gkey, []).append(rec)
        if not members and () not in groups:
            groups[()] = []
        for gkey in sorted(groups, key=ref_sort_key):
            recs = groups[gkey]
            row = [None] * n
            for p, v in zip(positions, gkey):
                row[p] = v
            row.append(grouping)
            for name, mtype, field in measures:
                row.append(ref_measure(recs, mtype, field))
            out.append(tuple(row))
    return out


def expect_table(key, sets, measures, table):
    dims = [key] if isinstance(key, str) else list(key)
    header = tuple(dims) + ('grouping',) + tuple(m[0] for m in measures)
    return [header] + reference(table, key, sets, measures)


# ---------------------------------------------------------------------
# random data generation
# ---------------------------------------------------------------------

_DIM_POOL = [None, 'a', 'b', 'c', 1, 2, 3]


def _rand_numeric(rng):
    choice = rng.random()
    if choice < 0.40:
        return rng.randint(-20, 20)
    if choice < 0.50:
        return rng.choice([10 ** 16, -10 ** 16, 10 ** 30, -(10 ** 30), 0])
    if choice < 0.70:
        return float(rng.randint(-20, 20))
    if choice < 0.85:
        return rng.choice([1e16, -1e16, 1.0, -1.0, -0.0, 0.0, 3.0, 0.1,
                           0.2])
    return rng.uniform(-100, 100)


def _rand_distinct(rng):
    choice = rng.random()
    if choice < 0.4:
        return rng.choice(['x', 'y', 'z'])
    if choice < 0.7:
        return rng.randint(0, 5)
    return float(rng.randint(0, 5))


def _rand_any(rng):
    return rng.choice([None, 1, 'x', 2.5, (1, 2)])


def rand_schema(rng):
    ndims = rng.randint(1, 3)
    dims = ['d%d' % i for i in range(ndims)]
    allsets = []
    for r in range(ndims + 1):
        allsets.extend(itertools.combinations(dims, r))
    rng.shuffle(allsets)
    nsets = rng.randint(1, min(4, len(allsets)))
    sets = [list(s) for s in allsets[:nsets]]
    if rng.random() < 0.5 and [] not in sets:
        sets[0] = []
    fields = ['m%d' % i for i in range(3)]
    measures = []
    for i in range(rng.randint(1, 4)):
        name = 'out%d' % i
        mtype = rng.choice(['count', 'sum', 'mean', 'min', 'max',
                            'countdistinct'])
        if mtype == 'count':
            field = rng.choice([None] + fields)
        else:
            field = rng.choice(fields)
        measures.append((name, mtype, field))
    return dims, sets, measures, fields


def rand_table(rng, dims, measures, fields, nrows):
    numeric = set()
    distinct = set()
    for (name, mtype, field) in measures:
        if field is None:
            continue
        if mtype in ('sum', 'mean', 'min', 'max'):
            numeric.add(field)
        elif mtype == 'countdistinct':
            distinct.add(field)
    header = list(dims) + list(fields)
    rows = [header]
    for _ in range(nrows):
        row = [rng.choice(_DIM_POOL) for _ in dims]
        for f in fields:
            if rng.random() < 0.15:
                row.append(None)
            elif f in numeric:
                row.append(_rand_numeric(rng))
            elif f in distinct:
                row.append(_rand_distinct(rng))
            else:
                row.append(_rand_any(rng))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# golden tests
# ---------------------------------------------------------------------

def test_groupingsets_golden():
    table = [('region', 'product', 'amount'),
             ('east', 'apple', 3),
             ('east', 'apple', 7),
             ('east', 'banana', 2),
             ('west', 'apple', None),
             ('west', 'banana', 4.5)]
    actual = groupingsets(table, ('region', 'product'),
                          [('region', 'product'), ('region',), ()],
                          [('total', 'sum', 'amount'),
                           ('n', 'count', None)])
    expect = [
        ('region', 'product', 'grouping', 'total', 'n'),
        ('east', 'apple', 0, 10, 2),
        ('east', 'banana', 0, 2, 1),
        ('west', 'apple', 0, None, 1),
        ('west', 'banana', 0, 4.5, 1),
        ('east', None, 1, 12, 3),
        ('west', None, 1, 4.5, 2),
        (None, None, 3, 16.5, 5),
    ]
    assert_tables_equal(expect, actual)
    # the view can be iterated repeatedly with identical results
    assert_tables_equal(expect, actual)


def test_groupstate_totable_matches_groupingsets():
    table = [('a', 'v'), ('x', 1), ('y', 2.5), ('x', None)]
    measures = [('t', 'sum', 'v'), ('n', 'count', None)]
    state = groupstate(table, 'a', [('a',), ()], measures, batch='b1')
    assert isinstance(state, GroupState)
    direct = groupingsets(table, 'a', [('a',), ()], measures)
    assert_tables_equal(list(state.totable()), list(direct))


def test_grouping_bits():
    table = [('a', 'b', 'c', 'v'), ('x', 'y', 'z', 1)]
    actual = groupingsets(table, ('a', 'b', 'c'),
                          [('a', 'b', 'c'), ('a', 'c'), ('b',), ()],
                          [('n', 'count', None)])
    expect = [
        ('a', 'b', 'c', 'grouping', 'n'),
        ('x', 'y', 'z', 0, 1),        # nothing aggregated away
        ('x', None, 'z', 2, 1),       # b aggregated away: 2 ** (3 - 1 - 1)
        (None, 'y', None, 5, 1),      # a and c away: 4 + 1
        (None, None, None, 7, 1),     # all away
    ]
    assert_tables_equal(expect, actual)


def test_none_dimension_distinguished_by_grouping():
    table = [('a', 'b', 'v'),
             (None, 'x', 1),
             ('p', None, 2),
             ('p', 'x', 3)]
    actual = groupingsets(table, ('a', 'b'), [('a', 'b'), ('a',)],
                          [('n', 'count', None)])
    expect = [
        ('a', 'b', 'grouping', 'n'),
        (None, 'x', 0, 1),    # genuine None value of dimension a
        ('p', None, 0, 1),    # genuine None value of dimension b
        ('p', 'x', 0, 1),
        (None, None, 1, 1),   # group of the None value of a, b away
        ('p', None, 1, 2),    # group of 'p', b aggregated away
    ]
    assert_tables_equal(expect, actual)


def test_sort_order_none_numbers_strings():
    table = [('a', 'v'), ('b', 1), (None, 2), (10, 3), ('a', 4),
             (2, 5), (None, 6)]
    actual = groupingsets(table, 'a', [('a',)], [('n', 'count', None)])
    rows = list(actual)[1:]
    assert [r[0] for r in rows] == [None, 2, 10, 'a', 'b']
    assert [r[2] for r in rows] == [2, 1, 1, 1, 1]


def test_empty_input():
    table = [('a', 'v')]
    measures = [('n', 'count', None), ('s', 'sum', 'v'),
                ('avg', 'mean', 'v'), ('lo', 'min', 'v'),
                ('hi', 'max', 'v'), ('d', 'countdistinct', 'v')]
    actual = groupingsets(table, 'a', [('a',), ()], measures)
    expect = [
        ('a', 'grouping', 'n', 's', 'avg', 'lo', 'hi', 'd'),
        (None, 1, 0, None, None, None, None, 0),
    ]
    assert_tables_equal(expect, actual)
    # non-empty sets yield no rows when there is no data
    actual = groupingsets(table, 'a', [('a',)], [('n', 'count', None)])
    assert_tables_equal([('a', 'grouping', 'n')], actual)


def test_count_semantics():
    table = [('a', 'v'), ('x', 1), ('x', None), ('y', None)]
    actual = groupingsets(table, 'a', [('a',), ()],
                          [('n', 'count', None), ('nv', 'count', 'v')])
    expect = [
        ('a', 'grouping', 'n', 'nv'),
        ('x', 0, 2, 1),
        ('y', 0, 1, 0),
        (None, 1, 3, 1),
    ]
    assert_tables_equal(expect, actual)
    # count of a field counts non-None values of any type
    table = [('a', 'v'), ('x', 's'), ('x', (1, 2)), ('x', None)]
    actual = groupingsets(table, 'a', [('a',)], [('n', 'count', 'v')])
    assert_tables_equal([('a', 'grouping', 'n'), ('x', 0, 2)], actual)


def test_sum_semantics():
    # all ints: exact int
    table = [('a', 'v'), ('x', 10 ** 30), ('x', 10 ** 30)]
    actual = groupingsets(table, 'a', [('a',)], [('t', 'sum', 'v')])
    assert_tables_equal([('a', 'grouping', 't'), ('x', 0, 2 * 10 ** 30)],
                        actual)
    # any float: float, exact sum rounded once (the exact sum of the
    # doubles 0.1 and 0.2 is 0.300000000000000016653..., which rounds to
    # 0.30000000000000004)
    table = [('a', 'v'), ('y', 0.1), ('y', 0.2), ('z', 1), ('z', 0.5)]
    actual = groupingsets(table, 'a', [('a',)], [('t', 'sum', 'v')])
    rows = list(actual)
    assert rows[1] == ('y', 0, 0.30000000000000004)
    assert repr(rows[1][2]) == '0.30000000000000004'
    assert rows[2] == ('z', 0, 1.5)
    assert type(rows[2][2]) is float
    # no values: None
    table = [('a', 'v'), ('x', None)]
    actual = groupingsets(table, 'a', [('a',)], [('t', 'sum', 'v')])
    assert_tables_equal([('a', 'grouping', 't'), ('x', 0, None)], actual)


def test_sum_order_sensitive_floats():
    # 1e16 + 1.0 - 1e16 is 0.0 with naive float addition, exactly 1.0
    # with a single rounding of the exact sum
    for perm in itertools.permutations([1e16, 1.0, -1e16]):
        table = [('a', 'v')] + [('x', v) for v in perm]
        actual = groupingsets(table, 'a', [()], [('t', 'sum', 'v')])
        assert list(actual)[1] == (None, 1, 1.0)
        # also when split into one batch per value and merged
        states = [groupstate([('a', 'v'), ('x', v)], 'a', [()],
                             [('t', 'sum', 'v')], batch='b%d' % i)
                  for i, v in enumerate(perm)]
        merged = states[0].merge(states[1]).merge(states[2])
        assert list(merged.totable())[1] == (None, 1, 1.0)


def test_mean_semantics():
    table = [('a', 'v'), ('x', 1), ('x', 2), ('y', 2), ('y', 4),
             ('z', None)]
    actual = groupingsets(table, 'a', [('a',)], [('m', 'mean', 'v')])
    rows = list(actual)
    assert rows[1] == ('x', 0, 1.5)
    assert type(rows[1][2]) is float
    # mean of ints is a float even when the result is integral
    assert rows[2] == ('y', 0, 3.0)
    assert type(rows[2][2]) is float
    assert rows[3] == ('z', 0, None)
    # mean of [1e16, 1.0, -1e16] is exactly 1/3 rounded once
    table = [('a', 'v'), ('x', 1e16), ('x', 1.0), ('x', -1e16)]
    actual = groupingsets(table, 'a', [('a',)], [('m', 'mean', 'v')])
    assert list(actual)[1] == ('x', 0, 1.0 / 3)


def test_sum_mean_overflow():
    table = [('a', 'v'), ('x', 1e308), ('x', 1e308)]
    with pytest.raises(OverflowError):
        list(groupingsets(table, 'a', [('a',)], [('t', 'sum', 'v')]))
    # mean of the same values does not overflow
    actual = groupingsets(table, 'a', [('a',)], [('t', 'mean', 'v')])
    assert list(actual)[1] == ('x', 0, 1e308)
    # mean of a huge int overflows
    table = [('a', 'v'), ('x', 10 ** 400)]
    with pytest.raises(OverflowError):
        list(groupingsets(table, 'a', [('a',)], [('t', 'mean', 'v')]))
    # but its sum is the exact int
    actual = groupingsets(table, 'a', [('a',)], [('t', 'sum', 'v')])
    assert list(actual)[1] == ('x', 0, 10 ** 400)


def test_min_max_semantics():
    table = [('a', 'v'), ('x', 1.5), ('x', 2), ('x', -1), ('y', None)]
    actual = groupingsets(table, 'a', [('a',)],
                          [('lo', 'min', 'v'), ('hi', 'max', 'v')])
    rows = list(actual)
    assert rows[1] == ('x', 0, -1, 2)
    assert type(rows[1][2]) is int and type(rows[1][3]) is int
    assert rows[2] == ('y', 0, None, None)


def test_min_max_int_float_ties():
    # 3 and 3.0 tie: the int wins, in any order, also across batches
    for vals in [(3, 3.0), (3.0, 3)]:
        table = [('a', 'v')] + [('x', v) for v in vals]
        actual = groupingsets(table, 'a', [('a',)],
                              [('lo', 'min', 'v'), ('hi', 'max', 'v')])
        row = list(actual)[1]
        assert row == ('x', 0, 3, 3)
        assert type(row[2]) is int and type(row[3]) is int
        states = [groupstate([('a', 'v'), ('x', v)], 'a', [('a',)],
                             [('lo', 'min', 'v'), ('hi', 'max', 'v')],
                             batch='b%d' % i)
                  for i, v in enumerate(vals)]
        for merged in (states[0].merge(states[1]),
                       states[1].merge(states[0])):
            row = list(merged.totable())[1]
            assert row == ('x', 0, 3, 3)
            assert type(row[2]) is int and type(row[3]) is int
    # a float alone stays a float
    table = [('a', 'v'), ('x', 3.0)]
    actual = groupingsets(table, 'a', [('a',)],
                          [('lo', 'min', 'v'), ('hi', 'max', 'v')])
    row = list(actual)[1]
    assert row == ('x', 0, 3.0, 3.0)
    assert type(row[2]) is float and type(row[3]) is float


def test_countdistinct_semantics():
    # 1 and 1.0 are the same value, '1' and 1 are not, None is ignored
    table = [('a', 'v'), ('x', 1), ('x', 1.0), ('x', '1'), ('x', None),
             ('x', 2), ('y', '1')]
    actual = groupingsets(table, 'a', [('a',)],
                          [('d', 'countdistinct', 'v')])
    assert_tables_equal([('a', 'grouping', 'd'),
                         ('x', 0, 3),
                         ('y', 0, 1)], actual)


def test_negative_zero_treated_as_zero():
    table = [('a', 'v'), ('x', -0.0), ('x', 0.0)]
    actual = groupingsets(table, 'a', [('a',)],
                          [('t', 'sum', 'v'), ('lo', 'min', 'v'),
                           ('hi', 'max', 'v'), ('d', 'countdistinct', 'v')])
    row = list(actual)[1]
    assert repr(row[2]) == '0.0'
    assert repr(row[3]) == '0.0'
    assert repr(row[4]) == '0.0'
    assert row[5] == 1


# ---------------------------------------------------------------------
# argument and data validation
# ---------------------------------------------------------------------

def _args(**kw):
    args = {'table': [('a', 'v'), ('x', 1)], 'key': 'a', 'sets': [('a',)],
            'measures': [('n', 'count', None)], 'batch': 'b1'}
    args.update(kw)
    return args


def test_key_argument_errors():
    with pytest.raises(ArgumentError):
        groupstate(**_args(key=[]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(key=('a', 'a')))
    with pytest.raises(ArgumentError):
        groupstate(**_args(key=5))
    with pytest.raises(ArgumentError):
        groupstate(**_args(key=('a', 5)))
    with pytest.raises(ArgumentError):
        groupstate(**_args(key=None))


def test_sets_argument_errors():
    with pytest.raises(ArgumentError):
        groupstate(**_args(sets=[]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(sets=None))
    with pytest.raises(ArgumentError):
        groupstate(**_args(sets='a'))
    with pytest.raises(ArgumentError):
        groupstate(**_args(sets=[('a', 'a')]))  # duplicate dim in set
    with pytest.raises(ArgumentError):
        groupstate(**_args(sets=[('a',), ('a',)]))  # duplicate set
    with pytest.raises(ArgumentError):
        # duplicate by membership, order irrelevant
        groupstate(**_args(key=('a', 'b'), sets=[('a', 'b'), ('b', 'a')],
                           table=[('a', 'b', 'v'), ('x', 'y', 1)]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(sets=[('zzz',)]))  # not a dimension


def test_measures_argument_errors():
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('n', 'count', None),
                                     ('n', 'sum', 'v')]))  # dup name
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('a', 'count', None)]))  # dim name
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('grouping', 'count', None)]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('n', 'bogus', 'v')]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('n', 'sum', None)]))  # needs field
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('n', 'count', 5)]))
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[('n', 'count')]))  # not a triple
    with pytest.raises(ArgumentError):
        groupstate(**_args(measures=[(5, 'count', None)]))


def test_batch_argument_errors():
    with pytest.raises(ArgumentError):
        groupstate(**_args(batch=''))
    with pytest.raises(ArgumentError):
        groupstate(**_args(batch=None))
    with pytest.raises(ArgumentError):
        groupstate(**_args(batch=5))


def test_field_selection_errors():
    with pytest.raises(FieldSelectionError):
        groupstate(**_args(key='zzz', sets=[('zzz',)]))
    with pytest.raises(FieldSelectionError):
        groupstate(**_args(key=('a', 'zzz')))
    with pytest.raises(FieldSelectionError):
        groupstate(**_args(measures=[('n', 'sum', 'zzz')]))
    with pytest.raises(FieldSelectionError):
        groupstate(**_args(measures=[('n', 'count', 'zzz')]))


def test_header_errors():
    # duplicate header fields
    with pytest.raises(ArgumentError):
        groupstate(**_args(table=[('a', 'a'), ('x', 1)]))
    # non-string header field
    with pytest.raises(ArgumentError):
        groupstate(**_args(table=[('a', 1), ('x', 1)]))
    # no header at all
    with pytest.raises(ArgumentError):
        groupstate(**_args(table=[]))


def test_row_width_errors():
    with pytest.raises(ArgumentError) as ei:
        groupstate(**_args(table=[('a', 'v'), ('x', 1), ('y', 2, 3)]))
    assert 'row 2' in str(ei.value)
    with pytest.raises(ArgumentError) as ei:
        groupstate(**_args(table=[('a', 'v'), ('x',)]))
    assert 'row 1' in str(ei.value)


def test_dimension_value_errors():
    for bad in (1.5, True, False, b'x', (1, 2), [1]):
        table = [('a', 'v'), ('x', 1), (bad, 2)]
        with pytest.raises(ArgumentError) as ei:
            groupstate(table, 'a', [('a',)], [('n', 'count', None)], 'b')
        assert 'row 2' in str(ei.value), bad


def test_measure_value_errors():
    bad_values = (float('nan'), float('inf'), -float('inf'), 's', True,
                  b'x', (1, 2))
    for mtype in ('sum', 'mean', 'min', 'max'):
        for bad in bad_values:
            table = [('a', 'v'), ('x', 1), ('x', bad)]
            with pytest.raises(ArgumentError) as ei:
                groupstate(table, 'a', [('a',)], [('m', mtype, 'v')], 'b')
            assert 'row 2' in str(ei.value), (mtype, bad)
    for bad in (float('nan'), float('inf'), True, b'x', (1, 2)):
        table = [('a', 'v'), ('x', 'ok'), ('x', bad)]
        with pytest.raises(ArgumentError) as ei:
            groupstate(table, 'a', [('a',)], [('m', 'countdistinct', 'v')],
                       'b')
        assert 'row 2' in str(ei.value), bad
    # None measure values are ignored, not errors
    table = [('a', 'v'), ('x', None), ('x', 1)]
    state = groupstate(table, 'a', [('a',)], [('m', 'sum', 'v')], 'b')
    assert list(state.totable())[1] == ('x', 0, 1)


def test_singleton_set_and_key_shorthands():
    table = [('a', 'v'), ('x', 1), ('y', 2)]
    # a bare string as a grouping set means the singleton set
    actual = groupingsets(table, 'a', ['a', ()], [('n', 'count', None)])
    expect = [('a', 'grouping', 'n'),
              ('x', 0, 1),
              ('y', 0, 1),
              (None, 1, 2)]
    assert_tables_equal(expect, actual)


# ---------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------

def test_merge_basic():
    t1 = [('a', 'v'), ('x', 1), ('y', 2)]
    t2 = [('a', 'v'), ('x', 3), ('z', 4.5)]
    measures = [('t', 'sum', 'v'), ('n', 'count', None)]
    s1 = groupstate(t1, 'a', [('a',), ()], measures, batch='b1')
    s2 = groupstate(t2, 'a', [('a',), ()], measures, batch='b2')
    merged = s1.merge(s2)
    assert isinstance(merged, GroupState)
    assert merged.batches == frozenset({'b1', 'b2'})
    expect = expect_table('a', [('a',), ()], measures, t1 + t2[1:])
    assert_tables_equal(expect, merged.totable())
    # inputs are unchanged
    assert s1.batches == frozenset({'b1'})
    assert_tables_equal(expect_table('a', [('a',), ()], measures, t1),
                        s1.totable())
    assert_tables_equal(expect_table('a', [('a',), ()], measures, t2),
                        s2.totable())
    # the result can be merged further
    s3 = groupstate([('a', 'v'), ('w', 1)], 'a', [('a',), ()], measures,
                    batch='b3')
    merged2 = merged.merge(s3)
    assert merged2.batches == frozenset({'b1', 'b2', 'b3'})
    expect = expect_table('a', [('a',), ()], measures,
                          t1 + t2[1:] + [('w', 1)])
    assert_tables_equal(expect, merged2.totable())


def test_merge_no_arguments():
    s1 = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',)],
                    [('n', 'count', None)], batch='b1')
    s2 = s1.merge()
    assert isinstance(s2, GroupState)
    assert s2 is not s1
    assert_tables_equal(list(s1.totable()), list(s2.totable()))
    assert s2.batches == frozenset({'b1'})


def test_merge_batch_overlap():
    measures = [('n', 'count', None)]
    s1 = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',)], measures,
                    batch='b1')
    s2 = groupstate([('a', 'v'), ('y', 1)], 'a', [('a',)], measures,
                    batch='b2')
    s3 = groupstate([('a', 'v'), ('z', 1)], 'a', [('a',)], measures,
                    batch='b1')
    with pytest.raises(ArgumentError):
        s1.merge(s3)
    with pytest.raises(ArgumentError):
        s1.merge(s2, s3)
    with pytest.raises(ArgumentError):
        s1.merge(s2).merge(s3)  # b1 present in the merged state
    with pytest.raises(ArgumentError):
        s1.merge(s2).merge(s2)  # b2 present in the merged state
    # inputs are unchanged after failed merges
    assert s1.batches == frozenset({'b1'})
    assert list(s1.totable())[1] == ('x', 0, 1)
    assert list(s2.totable())[1] == ('y', 0, 1)
    assert list(s3.totable())[1] == ('z', 0, 1)


def test_merge_incompatible_specs():
    measures = [('n', 'count', None)]
    base = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',), ()], measures,
                      batch='b1')
    # different key
    other = groupstate([('b', 'v'), ('x', 1)], 'b', [('b',), ()],
                       measures, batch='b2')
    with pytest.raises(ArgumentError):
        base.merge(other)
    # same dims, different key order
    other = groupstate([('a', 'c', 'v'), ('x', 'y', 1)], ('c', 'a'),
                       [('c', 'a'), ()], measures, batch='b2')
    with pytest.raises(ArgumentError):
        groupstate([('a', 'c', 'v'), ('x', 'y', 1)], ('a', 'c'),
                   [('a', 'c'), ()], measures, batch='b1').merge(other)
    # different sets
    other = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',)], measures,
                       batch='b2')
    with pytest.raises(ArgumentError):
        base.merge(other)
    # different measures
    other = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',), ()],
                       [('n', 'count', 'v')], batch='b2')
    with pytest.raises(ArgumentError):
        base.merge(other)
    other = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',), ()],
                       [('m', 'count', None)], batch='b2')
    with pytest.raises(ArgumentError):
        base.merge(other)
    # not a GroupState
    with pytest.raises(ArgumentError):
        base.merge(42)
    with pytest.raises(ArgumentError):
        base.merge(None)


def test_merge_sets_order_insensitive():
    t1 = [('a', 'v'), ('x', 1)]
    t2 = [('a', 'v'), ('y', 2)]
    measures = [('n', 'count', None)]
    s1 = groupstate(t1, 'a', [('a',), ()], measures, batch='b1')
    s2 = groupstate(t2, 'a', [(), ('a',)], measures, batch='b2')
    merged = s1.merge(s2)
    # output follows the order of the sets of the left operand
    expect = expect_table('a', [('a',), ()], measures, t1 + t2[1:])
    assert_tables_equal(expect, merged.totable())


def test_merge_tree_shape_independent():
    table = [('a', 'v'), ('x', 1e16), ('y', 1.0), ('x', -1e16),
             ('y', 3), ('x', 3.0), ('z', 0.1), ('z', 0.2)]
    measures = [('t', 'sum', 'v'), ('m', 'mean', 'v'),
                ('lo', 'min', 'v'), ('hi', 'max', 'v'),
                ('d', 'countdistinct', 'v'), ('n', 'count', None)]
    sets = [('a',), ()]
    states = [groupstate([('a', 'v'), row], 'a', sets, measures,
                         batch='b%d' % i)
              for i, row in enumerate(table[1:])]
    # left-leaning tree
    left = states[0]
    for s in states[1:]:
        left = left.merge(s)
    # right-leaning tree
    right = states[-1]
    for s in reversed(states[:-1]):
        right = s.merge(right)
    # balanced tree

    def balanced(items):
        if len(items) == 1:
            return items[0]
        mid = len(items) // 2
        return balanced(items[:mid]).merge(balanced(items[mid:]))
    bal = balanced(states)
    # all equal to a direct computation, bit for bit
    expect = expect_table('a', sets, measures, table)
    assert_tables_equal(expect, left.totable())
    assert_tables_equal(expect, right.totable())
    assert_tables_equal(expect, bal.totable())
    # and the serialized states are byte-identical
    assert left.dumps() == right.dumps() == bal.dumps()


# ---------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------

def _sample_state(batch='b1'):
    table = [('region', 'product', 'amount', 'tag'),
             ('east', 'apple', 3, 'u'),
             ('east', 'apple', 7.5, 'v'),
             ('east', 'banana', None, 'u'),
             (None, None, -0.0, 'u'),
             ('west', 'apple', 10 ** 30, 'w')]
    measures = [('total', 'sum', 'amount'), ('avg', 'mean', 'amount'),
                ('lo', 'min', 'amount'), ('hi', 'max', 'amount'),
                ('tags', 'countdistinct', 'tag'),
                ('vals', 'countdistinct', 'amount'),
                ('n', 'count', None), ('nv', 'count', 'amount')]
    sets = [('region', 'product'), ('region',), ()]
    return groupstate(table, ('region', 'product'), sets, measures,
                      batch=batch)


def test_dumps_roundtrip():
    state = _sample_state()
    data = state.dumps()
    assert isinstance(data, bytes)
    restored = loadgroupstate(data)
    assert isinstance(restored, GroupState)
    assert_tables_equal(list(state.totable()), list(restored.totable()))
    # round trip is stable
    assert restored.dumps() == data
    # merge compatibility is preserved
    other = _sample_state(batch='b2')
    assert_tables_equal(list(state.merge(other).totable()),
                        list(restored.merge(other).totable()))
    # batches are preserved
    assert restored.batches == state.batches
    with pytest.raises(ArgumentError):
        restored.merge(_sample_state(batch='b1'))
    # bytearray is accepted too
    assert_tables_equal(list(state.totable()),
                        list(loadgroupstate(bytearray(data)).totable()))


def test_loadgroupstate_errors():
    state = _sample_state()
    data = state.dumps()
    # not bytes
    with pytest.raises(ArgumentError):
        loadgroupstate('not bytes')
    with pytest.raises(ArgumentError):
        loadgroupstate(None)
    # too short / empty
    with pytest.raises(ArgumentError):
        loadgroupstate(b'')
    with pytest.raises(ArgumentError):
        loadgroupstate(b'PET')
    # bad magic
    with pytest.raises(ArgumentError):
        loadgroupstate(b'X' + data[1:])
    # bad version
    with pytest.raises(ArgumentError):
        loadgroupstate(data[:7] + bytes([99]) + data[8:])
    # truncated
    with pytest.raises(ArgumentError):
        loadgroupstate(data[:-1])
    with pytest.raises(ArgumentError):
        loadgroupstate(data[:16])
    # trailing garbage
    with pytest.raises(ArgumentError):
        loadgroupstate(data + b'x')
    # corrupt payload (same length)
    with pytest.raises(ArgumentError):
        loadgroupstate(data[:16] + b'###' + data[19:])
    # a pickle is not a GroupState and is not executed
    import pickle
    with pytest.raises(ArgumentError):
        loadgroupstate(pickle.dumps([1, 2, 3]))


def _make_payload(obj):
    raw = json.dumps(obj).encode('utf-8')
    return b'PETLGST' + bytes([1]) + struct.pack('>Q', len(raw)) + raw


def test_loadgroupstate_malformed_payloads():
    valid = json.loads(_sample_state().dumps()[16:].decode('utf-8'))

    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload([1, 2, 3]))
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload('hello'))
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload({}))
    # missing section
    obj = dict(valid)
    del obj['groups']
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # groups not aligned with sets
    obj = json.loads(json.dumps(valid))
    obj['groups'] = []
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # invalid group entry shape
    obj = json.loads(json.dumps(valid))
    obj['groups'][0][0] = [['east'], ['not', 'a', 'valid', 'accum', 'x']]
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # duplicate group key
    obj = json.loads(json.dumps(valid))
    obj['groups'][0].append(obj['groups'][0][0])
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # duplicate batches
    obj = json.loads(json.dumps(valid))
    obj['batches'] = ['b1', 'b1']
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # empty batches
    obj = json.loads(json.dumps(valid))
    obj['batches'] = []
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # duplicate grouping sets
    obj = json.loads(json.dumps(valid))
    obj['sets'] = [['region'], ['region']]
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # grouping set member not in key
    obj = json.loads(json.dumps(valid))
    obj['sets'] = [['region', 'zzz']]
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # bool is not a valid dimension value
    obj = json.loads(json.dumps(valid))
    obj['groups'][0][0][0][0] = True
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))
    # bool is not a valid count
    obj = json.loads(json.dumps(valid))
    obj['groups'][0][0][1][-1] = True
    with pytest.raises(ArgumentError):
        loadgroupstate(_make_payload(obj))


def test_dumps_load_in_subprocess(tmp_path):
    state = _sample_state()
    p = tmp_path / 'state.bin'
    p.write_bytes(state.dumps())
    code = (
        "import sys\n"
        "import petl\n"
        "with open(sys.argv[1], 'rb') as f:\n"
        "    gs = petl.loadgroupstate(f.read())\n"
        "for row in gs.totable():\n"
        "    print(repr(tuple(row)))\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(etl.__file__)))
    env = dict(os.environ)
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    out = subprocess.check_output([sys.executable, '-c', code, str(p)],
                                  env=env)
    expect = ''.join(repr(tuple(row)) + '\n' for row in state.totable())
    assert out.decode('utf-8') == expect


# ---------------------------------------------------------------------
# single pass over the source
# ---------------------------------------------------------------------

def test_single_pass_probe():
    class Probe(object):
        def __init__(self, rows):
            self.rows = rows
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.rows)

    rows = [('a', 'v'), ('x', 1), ('y', 2), ('x', 3)]
    probe = Probe(rows)
    # several grouping sets, still one iterator requested and one pass
    state = groupstate(probe, 'a', [('a',), ()],
                       [('n', 'count', None), ('t', 'sum', 'v')],
                       batch='b1')
    assert probe.iterations == 1
    first = list(state.totable())
    second = list(state.totable())
    assert first == second
    assert probe.iterations == 1
    # a one-shot generator source works

    def gen():
        yield ('a', 'v')
        yield ('x', 1)
        yield ('x', 2)
    state = groupstate(gen(), 'a', [('a',)], [('t', 'sum', 'v')], 'b')
    assert list(state.totable())[1] == ('x', 0, 3)


# ---------------------------------------------------------------------
# exports and Table methods
# ---------------------------------------------------------------------

def test_exports_and_table_methods():
    assert etl.groupstate is groupstate
    assert etl.groupingsets is groupingsets
    assert etl.loadgroupstate is loadgroupstate
    assert etl.GroupState is GroupState
    table = etl.wrap([('a', 'v'), ('x', 1), ('y', 2)])
    state = table.groupstate('a', [('a',)], [('n', 'count', None)], 'b1')
    assert isinstance(state, GroupState)
    result = table.groupingsets('a', [('a',)], [('n', 'count', None)])
    assert isinstance(result, etl.Table)
    assert list(result) == [('a', 'grouping', 'n'), ('x', 0, 1),
                            ('y', 0, 1)]


def test_groupstate_properties():
    state = groupstate([('a', 'v'), ('x', 1)], 'a', [('a',), ()],
                       [('n', 'count', None)], batch='b1')
    assert state.key == ('a',)
    assert state.sets == (frozenset({'a'}), frozenset())
    assert state.measures == (('n', 'count', None),)
    assert state.batches == frozenset({'b1'})
    assert 'GroupState' in repr(state)


# ---------------------------------------------------------------------
# randomized cross-checks against the reference implementation
# ---------------------------------------------------------------------

def test_random_against_reference():
    rng = random.Random(20261112)
    for trial in range(100):
        dims, sets, measures, fields = rand_schema(rng)
        table = rand_table(rng, dims, measures, fields, rng.randint(0, 40))
        if len(dims) == 1 and rng.random() < 0.5:
            key = dims[0]
        else:
            key = list(dims)
        expect = expect_table(key, sets, measures, table)
        actual = groupingsets(table, key, sets, measures)
        assert_tables_equal(expect, actual)
        # iterate twice for identical results
        assert_tables_equal(expect, actual)


def test_random_batches_and_merge_trees():
    rng = random.Random(987654321)
    for trial in range(60):
        dims, sets, measures, fields = rand_schema(rng)
        table = rand_table(rng, dims, measures, fields, rng.randint(0, 60))
        key = list(dims)
        # split the rows into random batches (some may be empty)
        rows = list(table[1:])
        rng.shuffle(rows)
        nbatches = rng.randint(1, 5)
        batches = [[] for _ in range(nbatches)]
        for r in rows:
            batches[rng.randrange(nbatches)].append(r)
        states = [groupstate([table[0]] + brows, key, sets, measures,
                             batch='batch-%d' % bi)
                  for bi, brows in enumerate(batches)]
        snapshots = [s.dumps() for s in states]
        # merge along a random tree
        pool = list(states)
        while len(pool) > 1:
            i = rng.randrange(len(pool))
            a = pool.pop(i)
            j = rng.randrange(len(pool))
            b = pool.pop(j)
            pool.append(a.merge(b))
        merged = pool[0]
        assert merged.batches == frozenset('batch-%d' % i
                                           for i in range(nbatches))
        expect = expect_table(key, sets, measures, table)
        assert_tables_equal(expect, merged.totable())
        # the input states were not modified by merging
        for s, snap in zip(states, snapshots):
            assert s.dumps() == snap
        # a dump/load round trip of the merged state is identical
        assert_tables_equal(expect, loadgroupstate(merged.dumps()).totable())
