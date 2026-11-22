"""Tests for sorting tables with NaN key values.

Sort contract: ascending order is MISSING < None < NaN < everything
else; all NaN values (float NaN, Decimal quiet NaN, any sign or payload)
are mutually co-equal and co-equal keys keep their input order (stable);
reverse inverts the order and stays stable. Co-equal for sorting does
not imply matchable - see the join tests.
"""

from __future__ import print_function, division, absolute_import

import math
from decimal import Decimal, localcontext, InvalidOperation

import petl as etl
from petl import MISSING
from petl.comparison import _isnan
from petl.test.helpers import eq_, ieq_nan, canonical_rows


def _table():
    # (key, tag) rows covering MISSING, None, NaN of several categories,
    # infinities, finite numbers and strings, in scrambled input order
    n1 = float('nan')
    n2 = Decimal('NaN7')
    n3 = -float('nan')
    return [('k', 'tag'),
            ('x', 'str'),
            (2.5, 'f2.5'),
            (n1, 'nan1'),
            (None, 'none'),
            (MISSING, 'missing'),
            (float('-inf'), 'ninf'),
            (n2, 'nan2'),
            (0, 'zero'),
            (float('inf'), 'inf'),
            (n3, 'nan3'),
            (1, 'one')]


def _expected_asc():
    # hand-written expected order: MISSING, None, NaN bucket (input
    # order), then numbers, then text
    tbl = {row[1]: row[0] for row in _table()[1:]}
    order = ['missing', 'none', 'nan1', 'nan2', 'nan3',
             'ninf', 'zero', 'one', 'f2.5', 'inf', 'str']
    return [('k', 'tag')] + [(tbl[tag], tag) for tag in order]


def _sort_variants(table, key='k'):
    """Run sort over the memory, spill and cache paths; every variant
    must give byte-identical results."""
    variants = {}
    variants['mem_nocache'] = list(etl.sort(table, key, buffersize=None,
                                            cache=False))
    variants['spill_nocache'] = list(etl.sort(table, key, buffersize=2,
                                              cache=False))
    mem_cached = etl.sort(table, key, buffersize=None, cache=True)
    variants['mem_cache_pass1'] = list(mem_cached)
    variants['mem_cache_pass2'] = list(mem_cached)
    spill_cached = etl.sort(table, key, buffersize=2, cache=True)
    variants['spill_cache_pass1'] = list(spill_cached)
    variants['spill_cache_pass2'] = list(spill_cached)
    return variants


def test_sort_nan_order_all_paths():
    expect = _expected_asc()
    for name, actual in _sort_variants(_table()).items():
        ieq_nan(expect, actual)


def test_sort_nan_reverse():
    # reverse inverts the group order but co-equal keys still keep their
    # original input order (Python stable reverse semantics), so the NaN
    # bucket is NOT itself reversed
    tbl = {row[1]: row[0] for row in _table()[1:]}
    order = ['str', 'inf', 'f2.5', 'one', 'zero', 'ninf',
             'nan1', 'nan2', 'nan3', 'none', 'missing']
    expect = [('k', 'tag')] + [(tbl[tag], tag) for tag in order]
    for reverse_variant in (dict(buffersize=None, cache=False),
                            dict(buffersize=2, cache=False),
                            dict(buffersize=2, cache=True)):
        actual = list(etl.sort(_table(), 'k', reverse=True,
                               **reverse_variant))
        ieq_nan(expect, actual)


def test_sort_nan_stability():
    # many co-equal NaN keys and duplicate finite keys: input order must
    # be preserved within each co-equal group, also through the spill
    # path and in reverse
    nans = [float('nan') for _ in range(4)]
    table = ([('k', 'seq')]
             + [(nans[0], 0), (1, 1), (nans[1], 2), (2, 3), (nans[2], 4),
                (1, 5), (nans[3], 6), (2, 7), (None, 8)])
    expect = [('k', 'seq'),
              (None, 8),
              (nans[0], 0), (nans[1], 2), (nans[2], 4), (nans[3], 6),
              (1, 1), (1, 5),
              (2, 3), (2, 7)]
    # reverse: co-equal groups in reverse order, rows within each group
    # still in original input order
    expect_rev = [('k', 'seq'),
                  (2, 3), (2, 7),
                  (1, 1), (1, 5),
                  (nans[0], 0), (nans[1], 2), (nans[2], 4), (nans[3], 6),
                  (None, 8)]
    for kwargs in (dict(buffersize=None, cache=False),
                   dict(buffersize=2, cache=False),
                   dict(buffersize=3, cache=True),
                   dict(buffersize=1, cache=False)):
        ieq_nan(expect, list(etl.sort(table, 'k', **kwargs)))
        ieq_nan(expect_rev, list(etl.sort(table, 'k', reverse=True,
                                          **kwargs)))


def test_sort_nan_compound_key():
    n1 = float('nan')
    n2 = Decimal('NaN')
    n3 = float('nan')
    n4 = Decimal('-NaN3')
    n5 = float('nan')
    table = [('a', 'b', 'tag'),
             (1, n1, 'p'),
             (n2, 1, 'q'),
             (1, 2, 'r'),
             (n3, n4, 's'),
             (0, n5, 't')]
    # hand-written expectation: NaN < everything at each component
    expect = [('a', 'b', 'tag'),
              (n3, n4, 's'),   # (nan, nan) < (nan, 1): second component
              (n2, 1, 'q'),    # nan < 0 on first component
              (0, n5, 't'),
              (1, n1, 'p'),    # (1, nan) < (1, 2)
              (1, 2, 'r')]
    for kwargs in (dict(buffersize=None, cache=False),
                   dict(buffersize=2, cache=False)):
        ieq_nan(expect, list(etl.sort(table, key=('a', 'b'), **kwargs)))
    # compound stability: co-equal compound keys keep input order
    table2 = [('a', 'b', 'tag'),
              (1, n1, 'first'),
              (1, n2, 'second'),
              (1, n3, 'third')]
    ieq_nan(table2, list(etl.sort(table2, key=('a', 'b'), buffersize=2,
                                  cache=False)))


def test_sort_nan_empty_and_allnan():
    # header-only table (no data rows)
    ieq_nan([('k', 'v')], list(etl.sort([('k', 'v')], 'k')))
    ieq_nan([('k', 'v')], list(etl.sort([('k', 'v')], 'k', buffersize=2)))
    # all-NaN table: stable, nothing reordered
    n1, n2, n3 = float('nan'), Decimal('NaN'), -float('nan')
    table = [('k', 'v'), (n1, 'a'), (n2, 'b'), (n3, 'c')]
    for kwargs in (dict(buffersize=None, cache=False),
                   dict(buffersize=1, cache=False)):
        ieq_nan(table, list(etl.sort(table, 'k', **kwargs)))


def test_sort_nan_output_values_unchanged():
    # output cells keep their original values and types; NaN verified by
    # category, sign and Decimal payload - also through the pickle spill
    n1 = -float('nan')
    n2 = Decimal('-NaN123')
    n3 = Decimal('NaN9')
    table = [('k', 'v'), (1, 'x'), (n1, 'a'), (n2, 'b'), (n3, 'c')]
    for kwargs in (dict(buffersize=None, cache=False),
                   dict(buffersize=1, cache=False)):
        actual = list(etl.sort(table, 'k', **kwargs))
        # NaN bucket first, in input order
        a1, a2, a3 = actual[1][0], actual[2][0], actual[3][0]
        assert type(a1) is float and math.copysign(1.0, a1) < 0
        assert isinstance(a2, Decimal) and a2.is_qnan()
        eq_(True, a2.is_signed())
        eq_((1, 2, 3), a2.as_tuple().digits)
        assert isinstance(a3, Decimal) and a3.is_qnan()
        eq_(False, a3.is_signed())
        eq_((9,), a3.as_tuple().digits)
        eq_((1, 'x'), actual[4])
        # no internal marker objects leak into the output
        for row in actual:
            for cell in row:
                assert type(cell).__name__ not in ('Comparable', '_Keyed')


def test_sort_nan_nonkey_field():
    # NaN in a non-key field does not affect ordering and passes through
    n1 = float('nan')
    n2 = Decimal('NaN4')
    table = [('k', 'v'), (2, n1), (1, n2), (3, 'x')]
    expect = [('k', 'v'), (1, n2), (2, n1), (3, 'x')]
    for kwargs in (dict(buffersize=None, cache=False),
                   dict(buffersize=1, cache=False)):
        actual = list(etl.sort(table, 'k', **kwargs))
        ieq_nan(expect, actual)
        if kwargs['buffersize'] is None:
            # memory path preserves object identity
            assert actual[1][1] is n2
            assert actual[2][1] is n1


def test_sort_nan_decimal_context():
    # sorting Decimal NaN keys works with InvalidOperation trapped or
    # not, gives identical results, and leaves context and flags alone
    table = [('k', 'v'),
             (Decimal('2'), 'two'),
             (Decimal('NaN'), 'nan'),
             (Decimal('-NaN7'), 'negnan'),
             (Decimal('1'), 'one')]
    expect = None
    for trap in (True, False):
        with localcontext() as ctx:
            ctx.traps[InvalidOperation] = trap
            ctx.clear_flags()
            actual = list(etl.sort(table, 'k', buffersize=2, cache=False))
            flags = [k for k, v in ctx.flags.items() if v]
            eq_([], flags)
            eq_(trap, ctx.traps[InvalidOperation])
            if expect is None:
                expect = actual
            else:
                ieq_nan(expect, actual)
    # hand-written order: NaN bucket (input order) then 1, 2
    eq_(['nan', 'negnan', 'one', 'two'], [r[1] for r in expect[1:]])


def test_mergesort_nan():
    n1, n2 = float('nan'), Decimal('NaN')
    t1 = [('k', 'v'), (3, 'a'), (n1, 'b'), (1, 'c')]
    t2 = [('k', 'v'), (2, 'd'), (n2, 'e'), (1, 'f')]
    expect = [('k', 'v'),
              (n1, 'b'), (n2, 'e'),
              (1, 'c'), (1, 'f'),
              (2, 'd'),
              (3, 'a')]
    # mergesort sorts its inputs first
    ieq_nan(expect, list(etl.mergesort(t1, t2, key='k', buffersize=2)))
    # presorted=True: inputs prepared per the documented order
    pre1 = [('k', 'v'), (n1, 'b'), (1, 'c'), (3, 'a')]
    pre2 = [('k', 'v'), (n2, 'e'), (1, 'f'), (2, 'd')]
    ieq_nan(expect, list(etl.mergesort(pre1, pre2, key='k',
                                       presorted=True)))
    # reverse: groups in reverse order, co-equal rows keep input order
    expect_r = [('k', 'v'),
                (3, 'a'),
                (2, 'd'),
                (1, 'c'), (1, 'f'),
                (n1, 'b'), (n2, 'e')]
    ieq_nan(expect_r, list(etl.mergesort(t1, t2, key='k', reverse=True,
                                         buffersize=2)))


def test_issorted_nan():
    n = float('nan')
    table = [('k', 'v'), (None, 1), (n, 2), (1, 3), (2, 4)]
    eq_(True, etl.issorted(table, key='k'))
    eq_(False, etl.issorted(table, key='k', reverse=True))
    eq_(True, etl.issorted(list(reversed(table[:1] + table[1:][::-1])) and
                           [table[0]] + list(reversed(table[1:])),
                           key='k', reverse=True))
    # two co-equal NaN keys are sorted (non-strict) but not strict
    table2 = [('k', 'v'), (n, 1), (float('nan'), 2)]
    eq_(True, etl.issorted(table2, key='k'))
    eq_(False, etl.issorted(table2, key='k', strict=True))
