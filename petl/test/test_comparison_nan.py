"""Tests for NaN handling in petl.comparison: the sort order places NaN
(float NaN and Decimal quiet NaN) after None and before all other values
with all NaNs mutually co-equal, while the relational rule
(key_contains_nan) marks any NaN-bearing key as unmatchable. The two
notions are deliberately separate.
"""

from __future__ import print_function, division, absolute_import

from decimal import Decimal, localcontext, InvalidOperation

from petl.test.helpers import eq_
from petl.comparison import Comparable, _isnan, key_contains_nan
from petl.missing import MISSING


def test_isnan_scalar():
    # float NaN, both signs
    assert _isnan(float('nan'))
    assert _isnan(-float('nan'))
    assert _isnan(float('inf') - float('inf'))
    # Decimal quiet NaN, signs and payloads
    assert _isnan(Decimal('NaN'))
    assert _isnan(Decimal('-NaN'))
    assert _isnan(Decimal('NaN123'))
    assert _isnan(Decimal('-NaN45'))
    # everything else is not NaN, including signaling NaN (out of scope)
    for v in (None, MISSING, 0, 1, -1, 0.0, -0.0, 1.5, float('inf'),
              float('-inf'), True, False, '', 'nan', b'nan',
              Decimal('0'), Decimal('-0'), Decimal('1.5'),
              Decimal('Infinity'), Decimal('-Infinity'),
              Decimal('sNaN'), (), (1, 2), [float('nan')]):
        assert not _isnan(v), repr(v)


def test_key_contains_nan():
    n = float('nan')
    dn = Decimal('NaN5')
    # scalar keys
    assert key_contains_nan(n)
    assert key_contains_nan(dn)
    # compound keys: NaN in any position
    assert key_contains_nan((n,))
    assert key_contains_nan((1, n))
    assert key_contains_nan((n, 1))
    assert key_contains_nan((1, 'a', dn))
    assert key_contains_nan((n, n))
    # no NaN anywhere
    assert not key_contains_nan(1)
    assert not key_contains_nan(None)
    assert not key_contains_nan(MISSING)
    assert not key_contains_nan('nan')
    assert not key_contains_nan(())
    assert not key_contains_nan((1, 2.5, 'a', None, MISSING))


def test_comparable_nan_sort_position():
    # ascending order: MISSING < None < NaN < numbers < text
    data = ['a', 1, float('nan'), None, MISSING, -1.5, Decimal('2.5')]
    a = sorted(data, key=Comparable)
    eq_([MISSING, None], a[:2])
    assert _isnan(a[2])
    eq_([-1.5, 1, Decimal('2.5'), 'a'], a[3:])


def test_comparable_nan_coequal_categories():
    # all NaN categories are mutually co-equal for ordering; co-equal
    # values keep input order (stable)
    n1 = float('nan')
    n2 = Decimal('NaN')
    n3 = Decimal('-NaN7')
    n4 = -float('nan')
    data = [n1, n2, n3, n4]
    a = sorted(data, key=Comparable)
    eq_(data, a)
    # co-equal across categories, signs and payloads
    eq_(True, Comparable(n1) == Comparable(n2))
    eq_(True, Comparable(n2) == Comparable(n4))
    eq_(True, Comparable(n3) == Comparable(n1))
    eq_(False, Comparable(n1) == Comparable(1))
    eq_(False, Comparable(1) == Comparable(n1))
    eq_(False, Comparable(n1) == Comparable(None))
    # neither less nor greater: a consistent total preorder
    for x in (n1, n2, n3, n4):
        for y in (n1, n2, n3, n4):
            eq_(False, Comparable(x) < Comparable(y))
            eq_(False, Comparable(x) > Comparable(y))
            eq_(True, Comparable(x) <= Comparable(y))
            eq_(True, Comparable(x) >= Comparable(y))


def test_comparable_nan_vs_others():
    n = float('nan')
    dn = Decimal('NaN')
    # NaN sorts after None, before everything else
    assert Comparable(None) < Comparable(n)
    assert not Comparable(n) < Comparable(None)
    assert Comparable(MISSING) < Comparable(n)
    assert not Comparable(n) < Comparable(MISSING)
    assert Comparable(n) < Comparable(0)
    assert Comparable(n) < Comparable(-float('inf'))
    assert Comparable(dn) < Comparable('a')
    assert not Comparable('a') < Comparable(dn)
    assert not Comparable(0) < Comparable(n)
    # ordinary equalities are preserved
    eq_(True, Comparable(1) == Comparable(1.0))
    eq_(True, Comparable(1) == Comparable(Decimal('1')))
    eq_(True, Comparable(1) == Comparable(True))
    eq_(True, Comparable(0.0) == Comparable(-0.0))
    eq_(True, Comparable(float('inf')) == Comparable(Decimal('Infinity')))
    # big integers stay exact - no float-conversion merge
    eq_(False, Comparable(2**53 + 1) == Comparable(float(2**53)))
    assert Comparable(float(2**53)) < Comparable(2**53 + 1)
    eq_(True, Comparable(10**100) == Comparable(10**100))


def test_comparable_compound_nan():
    n1 = float('nan')
    n2 = Decimal('NaN')
    # compound keys compare componentwise; NaN components are co-equal
    eq_(True, Comparable((1, n1)) == Comparable((1, n2)))
    eq_(False, Comparable((1, n1)) == Comparable((2, n1)))
    eq_(False, Comparable((1, n1)) == Comparable((1, 0)))
    assert Comparable((1, n1)) < Comparable((1, 0))
    assert Comparable((1, n1)) < Comparable((2, n1))
    assert not Comparable((1, n1)) < Comparable((1, n2))
    assert not Comparable((2, n1)) < Comparable((1, n2))
    # NaN in the first component dominates, like any value
    assert Comparable((n1, 5)) < Comparable((0, 0))
    assert not Comparable((0, 0)) < Comparable((n1, 5))


def test_comparable_wraps_original_unchanged():
    # the wrapper never replaces the wrapped value: no marker leakage
    n = float('nan')
    c = Comparable(n)
    assert c.inner is n
    dn = Decimal('-NaN123')
    c2 = Comparable(dn)
    assert c2.inner is dn
    row = (1, n)
    c3 = Comparable(row)
    assert c3.inner is row


def test_decimal_context_not_disturbed():
    # NaN comparisons must work with InvalidOperation trapped (the
    # default) or not, and must not mutate traps or status flags
    dn1 = Decimal('NaN')
    dn2 = Decimal('-NaN9')
    for trap in (True, False):
        with localcontext() as ctx:
            ctx.traps[InvalidOperation] = trap
            ctx.clear_flags()
            eq_(False, Comparable(dn1) < Comparable(dn2))
            eq_(True, Comparable(dn1) == Comparable(dn2))
            eq_(True, Comparable(dn1) < Comparable(1))
            eq_(False, Comparable(1) < Comparable(dn1))
            eq_(True, Comparable(None) < Comparable(dn1))
            eq_(False, Comparable(dn1) < Comparable(None))
            # finite decimal comparisons still work
            eq_(True, Comparable(Decimal('1.5')) < Comparable(2))
            eq_(True, Comparable(Decimal('1')) == Comparable(1))
            flags = [k for k, v in ctx.flags.items() if v]
            eq_([], flags)
            eq_(trap, ctx.traps[InvalidOperation])


def test_key_contains_nan_quiet_under_decimal_traps():
    # the predicate itself must not raise or set flags under either regime
    dn = Decimal('NaN')
    for trap in (True, False):
        with localcontext() as ctx:
            ctx.traps[InvalidOperation] = trap
            ctx.clear_flags()
            eq_(True, key_contains_nan(dn))
            eq_(True, key_contains_nan((1, dn)))
            eq_(False, key_contains_nan(Decimal('1.5')))
            flags = [k for k, v in ctx.flags.items() if v]
            eq_([], flags)
