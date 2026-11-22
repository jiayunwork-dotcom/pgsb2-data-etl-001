from __future__ import absolute_import, print_function, division

import math
import os
import sys
from collections import Counter
from decimal import Decimal

import pytest

from petl.compat import izip_longest


def eq_(expect, actual, msg=None):
    """Test when two values from a python variable are exactly equals (==)"""
    assert expect == actual, msg or ('%r != %s' % (expect, actual))


def assert_almost_equal(first, second, places=None, msg=None):
    """Test when the values are aproximatedly equals by a places exponent"""
    vabs = None if places is None else 10 ** (- places)
    # Use the == operator to trigger pytest.approx comparison logic
    assert second == pytest.approx(first, abs=vabs), msg


def ieq(expect, actual, cast=None):
    """Test when values of a iterable are equals for each row and column"""
    ie = iter(expect)
    ia = iter(actual)
    ir = 0
    for re, ra in izip_longest(ie, ia, fillvalue=None):
        if cast:
            ra = cast(ra)
        if re is None and ra is None:
            continue
        if type(re) in (int, float, bool, str):
            eq_(re, ra)
            continue
        _ieq_row(re, ra, ir)
        ir = ir + 1


def _ieq_row(re, ra, ir):
    assert ra is not None, "Expected row #%d is None, but result row is not None" % ir
    assert re is not None, "Expected row #%d is not None, but result row is None" % ir
    ic = 0
    for ve, va in izip_longest(re, ra, fillvalue=None):
        if isinstance(ve, list):
            for je, ja in izip_longest(ve, va, fillvalue=None):
                _ieq_col(je, ja, re, ra, ir, ic)
        elif not isinstance(ve, dict):
            _ieq_col(ve, va, re, ra, ir, ic)
        ic = ic + 1


def _ieq_col(ve, va, re, ra, ir, ic):
    """Print two values when they aren't exactly equals (==)"""
    try:
        eq_(ve, va)
    except AssertionError as ea:
        # Show the values but only when they differ
        print('\nrow #%d' % ir, re, ' != ', ra, file=sys.stderr)
        print('col #%d: ' % ic, ve, ' != ', va, file=sys.stderr)
        raise ea


def ieq2(expect, actual, cast=None):
    """Test when iterables values are equals twice looking for side effects"""
    ieq(expect, actual, cast)
    ieq(expect, actual, cast)


def nan_token(cell):
    """Return a canonical, comparable token for a NaN cell value.

    NaN cells are identified by category (float vs Decimal), sign and -
    for Decimal - payload, so tests can assert that NaN values travelled
    the data paths unchanged without ever relying on NaN == NaN. Any
    non-NaN value is returned unchanged.
    """
    if isinstance(cell, float) and cell != cell:
        return ('_NAN_', 'float', math.copysign(1.0, cell) < 0.0)
    if isinstance(cell, Decimal) and cell.is_qnan():
        return ('_NAN_', 'decimal', cell.is_signed(), cell.as_tuple().digits)
    return cell


def canonical_rows(rows):
    """Materialise rows, replacing NaN cells with canonical tokens."""
    return [tuple(nan_token(cell) for cell in row) for row in rows]


def ieq_nan(expect, actual):
    """Like :func:`ieq`, but NaN cells compare by category, sign and payload."""
    ieq(canonical_rows(expect), canonical_rows(actual))


def row_multiset(rows):
    """Canonical-row multiset, for order-independent comparison with
    exact duplicate counts."""
    return Counter(canonical_rows(rows))


def get_env_vars_named(prefix, remove_prefix=True):
    """Get all named variables starting with prefix"""
    res = {}
    varlen = len(prefix)
    for varname, varvalue in os.environ.items():
        if varname.upper().startswith(prefix.upper()):
            if remove_prefix:
                varname = varname[varlen:]
            res[varname] = varvalue
    if len(res) == 0:
        return None
    return res
