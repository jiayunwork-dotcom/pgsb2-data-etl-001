from __future__ import absolute_import, print_function, division


import operator
from decimal import Decimal
from functools import partial

from petl.compat import text_type, binary_type, numeric_types
from petl.missing import MISSING


def _isnan(obj):
    """Return True for the NaN values petl treats as unmatchable: Python
    float NaN and Decimal quiet NaN (either sign, any payload). Signaling
    NaN and all other values return False.

    The checks are deliberately quiet: no arithmetic or comparison is
    performed on Decimal operands, so the caller's decimal context (trap
    settings and status flags) is never consulted or modified.
    """
    # float first - cheap and the common case; ``obj != obj`` is quiet
    if isinstance(obj, float):
        return obj != obj
    # Decimal quiet NaN only; is_qnan() does not touch the decimal context
    if isinstance(obj, Decimal):
        return obj.is_qnan()
    return False


def key_contains_nan(key):
    """Return True if a (possibly compound) join/sort key holds a NaN value.

    `key` is either a bare cell value or a tuple of cell values, matching
    the keys produced by :func:`operator.itemgetter`. NaN means float NaN
    or Decimal quiet NaN (either sign, any payload); see :func:`_isnan`.

    A key containing a NaN component can never match another key in a
    join, because NaN is not equal to anything - not even to itself or
    to the identical NaN object. This is the relational-matching rule,
    which is deliberately distinct from the sort order (under which all
    NaN values are mutually co-equal, see :class:`Comparable`).
    """
    if _isnan(key):
        return True
    if isinstance(key, tuple):
        for component in key:
            if _isnan(component):
                return True
    return False


class Comparable(object):
    """Wrapper class to allow for flexible comparison of objects of different
    types, preserving the relaxed sorting behaviour of Python 2 with
    additional flexibility to allow for comparison of arbitrary objects with
    the `None` value (for example, the date and time objects from the standard
    library cannot be directly compared with `None` in Python 2).

    The ascending sort order is: :data:`petl.MISSING`, then `None`, then
    NaN values (float NaN and Decimal quiet NaN, all mutually co-equal
    regardless of category, sign or payload), then everything else under
    the relaxed rules. Co-equal NaN values keep their input order (the
    sort is stable). Note that co-equal for sorting does **not** mean
    matchable: joins treat any key containing a NaN component as never
    equal to another key - see :func:`key_contains_nan`.

    """

    __slots__ = ['obj', 'inner', 'isnan']

    def __init__(self, obj):
        # store wrapped object unchanged
        self.inner = obj
        # handle lists and tuples
        if isinstance(obj, (list, tuple)):
            obj = tuple(Comparable(o) for o in obj)
            # a container is never itself NaN; any NaN components carry
            # their own flag on the wrapped components
            self.isnan = False
        else:
            # cache NaN-ness once per wrapped value: sorting performs
            # O(n log n) comparisons per key and compound keys re-compare
            # components repeatedly, so this must be cheap to read
            self.isnan = _isnan(obj)
        self.obj = obj

    def __lt__(self, other):

        # convenience
        obj = self.obj
        obj_isnan = self.isnan
        if isinstance(other, Comparable):
            other_isnan = other.isnan
            other = other.obj
        else:
            other_isnan = _isnan(other)

        # MISSING < everything else, including None; only MISSING equals
        # MISSING. Note equality is handled by __eq__, which keeps the
        # marker's identity semantics, so here MISSING < MISSING is False.
        if obj is MISSING:
            return other is not MISSING
        if other is MISSING:
            return False

        # None < everything else (except MISSING)
        if other is None:
            return False
        if obj is None:
            return True

        # NaN < everything else (except MISSING and None); any two NaN
        # values are mutually co-equal for ordering. This check must come
        # before any native comparison: ``Decimal('NaN') < x`` raises
        # InvalidOperation under the default decimal context, and native
        # ``nan < x`` is always False, which would not give a total order.
        if obj_isnan or other_isnan:
            return obj_isnan and not other_isnan

        # numbers < everything else (except None)
        if isinstance(obj, numeric_types) \
                and not isinstance(other, numeric_types):
            return True
        if not isinstance(obj, numeric_types) \
                and isinstance(other, numeric_types):
            return False

        # binary < unicode
        if isinstance(obj, text_type) and isinstance(other, binary_type):
            return False
        if isinstance(obj, binary_type) and isinstance(other, text_type):
            return True

        try:
            # attempt native comparison
            return obj < other

        except TypeError:
            # fall back to comparing type names
            return _typestr(obj) < _typestr(other)

    def __eq__(self, other):
        if isinstance(other, Comparable):
            other = other.obj
        # NaN values are co-equal under the sort order (any category,
        # sign or payload), so the stability machinery - the heapq.merge
        # order tie-break, max() first-win in the reverse merge, and
        # itertools.groupby - sees a consistent total preorder. This is
        # deliberately NOT the join-matching rule: joins never match NaN
        # keys (see key_contains_nan). The check also keeps Decimal NaN
        # away from native ``<``/``>`` comparisons, which would raise
        # InvalidOperation under the default decimal context.
        if self.isnan:
            return _isnan(other)
        return self.obj == other

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)

    def __str__(self):
        return str(self.obj)

    def __unicode__(self):
        return text_type(self.obj)

    def __repr__(self):
        return 'Comparable(' + repr(self.obj) + ')'

    def __iter__(self, *args, **kwargs):
        return iter(self.obj, *args, **kwargs)

    def __len__(self):
        return len(self.obj)

    def __getitem__(self, item):
        return self.obj.__getitem__(item)


def _typestr(x):
    # attempt to preserve Python 2 name orderings
    if isinstance(x, binary_type):
        return 'str'
    if isinstance(x, text_type):
        return 'unicode'
    return type(x).__name__


def comparable_itemgetter(*args):
    getter = operator.itemgetter(*args)
    getter_with_default = _itemgetter_with_default(*args)

    def _getter_with_fallback(obj):
        try:
            return getter(obj)
        except (IndexError, KeyError):
            return getter_with_default(obj)
    g = lambda x: Comparable(_getter_with_fallback(x))
    return g


def _itemgetter_with_default(*args):
    """ itemgetter compatible with `operator.itemgetter` behavior, filling missing
    values with default instead of raising IndexError or KeyError """
    def _get_default(obj, item, default):
        try:
            return obj[item]
        except (IndexError, KeyError):
            return default
    if len(args) == 1:
        return partial(_get_default, item=args[0], default=None)
    return lambda obj: tuple(_get_default(obj, item=item, default=None) for item in args)
