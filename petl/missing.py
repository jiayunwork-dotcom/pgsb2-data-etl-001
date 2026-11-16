"""Sparse field-presence support.

This module defines :data:`petl.MISSING`, a public singleton marker that
means *the field is absent from the source record*.  It is deliberately a
distinct object rather than another use of ``None``: an upstream object that
omits a key entirely is not the same as one that carries an explicit
``null``.

The marker is designed to travel through every petl data path unchanged:

* it is hashable and has identity-based equality, so it can be used as a
  dictionary key and inside compound tuple keys;
* :func:`copy.copy`, :func:`copy.deepcopy` and :mod:`pickle` round-trips all
  return the same singleton, including after an external sort has spilled
  rows to a temporary pickle file;
* ordinary strings (including a string whose text is ``"MISSING"``), ``None``
  and arbitrary user objects never compare equal to it.

Rows stay ordinary ``tuple``/``list`` objects: the marker is simply a cell
value and every existing table access pattern keeps working.
"""

from __future__ import absolute_import, print_function, division


class _MissingMarker(object):
    """Singleton type for :data:`petl.MISSING`.

    Equality and hashing deliberately inherit identity semantics: two
    references are equal only if they are this exact object.  This keeps the
    marker distinct from ``None``, strings (even ones spelled ``"MISSING"``)
    and user objects, while remaining safe to use as a dictionary key.
    """

    __slots__ = ()

    # A marker is a flag, not a value; behave like None in boolean contexts.
    def __bool__(self):
        return False

    # Python 2 spelling (kept for the package's legacy compatibility layer).
    __nonzero__ = __bool__

    def __repr__(self):
        return 'MISSING'

    def __str__(self):
        return 'MISSING'

    def __unicode__(self):
        return u'MISSING'

    # Copying a marker yields the same marker - there is nothing to duplicate.
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        # Pickle (including across separate Python processes, shelve files and
        # external-sort chunk files) reconstructs the module-level singleton
        # instead of creating a second, unrelated instance.
        return (_reconstruct_missing, ())


def _reconstruct_missing():
    return MISSING


#: Public marker for a field that is absent from the source record.
MISSING = _MissingMarker()


def key_contains_missing(key):
    """Return True if a (possibly compound) join/sort key holds MISSING.

    `key` is either a bare cell value or a tuple of cell values, matching the
    keys produced by :func:`operator.itemgetter`.
    """
    if key is MISSING:
        return True
    if isinstance(key, tuple):
        for component in key:
            if component is MISSING:
                return True
    return False
