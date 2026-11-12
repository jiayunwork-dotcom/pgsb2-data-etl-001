from __future__ import absolute_import, print_function, division

import logging
import os
import tempfile
from collections import Counter
from petl.compat import next, pickle, text_type, string_types
from petl.comparison import Comparable
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import header, Table
from petl.transform.sorts import sort, _heapqmergesorted, _iterchunk
from petl.transform.basics import cut
import petl.config as config


logger = logging.getLogger(__name__)
debug = logger.debug


def complement(a, b, presorted=False, buffersize=None, tempdir=None,
               cache=True, strict=False):
    """
    Return rows in `a` that are not in `b`. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['x', 'y', 'z'],
        ...      ['B', 2, False],
        ...      ['A', 9, False],
        ...      ['B', 3, True],
        ...      ['C', 9, True]]
        >>> aminusb = etl.complement(a, b)
        >>> aminusb
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

        >>> bminusa = etl.complement(b, a)
        >>> bminusa
        +-----+---+-------+
        | x   | y | z     |
        +=====+===+=======+
        | 'A' | 9 | False |
        +-----+---+-------+
        | 'B' | 3 | True  |
        +-----+---+-------+

    Note that the field names of each table are ignored - rows are simply
    compared following a lexical sort. See also the
    :func:`petl.transform.setops.recordcomplement` function.

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Note that the default behaviour is not strictly set-like, because 
    duplicate rows are counted separately, e.g.::
    
        >>> a = [['foo', 'bar'],
        ...      ['A', 1],
        ...      ['B', 2],
        ...      ['B', 2],
        ...      ['C', 7]]
        >>> b = [['foo', 'bar'],
        ...      ['B', 2]]
        >>> aminusb = etl.complement(a, b)
        >>> aminusb
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'B' |   2 |
        +-----+-----+
        | 'C' |   7 |
        +-----+-----+

    This behaviour can be changed with the `strict` keyword argument, e.g.::

        >>> aminusb = etl.complement(a, b, strict=True)
        >>> aminusb
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'C' |   7 |
        +-----+-----+

    .. versionchanged:: 1.1.0
    
    If `strict` is `True` then strict set-like behaviour is used, i.e., 
    only rows in `a` not found in `b` are returned.

    """

    return ComplementView(a, b, presorted=presorted, buffersize=buffersize,
                          tempdir=tempdir, cache=cache, strict=strict)


Table.complement = complement


class ComplementView(Table):

    def __init__(self, a, b, presorted=False, buffersize=None, tempdir=None,
                 cache=True, strict=False):
        if presorted:
            self.a = a
            self.b = b
        else:
            self.a = sort(a, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)
            self.b = sort(b, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)
        self.strict = strict

    def __iter__(self):
        return itercomplement(self.a, self.b, self.strict)


def itercomplement(ta, tb, strict):
    # coerce rows to tuples to ensure hashable and comparable
    ita = (tuple(row) for row in iter(ta))
    itb = (tuple(row) for row in iter(tb))
    ahdr = tuple(next(ita))
    next(itb)  # ignore b fields
    yield ahdr

    try:
        a = next(ita)
    except StopIteration:
        pass
    else:
        try:
            b = next(itb)
        except StopIteration:
            yield a
            for row in ita:
                yield row
        else:
            # we want the elements in a that are not in b
            while True:
                if b is None or Comparable(a) < Comparable(b):
                    yield a
                    try:
                        a = next(ita)
                    except StopIteration:
                        break
                elif a == b:
                    try:
                        a = next(ita)
                    except StopIteration:
                        break
                    if not strict:
                        try:
                            b = next(itb)
                        except StopIteration:
                            b = None
                else:
                    try:
                        b = next(itb)
                    except StopIteration:
                        b = None


def recordcomplement(a, b, buffersize=None, tempdir=None, cache=True,
                     strict=False):
    """
    Find records in `a` that are not in `b`. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['bar', 'foo', 'baz'],
        ...      [2, 'B', False],
        ...      [9, 'A', False],
        ...      [3, 'B', True],
        ...      [9, 'C', True]]
        >>> aminusb = etl.recordcomplement(a, b)
        >>> aminusb
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

        >>> bminusa = etl.recordcomplement(b, a)
        >>> bminusa
        +-----+-----+-------+
        | bar | foo | baz   |
        +=====+=====+=======+
        |   3 | 'B' | True  |
        +-----+-----+-------+
        |   9 | 'A' | False |
        +-----+-----+-------+

    Note that both tables must have the same set of fields, but that the order
    of the fields does not matter. See also the
    :func:`petl.transform.setops.complement` function.

    See also the discussion of the `buffersize`, `tempdir` and `cache` arguments
    under the :func:`petl.transform.sorts.sort` function.

    """

    # TODO possible with only one pass?

    ha = header(a)
    hb = header(b)
    assert set(ha) == set(hb), 'both tables must have the same set of fields'
    # make sure fields are in the same order
    bv = cut(b, *ha)
    return complement(a, bv, buffersize=buffersize, tempdir=tempdir,
                      cache=cache, strict=strict)


Table.recordcomplement = recordcomplement


def diff(a, b, presorted=False, buffersize=None, tempdir=None, cache=True,
         strict=False):
    """
    Find the difference between rows in two tables. Returns a pair of tables.
    E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['x', 'y', 'z'],
        ...      ['B', 2, False],
        ...      ['A', 9, False],
        ...      ['B', 3, True],
        ...      ['C', 9, True]]
        >>> added, subtracted = etl.diff(a, b)
        >>> # rows in b not in a
        ... added
        +-----+---+-------+
        | x   | y | z     |
        +=====+===+=======+
        | 'A' | 9 | False |
        +-----+---+-------+
        | 'B' | 3 | True  |
        +-----+---+-------+

        >>> # rows in a not in b
        ... subtracted
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

    Convenient shorthand for ``(complement(b, a), complement(a, b))``. See also
    :func:`petl.transform.setops.complement`.

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    .. versionchanged:: 1.1.0

    If `strict` is `True` then strict set-like behaviour is used.

    """

    if not presorted:
        a = sort(a)
        b = sort(b)
    added = complement(b, a, presorted=True, buffersize=buffersize,
                       tempdir=tempdir, cache=cache, strict=strict)
    subtracted = complement(a, b, presorted=True, buffersize=buffersize,
                            tempdir=tempdir, cache=cache, strict=strict)
    return added, subtracted


Table.diff = diff


def recorddiff(a, b, buffersize=None, tempdir=None, cache=True, strict=False):
    """
    Find the difference between records in two tables. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['bar', 'foo', 'baz'],
        ...      [2, 'B', False],
        ...      [9, 'A', False],
        ...      [3, 'B', True],
        ...      [9, 'C', True]]
        >>> added, subtracted = etl.recorddiff(a, b)
        >>> added
        +-----+-----+-------+
        | bar | foo | baz   |
        +=====+=====+=======+
        |   3 | 'B' | True  |
        +-----+-----+-------+
        |   9 | 'A' | False |
        +-----+-----+-------+

        >>> subtracted
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

    Convenient shorthand for
    ``(recordcomplement(b, a), recordcomplement(a, b))``. See also
    :func:`petl.transform.setops.recordcomplement`.

    See also the discussion of the `buffersize`, `tempdir` and `cache`
    arguments under the :func:`petl.transform.sorts.sort` function.

    .. versionchanged:: 1.1.0

    If `strict` is `True` then strict set-like behaviour is used.

    """

    added = recordcomplement(b, a, buffersize=buffersize, tempdir=tempdir,
                             cache=cache, strict=strict)
    subtracted = recordcomplement(a, b, buffersize=buffersize, tempdir=tempdir,
                                  cache=cache, strict=strict)
    return added, subtracted


Table.recorddiff = recorddiff


def intersection(a, b, presorted=False, buffersize=None, tempdir=None,
                 cache=True):
    """
    Return rows in `a` that are also in `b`. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, True],
        ...           ['C', 7, False],
        ...           ['B', 2, False],
        ...           ['C', 9, True]]
        >>> table2 = [['x', 'y', 'z'],
        ...           ['B', 2, False],
        ...           ['A', 9, False],
        ...           ['B', 3, True],
        ...           ['C', 9, True]]
        >>> table3 = etl.intersection(table1, table2)
        >>> table3
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'B' |   2 | False |
        +-----+-----+-------+
        | 'C' |   9 | True  |
        +-----+-----+-------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    """

    return IntersectionView(a, b, presorted=presorted, buffersize=buffersize,
                            tempdir=tempdir, cache=cache)


Table.intersection = intersection


class IntersectionView(Table):

    def __init__(self, a, b, presorted=False, buffersize=None, tempdir=None,
                 cache=True):
        if presorted:
            self.a = a
            self.b = b
        else:
            self.a = sort(a, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)
            self.b = sort(b, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)

    def __iter__(self):
        return iterintersection(self.a, self.b)


def iterintersection(a, b):
    ita = iter(a)
    itb = iter(b)
    ahdr = next(ita)
    next(itb)  # ignore b header
    yield tuple(ahdr)
    try:
        a = tuple(next(ita))
        b = tuple(next(itb))
        while True:
            if Comparable(a) < Comparable(b):
                a = tuple(next(ita))
            elif a == b:
                yield a
                a = tuple(next(ita))
                b = tuple(next(itb))
            else:
                b = tuple(next(itb))
    except StopIteration:
        pass


def hashcomplement(a, b, strict=False):
    """
    Alternative implementation of :func:`petl.transform.setops.complement`,
    where the complement is executed by constructing an in-memory set for all
    rows found in the right hand table, then iterating over rows from the
    left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.
    
    .. versionchanged:: 1.1.0
    
    If `strict` is `True` then strict set-like behaviour is used, i.e., 
    only rows in `a` not found in `b` are returned.

    """

    return HashComplementView(a, b, strict=strict)


Table.hashcomplement = hashcomplement


class HashComplementView(Table):

    def __init__(self, a, b, strict=False):
        self.a = a
        self.b = b
        self.strict = strict

    def __iter__(self):
        return iterhashcomplement(self.a, self.b, self.strict)


def iterhashcomplement(a, b, strict):
    ita = iter(a)
    ahdr = next(ita)
    yield tuple(ahdr)
    itb = iter(b)
    next(itb)  # discard b header, assume same as a

    # N.B., need to account for possibility of duplicate rows
    bcnt = Counter(tuple(row) for row in itb)
    for ar in ita:
        t = tuple(ar)
        if bcnt[t] > 0:
            if not strict:
                bcnt[t] -= 1
        else:
            yield t


def hashintersection(a, b):
    """
    Alternative implementation of
    :func:`petl.transform.setops.intersection`, where the intersection
    is executed by constructing an in-memory set for all rows found in the
    right hand table, then iterating over rows from the left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.

    """

    return HashIntersectionView(a, b)


Table.hashintersection = hashintersection


class HashIntersectionView(Table):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __iter__(self):
        return iterhashintersection(self.a, self.b)


def iterhashintersection(a, b):
    ita = iter(a)
    ahdr = next(ita)
    yield tuple(ahdr)
    itb = iter(b)
    next(itb)  # discard b header, assume same as a

    # N.B., need to account for possibility of duplicate rows
    bcnt = Counter(tuple(row) for row in itb)
    for ar in ita:
        t = tuple(ar)
        if bcnt[t] > 0:
            yield t
            bcnt[t] -= 1


def keydiff(before, after, key, presorted=False, buffersize=None,
            tempdir=None):
    """
    Return a stream of changes between two table snapshots, keyed by one or
    more identifying fields. E.g.::

        >>> import petl as etl
        >>> before = [['id', 'name', 'score'],
        ...           [1, 'Alice', 10],
        ...           [2, 'Bob', 20],
        ...           [3, 'Carol', 30]]
        >>> after = [['id', 'name', 'score'],
        ...          [1, 'Alice', 11],
        ...          [4, 'Dave', 40],
        ...          [2, 'Bob', 20]]
        >>> changes = etl.keydiff(before, after, 'id')
        >>> changes
        +----------+------+------------------+------------------+------------+
        | op       | key  | before           | after            | changed    |
        +==========+======+==================+==================+============+
        | 'update' | (1,) | (1, 'Alice', 10) | (1, 'Alice', 11) | ('score',) |
        +----------+------+------------------+------------------+------------+
        | 'delete' | (3,) | (3, 'Carol', 30) | None             | ()         |
        +----------+------+------------------+------------------+------------+
        | 'insert' | (4,) | None             | (4, 'Dave', 40)  | ()         |
        +----------+------+------------------+------------------+------------+
        <BLANKLINE>

    The ``key`` argument must be a field name or a non-empty sequence of
    field names. Both input tables must contain all key fields, must have
    unique field names, and must have the same set of field names; column
    order may differ. Rows are compared by field name, but the ``before`` and
    ``after`` columns always contain the complete row interpreted using
    *before*'s header order.

    The output has a fixed header ``('op', 'key', 'before', 'after',
    'changed')``. The ``op`` column is one of ``'insert'``, ``'update'`` or
    ``'delete'``. The ``key`` column always holds a tuple of key values, even
    for a single key field. ``before`` and ``after`` hold complete row
    tuples, with ``None`` for the side where the record does not exist.
    ``changed`` is a tuple of non-key field names whose values differ; its
    order follows *before*'s header. It is empty for inserts and deletes.
    Rows that are identical on both sides produce no output.

    Each side must have unique keys; a duplicate key raises
    :class:`petl.errors.ArgumentError` identifying the side and the original
    data row number (counting from 1 after the header). Property values are
    compared using Python equality, so unhashable values such as lists and
    dicts are supported.

    Output is ordered by ascending key. By default the inputs may be
    unordered; rows are sorted externally in chunks of ``buffersize`` rows
    (see :func:`petl.transform.sorts.sort`) and temporary files are written
    under ``tempdir``. Set ``presorted=True`` when both sides are already
    sorted by key and ordered ascending under petl's comparison semantics,
    which allows a single streaming pass over each side.

    .. versionadded:: 1.7.32

    """

    return KeyDiffView(before, after, key, presorted=presorted,
                       buffersize=buffersize, tempdir=tempdir)


Table.keydiff = keydiff


KEYDIFF_HEADER = ('op', 'key', 'before', 'after', 'changed')

# Cap the number of temporary chunk files opened at once during the merge
# phase. This mirrors the bounded fan-in used by external merge sorts and
# keeps file-descriptor use constant for very large inputs.
KEYDIFF_MAX_OPEN_CHUNKS = 64


class KeyDiffView(Table):

    def __init__(self, before, after, key, presorted=False, buffersize=None,
                 tempdir=None):
        self.before = before
        self.after = after
        self.key = key
        self.presorted = presorted
        self.buffersize = buffersize
        self.tempdir = tempdir

    def __iter__(self):
        return iterkeydiff(self.before, self.after, self.key,
                           presorted=self.presorted,
                           buffersize=self.buffersize,
                           tempdir=self.tempdir)


def _keydiff_close(obj):
    """Close an iterator if it supports ``close()``.

    Cleanup must never mask an original exception, so any error raised by
    ``close()`` is logged and suppressed.
    """
    if obj is None:
        return
    close = getattr(obj, 'close', None)
    if close is None:
        return
    try:
        close()
    except Exception as e:  # pragma: no cover - defensive
        debug('exception closing %r: %s', obj, e)


def _keydiff_fieldnames(hdr):
    # Field lookup follows petl's own field-selection convention, which
    # normalises header names to text for matching.
    return [text_type(f) for f in hdr]


def _keydiff_has_duplicate_names(hdr):
    # Match petl's field-selection convention, which compares header values
    # as text. This also catches headers such as (1, '1') that would be
    # ambiguous for key lookup.
    names = _keydiff_fieldnames(hdr)
    for i, name in enumerate(names):
        if name in names[:i]:
            return True
    return False


def _keydiff_same_field_set(ha, hb):
    na = _keydiff_fieldnames(ha)
    nb = _keydiff_fieldnames(hb)
    if len(na) != len(nb):
        return False
    return all(name in nb for name in na)


def _keydiff_keynames(key):
    if key is None:
        raise ArgumentError('key is required')
    if isinstance(key, string_types):
        return (key,)
    if isinstance(key, (list, tuple)):
        if len(key) == 0:
            raise ArgumentError('key must not be empty')
        names = tuple(key)
        for name in names:
            if not isinstance(name, string_types):
                raise ArgumentError(
                    'key must be a field name or a sequence of field names, '
                    'got %r' % (name,))
        # Compare as petl compares field names, so a key such as
        # ('1', 1) is recognised as repeating the same field.
        if len(set(text_type(n) for n in names)) != len(names):
            raise ArgumentError('key contains duplicate field names: %r'
                                % (names,))
        return names
    raise ArgumentError(
        'key must be a field name or a sequence of field names, got %r'
        % (key,))


def _keydiff_indices(hdr, keynames, side):
    fields = _keydiff_fieldnames(hdr)
    indices = []
    for name in keynames:
        try:
            indices.append(fields.index(name))
        except ValueError:
            raise FieldSelectionError(
                '%r (not found in %s table header %r)'
                % (name, side, tuple(hdr)))
    if len(set(indices)) != len(indices):
        raise ArgumentError('key contains duplicate field names: %r'
                            % (keynames,))
    return tuple(indices)


def _keydiff_projection(before_hdr, after_hdr):
    fields = _keydiff_fieldnames(after_hdr)
    return tuple(fields.index(name)
                 for name in _keydiff_fieldnames(before_hdr))


def _keydiff_compare(a, b):
    if a == b:
        return 0
    if a < b:
        return -1
    return 1


def _keydiff_make_key(row, indices):
    return tuple(row[i] for i in indices)


def _keydiff_comparable(rawkey, multiple):
    if multiple:
        return Comparable(rawkey)
    return Comparable(rawkey[0])


def _keydiff_check_width(row, width, side, rownumber):
    if len(row) != width:
        raise ArgumentError(
            '%s table row %d has %d fields, expected %d'
            % (side, rownumber, len(row), width))


def _keydiff_raise_duplicate(side, rownumber, rawkey, firstrow):
    raise ArgumentError(
        '%s table has duplicate key %r at row %d (first seen at row %d)'
        % (side, rawkey, rownumber, firstrow))


def _keydiff_raise_regression(side, rownumber, rawkey, previouskey):
    raise ArgumentError(
        '%s table is not sorted by key: key %r at row %d follows %r'
        % (side, rawkey, rownumber, previouskey))


def _keydiff_presorted_rows(it, width, keyindices, multiple, side):
    """Yield checked records from an already sorted side, once only.

    At most one record per side is retained by the caller, so the memory
    overhead does not grow with the input size.
    """
    previous = None
    previouskey = None
    previousrow = None
    rownumber = 0
    for row in it:
        rownumber += 1
        row = tuple(row)
        _keydiff_check_width(row, width, side, rownumber)
        rawkey = _keydiff_make_key(row, keyindices)
        cmpkey = _keydiff_comparable(rawkey, multiple)
        if previous is not None:
            cmp = _keydiff_compare(previous, cmpkey)
            if cmp > 0:
                _keydiff_raise_regression(side, rownumber, rawkey, previouskey)
            if cmp == 0:
                _keydiff_raise_duplicate(side, rownumber, rawkey, previousrow)
        yield (cmpkey, rownumber, rawkey, row)
        previous = cmpkey
        previouskey = rawkey
        previousrow = rownumber


def _keydiff_sortkey(record):
    return record[0]


def _keydiff_read_chunk(it, state, size, width, keyindices, multiple, side):
    """Read up to ``size`` records, returning ``(records, exhausted)``.

    ``state`` is a one-item list holding the original data row number, so
    that row numbers are always relative to the first data row after the
    header, regardless of how many chunks have been read.
    """
    records = []
    while size is None or len(records) < size:
        state[0] += 1
        try:
            row = next(it)
        except StopIteration:
            state[0] -= 1
            return records, True
        row = tuple(row)
        _keydiff_check_width(row, width, side, state[0])
        rawkey = _keydiff_make_key(row, keyindices)
        records.append((_keydiff_comparable(rawkey, multiple), state[0],
                        rawkey, row))
    return records, False


def _keydiff_write_chunk(records, tempdir):
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=tempdir, delete=False,
                                         mode='wb') as f:
            name = f.name
            for record in records:
                pickle.dump(record, f, protocol=-1)
            f.flush()
    except Exception:
        # never leave a partially written chunk behind
        if name is not None:
            _keydiff_remove_tempfiles([name])
        raise
    return name


def _keydiff_remove_tempfiles(names):
    for name in names:
        try:
            os.unlink(name)
        except Exception as e:  # pragma: no cover - defensive
            debug('exception deleting temporary file %s: %s', name, e)


def _keydiff_checked(records, side):
    """Check duplicate keys on a stream already sorted by key."""
    previous = None
    previousrow = None
    for record in records:
        cmpkey = record[0]
        if previous is not None and _keydiff_compare(previous, cmpkey) == 0:
            _keydiff_raise_duplicate(side, record[1], record[2], previousrow)
        yield record
        previous = cmpkey
        previousrow = record[1]


def _keydiff_merge_group(names, tempdir):
    """Merge one group of sorted chunk files into a new sorted chunk file."""
    chunkiters = []
    merged = None
    try:
        chunkiters = [_iterchunk(name) for name in names]
        merged = _heapqmergesorted(_keydiff_sortkey, *chunkiters)
        return _keydiff_write_chunk_stream(merged, tempdir)
    finally:
        _keydiff_close(merged)
        for chunkiter in chunkiters:
            _keydiff_close(chunkiter)


def _keydiff_write_chunk_stream(records, tempdir):
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=tempdir, delete=False,
                                         mode='wb') as f:
            name = f.name
            for record in records:
                pickle.dump(record, f, protocol=-1)
            f.flush()
    except Exception:
        if name is not None:
            _keydiff_remove_tempfiles([name])
        raise
    return name


def _keydiff_reduce_tempfiles(tempfiles, tempdir, maxopen):
    """Merge groups of chunk files until at most ``maxopen`` remain.

    This bounds the number of simultaneously open temporary files, so a
    very large input does not exhaust the OS file-descriptor limit.
    Returns the list of remaining (sorted) chunk file names.
    """
    while len(tempfiles) > maxopen:
        mergedfiles = []
        try:
            for start in range(0, len(tempfiles), maxopen):
                group = tempfiles[start:start + maxopen]
                mergedfiles.append(_keydiff_merge_group(group, tempdir))
        except Exception:
            _keydiff_remove_tempfiles(mergedfiles)
            raise
        _keydiff_remove_tempfiles(tempfiles)
        tempfiles = mergedfiles
    return tempfiles


def _keydiff_sorted_rows(it, width, keyindices, multiple, side, buffersize,
                         tempdir):
    """Externally sort one side's records while preserving row numbers."""
    if buffersize is None:
        buffersize = config.sort_buffersize
    if buffersize is not None and buffersize <= 0:
        raise ArgumentError(
            'buffersize must be a positive integer or None, got %r'
            % (buffersize,))

    state = [0]
    tempfiles = []
    chunkiters = []
    try:
        records, exhausted = _keydiff_read_chunk(
            it, state, buffersize, width, keyindices, multiple, side)
        records.sort(key=_keydiff_sortkey)
        if buffersize is None or exhausted:
            for record in _keydiff_checked(records, side):
                yield record
            return

        while True:
            tempfiles.append(_keydiff_write_chunk(records, tempdir))
            records, exhausted = _keydiff_read_chunk(
                it, state, buffersize, width, keyindices, multiple, side)
            records.sort(key=_keydiff_sortkey)
            if exhausted:
                if records:
                    tempfiles.append(_keydiff_write_chunk(records, tempdir))
                break

        tempfiles = _keydiff_reduce_tempfiles(tempfiles, tempdir,
                                              KEYDIFF_MAX_OPEN_CHUNKS)
        chunkiters = [_iterchunk(name) for name in tempfiles]
        merged = _heapqmergesorted(_keydiff_sortkey, *chunkiters)
        try:
            for record in _keydiff_checked(merged, side):
                yield record
        finally:
            _keydiff_close(merged)
    finally:
        for chunkiter in chunkiters:
            _keydiff_close(chunkiter)
        _keydiff_remove_tempfiles(tempfiles)


def _keydiff_record_changed(before_row, after_row, before_hdr,
                            nonkey_positions):
    return tuple(before_hdr[i] for i in nonkey_positions
                 if before_row[i] != after_row[i])


def _keydiff_changes(bit, ait, before_hdr, nonkey_positions,
                     after_projection):
    """Merge two duplicate-free, key-sorted record streams into change rows."""
    try:
        b = next(bit)
    except StopIteration:
        b = None
    try:
        a = next(ait)
    except StopIteration:
        a = None

    while b is not None and a is not None:
        comparison = _keydiff_compare(b[0], a[0])
        if comparison < 0:
            yield ('delete', b[2], b[3], None, ())
            try:
                b = next(bit)
            except StopIteration:
                b = None
        elif comparison > 0:
            yield ('insert', a[2], None,
                   tuple(a[3][i] for i in after_projection), ())
            try:
                a = next(ait)
            except StopIteration:
                a = None
        else:
            after_row = tuple(a[3][i] for i in after_projection)
            changed = _keydiff_record_changed(b[3], after_row, before_hdr,
                                              nonkey_positions)
            if changed:
                yield ('update', b[2], b[3], after_row, changed)
            try:
                b = next(bit)
            except StopIteration:
                b = None
            try:
                a = next(ait)
            except StopIteration:
                a = None

    while b is not None:
        yield ('delete', b[2], b[3], None, ())
        try:
            b = next(bit)
        except StopIteration:
            b = None
    while a is not None:
        yield ('insert', a[2], None,
               tuple(a[3][i] for i in after_projection), ())
        try:
            a = next(ait)
        except StopIteration:
            a = None


def iterkeydiff(before, after, key, presorted=False, buffersize=None,
                tempdir=None):
    keynames = _keydiff_keynames(key)
    itb = None
    ita = None
    bit = None
    ait = None
    try:
        # Exactly one source iterator per side; the header is read from the
        # same iterator that supplies the data, never re-opened.
        itb = iter(before)
        try:
            hb = tuple(next(itb))
        except StopIteration:
            raise ArgumentError('before table has no header')
        ita = iter(after)
        try:
            ha = tuple(next(ita))
        except StopIteration:
            raise ArgumentError('after table has no header')

        if _keydiff_has_duplicate_names(hb):
            raise ArgumentError(
                'before table has duplicate field names: %r' % (hb,))
        if _keydiff_has_duplicate_names(ha):
            raise ArgumentError(
                'after table has duplicate field names: %r' % (ha,))
        if not _keydiff_same_field_set(hb, ha):
            raise ArgumentError(
                'before and after tables must have the same set of fields; '
                'before: %r, after: %r' % (hb, ha))

        kb = _keydiff_indices(hb, keynames, 'before')
        ka = _keydiff_indices(ha, keynames, 'after')
        after_projection = _keydiff_projection(hb, ha)
        keypositions = set(kb)
        nonkey_positions = tuple(i for i in range(len(hb))
                                 if i not in keypositions)
        multiple = len(keynames) > 1

        if presorted:
            bit = _keydiff_presorted_rows(itb, len(hb), kb, multiple,
                                          'before')
            ait = _keydiff_presorted_rows(ita, len(ha), ka, multiple,
                                          'after')
        else:
            bit = _keydiff_sorted_rows(itb, len(hb), kb, multiple, 'before',
                                       buffersize, tempdir)
            ait = _keydiff_sorted_rows(ita, len(ha), ka, multiple, 'after',
                                       buffersize, tempdir)

        yield KEYDIFF_HEADER
        for change in _keydiff_changes(bit, ait, hb, nonkey_positions,
                                       after_projection):
            yield change
    finally:
        _keydiff_close(bit)
        _keydiff_close(ait)
        _keydiff_close(itb)
        _keydiff_close(ita)
