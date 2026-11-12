"""
Grouping sets with mergeable partial aggregation states.

The :func:`groupstate` function aggregates a table over several grouping
sets (e.g., by region and product, by region only, and grand total) in a
single pass over the data and returns a :class:`GroupState` object holding
the aggregated result. A :class:`GroupState` can be:

* rendered as a table with :meth:`GroupState.totable`,
* merged with other compatible states covering disjoint batches with
  :meth:`GroupState.merge`, so that, e.g., daily partial aggregates can be
  combined into monthly ones without re-reading the detail rows, and
* serialized to bytes with :meth:`GroupState.dumps` and restored with
  :func:`loadgroupstate`, also in another process.

The :func:`groupingsets` function is a convenience shortcut equivalent to
calling :func:`groupstate` and immediately rendering the result with
:meth:`GroupState.totable`.

"""

from __future__ import absolute_import, print_function, division


import json
import math
import struct
from fractions import Fraction


from petl.compat import string_types
from petl.errors import ArgumentError, FieldSelectionError
from petl.comparison import Comparable
from petl.util.base import Table


__all__ = ['GroupState', 'groupingsets', 'groupstate', 'loadgroupstate']


_COUNT = 'count'
_SUM = 'sum'
_MEAN = 'mean'
_MIN = 'min'
_MAX = 'max'
_COUNTDISTINCT = 'countdistinct'
_MEASURE_TYPES = (_COUNT, _SUM, _MEAN, _MIN, _MAX, _COUNTDISTINCT)

_GROUPING_FIELD = 'grouping'

# serialization format: magic + version byte + 8-byte big-endian payload
# length + JSON payload
_MAGIC = b'PETLGST'
_FORMAT_VERSION = 1
_HEADER_LEN = len(_MAGIC) + 1 + 8

# batch label used internally by groupingsets(), which never gets merged
_DEFAULT_BATCH = '-'


def groupstate(table, key, sets, measures, batch):
    """
    Aggregate `table` over several grouping sets in a single pass and return
    a :class:`GroupState` holding the result. E.g.::

        >>> import petl as etl
        >>> table1 = [['region', 'product', 'amount'],
        ...           ['east', 'apple', 3],
        ...           ['east', 'apple', 7],
        ...           ['east', 'banana', 2],
        ...           ['west', 'apple', None],
        ...           ['west', 'banana', 4.5]]
        >>> state = etl.groupstate(table1, key=('region', 'product'),
        ...                        sets=[('region', 'product'), ('region',), ()],
        ...                        measures=[('total', 'sum', 'amount'),
        ...                                  ('n', 'count', None)],
        ...                        batch='2026-11-01')
        >>> etl.lookall(state.totable())
        +--------+----------+----------+-------+---+
        | region | product  | grouping | total | n |
        +========+==========+==========+=======+===+
        | 'east' | 'apple'  |        0 |    10 | 2 |
        +--------+----------+----------+-------+---+
        | 'east' | 'banana' |        0 |     2 | 1 |
        +--------+----------+----------+-------+---+
        | 'west' | 'apple'  |        0 | None  | 1 |
        +--------+----------+----------+-------+---+
        | 'west' | 'banana' |        0 |   4.5 | 1 |
        +--------+----------+----------+-------+---+
        | 'east' | None     |        1 |    12 | 3 |
        +--------+----------+----------+-------+---+
        | 'west' | None     |        1 |   4.5 | 2 |
        +--------+----------+----------+-------+---+
        | None   | None     |        3 |  16.5 | 5 |
        +--------+----------+----------+-------+---+

    The `key` argument is a single field name or a non-empty sequence of
    unique field names; these are the *dimensions*. The order of the
    dimensions determines the order of the output columns and the bit
    positions of the `grouping` column (see below).

    The `sets` argument is a non-empty sequence of *grouping sets*. Each
    grouping set is composed of dimensions from `key` and may be empty (the
    empty set produces a grand total row). A grouping set may not contain
    the same dimension twice, and two grouping sets may not have the same
    members.

    The `measures` argument is a non-empty sequence of ``(name, type,
    field)`` triples, where `name` is the output field name (names must be
    unique and must not collide with a dimension name or ``'grouping'``),
    `type` is one of ``'count'``, ``'sum'``, ``'mean'``, ``'min'``,
    ``'max'`` or ``'countdistinct'``, and `field` is the name of the field
    to aggregate. For ``'count'`` the field may be ``None``, meaning rows
    are counted; if a field is given, the non-``None`` values of that field
    are counted. All other measure types require a field.

    The `batch` argument is a non-empty string labelling which batch of
    detail rows this partial aggregate covers; it is used to detect
    overlapping batches when states are merged (see :class:`GroupState`).

    The source table must have a header of unique string field names, and
    every data row must match the header width. Dimension values must be
    ``str``, ``int`` (not ``bool``) or ``None``. Measure values of ``None``
    are ignored (except that ``'count'`` without a field counts every row).
    The ``'sum'``, ``'mean'``, ``'min'`` and ``'max'`` measures accept
    finite ``int`` and ``float`` values (not ``bool``); ``'countdistinct'``
    accepts ``str`` and finite ``int`` and ``float`` values. The float
    value ``-0.0`` is treated as ``0.0``. Violations are reported as
    :class:`petl.errors.ArgumentError` including the data row number
    (counting 1 for the first row after the header); missing dimension or
    measure fields are reported as :class:`petl.errors.FieldSelectionError`;
    other invalid arguments are reported as
    :class:`petl.errors.ArgumentError`.

    The measures are defined as follows:

    * ``count`` - the number of rows (field is ``None``) or the number of
      non-``None`` values; the result is an ``int``.
    * ``sum`` - ``None`` if there are no non-``None`` values. If all values
      are ``int`` the result is the exact ``int`` sum. If any value is a
      ``float`` the result is a ``float`` equal to the exact sum of all
      values rounded only once (round half to even).
    * ``mean`` - ``None`` if there are no values, otherwise a ``float``
      equal to the exact sum divided by the number of values, rounded only
      once (round half to even). A ``sum`` or ``mean`` result that does not
      fit in a ``float`` raises :exc:`OverflowError`.
    * ``min`` / ``max`` - ``None`` if there are no values. If the extreme
      value is tied between an ``int`` and a ``float`` of equal value, the
      ``int`` is reported.
    * ``countdistinct`` - the number of distinct values, compared with
      Python equality, so ``1`` and ``1.0`` count as the same value while
      ``'1'`` and ``1`` do not; ``None`` is not counted.

    The table is read exactly once, through a single iterator, no matter
    how many grouping sets are given, and memory usage is proportional to
    the number of output groups and distinct values, not to the number of
    data rows.

    See also :func:`groupingsets`, :class:`GroupState`.

    """

    dims = _normalize_key(key)
    gsets = _normalize_sets(sets, dims)
    mspecs = _normalize_measures(measures, dims)
    mtypes = tuple(m[1] for m in mspecs)
    batches = frozenset((_normalize_batch(batch),))

    it = iter(table)
    try:
        header = tuple(next(it))
    except StopIteration:
        raise ArgumentError('expected a table with a header row, '
                            'found an empty table')
    _validate_header(header)

    # resolve dimension and measure field positions in the header
    dimpos = tuple(_fieldpos(header, d) for d in dims)
    mpos = tuple(None if f is None else _fieldpos(header, f)
                 for (_, _, f) in mspecs)

    # for each grouping set, the positions of its dimensions, in key order
    setdimpos = tuple(tuple(i for i, d in enumerate(dims) if d in gset)
                      for gset in gsets)

    # aggregate the whole table in a single pass
    state = [{} for _ in gsets]
    for rownum, row in enumerate(it, start=1):
        if len(row) != len(header):
            raise ArgumentError('row %d: expected %d fields, found %d'
                                % (rownum, len(header), len(row)))
        dimvals = tuple(_check_dimension_value(dims[i], row[dimpos[i]],
                                               rownum)
                        for i in range(len(dims)))
        mvals = tuple(_extract_measure_value(mspecs[i], mpos[i], row,
                                             rownum)
                      for i in range(len(mspecs)))
        for si, positions in enumerate(setdimpos):
            gkey = tuple(dimvals[p] for p in positions)
            groups = state[si]
            accums = groups.get(gkey)
            if accums is None:
                accums = [_new_accumulator(t) for t in mtypes]
                groups[gkey] = accums
            for mi, mtype in enumerate(mtypes):
                value = mvals[mi]
                if value is not None:
                    _add_value(accums, mi, mtype, value)

    return GroupState(dims, gsets, mspecs, batches, state)


def groupingsets(table, key, sets, measures):
    """
    Aggregate `table` over several grouping sets and return the result as a
    table. This is a convenience shortcut equivalent to
    ``groupstate(table, key, sets, measures, batch).totable()`` for the
    case where the result is computed in one go and never merged. See
    :func:`groupstate` for the full documentation of the arguments, the
    measure semantics and the output layout. E.g.::

        >>> import petl as etl
        >>> table1 = [['region', 'amount'],
        ...           ['east', 3],
        ...           ['east', 7],
        ...           ['west', 4]]
        >>> etl.groupingsets(table1, key='region',
        ...                  sets=[('region',), ()],
        ...                  measures=[('total', 'sum', 'amount'),
        ...                            ('n', 'count', None)])
        +--------+----------+-------+---+
        | region | grouping | total | n |
        +========+==========+=======+===+
        | 'east' |        0 |    10 | 2 |
        +--------+----------+-------+---+
        | 'west' |        0 |     4 | 1 |
        +--------+----------+-------+---+
        | None   |        1 |    14 | 3 |
        +--------+----------+-------+---+

    The returned table can be iterated repeatedly with identical results.

    """

    return groupstate(table, key, sets, measures,
                      batch=_DEFAULT_BATCH).totable()


def loadgroupstate(data):
    """
    Restore a :class:`GroupState` from `data`, which must be ``bytes``
    produced by :meth:`GroupState.dumps` (possibly in another process).
    The restored state behaves exactly like the original, both in
    :meth:`GroupState.totable` results and in which states it can be merged
    with. Data that was not produced by :meth:`GroupState.dumps`, has an
    unsupported format version, or is truncated or otherwise malformed is
    reported as :class:`petl.errors.ArgumentError`. The format is a
    self-describing byte format; loading never executes arbitrary code
    (it is not based on :mod:`pickle`).

    """

    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise ArgumentError('expected bytes, found %s'
                            % type(data).__name__)
    if len(data) < _HEADER_LEN:
        raise ArgumentError('not a serialized GroupState: data too short')
    if not data.startswith(_MAGIC):
        raise ArgumentError('not a serialized GroupState: bad magic')
    version = data[len(_MAGIC)]
    if version != _FORMAT_VERSION:
        raise ArgumentError('unsupported GroupState format version: %r'
                            % (version,))
    (length,) = struct.unpack('>Q', data[len(_MAGIC) + 1:_HEADER_LEN])
    payload = data[_HEADER_LEN:]
    if len(payload) != length:
        raise ArgumentError('not a serialized GroupState: truncated or '
                            'corrupt data')
    try:
        obj = json.loads(payload.decode('utf-8'))
    except ValueError:
        raise ArgumentError('not a serialized GroupState: malformed '
                            'payload')
    return _state_from_payload(obj)


class GroupState(object):
    """
    An immutable partial aggregation result produced by :func:`groupstate`
    or :func:`loadgroupstate`, covering a specific set of batches of
    detail rows.

    A state can be rendered as a table with :meth:`totable`, merged with
    other states with :meth:`merge`, and serialized to bytes with
    :meth:`dumps`.

    Two states can only be merged if they were built with exactly the same
    `key` and `measures`, and with grouping sets having the same members
    (the order in which the sets were given does not matter); otherwise
    :meth:`merge` raises :class:`petl.errors.ArgumentError`. In addition,
    the batches covered by the states being merged must be pairwise
    disjoint - merging states that share a batch raises
    :class:`petl.errors.ArgumentError` and leaves all inputs unchanged.

    Merging never modifies any of the inputs, and the resulting state can
    be merged further. No matter in which order or tree shape the states
    of a collection of batches are merged, the final :meth:`totable`
    result is identical - bit for bit, including ``int`` versus ``float``
    types - to aggregating the concatenated detail rows of all those
    batches directly with :func:`groupstate`.

    """

    def __init__(self, dims, gsets, mspecs, batches, state):
        self._dims = tuple(dims)
        self._sets = tuple(gsets)
        self._measures = tuple(mspecs)
        self._batches = frozenset(batches)
        self._state = state

    @property
    def key(self):
        """The dimension field names, as a tuple, in key order."""
        return self._dims

    @property
    def sets(self):
        """The grouping sets, as a tuple of frozensets of dimension names,
        in the order the sets were given."""
        return self._sets

    @property
    def measures(self):
        """The measures, as a tuple of ``(name, type, field)`` triples."""
        return self._measures

    @property
    def batches(self):
        """The set of batch labels covered by this state."""
        return self._batches

    def __repr__(self):
        return 'GroupState(key=%r, sets=%r, measures=%r, batches=%r)' % (
            self._dims, self._sets, self._measures,
            sorted(self._batches))

    def merge(self, *others):
        """
        Return a new :class:`GroupState` combining this state with
        `others`. All states must have been built with the same `key` and
        `measures`, and with grouping sets having the same members, and
        their batches must be pairwise disjoint; otherwise
        :class:`petl.errors.ArgumentError` is raised. None of the inputs
        is modified, and the result can be merged further.

        """

        for other in others:
            if not isinstance(other, GroupState):
                raise ArgumentError('expected a GroupState, found %r'
                                    % (other,))
            if other._dims != self._dims:
                raise ArgumentError('cannot merge states with different '
                                    'keys: %r vs %r'
                                    % (self._dims, other._dims))
            if other._measures != self._measures:
                raise ArgumentError('cannot merge states with different '
                                    'measures: %r vs %r'
                                    % (self._measures, other._measures))
            if frozenset(other._sets) != frozenset(self._sets):
                raise ArgumentError('cannot merge states with different '
                                    'grouping sets: %r vs %r'
                                    % (self._sets, other._sets))

        # batches of all inputs must be pairwise disjoint
        seen = set(self._batches)
        for other in others:
            overlap = seen & other._batches
            if overlap:
                raise ArgumentError('cannot merge states with overlapping '
                                    'batches: %r' % (sorted(overlap),))
            seen |= other._batches

        mtypes = tuple(m[1] for m in self._measures)

        # copy this state's groups
        merged = []
        for si in range(len(self._sets)):
            groups = {}
            for gkey, accums in self._state[si].items():
                groups[gkey] = [_copy_accumulator(a, t)
                                for a, t in zip(accums, mtypes)]
            merged.append(groups)

        # fold in the other states, matching grouping sets by membership
        for other in others:
            othermap = dict(zip(other._sets, other._state))
            for si, gset in enumerate(self._sets):
                groups = merged[si]
                for gkey, accums in othermap[gset].items():
                    existing = groups.get(gkey)
                    if existing is None:
                        groups[gkey] = [_copy_accumulator(a, t)
                                        for a, t in zip(accums, mtypes)]
                    else:
                        _merge_accumulators(existing, accums, mtypes)

        return GroupState(self._dims, self._sets, self._measures,
                          frozenset(seen), merged)

    def totable(self):
        """
        Return the aggregated result as a table. The header is the
        dimension names, then ``'grouping'``, then the measure names. Rows
        are emitted grouping set by grouping set, in the order the sets
        were given; within each set there is one row per combination of
        dimension values seen in the data, sorted ascending by the set's
        dimension values using petl's sort comparison semantics (``None``
        sorts first). Dimensions that are not part of a set are ``None``
        in its rows. The empty grouping set always yields exactly one
        grand total row, even when there is no data at all; a non-empty
        set yields no rows when there is no data.

        The ``grouping`` column is an integer bit mask: with ``n``
        dimensions, bit ``2 ** (n - 1 - i)`` is set when the ``i``-th
        dimension of `key` (counting from 0) has been aggregated away. So
        a detail row of the full grouping set has ``grouping == 0``, and
        the grand total row of the empty set has all ``n`` bits set. This
        distinguishes a genuine ``None`` dimension value from a dimension
        that was aggregated away.

        The returned table is a view over this state and can be iterated
        repeatedly with identical results.

        """

        return GroupStateTable(self._dims, self._sets, self._measures,
                               self._state)

    def dumps(self):
        """
        Serialize this state to ``bytes``, which can be restored with
        :func:`loadgroupstate`, also in another process. The restored
        state is indistinguishable from the original, both in
        :meth:`totable` results and in which states it can be merged with.

        """

        mtypes = tuple(m[1] for m in self._measures)
        groups = []
        for si in range(len(self._sets)):
            entries = []
            ordered = sorted(self._state[si].items(),
                             key=lambda kv: Comparable(kv[0]))
            for gkey, accums in ordered:
                entries.append([list(gkey),
                                [_dump_accumulator(a, t)
                                 for a, t in zip(accums, mtypes)]])
            groups.append(entries)
        payload = {
            'key': list(self._dims),
            'sets': [[d for d in self._dims if d in gset]
                     for gset in self._sets],
            'measures': [list(m) for m in self._measures],
            'batches': sorted(self._batches),
            'groups': groups,
        }
        data = json.dumps(payload, ensure_ascii=True,
                          separators=(',', ':')).encode('utf-8')
        return _MAGIC + bytes([_FORMAT_VERSION]) \
            + struct.pack('>Q', len(data)) + data


class GroupStateTable(Table):

    def __init__(self, dims, gsets, mspecs, state):
        self._dims = dims
        self._sets = gsets
        self._measures = mspecs
        self._state = state

    def __iter__(self):
        dims = self._dims
        n = len(dims)
        mtypes = tuple(m[1] for m in self._measures)
        names = tuple(m[0] for m in self._measures)
        yield dims + (_GROUPING_FIELD,) + names
        for si, gset in enumerate(self._sets):
            positions = tuple(i for i, d in enumerate(dims) if d in gset)
            grouping = 0
            for i in range(n):
                if dims[i] not in gset:
                    grouping |= 2 ** (n - 1 - i)
            groups = self._state[si]
            if not gset and not groups:
                # the empty grouping set always yields a grand total row
                accums = [_new_accumulator(t) for t in mtypes]
                yield _output_row(n, (), positions, grouping, accums,
                                  mtypes)
                continue
            for gkey in sorted(groups, key=Comparable):
                yield _output_row(n, gkey, positions, grouping,
                                  groups[gkey], mtypes)


def _output_row(n, gkey, positions, grouping, accums, mtypes):
    row = [None] * n
    for pos, val in zip(positions, gkey):
        row[pos] = val
    row.append(grouping)
    for a, t in zip(accums, mtypes):
        row.append(_render_accumulator(a, t))
    return tuple(row)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _norm_float(value):
    # treat -0.0 as 0.0
    return 0.0 if value == 0.0 else value


def _normalize_key(key):
    if isinstance(key, string_types):
        dims = (key,)
    elif isinstance(key, (list, tuple)):
        dims = tuple(key)
    else:
        raise ArgumentError('key must be a field name or a non-empty '
                            'sequence of field names, found %r' % (key,))
    if len(dims) == 0:
        raise ArgumentError('key must contain at least one dimension')
    for d in dims:
        if not isinstance(d, string_types):
            raise ArgumentError('key dimensions must be field name '
                                'strings, found %r' % (d,))
    if len(set(dims)) != len(dims):
        raise ArgumentError('key dimensions must be unique, found %r'
                            % (key,))
    return dims


def _normalize_sets(sets, dims):
    if isinstance(sets, string_types) \
            or not isinstance(sets, (list, tuple)) or len(sets) == 0:
        raise ArgumentError('sets must be a non-empty sequence of '
                            'grouping sets, found %r' % (sets,))
    dimset = set(dims)
    normed = []
    seen = set()
    for s in sets:
        if isinstance(s, string_types):
            members = (s,)
        elif isinstance(s, (list, tuple, set, frozenset)):
            members = tuple(s)
        else:
            raise ArgumentError('each grouping set must be a sequence of '
                                'dimension names, found %r' % (s,))
        for m in members:
            if m not in dimset:
                raise ArgumentError('grouping set member %r is not a '
                                    'dimension of key %r' % (m, dims))
        fs = frozenset(members)
        if len(fs) != len(members):
            raise ArgumentError('duplicate dimension in grouping set %r'
                                % (s,))
        if fs in seen:
            raise ArgumentError('duplicate grouping set %r' % (s,))
        seen.add(fs)
        normed.append(fs)
    return tuple(normed)


def _normalize_measures(measures, dims):
    if isinstance(measures, string_types) \
            or not isinstance(measures, (list, tuple)) \
            or len(measures) == 0:
        raise ArgumentError('measures must be a non-empty sequence of '
                            '(name, type, field) triples, found %r'
                            % (measures,))
    reserved = set(dims) | {_GROUPING_FIELD}
    normed = []
    names = set()
    for m in measures:
        if not isinstance(m, (list, tuple)) or len(m) != 3:
            raise ArgumentError('each measure must be a (name, type, '
                                'field) triple, found %r' % (m,))
        name, mtype, field = m
        if not isinstance(name, string_types):
            raise ArgumentError('measure name must be a string, found %r'
                                % (name,))
        if name in names:
            raise ArgumentError('duplicate measure name %r' % (name,))
        if name in reserved:
            raise ArgumentError('measure name %r conflicts with a '
                                'dimension name or %r'
                                % (name, _GROUPING_FIELD))
        if mtype not in _MEASURE_TYPES:
            raise ArgumentError('invalid measure type %r, expected one of '
                                '%r' % (mtype, _MEASURE_TYPES))
        if field is None:
            if mtype != _COUNT:
                raise ArgumentError('measure %r of type %r requires a '
                                    'field' % (name, mtype))
        elif not isinstance(field, string_types):
            raise ArgumentError('measure field must be a field name '
                                'string or None, found %r' % (field,))
        names.add(name)
        normed.append((name, mtype, field))
    return tuple(normed)


def _normalize_batch(batch):
    if not isinstance(batch, string_types) or not batch:
        raise ArgumentError('batch must be a non-empty string, found %r'
                            % (batch,))
    return batch


def _validate_header(header):
    for h in header:
        if not isinstance(h, string_types):
            raise ArgumentError('header fields must be strings, found %r'
                                % (h,))
    if len(set(header)) != len(header):
        raise ArgumentError('header fields must be unique, found %r'
                            % (header,))


def _fieldpos(header, field):
    try:
        return header.index(field)
    except ValueError:
        raise FieldSelectionError(field)


def _check_dimension_value(name, value, rownum):
    if value is None or isinstance(value, string_types) or _is_int(value):
        return value
    raise ArgumentError('row %d: dimension %r has value %r of type %s, '
                        'expected str, int or None'
                        % (rownum, name, value, type(value).__name__))


def _extract_measure_value(spec, pos, row, rownum):
    name, mtype, _field = spec
    if mtype == _COUNT:
        # count rows (no field) or count non-None values (field given);
        # values are not type-checked
        if pos is None:
            return 1
        return None if row[pos] is None else 1
    value = row[pos]
    if value is None:
        return None
    if mtype in (_SUM, _MEAN, _MIN, _MAX):
        if _is_int(value):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return _norm_float(value)
        raise ArgumentError('row %d: measure %r has value %r of type %s, '
                            'expected a finite int or float'
                            % (rownum, name, value, type(value).__name__))
    # countdistinct
    if _is_int(value) or isinstance(value, string_types):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return _norm_float(value)
    raise ArgumentError('row %d: measure %r has value %r of type %s, '
                        'expected str or a finite int or float'
                        % (rownum, name, value, type(value).__name__))


def _new_accumulator(mtype):
    if mtype == _COUNT:
        return 0
    if mtype in (_SUM, _MEAN):
        # number of values, number of float values, exact int sum,
        # exact float sum
        return [0, 0, 0, Fraction(0)]
    if mtype in (_MIN, _MAX):
        return None
    if mtype == _COUNTDISTINCT:
        return set()
    raise AssertionError(mtype)  # pragma: no cover


def _add_value(accums, mi, mtype, value):
    a = accums[mi]
    if mtype == _COUNT:
        accums[mi] = a + 1
    elif mtype in (_SUM, _MEAN):
        a[0] += 1
        if isinstance(value, float):
            a[1] += 1
            a[3] += Fraction(value)
        else:
            a[2] += value
    elif mtype == _MIN:
        if a is None or value < a \
                or (value == a and isinstance(a, float) and _is_int(value)):
            accums[mi] = value
    elif mtype == _MAX:
        if a is None or value > a \
                or (value == a and isinstance(a, float) and _is_int(value)):
            accums[mi] = value
    else:  # countdistinct
        a.add(value)


def _copy_accumulator(a, mtype):
    if mtype in (_SUM, _MEAN):
        return list(a)
    if mtype == _COUNTDISTINCT:
        return set(a)
    return a  # ints, floats and None are immutable


def _merge_extreme(a, b, ismin):
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        # an int/float tie resolves to the int
        if _is_int(a):
            return a
        if _is_int(b):
            return b
        return a
    if ismin:
        return min(a, b)
    return max(a, b)


def _merge_accumulators(dst, src, mtypes):
    for mi, mtype in enumerate(mtypes):
        a = dst[mi]
        b = src[mi]
        if mtype == _COUNT:
            dst[mi] = a + b
        elif mtype in (_SUM, _MEAN):
            a[0] += b[0]
            a[1] += b[1]
            a[2] += b[2]
            a[3] += b[3]
        elif mtype == _MIN:
            dst[mi] = _merge_extreme(a, b, True)
        elif mtype == _MAX:
            dst[mi] = _merge_extreme(a, b, False)
        else:  # countdistinct
            dst[mi] = a | b


def _render_accumulator(a, mtype):
    if mtype == _COUNT:
        return a
    if mtype == _SUM:
        n, nf, isum, fsum = a
        if n == 0:
            return None
        if nf == 0:
            return isum
        # exact sum rounded only once; raises OverflowError if the
        # result does not fit in a float
        return float(Fraction(isum) + fsum)
    if mtype == _MEAN:
        n, nf, isum, fsum = a
        if n == 0:
            return None
        return float((Fraction(isum) + fsum) / n)
    if mtype in (_MIN, _MAX):
        return a
    return len(a)  # countdistinct


def _distinct_sort_key(value):
    # deterministic order for serialized distinct values: numbers first
    # (by value), then strings
    if isinstance(value, string_types):
        return (1, value)
    return (0, value)


def _dump_accumulator(a, mtype):
    if mtype == _COUNT:
        return a
    if mtype in (_SUM, _MEAN):
        n, nf, isum, fsum = a
        return [n, nf, isum, [fsum.numerator, fsum.denominator]]
    if mtype in (_MIN, _MAX):
        return a
    return sorted(a, key=_distinct_sort_key)  # countdistinct


def _state_from_payload(obj):
    if not isinstance(obj, dict):
        raise ArgumentError('malformed GroupState payload: expected an '
                            'object')
    try:
        key = obj['key']
        sets = obj['sets']
        measures = obj['measures']
        batches = obj['batches']
        groups = obj['groups']
    except (KeyError, TypeError):
        raise ArgumentError('malformed GroupState payload: missing '
                            'section')
    dims = _normalize_key(key)
    gsets = _normalize_sets(sets, dims)
    mspecs = _normalize_measures(measures, dims)
    mtypes = tuple(m[1] for m in mspecs)
    if not isinstance(batches, list) or len(batches) == 0:
        raise ArgumentError('malformed GroupState payload: invalid '
                            'batches')
    for b in batches:
        if not isinstance(b, string_types) or not b:
            raise ArgumentError('malformed GroupState payload: invalid '
                                'batch %r' % (b,))
    if len(set(batches)) != len(batches):
        raise ArgumentError('malformed GroupState payload: duplicate '
                            'batches')
    if not isinstance(groups, list) or len(groups) != len(gsets):
        raise ArgumentError('malformed GroupState payload: invalid '
                            'groups')
    state = []
    for si, entries in enumerate(groups):
        if not isinstance(entries, list):
            raise ArgumentError('malformed GroupState payload: invalid '
                                'groups')
        gmap = {}
        expected_len = len(gsets[si])
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2 \
                    or not isinstance(entry[0], list) \
                    or not isinstance(entry[1], list) \
                    or len(entry[0]) != expected_len \
                    or len(entry[1]) != len(mspecs):
                raise ArgumentError('malformed GroupState payload: '
                                    'invalid group entry')
            gkey = tuple(_load_dimension_value(v) for v in entry[0])
            if gkey in gmap:
                raise ArgumentError('malformed GroupState payload: '
                                    'duplicate group key')
            gmap[gkey] = [_load_accumulator(v, t)
                          for v, t in zip(entry[1], mtypes)]
        state.append(gmap)
    return GroupState(dims, gsets, mspecs, frozenset(batches), state)


def _load_dimension_value(value):
    if value is None or isinstance(value, string_types) or _is_int(value):
        return value
    raise ArgumentError('malformed GroupState payload: invalid dimension '
                        'value %r' % (value,))


def _load_measure_value(value):
    if _is_int(value) or isinstance(value, string_types):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return _norm_float(value)
    raise ArgumentError('malformed GroupState payload: invalid measure '
                        'value %r' % (value,))


def _load_accumulator(value, mtype):
    if mtype == _COUNT:
        if _is_int(value) and value >= 0:
            return value
        raise ArgumentError('malformed GroupState payload: invalid count '
                            '%r' % (value,))
    if mtype in (_SUM, _MEAN):
        ok = isinstance(value, list) and len(value) == 4 \
            and all(_is_int(v) and v >= 0 for v in value[:2]) \
            and _is_int(value[2]) \
            and isinstance(value[3], list) and len(value[3]) == 2 \
            and all(_is_int(v) for v in value[3]) and value[3][1] > 0
        if not ok:
            raise ArgumentError('malformed GroupState payload: invalid '
                                'sum accumulator %r' % (value,))
        return [value[0], value[1], value[2],
                Fraction(value[3][0], value[3][1])]
    if mtype in (_MIN, _MAX):
        if value is None:
            return None
        return _load_measure_value(value)
    # countdistinct
    if not isinstance(value, list):
        raise ArgumentError('malformed GroupState payload: invalid '
                            'distinct values %r' % (value,))
    return {_load_measure_value(v) for v in value}


Table.groupstate = groupstate
Table.groupingsets = groupingsets
