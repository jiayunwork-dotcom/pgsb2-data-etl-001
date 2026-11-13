from __future__ import absolute_import, print_function, division


import itertools
import collections
import heapq
import logging
import operator
import os
from collections import namedtuple
from tempfile import NamedTemporaryFile
from petl.compat import next, text_type, string_types, pickle


import petl.config as config
from petl.errors import ArgumentError
from petl.comparison import Comparable, comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
    header, data, asindices
from petl.transform.sorts import sort


logger = logging.getLogger(__name__)


def melt(table, key=None, variables=None, variablefield='variable',
         valuefield='value'):
    """
    Reshape a table, melting fields into data. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'gender', 'age'],
        ...           [1, 'F', 12],
        ...           [2, 'M', 17],
        ...           [3, 'M', 16]]
        >>> table2 = etl.melt(table1, 'id')
        >>> table2.lookall()
        +----+----------+-------+
        | id | variable | value |
        +====+==========+=======+
        |  1 | 'gender' | 'F'   |
        +----+----------+-------+
        |  1 | 'age'    |    12 |
        +----+----------+-------+
        |  2 | 'gender' | 'M'   |
        +----+----------+-------+
        |  2 | 'age'    |    17 |
        +----+----------+-------+
        |  3 | 'gender' | 'M'   |
        +----+----------+-------+
        |  3 | 'age'    |    16 |
        +----+----------+-------+

        >>> # compound keys are supported
        ... table3 = [['id', 'time', 'height', 'weight'],
        ...           [1, 11, 66.4, 12.2],
        ...           [2, 16, 53.2, 17.3],
        ...           [3, 12, 34.5, 9.4]]
        >>> table4 = etl.melt(table3, key=['id', 'time'])
        >>> table4.lookall()
        +----+------+----------+-------+
        | id | time | variable | value |
        +====+======+==========+=======+
        |  1 |   11 | 'height' |  66.4 |
        +----+------+----------+-------+
        |  1 |   11 | 'weight' |  12.2 |
        +----+------+----------+-------+
        |  2 |   16 | 'height' |  53.2 |
        +----+------+----------+-------+
        |  2 |   16 | 'weight' |  17.3 |
        +----+------+----------+-------+
        |  3 |   12 | 'height' |  34.5 |
        +----+------+----------+-------+
        |  3 |   12 | 'weight' |   9.4 |
        +----+------+----------+-------+

        >>> # a subset of variable fields can be selected
        ... table5 = etl.melt(table3, key=['id', 'time'],
        ...                   variables=['height'])
        >>> table5.lookall()
        +----+------+----------+-------+
        | id | time | variable | value |
        +====+======+==========+=======+
        |  1 |   11 | 'height' |  66.4 |
        +----+------+----------+-------+
        |  2 |   16 | 'height' |  53.2 |
        +----+------+----------+-------+
        |  3 |   12 | 'height' |  34.5 |
        +----+------+----------+-------+

    See also :func:`petl.transform.reshape.recast`.

    """

    return MeltView(table, key=key, variables=variables,
                    variablefield=variablefield,
                    valuefield=valuefield)


Table.melt = melt


class MeltView(Table):

    def __init__(self, source, key=None, variables=None,
                 variablefield='variable', valuefield='value'):
        self.source = source
        self.key = key
        self.variables = variables
        self.variablefield = variablefield
        self.valuefield = valuefield

    def __iter__(self):
        return itermelt(self.source, self.key, self.variables,
                        self.variablefield, self.valuefield)


def itermelt(source, key, variables, variablefield, valuefield):
    if key is None and variables is None:
        raise ValueError('either key or variables must be specified')

    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return

    # determine key and variable field indices
    key_indices = variables_indices = None
    if key is not None:
        key_indices = asindices(hdr, key)
    if variables is not None:
        if not isinstance(variables, (list, tuple)):
            variables = (variables,)
        variables_indices = asindices(hdr, variables)

    if key is None:
        # assume key is fields not in variables
        selected = {i + len(hdr) if i < 0 else i for i in variables_indices}
        key_indices = [i for i in range(len(hdr))
                       if i not in selected]
    if variables is None:
        # assume variables are fields not in key
        selected = {i + len(hdr) if i < 0 else i for i in key_indices}
        variables_indices = [i for i in range(len(hdr))
                             if i not in selected]
        variables = [hdr[i] for i in variables_indices]

    getkey = rowgetter(*key_indices)

    # determine the output fields
    outhdr = [hdr[i] for i in key_indices]
    outhdr.append(variablefield)
    outhdr.append(valuefield)
    yield tuple(outhdr)

    # construct the output data
    for row in it:
        k = getkey(row)
        for v, i in zip(variables, variables_indices):
            try:
                o = list(k)  # populate with key values initially
                o.append(v)  # add variable
                o.append(row[i])  # add value
                yield tuple(o)
            except IndexError:
                # row is missing this value, and melt() should yield no row
                pass


def recast(table, key=None, variablefield='variable', valuefield='value',
           samplesize=1000, reducers=None, missing=None):
    """
    Recast molten data. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'variable', 'value'],
        ...           [3, 'age', 16],
        ...           [1, 'gender', 'F'],
        ...           [2, 'gender', 'M'],
        ...           [2, 'age', 17],
        ...           [1, 'age', 12],
        ...           [3, 'gender', 'M']]
        >>> table2 = etl.recast(table1)
        >>> table2
        +----+-----+--------+
        | id | age | gender |
        +====+=====+========+
        |  1 |  12 | 'F'    |
        +----+-----+--------+
        |  2 |  17 | 'M'    |
        +----+-----+--------+
        |  3 |  16 | 'M'    |
        +----+-----+--------+

        >>> # specifying variable and value fields
        ... table3 = [['id', 'vars', 'vals'],
        ...           [3, 'age', 16],
        ...           [1, 'gender', 'F'],
        ...           [2, 'gender', 'M'],
        ...           [2, 'age', 17],
        ...           [1, 'age', 12],
        ...           [3, 'gender', 'M']]
        >>> table4 = etl.recast(table3, variablefield='vars', valuefield='vals')
        >>> table4
        +----+-----+--------+
        | id | age | gender |
        +====+=====+========+
        |  1 |  12 | 'F'    |
        +----+-----+--------+
        |  2 |  17 | 'M'    |
        +----+-----+--------+
        |  3 |  16 | 'M'    |
        +----+-----+--------+

        >>> # if there are multiple values for each key/variable pair, and no
        ... # reducers function is provided, then all values will be listed
        ... table6 = [['id', 'time', 'variable', 'value'],
        ...           [1, 11, 'weight', 66.4],
        ...           [1, 14, 'weight', 55.2],
        ...           [2, 12, 'weight', 53.2],
        ...           [2, 16, 'weight', 43.3],
        ...           [3, 12, 'weight', 34.5],
        ...           [3, 17, 'weight', 49.4]]
        >>> table7 = etl.recast(table6, key='id')
        >>> table7
        +----+--------------+
        | id | weight       |
        +====+==============+
        |  1 | [66.4, 55.2] |
        +----+--------------+
        |  2 | [53.2, 43.3] |
        +----+--------------+
        |  3 | [34.5, 49.4] |
        +----+--------------+

        >>> # multiple values can be reduced via an aggregation function
        ... def mean(values):
        ...     return float(sum(values)) / len(values)
        ...
        >>> table8 = etl.recast(table6, key='id', reducers={'weight': mean})
        >>> table8
        +----+--------------------+
        | id | weight             |
        +====+====================+
        |  1 | 60.800000000000004 |
        +----+--------------------+
        |  2 |              48.25 |
        +----+--------------------+
        |  3 |              41.95 |
        +----+--------------------+

        >>> # missing values are padded with whatever is provided via the
        ... # missing keyword argument (None by default)
        ... table9 = [['id', 'variable', 'value'],
        ...           [1, 'gender', 'F'],
        ...           [2, 'age', 17],
        ...           [1, 'age', 12],
        ...           [3, 'gender', 'M']]
        >>> table10 = etl.recast(table9, key='id')
        >>> table10
        +----+------+--------+
        | id | age  | gender |
        +====+======+========+
        |  1 |   12 | 'F'    |
        +----+------+--------+
        |  2 |   17 | None   |
        +----+------+--------+
        |  3 | None | 'M'    |
        +----+------+--------+

    Note that the table is scanned once to discover variables, then a second
    time to reshape the data and recast variables as fields. How many rows are
    scanned in the first pass is determined by the `samplesize` argument.
    Discovered variable names are sorted using PETL's mixed-type ordering,
    so names such as integers and strings can be recast together. Names
    explicitly supplied in a `variablefield` dictionary retain their order.

    When the result is iterated, invalid key, variable or value field names
    raise :class:`petl.errors.ArgumentError`. The value field must exist in the
    input header and must not also be selected as a key or variable field.
    This also applies to an empty table with no fields; define the expected
    header before recasting it.

    The sorted rows and discovered fields are cached after the first data
    read. Later iterations use this cache, as with :func:`sort`. Create a new
    recast view to read the source again. Cached rows are shallow tuple
    copies, so changing fields in a source row does not change the cache.
    Mutable objects stored in cells are not copied by this step. Do not
    change these objects while using the view. Do not change source fields
    while iterators are active.

    See also :func:`petl.transform.reshape.melt`.

    """

    return RecastView(table, key=key, variablefield=variablefield,
                      valuefield=valuefield, samplesize=samplesize,
                      reducers=reducers, missing=missing)


Table.recast = recast


class RecastView(Table):

    def __init__(self, source, key=None, variablefield='variable',
                 valuefield='value', samplesize=1000, reducers=None,
                 missing=None):
        self.source = source
        self.key = key
        self.variablefield = variablefield
        self.valuefield = valuefield
        self.samplesize = samplesize
        if reducers is None:
            self.reducers = dict()
        else:
            self.reducers = reducers
        self.missing = missing
        self._cache = []

    def __iter__(self):
        return iterrecast(self.source, self.key, self.variablefield,
                          self.valuefield, self.samplesize, self.reducers,
                          self.missing, self._cache)


class _RecastSource(object):

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        for row in self.source:
            yield tuple(row)


def _prepare_recast(source, key, variablefield, valuefield, samplesize):
    """Find the output fields and prepare a lazy sort."""

    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))

    # normalise some stuff
    keyfields = key
    variablefields = variablefield  # N.B., could be more than one

    # normalise key fields
    if keyfields and not isinstance(keyfields, (list, tuple)):
        keyfields = (keyfields,)

    # normalise variable fields
    if variablefields:
        if isinstance(variablefields, dict):
            pass  # handle this later
        elif not isinstance(variablefields, (list, tuple)):
            variablefields = (variablefields,)

    # infer key fields
    if not keyfields:
        # assume keyfields is fields not in variables
        keyfields = [f for f in flds
                     if f not in variablefields and f != valuefield]

    # infer key fields
    if not variablefields:
        # assume variables are fields not in keyfields
        variablefields = [f for f in flds
                          if f not in keyfields and f != valuefield]

    # Validate user-supplied fields even when assertions are disabled.
    if valuefield not in flds:
        raise ArgumentError('invalid value field: %s' % valuefield)
    if valuefield in keyfields:
        raise ArgumentError('value field cannot be a key field')
    if valuefield in variablefields:
        raise ArgumentError('value field cannot be a variable field')
    for f in keyfields:
        if f not in flds:
            raise ArgumentError('invalid key field: %s' % f)
    for f in variablefields:
        if f not in flds:
            raise ArgumentError('invalid variable field: %s' % f)

    # we'll need these later
    valueindex = flds.index(valuefield)
    keyindices = [flds.index(f) for f in keyfields]
    variableindices = [flds.index(f) for f in variablefields]

    # determine the actual variable names to be cast as fields
    if isinstance(variablefields, dict):
        # user supplied dictionary
        variables = variablefields
    else:
        variables = collections.defaultdict(set)
        # sample the data to discover variables to be cast as fields
        for row in itertools.islice(it, 0, samplesize):
            for i, f in zip(variableindices, variablefields):
                variables[f].add(row[i])
        for f in variables:
            # turn from sets to sorted lists
            variables[f] = sorted(variables[f], key=Comparable)

    # Keep the sample order when the sorted rows are read again.
    variables = [tuple(variables[f]) for f in variablefields]
    outhdr = tuple(keyfields) + tuple(itertools.chain.from_iterable(variables))
    sorted_source = sort(_RecastSource(source), key=keyfields)
    return (sorted_source, outhdr, keyindices, variableindices, variables,
            valueindex)


def iterrecast(source, key, variablefield, valuefield,
               samplesize, reducers, missing, cache=None):
    if cache:
        prepared = cache[0]
    else:
        prepared = _prepare_recast(source, key, variablefield, valuefield,
                                  samplesize)
        if prepared is None:
            return
    yield prepared[1]

    # Another iterator can fill the cache after this header is read.
    if cache:
        if prepared[1:] != cache[0][1:]:
            raise ArgumentError('recast fields changed during iteration')
        prepared = cache[0]
    source, outhdr, keyindices, variableindices, variables, valueindex = prepared
    it = itertools.islice(source, 1, None)
    # A data read completes the sort. Do not retain a failed or unread sort.
    first = list(itertools.islice(it, 1))
    if cache is not None and not cache:
        cache.append(prepared)
    it = itertools.chain(first, it)
    getsortablekey = comparable_itemgetter(*keyindices)
    getactualkey = operator.itemgetter(*keyindices)

    # process sorted data in newfields
    groups = itertools.groupby(it, key=getsortablekey)
    for _, group in groups:
        # may need to iterate over the group more than once
        group = list(group)
        # N.B., key returned by groupby may be wrapped as SortableItem, we want
        # to output the actual key value, get it from the first row in the group
        key_value = getactualkey(group[0])
        if len(keyindices) > 1:
            out_row = list(key_value)
        else:
            out_row = [key_value]
        for i, field_variables in zip(variableindices, variables):
            for variable in field_variables:
                # collect all values for the current variable
                vals = [r[valueindex] for r in group if r[i] == variable]
                if len(vals) == 0:
                    val = missing
                elif len(vals) == 1:
                    val = vals[0]
                else:
                    if variable in reducers:
                        redu = reducers[variable]
                    else:
                        redu = list  # list all values
                    val = redu(vals)
                out_row.append(val)
        yield tuple(out_row)


def recastby(table, key, variablefield, valuefield, fields,
             duplicates='error', unknown='error', presorted=False,
             buffersize=None, tempdir=None, missing=None):
    """
    Recast molten data into a fixed set of fields, with explicitly controlled
    behaviour for repeated and unknown variables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'variable', 'value'],
        ...           [1, 'age', 12],
        ...           [1, 'gender', 'F'],
        ...           [2, 'age', 17],
        ...           [2, 'gender', 'M'],
        ...           [1, 'age', 13]]
        >>> table2 = list(etl.recastby(table1, 'id', 'variable', 'value',
        ...                            ['age', 'gender']))
        Traceback (most recent call last):
            ...
        petl.errors.ArgumentError: argument error: duplicate indicator 'age' \
for key 1 at data row 5 (first seen at data row 1)

        >>> table3 = etl.recastby(table1, 'id', 'variable', 'value',
        ...                       ['age', 'gender'], duplicates='last')
        >>> table3
        +----+-----+--------+
        | id | age | gender |
        +====+=====+========+
        |  1 |   13 | 'F'    |
        +----+-----+--------+
        |  2 |   17 | 'M'    |
        +----+-----+--------+

        >>> # unknown variables are rejected by default ...
        ... table4 = [['id', 'variable', 'value'],
        ...           [1, 'age', 12],
        ...           [1, 'colour', 'blue']]
        >>> etl.recastby(table4, 'id', 'variable', 'value', ['age']).look()
        Traceback (most recent call last):
            ...
        petl.errors.ArgumentError: argument error: unknown indicator 'colour' \
for key 1 at data row 2

        >>> # ... but can be ignored
        ... table5 = etl.recastby(table4, 'id', 'variable', 'value', ['age'],
        ...                        unknown='ignore')
        >>> table5
        +----+-----+
        | id | age |
        +====+=====+
        |  1 |  12 |
        +----+-----+

        >>> # compound keys and a missing value are supported
        ... table6 = [['id', 'time', 'variable', 'value'],
        ...           [1, 11, 'height', 66.4],
        ...           [1, 11, 'weight', 12.2],
        ...           [2, 12, 'height', 53.2]]
        >>> table7 = etl.recastby(table6, ['id', 'time'], 'variable', 'value',
        ...                        ['height', 'weight'])
        >>> table7
        +----+------+--------+--------+
        | id | time | height | weight |
        +====+======+========+========+
        |  1 |   11 |   66.4 |   12.2 |
        +----+------+--------+--------+
        |  2 |   12 |   53.2 |   None |
        +----+------+--------+--------+

    The `key` argument is a field name or a non-empty sequence of field names.
    `variablefield` and `valuefield` are single field names. All three must
    exist in the input header and must not overlap. The `fields` argument is
    a non-empty sequence of unique field names giving the variables to be
    recast and the order of the output columns. A single field name may also
    be given. Each name must be a string and must not clash with a key output
    field name. `fields` may be a one-shot iterator: it is not consumed when
    the view is created or `iter()` is called, only when the first result
    row is actually requested. The resulting column order is then reused for
    later iterations of the same view.

    Exactly one output row is produced for each distinct key, in the order
    given on the command line, followed by one column per name in `fields`.
    When the input is not presorted, output keys are ordered using PETL's
    usual mixed-type comparison semantics (see :class:`petl.comparison.
    Comparable`), so `None` can be used as a key value. When the input is
    presorted, keys must already be in ascending order; a key that goes
    backwards raises :class:`petl.errors.ArgumentError` reporting the original
    data row number (data rows are numbered from 1).

    When the same key and variable occur more than once, `duplicates` controls
    the result: `'error'` (the default) raises :class:`petl.errors.ArgumentError`,
    `'first'` keeps the first value in the original input order, and `'last'`
    keeps the last value in the original input order. `None` is treated as a
    real value rather than as an unset marker.

    Variables not named in `fields` are handled according to `unknown`:
    `'error'` (the default) raises :class:`petl.errors.ArgumentError`, while
    `'ignore'` skips such rows. A key that only has unknown variables is still
    emitted as a row, with every output variable column set to `missing`.

    By default the table is sorted by key. `buffersize` limits the number of
    rows held in memory while sorting and `tempdir` selects the directory used
    for temporary spill files; with a `buffersize` of `None`, the value of
    ``petl.config.sort_buffersize`` is used (and `None` there means sort
    entirely in memory). Rows read from the source retain their original input
    order after spilling and merging, so `duplicates='first'` and
    `duplicates='last'` are stable. The sort is performed for each result
    iteration; results are not cached across iterations. This means a
    re-readable source yields consistent results, but a one-shot source is
    not made replayable.

    With `presorted=True`, each input row is consumed exactly once and only
    one row needs to be held in memory while a group is being assembled. The
    previous group is emitted as soon as the first row of the next group has
    been read. With a fixed number of output columns, memory used for a group
    does not grow with the number of its input rows.

    Iterating the source is lazy: constructing the view and calling `iter()`
    do not read the source or consume `fields`. Each result iteration requests
    one iterator from the source. The source iterator is closed, and any
    temporary sort files are removed, when the result is exhausted, when an
    exception occurs, or when the result iterator is explicitly closed.
    Cleanup never replaces an exception raised by the source. Do not change
    source fields while result iterators are active. Rows are copied as
    shallow tuples, as with :func:`petl.transform.reshape.recast`; mutable
    objects stored in cells are not copied.

    This function complements :func:`petl.transform.reshape.recast`, whose
    signature, sampling and `reducers` behaviour are unchanged.

    Performance and compatibility trade-offs:

    * With `presorted=True` the transform is streaming: it needs O(len(key) +
      len(fields)) extra memory per group aside from the one next row held to
      detect the group boundary, and it avoids all sorting and spilling. This
      is the recommended mode for large inputs already ordered by key. It
      trusts the caller's ordering, so it additionally checks for a key that
      goes backwards and reports it with the original data row number. A
      duplicated or unknown indicator is still reported with its original
      data row number, but only while its own group is held; the group is
      emitted before any later group is read.

    * Without `presorted`, every data row is read and (unless it fits in the
      sort buffer) spilled to disk before any result row after the header can
      be produced. The whole input is traversed once, so duplicates and
      unknown indicators anywhere in the table are diagnosed with their
      original row numbers, and a backwards key is not an error because the
      transform sorts. Memory is bounded by `buffersize` (plus one group's
      state during the merge). The cost is latency, not extra source passes:
      the source is read exactly once per result iteration, but the first
      data row is delayed until the input has been sorted.

    * Spilled chunks are merged in several passes with a bounded fan-in
      (approximately `buffersize`, capped at a small constant). A merge
      therefore keeps only a fixed number of run files open and a fixed
      number of buffered records, no matter how many chunks the input
      produced. Each run file is opened lazily and closed and unlinked as
      soon as it has been consumed; intermediate merge runs are removed in
      the same way, so the temporary directory does not accumulate per-pass
      copies of the data. The total number of temporary files may grow with
      the input size, but the number open at any time and the number of
      retained rows do not.

    * Unlike :func:`petl.transform.reshape.recast`, this transform does not
      guess fields, does not sample, and never caches its result. Callers that
      need repeated, cheap passes should materialise the result (for example
      with :func:`petl.util.materialise.cache`) rather than rely on this view.

    * The result emits exactly the key columns in `key` order plus exactly
      `fields`, so its header is fixed up front rather than discovered from a
      sample. This is why an empty input (header only) still emits a
      complete output header, while a completely empty input (no header) is
      rejected.

    """

    return RecastByView(table, key, variablefield, valuefield, fields,
                        duplicates=duplicates, unknown=unknown,
                        presorted=presorted, buffersize=buffersize,
                        tempdir=tempdir, missing=missing)


Table.recastby = recastby


class RecastByView(Table):

    def __init__(self, source, key, variablefield, valuefield, fields,
                 duplicates='error', unknown='error', presorted=False,
                 buffersize=None, tempdir=None, missing=None):
        # Reject invalid modes immediately: these do not depend on the
        # (possibly one-shot) `fields` argument or on the input header.
        _recastby_validate_modes(duplicates, unknown)
        if buffersize is not None:
            _recastby_check_buffersize(buffersize)
        self.source = source
        self.key = key
        self.variablefield = variablefield
        self.valuefield = valuefield
        self.fields = fields
        # N.B., `duplicates` is also a fluent Table method; storing it as an
        # instance attribute would shadow the method.
        self._duplicates = duplicates
        self.unknown = unknown
        self.presorted = presorted
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.missing = missing
        # `fields` may be a one-shot iterator, so only resolve it on the
        # first actual iteration, and reuse the resolved order afterwards.
        self._fields_ready = False
        self._fields = None
        self._fields_error = None

    def __iter__(self):
        return iterrecastby(self.source, self.key, self.variablefield,
                            self.valuefield, self.fields, self._duplicates,
                            self.unknown, self.presorted, self.buffersize,
                            self.tempdir, self.missing, self)

    def _resolve_fields(self, keyfields):
        if not self._fields_ready:
            try:
                self._fields = _prepare_recastby_fields(self.fields, keyfields)
            except Exception as e:
                self._fields_error = e
            self._fields_ready = True
        if self._fields_error is not None:
            raise self._fields_error
        return self._fields


_RECASTBY_UNSET = object()
_RECASTBY_DUPLICATES = ('error', 'first', 'last')
_RECASTBY_UNKNOWN = ('error', 'ignore')
# Upper bound on the number of sorted runs merged in one pass. This bounds
# the number of temporary run files open at the same time and the number of
# records buffered while merging, independently of how many chunks the input
# spills to. The effective fan-in is further limited by the sort buffer size
# (see _recastby_merge_fanin), so the merge never retains more records than
# the in-memory sort does by more than a constant factor.
_RECASTBY_MERGE_FANIN = 32


def _recastby_merge_fanin(buffersize):
    # At least two runs are needed to make progress; otherwise keep the
    # fan-in proportional to the sort buffer, capped at the constant above.
    if buffersize is None:
        return _RECASTBY_MERGE_FANIN
    return max(2, min(_RECASTBY_MERGE_FANIN, buffersize))


def _recastby_validate_modes(duplicates, unknown):
    if duplicates not in _RECASTBY_DUPLICATES:
        raise ArgumentError(
            "invalid duplicates mode: %r (expected 'error', 'first' or "
            "'last')" % (duplicates,))
    if unknown not in _RECASTBY_UNKNOWN:
        raise ArgumentError(
            "invalid unknown mode: %r (expected 'error' or 'ignore')"
            % (unknown,))


def _recastby_check_buffersize(buffersize):
    if not isinstance(buffersize, int) or isinstance(buffersize, bool) \
            or buffersize < 1:
        raise ArgumentError(
            'buffersize must be a positive integer or None, got %r'
            % (buffersize,))


def _recastby_buffersize(buffersize):
    if buffersize is None:
        buffersize = config.sort_buffersize
    if buffersize is None:
        return None
    _recastby_check_buffersize(buffersize)
    return buffersize


def _recastby_header_fields(hdr):
    flds = list(map(text_type, hdr))
    for i, f in enumerate(flds):
        if f in flds[:i]:
            raise ArgumentError('duplicate header field: %r' % (f,))
    return flds


def _recastby_keyfields(key):
    if isinstance(key, string_types):
        keyfields = (key,)
    else:
        try:
            keyfields = tuple(key)
        except TypeError:
            raise ArgumentError(
                'key must be a field name or a sequence of field names')
    if not keyfields:
        raise ArgumentError('key must contain at least one field name')
    for i, f in enumerate(keyfields):
        if not isinstance(f, string_types):
            raise ArgumentError('invalid key field: %r' % (f,))
        if f in keyfields[:i]:
            raise ArgumentError('duplicate key field: %r' % (f,))
    return keyfields


def _recastby_single_field(name, label):
    if not isinstance(name, string_types):
        raise ArgumentError('%s must be a single field name: %r'
                            % (label, name))
    return name


def _recastby_index(flds, name, label):
    try:
        return flds.index(name)
    except ValueError:
        raise ArgumentError('invalid %s: %s' % (label, name))


def _prepare_recastby_fields(fields, keyfields):
    if isinstance(fields, string_types):
        names = (fields,)
    else:
        if fields is None:
            raise ArgumentError('fields must be a sequence of field names')
        try:
            names = tuple(fields)
        except TypeError:
            raise ArgumentError(
                'fields must be a sequence of field names')
    if not names:
        raise ArgumentError('fields must contain at least one field name')
    for f in names:
        if not isinstance(f, string_types):
            raise ArgumentError(
                'fields must be field names (strings): %r' % (f,))
    seen = set()
    for f in names:
        if f in seen:
            raise ArgumentError('duplicate field in fields: %r' % (f,))
        seen.add(f)
    for f in names:
        if f in keyfields:
            raise ArgumentError(
                'field %r conflicts with a key field' % (f,))
    return names


def _recastby_display_key(keytuple):
    if len(keytuple) == 1:
        return keytuple[0]
    return keytuple


def _recastby_check_width(row, rownum, width):
    if len(row) != width:
        raise ArgumentError(
            'data row %d has %d values but the header has %d fields'
            % (rownum, len(row), width))


def _recastby_sortkey(record):
    # record is (keytuple, original row number, row); the row number makes
    # the order of equal keys independent of chunk spilling and merging.
    return (Comparable(record[0]), record[1])


class _RecastByKeyed(namedtuple('_RecastByKeyed', ['key', 'obj'])):

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    def __ne__(self, other):
        return self.key != other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key


def _safe_close(obj, what='iterator'):
    close = getattr(obj, 'close', None)
    if close is None:
        return
    try:
        close()
    except Exception:
        logger.debug('exception closing %s', what, exc_info=True)


def _iter_presorted_records(it, width, keyindices):
    rownum = 0
    for row in it:
        rownum += 1
        row = tuple(row)
        _recastby_check_width(row, rownum, width)
        yield (tuple(row[i] for i in keyindices), rownum, row)


def _read_recastby_chunk(it, width, keyindices, buffersize, start):
    records = []
    rownum = start
    for row in itertools.islice(it, 0, buffersize):
        rownum += 1
        row = tuple(row)
        _recastby_check_width(row, rownum, width)
        records.append((tuple(row[i] for i in keyindices), rownum, row))
    return records, rownum


def _pickle_dump_recastby_record(record, f):
    # Separate indirection so tests can inject controlled write failures.
    pickle.dump(record, f, protocol=-1)


def _dump_recastby_chunk(records, tempdir, live):
    f = NamedTemporaryFile(dir=tempdir, delete=False, mode='wb')
    # record the name before writing so a failed write is still cleaned up
    live.add(f.name)
    try:
        for record in records:
            _pickle_dump_recastby_record(record, f)
        f.flush()
    finally:
        _safe_close(f, 'sort chunk')


def _open_recastby_chunk_file(fn):
    # Separate indirection so tests can inject controlled read failures;
    # also the single place a run file is opened.
    return open(fn, 'rb')


def _iter_recastby_chunk_file(fn):
    f = _open_recastby_chunk_file(fn)
    try:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return
    finally:
        _safe_close(f, 'sort chunk file')


def _iter_recastby_run_file(fn, live):
    # the run file is opened lazily, when this iterator is first advanced,
    # and removed as soon as the run has been (or stops being) consumed.
    # Once it is gone, its name is dropped from `live`, so the cleanup
    # registry only holds files that still exist.
    it = _iter_recastby_chunk_file(fn)
    try:
        for record in it:
            yield record
    finally:
        _safe_close(it, 'sort chunk file')
        if _unlink_recastby_file(fn):
            live.discard(fn)


def _keyed_recastby_iter(it):
    for record in it:
        yield _RecastByKeyed(_recastby_sortkey(record), record)


def _merge_recastby_chunk_files(filenames, live):
    """Merge at most ``_RECASTBY_MERGE_FANIN`` sorted run files.

    Every run file is opened lazily and closed and unlinked once it is
    exhausted, so a merge never keeps more than the fan-in number of
    temporary files (and one buffered record per file) alive. Callers bound
    the number of runs passed to a single call, independently of the total
    number of spilled chunks.
    """
    runiters = []
    keyediters = []
    try:
        runiters = [_iter_recastby_run_file(fn, live) for fn in filenames]
        keyediters = [_keyed_recastby_iter(it) for it in runiters]
        for item in heapq.merge(*keyediters):
            yield item.obj
    finally:
        for it in keyediters:
            _safe_close(it, 'sort chunk')
        for it in runiters:
            _safe_close(it, 'sort chunk file')


def _dump_recastby_merged_chunk(filenames, tempdir, live):
    # Merge one group of runs into a new run file. The inputs are removed
    # by _merge_recastby_chunk_files as they are consumed; the new name is
    # registered in `live` before it is written so that a failed write is
    # still cleaned up by the caller.
    f = NamedTemporaryFile(dir=tempdir, delete=False, mode='wb')
    live.add(f.name)
    mergeiter = _merge_recastby_chunk_files(filenames, live)
    try:
        for record in mergeiter:
            _pickle_dump_recastby_record(record, f)
        f.flush()
    finally:
        _safe_close(mergeiter, 'sort merge')
        _safe_close(f, 'sort chunk')
    return f.name


def _reduce_recastby_runs(filenames, tempdir, live, fanin):
    # Multi-pass reduction with a bounded fan-in: each pass merges groups
    # of at most `fanin` runs into a single run file and deletes the
    # inputs, so neither the number of open files nor the number of rows
    # held in memory grows with the number of chunks.
    # N.B., `live` may alias `filenames`; snapshot each pass's inputs so
    # newly created runs are not picked up by the same pass.
    inputs = list(filenames)
    while len(inputs) > fanin:
        reduced = []
        for start in range(0, len(inputs), fanin):
            group = inputs[start:start + fanin]
            if len(group) == 1:
                # nothing to merge, carry the run over unchanged
                reduced.append(group[0])
            else:
                reduced.append(
                    _dump_recastby_merged_chunk(group, tempdir, live))
        inputs = reduced
    return inputs


def _unlink_recastby_file(fn):
    try:
        os.unlink(fn)
    except Exception:
        # Cleanup must never replace an exception from the source or
        # from the caller's iteration.
        logger.debug('exception deleting %s', fn, exc_info=True)
        return False
    return True


def _unlink_recastby_files(filenames):
    for fn in filenames:
        _unlink_recastby_file(fn)


def _iter_sorted_recastby_records(it, width, keyindices, buffersize,
                                  tempdir):
    # `live` accumulates every file name created for this result iteration,
    # including intermediate merge runs, so that cleanup can remove anything
    # still present after an error or an early close. Names of runs that have
    # already been consumed stay in the set; unlinking them again is
    # harmless.
    live = set()
    try:
        chunk, rownum = _read_recastby_chunk(
            it, width, keyindices, buffersize, 0)
        if not chunk:
            return
        chunk.sort(key=_recastby_sortkey)
        if buffersize is None or len(chunk) < buffersize:
            for record in chunk:
                yield record
            return

        # The first chunk filled the buffer, so there may be more data.
        while chunk:
            _dump_recastby_chunk(chunk, tempdir, live)
            chunk, rownum = _read_recastby_chunk(
                it, width, keyindices, buffersize, rownum)
            if chunk:
                chunk.sort(key=_recastby_sortkey)

        filenames = _reduce_recastby_runs(
            live, tempdir, live, _recastby_merge_fanin(buffersize))
        mergeiter = _merge_recastby_chunk_files(filenames, live)
        try:
            for record in mergeiter:
                yield record
        finally:
            _safe_close(mergeiter, 'sort merge')
    finally:
        _unlink_recastby_files(live)


def _recastby_field_index(indicator, field_index, fields):
    try:
        return field_index.get(indicator)
    except TypeError:
        # an unknown variable may itself be unhashable, e.g. a list or dict
        for i, f in enumerate(fields):
            if f == indicator:
                return i
        return None


def _recastby_output(keytuple, values, missing):
    out = list(keytuple)
    for value in values:
        out.append(missing if value is _RECASTBY_UNSET else value)
    return tuple(out)


def _iter_recastby_groups(records, variableindex, valueindex, fields,
                          duplicates, unknown, missing, check_order):
    unset = _RECASTBY_UNSET
    nfields = len(fields)
    field_index = {f: i for i, f in enumerate(fields)}
    prev_key = unset
    prev_comparable = None
    values = None
    seen_rows = None
    for keytuple, rownum, row in records:
        if prev_key is unset:
            prev_key = keytuple
            prev_comparable = Comparable(keytuple)
            values = [unset] * nfields
            seen_rows = [-1] * nfields
        elif keytuple != prev_key:
            if check_order:
                current_comparable = Comparable(keytuple)
                if current_comparable < prev_comparable:
                    raise ArgumentError(
                        'key out of order at data row %d: %r follows %r'
                        % (rownum, _recastby_display_key(keytuple),
                           _recastby_display_key(prev_key)))
            yield _recastby_output(prev_key, values, missing)
            prev_key = keytuple
            prev_comparable = Comparable(keytuple)
            values = [unset] * nfields
            seen_rows = [-1] * nfields

        indicator = row[variableindex]
        idx = _recastby_field_index(indicator, field_index, fields)
        if idx is None:
            if unknown == 'ignore':
                continue
            raise ArgumentError(
                'unknown indicator %r for key %r at data row %d'
                % (indicator, _recastby_display_key(keytuple), rownum))

        if seen_rows[idx] >= 0:
            if duplicates == 'error':
                raise ArgumentError(
                    'duplicate indicator %r for key %r at data row %d '
                    '(first seen at data row %d)'
                    % (indicator, _recastby_display_key(keytuple), rownum,
                       seen_rows[idx]))
            elif duplicates == 'last':
                values[idx] = row[valueindex]
            # duplicates == 'first': keep the existing value, including None
        else:
            seen_rows[idx] = rownum
            values[idx] = row[valueindex]

    if prev_key is not unset:
        yield _recastby_output(prev_key, values, missing)


def iterrecastby(source, key, variablefield, valuefield, fields, duplicates,
                 unknown, presorted, buffersize, tempdir, missing, view):
    _recastby_validate_modes(duplicates, unknown)
    if not presorted:
        buffersize = _recastby_buffersize(buffersize)

    it = iter(source)  # exactly one source iterator per result iteration
    records = None
    try:
        try:
            hdr = next(it)
        except StopIteration:
            raise ArgumentError('no header row')
        hdr = tuple(hdr)
        width = len(hdr)
        flds = _recastby_header_fields(hdr)

        keyfields = _recastby_keyfields(key)
        keyindices = [_recastby_index(flds, f, 'key field')
                      for f in keyfields]
        variablefield = _recastby_single_field(variablefield, 'variablefield')
        valuefield = _recastby_single_field(valuefield, 'valuefield')
        variableindex = _recastby_index(flds, variablefield,
                                        'variable field')
        valueindex = _recastby_index(flds, valuefield, 'value field')

        if variableindex in keyindices:
            raise ArgumentError('variable field cannot be a key field')
        if valueindex in keyindices:
            raise ArgumentError('value field cannot be a key field')
        if variableindex == valueindex:
            raise ArgumentError(
                'variable field and value field must be different')

        fields = view._resolve_fields(keyfields)
        outhdr = tuple(keyfields) + tuple(fields)
        yield outhdr

        if presorted:
            records = _iter_presorted_records(it, width, keyindices)
        else:
            records = _iter_sorted_recastby_records(
                it, width, keyindices, buffersize, tempdir)
        for outrow in _iter_recastby_groups(
                records, variableindex, valueindex, fields, duplicates,
                unknown, missing, presorted):
            yield outrow
    finally:
        if records is not None:
            _safe_close(records, 'record iterator')
        _safe_close(it, 'source iterator')


def transpose(table):
    """
    Transpose rows into columns. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple'],
        ...           [5, 'yellow'],
        ...           [7, 'orange']]
        >>> table2 = etl.transpose(table1)
        >>> table2
        +----------+--------+-------+----------+----------+----------+
        | id       | 1      | 2     | 3        | 5        | 7        |
        +==========+========+=======+==========+==========+==========+
        | 'colour' | 'blue' | 'red' | 'purple' | 'yellow' | 'orange' |
        +----------+--------+-------+----------+----------+----------+

    See also :func:`petl.transform.reshape.recast`.

    """

    return TransposeView(table)


Table.transpose = transpose


class TransposeView(Table):

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return itertranspose(self.source)


def itertranspose(source):
    hdr = header(source)
    its = [iter(source) for _ in hdr]
    for i in range(len(hdr)):
        yield tuple(row[i] for row in its[i])


def pivot(table, f1, f2, f3, aggfun, missing=None,
          presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Construct a pivot table. E.g.::

        >>> import petl as etl
        >>> table1 = [['region', 'gender', 'style', 'units'],
        ...           ['east', 'boy', 'tee', 12],
        ...           ['east', 'boy', 'golf', 14],
        ...           ['east', 'boy', 'fancy', 7],
        ...           ['east', 'girl', 'tee', 3],
        ...           ['east', 'girl', 'golf', 8],
        ...           ['east', 'girl', 'fancy', 18],
        ...           ['west', 'boy', 'tee', 12],
        ...           ['west', 'boy', 'golf', 15],
        ...           ['west', 'boy', 'fancy', 8],
        ...           ['west', 'girl', 'tee', 6],
        ...           ['west', 'girl', 'golf', 16],
        ...           ['west', 'girl', 'fancy', 1]]
        >>> table2 = etl.pivot(table1, 'region', 'gender', 'units', sum)
        >>> table2
        +--------+-----+------+
        | region | boy | girl |
        +========+=====+======+
        | 'east' |  33 |   29 |
        +--------+-----+------+
        | 'west' |  35 |   23 |
        +--------+-----+------+

        >>> table3 = etl.pivot(table1, 'region', 'style', 'units', sum)
        >>> table3
        +--------+-------+------+-----+
        | region | fancy | golf | tee |
        +========+=======+======+=====+
        | 'east' |    25 |   22 |  15 |
        +--------+-------+------+-----+
        | 'west' |     9 |   31 |  18 |
        +--------+-------+------+-----+

        >>> table4 = etl.pivot(table1, 'gender', 'style', 'units', sum)
        >>> table4
        +--------+-------+------+-----+
        | gender | fancy | golf | tee |
        +========+=======+======+=====+
        | 'boy'  |    15 |   29 |  24 |
        +--------+-------+------+-----+
        | 'girl' |    19 |   24 |   9 |
        +--------+-------+------+-----+

    See also :func:`petl.transform.reshape.recast`.

    """

    return PivotView(table, f1, f2, f3, aggfun, missing=missing,
                     presorted=presorted, buffersize=buffersize,
                     tempdir=tempdir, cache=cache)


Table.pivot = pivot


class PivotView(Table):

    def __init__(self, source, f1, f2, f3, aggfun, missing=None,
                 presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.source = source
        else:
            self.source = sort(source, key=(f1, f2), buffersize=buffersize,
                               tempdir=tempdir, cache=cache)
        self.f1, self.f2, self.f3 = f1, f2, f3
        self.aggfun = aggfun
        self.missing = missing

    def __iter__(self):
        return iterpivot(self.source, self.f1, self.f2, self.f3, self.aggfun,
                         self.missing)


def iterpivot(source, f1, f2, f3, aggfun, missing):

    # first pass - collect fields
    f2vals = set(itervalues(source, f2))  # TODO only make one pass
    f2vals = list(f2vals)
    f2vals.sort()
    outhdr = [f1]
    outhdr.extend(f2vals)
    yield tuple(outhdr)

    # second pass - generate output
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    f1i = flds.index(f1)
    f2i = flds.index(f2)
    f3i = flds.index(f3)
    for v1, v1rows in itertools.groupby(it, key=operator.itemgetter(f1i)):
        outrow = [v1] + [missing] * len(f2vals)
        for v2, v12rows in itertools.groupby(v1rows,
                                             key=operator.itemgetter(f2i)):
            aggval = aggfun([row[f3i] for row in v12rows])
            outrow[1 + f2vals.index(v2)] = aggval
        yield tuple(outrow)


def flatten(table):
    """
    Convert a table to a sequence of values in row-major order. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, True],
        ...           ['C', 7, False],
        ...           ['B', 2, False],
        ...           ['C', 9, True]]
        >>> list(etl.flatten(table1))
        ['A', 1, True, 'C', 7, False, 'B', 2, False, 'C', 9, True]

    See also :func:`petl.transform.reshape.unflatten`.

    """

    return FlattenView(table)


Table.flatten = flatten


class FlattenView(Table):

    def __init__(self, table):
        self.table = table

    def __iter__(self):
        for row in data(self.table):
            for value in row:
                yield value


def unflatten(*args, **kwargs):
    """
    Convert a sequence of values in row-major order into a table. E.g.::

        >>> import petl as etl
        >>> a = ['A', 1, True, 'C', 7, False, 'B', 2, False, 'C', 9]
        >>> table1 = etl.unflatten(a, 3)
        >>> table1
        +-----+----+-------+
        | f0  | f1 | f2    |
        +=====+====+=======+
        | 'A' |  1 | True  |
        +-----+----+-------+
        | 'C' |  7 | False |
        +-----+----+-------+
        | 'B' |  2 | False |
        +-----+----+-------+
        | 'C' |  9 | None  |
        +-----+----+-------+

        >>> # a table and field name can also be provided as arguments
        ... table2 = [['lines'],
        ...           ['A'],
        ...           [1],
        ...           [True],
        ...           ['C'],
        ...           [7],
        ...           [False],
        ...           ['B'],
        ...           [2],
        ...           [False],
        ...           ['C'],
        ...           [9]]
        >>> table3 = etl.unflatten(table2, 'lines', 3)
        >>> table3
        +-----+----+-------+
        | f0  | f1 | f2    |
        +=====+====+=======+
        | 'A' |  1 | True  |
        +-----+----+-------+
        | 'C' |  7 | False |
        +-----+----+-------+
        | 'B' |  2 | False |
        +-----+----+-------+
        | 'C' |  9 | None  |
        +-----+----+-------+

    See also :func:`petl.transform.reshape.flatten`.

    """

    return UnflattenView(*args, **kwargs)


Table.unflatten = unflatten


class UnflattenView(Table):

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            self.input = args[0]
            self.period = args[1]
        elif len(args) == 3:
            self.input = values(args[0], args[1])
            self.period = args[2]
        else:
            assert False, 'invalid arguments'
        self.missing = kwargs.get('missing', None)

    def __iter__(self):
        inpt = self.input
        period = self.period
        missing = self.missing

        # generate header row
        outhdr = tuple('f%s' % i for i in range(period))
        yield outhdr

        # generate data rows
        row = list()
        for v in inpt:
            if len(row) < period:
                row.append(v)
            else:
                yield tuple(row)
                row = [v]

        # deal with last row
        if len(row) > 0:
            if len(row) < period:
                row.extend([missing] * (period - len(row)))
            yield tuple(row)
