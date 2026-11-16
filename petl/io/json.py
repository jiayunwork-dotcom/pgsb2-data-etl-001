# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

# standard library dependencies
import io
import json
import inspect
from itertools import chain, islice
from json.encoder import JSONEncoder
from os import unlink
from tempfile import NamedTemporaryFile

from petl.compat import PY2
from petl.compat import pickle
from petl.errors import ArgumentError
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.missing import MISSING
# internal dependencies
from petl.util.base import data, Table, dicts as _dicts


def fromjson(source, *args, **kwargs):
    """
    Extract data from a JSON file. The file must contain a JSON array as
    the top level object, and each member of the array will be treated as a
    row of data. E.g.::

        >>> import petl as etl
        >>> data = '''
        ... [{"foo": "a", "bar": 1},
        ... {"foo": "b", "bar": 2},
        ... {"foo": "c", "bar": 2}]
        ... '''
        >>> with open('example.file1.json', 'w') as f:
        ...     f.write(data)
        ...
        74
        >>> table1 = etl.fromjson('example.file1.json', header=['foo', 'bar'])
        >>> table1
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+
        | 'c' |   2 |
        +-----+-----+

    Setting argument `lines` to `True` will enable to
    infer the document as a JSON lines document. For more details about JSON lines
    please visit https://jsonlines.org/.

    An empty JSON lines file produces a table with no data rows. If no
    `header` is supplied, the table has an empty header tuple.

    An explicit `header` iterable is read once on first iteration and reused
    for every row and subsequent pass over the table.

    Additional arguments are passed to :func:`json.load`, or to
    :func:`json.loads` for each JSON line. For example, ``parse_float=Decimal``
    preserves decimal precision in both formats. Decoder options also apply
    when discovering the header from the first JSON line. The decoded first
    record is reused as the first data row.

        >>> import petl as etl
        >>> data_with_jlines = '''{"name": "Gilbert", "wins": [["straight", "7S"], ["one pair", "10H"]]}
        ... {"name": "Alexa", "wins": [["two pair", "4S"], ["two pair", "9S"]]}
        ... {"name": "May", "wins": []}
        ... {"name": "Deloise", "wins": [["three of a kind", "5S"]]}'''
        ...
        >>> with open('example.file2.json', 'w') as f:
        ...     f.write(data_with_jlines)
        ...
        223
        >>> table2 = etl.fromjson('example.file2.json', lines=True)
        >>> table2
        +-----------+-------------------------------------------+
        | name      | wins                                      |
        +===========+===========================================+
        | 'Gilbert' | [['straight', '7S'], ['one pair', '10H']] |
        +-----------+-------------------------------------------+
        | 'Alexa'   | [['two pair', '4S'], ['two pair', '9S']]  |
        +-----------+-------------------------------------------+
        | 'May'     | []                                        |
        +-----------+-------------------------------------------+
        | 'Deloise' | [['three of a kind', '5S']]               |
        +-----------+-------------------------------------------+

    If your JSON file does not fit this structure, you will need to parse it
    via :func:`json.load` and select the array to treat as the data, see also
    :func:`petl.io.json.fromdicts`.

    Set `preserve_missing` to ``True`` to distinguish an object that omits a
    key from one that carries an explicit ``null``: missing keys become the
    :data:`petl.MISSING` marker while explicit ``null`` values stay ``None``.
    Enabling `preserve_missing` together with a non-``None`` `missing`
    argument raises :class:`petl.errors.ArgumentError`.

    .. versionchanged:: 1.1.0

    If no `header` is specified, fields will be discovered by sampling keys
    from the first `sample` objects in `source`. The header will be
    constructed from keys in the order discovered. Note that this
    ordering may not be stable, and therefore it may be advisable to specify
    an explicit `header` or to use another function like
    :func:`petl.transform.headers.sortheader` on the resulting table to
    guarantee stability.

    With `sample=1`, only the first object's keys are used. An empty input
    array produces an empty header and no data rows.

    """

    source = read_source_from_arg(source)
    return JsonView(source, *args, **kwargs)


class JsonView(Table):
    def __init__(self, source, *args, **kwargs):
        self.source = source
        self.missing = kwargs.pop('missing', None)
        self.preserve_missing = kwargs.pop('preserve_missing', False)
        if self.preserve_missing and self.missing is not None:
            raise ArgumentError(
                'preserve_missing cannot be combined with a non-None missing '
                'argument'
            )
        self._header = kwargs.pop('header', None)
        self.sample = kwargs.pop('sample', 1000)
        self.lines = kwargs.pop('lines', False)
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        if self._header is not None:
            self._header = tuple(self._header)
        with self.source.open('rb') as f:
            if not PY2:
                # wrap buffer for text IO
                f = io.TextIOWrapper(f, encoding='utf-8', newline='',
                                     write_through=True)
            try:
                if self.lines:
                    for row in iterjlines(f, self._header, self.missing,
                                         self.args, self.kwargs,
                                         preserve_missing=self.preserve_missing):
                        yield row
                else:
                    dicts = json.load(f, *self.args, **self.kwargs)
                    for row in iterdicts(dicts, self._header, self.sample,
                                         self.missing,
                                         preserve_missing=self.preserve_missing):
                        yield row
            finally:
                if not PY2:
                    f.detach()


def fromdicts(dicts, header=None, sample=1000, missing=None,
              preserve_missing=False):
    """
    View a sequence of Python :class:`dict` as a table. E.g.::

        >>> import petl as etl
        >>> dicts = [{"foo": "a", "bar": 1},
        ...          {"foo": "b", "bar": 2},
        ...          {"foo": "c", "bar": 2}]
        >>> table1 = etl.fromdicts(dicts, header=['foo', 'bar'])
        >>> table1
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+
        | 'c' |   2 |
        +-----+-----+

    An explicit `header` iterable is read once on first iteration and reused
    for every row and subsequent pass over the table.

    Argument `dicts` can also be a generator, the output of generator
    is iterated and cached using a temporary file to support further
    transforms and multiple passes of the table:

        >>> import petl as etl
        >>> dicts = ({"foo": chr(ord("a")+i), "bar":i+1} for i in range(3))
        >>> table1 = etl.fromdicts(dicts, header=['foo', 'bar'])
        >>> table1
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+
        | 'c' |   3 |
        +-----+-----+

    If `header` is not specified, `sample` items from `dicts` will be
    inspected to discovery dictionary keys. Note that the order in which
    dictionary keys are discovered may not be stable,

    A sample size of one discovers the first dictionary's keys. Empty
    inputs produce an empty header and no data rows, including with
    `sample=1`.

    Set `preserve_missing` to ``True`` to keep the difference between a
    dictionary that lacks a key and one that holds an explicit ``None``:
    absent keys are emitted as the :data:`petl.MISSING` marker, explicit
    ``None`` values stay ``None`` and every other value is passed through
    unchanged. Because a missing field would otherwise be filled with the
    `missing` argument, combining `preserve_missing=True` with a non-``None``
    `missing` raises :class:`petl.errors.ArgumentError`. Header discovery via
    `sample` and the treatment of unknown fields are unchanged.

    See also :func:`petl.io.json.fromjson`.

    .. versionchanged:: 1.1.0

    If no `header` is specified, fields will be discovered by sampling keys
    from the first `sample` dictionaries in `dicts`. The header will be
    constructed from keys in the order discovered. Note that this
    ordering may not be stable, and therefore it may be advisable to specify
    an explicit `header` or to use another function like
    :func:`petl.transform.headers.sortheader` on the resulting table to
    guarantee stability.

    .. versionchanged:: 1.7.5

    Full support of generators passed as `dicts` has been added, leveraging
    `itertools.tee`.

    .. versionchanged:: 1.7.11

    Generator support has been modified to use temporary file cache
    instead of `itertools.tee` due to high memory usage.

    """
    if preserve_missing and missing is not None:
        raise ArgumentError(
            'preserve_missing cannot be combined with a non-None missing '
            'argument'
        )
    view = DictsGeneratorView if inspect.isgenerator(dicts) else DictsView
    return view(dicts, header=header, sample=sample, missing=missing,
                preserve_missing=preserve_missing)


class _DictsAttr(object):
    """Reads as the input data, calls as the inherited Table.dicts() method.

    Only exists to keep `.dicts` working both ways in the v1.x series; v2 can
    drop it along with the property below.
    """

    def __init__(self, view):
        self._view = view

    def __call__(self, *sliceargs, **kwargs):
        return _dicts(self._view, *sliceargs, **kwargs)

    def __iter__(self):
        return iter(self._view._dicts)

    def __next__(self):
        return next(self._view._dicts)

    next = __next__

    def __len__(self):
        return len(self._view._dicts)

    def __getitem__(self, item):
        return self._view._dicts[item]

    def __eq__(self, other):
        return self._view._dicts == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._view._dicts)

    def __repr__(self):
        return repr(self._view._dicts)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._view._dicts, name)


class DictsView(Table):

    def __init__(self, dicts, header=None, sample=1000, missing=None,
                 preserve_missing=False):
        self._dicts = dicts
        self._header = header
        self.sample = sample
        self.missing = missing
        self.preserve_missing = preserve_missing

    def __iter__(self):
        if self._header is not None:
            self._header = tuple(self._header)
        for row in iterdicts(self._dicts, self._header, self.sample,
                             self.missing,
                             preserve_missing=self.preserve_missing):
            yield row

    @property
    def dicts(self):
        # `.dicts` was the raw input data, which shadowed Table.dicts() (#643).
        # A property on the subclass wins over the inherited method, so both
        # uses keep working until v2.
        return _DictsAttr(self)

    @dicts.setter
    def dicts(self, value):
        self._dicts = value


class DictsGeneratorView(DictsView):

    def __init__(self, dicts, header=None, sample=1000, missing=None,
                 preserve_missing=False):
        super(DictsGeneratorView, self).__init__(
            dicts, header, sample, missing, preserve_missing
        )
        self._filecache = None
        self._cached = 0

    def __iter__(self):
        if self._header is None:
            self._determine_header()
        else:
            self._header = tuple(self._header)
        yield self._header

        if not self._filecache:
            if PY2:
                self._filecache = NamedTemporaryFile(delete=False, mode='wb+', bufsize=0)
            else:
                self._filecache = NamedTemporaryFile(delete=False, mode='wb+', buffering=0)

        position = 0
        it = iter(self._dicts)
        while True:
            if position < self._cached:
                self._filecache.seek(position)
                row = pickle.load(self._filecache)
                position = self._filecache.tell()
                yield row
                continue
            try:
                o = next(it)
            except StopIteration:
                break
            row = tuple(_dict_cell(o, f, self.missing,
                                   self.preserve_missing)
                        for f in self._header)
            self._filecache.seek(self._cached)
            pickle.dump(row, self._filecache, protocol=-1)
            self._cached = position = self._filecache.tell()
            yield row

    def _determine_header(self):
        it = iter(self._dicts)
        header = list()
        peek = list(islice(it, self.sample))
        it = chain(peek, it)
        self._dicts = it
        for o in peek:
            if hasattr(o, 'keys'):
                header += [k for k in o.keys() if k not in header]
        self._header = tuple(header)
        return it

    def __del__(self):
        if self._filecache:
            self._filecache.close()
            unlink(self._filecache.name)


def _dict_cell(o, field, missing, preserve_missing):
    """Extract one cell from a source dictionary.

    With `preserve_missing`, an absent key yields the MISSING marker while an
    explicit ``None`` is returned unchanged. Otherwise the historical
    ``dict.get(field, missing)`` behaviour is preserved.
    """
    if preserve_missing:
        if field in o:
            return o[field]
        return MISSING
    return o.get(field, missing)


def iterjlines(f, header, missing, args=(), kwargs=None,
               preserve_missing=False):
    if kwargs is None:
        kwargs = {}
    it = iter(f)
    first = ()

    if header is None:
        header = list()
        try:
            peek = next(it)
        except StopIteration:
            yield tuple()
            return
        json_obj = json.loads(peek, *args, **kwargs)
        first = (json_obj,)
        if hasattr(json_obj, 'keys'):
            header += [k for k in json_obj.keys() if k not in header]
    yield tuple(header)

    for json_obj in chain(first, (json.loads(o, *args, **kwargs) for o in it)):
        yield tuple(_dict_cell(json_obj, f, missing, preserve_missing)
                    for f in header)


def iterdicts(dicts, header, sample, missing, preserve_missing=False):
    it = iter(dicts)

    # determine header row
    if header is None:
        # discover fields
        header = list()
        peek = list(islice(it, sample))
        it = chain(peek, it)
        for o in peek:
            if hasattr(o, 'keys'):
                header += [k for k in o.keys() if k not in header]
    yield tuple(header)

    # generate data rows
    for o in it:
        yield tuple(_dict_cell(o, f, missing, preserve_missing)
                    for f in header)


def tojson(table, source=None, prefix=None, suffix=None, omit_missing=False,
           *args, **kwargs):
    """
    Write a table in JSON format, with rows output as JSON objects. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> etl.tojson(table1, 'example.file3.json', sort_keys=True)
        >>> # check what it did
        ... print(open('example.file3.json').read())
        [{"bar": 1, "foo": "a"}, {"bar": 2, "foo": "b"}, {"bar": 2, "foo": "c"}]



    Setting argument `lines` to `True` will enable to
    infer the writing format as a JSON lines . For more details about JSON lines
    please visit https://jsonlines.org/.

        >>> import petl as etl
        >>> table1 = [['name', 'wins'],
        ...           ['Gilbert', [['straight', '7S'], ['one pair', '10H']]],
        ...           ['Alexa', [['two pair', '4S'], ['two pair', '9S']]],
        ...           ['May', []],
        ...           ['Deloise',[['three of a kind', '5S']]]]
        >>> etl.tojson(table1, 'example.file3.jsonl', lines = True, sort_keys=True)
        >>> # check what it did
        ... print(open('example.file3.jsonl').read())
        {"name": "Gilbert", "wins": [["straight", "7S"], ["one pair", "10H"]]}
        {"name": "Alexa", "wins": [["two pair", "4S"], ["two pair", "9S"]]}
        {"name": "May", "wins": []}
        {"name": "Deloise", "wins": [["three of a kind", "5S"]]}

    Set `omit_missing` to ``True`` to round-trip sparse tables read with
    ``preserve_missing=True``: cells holding the :data:`petl.MISSING` marker
    are left out of the output object, while explicit ``None`` cells are
    still written as ``null`` and every other value uses the normal JSON
    encoding. Both array and JSON lines output support this. If
    `omit_missing` is left at its default and a MISSING cell is encountered,
    a :class:`TypeError` is raised rather than silently writing ``null`` or
    the string ``"MISSING"``.

    Note that this is currently not streaming, all data is loaded into memory
    before being written to the file.

    """

    obj = [_json_object(hdr, row, omit_missing)
           for hdr, row in _iter_json_objects(table)]
    _writejson(source, obj, prefix, suffix, *args, **kwargs)


Table.tojson = tojson


def _iter_json_objects(table):
    """Yield ``(header, row)`` pairs for each data row of `table`."""
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return
    for row in it:
        yield hdr, row


def _json_object(hdr, row, omit_missing):
    """Build a plain dict for one row, honouring the MISSING contract."""
    obj = dict()
    for i, f in enumerate(hdr):
        if i < len(row):
            value = row[i]
        else:
            # match the historical short-row behaviour of Table.dicts()
            value = None
        if value is MISSING:
            if omit_missing:
                continue
            raise TypeError(
                'petl.MISSING value for field %r cannot be encoded as JSON; '
                'use tojson(..., omit_missing=True) to omit the key'
                % (f,)
            )
        obj[f] = value
    return obj


def tojsonarrays(table, source=None, prefix=None, suffix=None,
                 output_header=False, *args, **kwargs):
    """
    Write a table in JSON format, with rows output as JSON arrays. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> etl.tojsonarrays(table1, 'example.file4.json')
        >>> # check what it did
        ... print(open('example.file4.json').read())
        [["a", 1], ["b", 2], ["c", 2]]

    Note that this is currently not streaming, all data is loaded into memory
    before being written to the file.

    """

    if output_header:
        obj = [row for row in table]
    else:
        obj = [row for row in data(table)]
    _writejson(source, obj, prefix, suffix, *args, **kwargs)


Table.tojsonarrays = tojsonarrays


def _writejson(source, obj, prefix, suffix, *args, **kwargs):
    lines = kwargs.pop('lines', False)
    encoder = JSONEncoder(*args, **kwargs)
    source = write_source_from_arg(source)
    with source.open('wb') as f:
        if PY2:
            # write directly to buffer
            _writeobj(encoder, obj, f, prefix, suffix, lines=lines)
        else:
            # wrap buffer for text IO
            f = io.TextIOWrapper(f, encoding='utf-8', newline='',
                                 write_through=True)
            try:
                _writeobj(encoder, obj, f, prefix, suffix, lines=lines)
            finally:
                f.detach()


def _writeobj(encoder, obj, f, prefix, suffix, lines=False):
    if prefix is not None:
        f.write(prefix)
    if lines:
        for rec in obj:
            for chunk in encoder.iterencode(rec):
                f.write(chunk)
            f.write('\n')
    else:
        for chunk in encoder.iterencode(obj):
            f.write(chunk)
    if suffix is not None:
        f.write(suffix)
