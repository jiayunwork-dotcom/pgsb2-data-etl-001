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
from petl.io.json_streaming import decode_streaming_records
from petl.io.sources import read_source_from_arg, write_source_from_arg
# internal dependencies
from petl.util.base import data, Table, dicts as _dicts


def _positive_int_option(name, value):
    # bool is a subclass of int; reject it explicitly so True is not a
    # silently accepted "size"
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArgumentError(
            '%s must be a positive integer, found %r' % (name, value))
    return value


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

    Set `streaming` to `True` to read a top level array incrementally
    instead of loading the whole document into memory before the first row.
    Streaming mode accepts UTF-8 encoded files whose top level value is a
    JSON array of JSON objects; it works with local files, compressed files
    and any Source whose ``open('rb')`` returns a binary context manager.
    The stream does not need to be seekable and a single ``read(n)`` may
    return fewer than ``n`` bytes.

    `chunksize` is the maximum number of bytes requested from the binary
    source per read (default 65536) and `max_record_chars` caps the number
    of decoded text characters allowed for a single object, counted from
    its opening to closing brace and including internal whitespace but not
    the whitespace between objects (default 8388608); an object exactly at
    the limit is accepted. Memory use is bounded by `chunksize`,
    `max_record_chars` and `sample`, not by the total file size.

    Streaming mode supports the standard decoder options `parse_float`,
    `parse_int`, `parse_constant`, `object_hook`, `object_pairs_hook` and
    `strict`; they apply to header sampling as well as data rows and each
    sampled object is decoded only once. A record (including the result of
    an `object_hook` or `object_pairs_hook`) must still be a dict. Custom
    `cls` decoders and extra positional arguments are rejected with an
    :class:`petl.errors.ArgumentError` rather than silently ignored.
    `streaming=True` cannot be combined with `lines=True`.

        >>> import petl as etl
        >>> big = '[{"foo": "a"}, {"foo": "b"}]'
        >>> with open('example.file5.json', 'w') as f:
        ...     f.write(big)
        ...
        28
        >>> table3 = etl.fromjson('example.file5.json', streaming=True)
        >>> list(table3)
        [('foo',), ('a',), ('b',)]

    Syntax errors are raised as :class:`json.JSONDecodeError` with line and
    column positions based on the original input text, UTF-8 decoding
    failures remain :class:`UnicodeDecodeError`, and record type and length
    violations raise :class:`petl.errors.ArgumentError`. Results already
    yielded before a malformed suffix are not withdrawn.

    Delivered rows and the reported error do not depend on how the binary
    stream is split into reads: the same byte string always yields the same
    complete prefix of records and the same earliest error, even when a
    single read returns one byte at a time. Error ordering inside one
    unfinished object is deterministic: if a JSON syntax error has already
    become definite before the first bad UTF-8 byte, a
    :class:`~json.JSONDecodeError` is raised; otherwise the
    :class:`UnicodeDecodeError` wins and the illegal bytes are never
    replaced or ignored. A prefix that is merely incomplete -- truncated
    JSON or a multi-byte UTF-8 character split across reads -- is not an
    error until the source reaches EOF. When header sampling is enabled,
    neither the header nor any row is delivered if an error occurs before
    the sample is complete; once sampling has completed, the fully read
    records are delivered in order and a later error cannot withdraw them.
    JSONDecodeError positions (``pos``, ``lineno``, ``colno`` and the
    position in the message) count decoded Unicode characters, never raw
    byte offsets; the UnicodeDecodeError keeps its real illegal bytes while
    its relative start/end is not remapped to a file offset.

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
        self._header = kwargs.pop('header', None)
        self.sample = kwargs.pop('sample', 1000)
        self.lines = kwargs.pop('lines', False)
        self.streaming = kwargs.pop('streaming', False)
        self.chunksize = _positive_int_option(
            'chunksize', kwargs.pop('chunksize', 65536))
        self.max_record_chars = _positive_int_option(
            'max_record_chars',
            kwargs.pop('max_record_chars', 8388608))
        # these are validated before the source is ever opened, so an
        # unsupported combination fails at view construction time
        if self.streaming and self.lines:
            raise ArgumentError(
                'streaming=True cannot be combined with lines=True; JSON '
                'lines are already read incrementally')
        if self.streaming:
            if args:
                raise ArgumentError(
                    'streaming mode does not accept extra positional '
                    'arguments; use decoder keyword arguments instead')
            if 'cls' in kwargs:
                raise ArgumentError(
                    'streaming mode does not support a custom decoder cls')
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        if self._header is not None:
            self._header = tuple(self._header)
        if self.streaming:
            for row in self._iter_streaming():
                yield row
            return
        with self.source.open('rb') as f:
            if not PY2:
                # wrap buffer for text IO
                f = io.TextIOWrapper(f, encoding='utf-8', newline='',
                                     write_through=True)
            try:
                if self.lines:
                    for row in iterjlines(f, self._header, self.missing,
                                         self.args, self.kwargs):
                        yield row
                else:
                    dicts = json.load(f, *self.args, **self.kwargs)
                    for row in iterdicts(dicts, self._header, self.sample,
                                         self.missing):
                        yield row
            finally:
                if not PY2:
                    f.detach()

    def _iter_streaming(self):
        # Mirror iterdicts(): an explicit header (even an empty one) is
        # consumed once here and reused; otherwise sample the leading
        # objects for keys while keeping the decoded objects as rows so
        # nothing is decoded twice.
        if self._header is None:
            if self.sample < 0:
                raise ValueError('sample must be non-negative')
            header = None
            sampled = []
        else:
            header = tuple(self._header)
            sampled = None
            yield header

        with self.source.open('rb') as f:
            records = decode_streaming_records(
                f, self.chunksize, self.max_record_chars, self.kwargs)
            if header is None:
                discovered = []
                for record, _, _ in islice(records, self.sample):
                    sampled.append(record)
                    for key in record.keys():
                        if key not in discovered:
                            discovered.append(key)
                header = tuple(discovered)
                yield header
                iterator = chain(sampled,
                                 (record for record, _, _ in records))
            else:
                iterator = (record for record, _, _ in records)

            for record in iterator:
                yield tuple(record.get(field, self.missing)
                            for field in header)


def fromdicts(dicts, header=None, sample=1000, missing=None):
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
    view = DictsGeneratorView if inspect.isgenerator(dicts) else DictsView
    return view(dicts, header=header, sample=sample, missing=missing)


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

    def __init__(self, dicts, header=None, sample=1000, missing=None):
        self._dicts = dicts
        self._header = header
        self.sample = sample
        self.missing = missing

    def __iter__(self):
        if self._header is not None:
            self._header = tuple(self._header)
        for row in iterdicts(self._dicts, self._header, self.sample, self.missing):  # noqa: UP028 - Python 2 compatibility
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

    def __init__(self, dicts, header=None, sample=1000, missing=None):
        super(DictsGeneratorView, self).__init__(dicts, header, sample, missing)
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
            row = tuple(o.get(f, self.missing) for f in self._header)
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


def iterjlines(f, header, missing, args=(), kwargs=None):
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
        yield tuple(json_obj[f] if f in json_obj else missing for f in header)


def iterdicts(dicts, header, sample, missing):
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
        yield tuple(o.get(f, missing) for f in header)


def tojson(table, source=None, prefix=None, suffix=None, *args, **kwargs):
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

    Note that this is currently not streaming, all data is loaded into memory
    before being written to the file.

    """

    obj = [row for row in _dicts(table)]
    _writejson(source, obj, prefix, suffix, *args, **kwargs)


Table.tojson = tojson


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
