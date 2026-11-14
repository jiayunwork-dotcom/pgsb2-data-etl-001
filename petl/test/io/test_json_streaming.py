# -*- coding: utf-8 -*-
"""Tests for the optional streaming array reader in
:func:`petl.io.json.fromjson` (``streaming=True``).

The streaming reader must work against binary streams that cannot seek,
return fewer bytes than requested on each read, or hand back an occasional
empty read before more data arrives. The helper streams below model those
conditions deterministically.
"""
from __future__ import absolute_import, print_function, division

import bz2
import gzip
import io
import json
from collections import OrderedDict
from decimal import Decimal
from json import JSONDecodeError

import pytest

import petl as etl
from petl import fromjson
from petl.errors import ArgumentError
from petl.io import json as json_module
from petl.io.json_streaming import (
    _ChunkedByteStream,
    _iter_array_records,
    _EMPTY_READ_EOF,
)
from petl.test.helpers import ieq


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

class ScriptedStream(io.RawIOBase):
    """Non-seekable binary stream driven by a queue of chunks.

    Each ``read(n)`` returns the next queued chunk, so a chunk shorter than
    ``n`` is a short read; an ``b''`` chunk models a transient empty read.
    Once the queue is exhausted every read returns ``b''`` (real EOF).
    ``seek`` is forbidden and recorded.
    """

    def __init__(self, chunks, boom_at=None):
        self._chunks = list(chunks)
        self.boom_at = boom_at
        self.reads = 0
        self.requests = []
        self.seek_calls = 0

    def readable(self):
        return True

    def seekable(self):
        return False

    def seek(self, *args, **kwargs):
        self.seek_calls += 1
        raise AssertionError('streaming reader must not seek')

    def read(self, n=-1):
        self.reads += 1
        assert n is not None and n >= 0, 'read() must be bounded'
        self.requests.append(n)
        if self.boom_at is not None and self.reads >= self.boom_at:
            raise OSError('stream exploded')
        if not self._chunks:
            return b''
        return self._chunks.pop(0)


class ScriptedSource(object):
    """Source returning a fresh stream per ``open('rb')``, like a file."""

    def __init__(self, make_chunks):
        self.make_chunks = make_chunks
        self.opens = 0
        self.closes = 0
        self.streams = []

    def open(self, mode='rb'):
        assert mode == 'rb'
        self.opens += 1
        stream = ScriptedStream(self.make_chunks())
        self.streams.append(stream)
        source = self

        class Context(object):
            def __enter__(self):
                return stream

            def __exit__(self, *exc):
                source.closes += 1
                return False

        return Context()


def chunked(data, size, empty_at=()):
    """Split ``data`` into ``size`` byte chunks with optional empty reads."""
    pieces = [data[i:i + size] for i in range(0, len(data), size)]
    chunks = []
    for index, piece in enumerate(pieces):
        if index in empty_at:
            chunks.append(b'')
        chunks.append(piece)
    return chunks


def write_bytes(tmpdir, data, name='data.json'):
    path = tmpdir.join(name)
    path.write_binary(data)
    return str(path)


RECORDS = b'[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]'
EXPECTED = [('a',), (1,), (2,), (3,), (4,)]


# --------------------------------------------------------------------------
# Basic equivalence across chunking and short reads
# --------------------------------------------------------------------------

@pytest.mark.parametrize('size', [1, 2, 3, 5, 1000])
@pytest.mark.parametrize('chunksize', [1, 4, 64])
def test_streaming_matches_reference_across_chunk_sizes(size, chunksize):
    source = ScriptedSource(lambda: chunked(RECORDS, size))
    actual = fromjson(source, streaming=True, chunksize=chunksize)
    ieq(EXPECTED, actual)


@pytest.mark.parametrize('empty_at', [set(), {0}, {3}, {0, 2, 6}])
def test_streaming_tolerates_transient_empty_reads(empty_at):
    source = ScriptedSource(lambda: chunked(RECORDS, 2, empty_at))
    ieq(EXPECTED, fromjson(source, streaming=True, chunksize=16))


def test_streaming_chunksize_one(tmpdir):
    path = write_bytes(tmpdir, RECORDS)
    ieq(EXPECTED, fromjson(path, streaming=True, chunksize=1))


def test_streaming_never_seeks_or_reads_unbounded():
    source = ScriptedSource(lambda: chunked(RECORDS, 1))
    # a single traversal: list() would add a __len__ pass and a second open
    for _ in fromjson(source, streaming=True, chunksize=7):
        pass
    assert source.opens == 1 and source.closes == 1
    assert all(stream.seek_calls == 0 for stream in source.streams)
    assert all(max(stream.requests) == 7 for stream in source.streams)


def test_first_rows_available_before_tail_is_read():
    class BlockingStream(io.RawIOBase):
        def __init__(self, head, tail):
            self._head = io.BytesIO(head)
            self._tail = io.BytesIO(tail)
            self._in_tail = False

        def readable(self):
            return True

        def seekable(self):
            return False

        def read(self, n=-1):
            if not self._in_tail:
                piece = self._head.read(n)
                if piece:
                    return piece
                self._in_tail = True
                raise Blocked()
            return self._tail.read(n)

    class Blocked(Exception):
        pass

    class Source(object):
        def open(self, mode='rb'):
            stream = BlockingStream(b'[{"a": 1}, ', b'{"a": 2}]')

            class Context(object):
                def __enter__(self):
                    return stream

                def __exit__(self, *exc):
                    return False

            return Context()

    # an explicit header means no sampling read-ahead: the first record is
    # produced before the unread tail is touched
    rows = iter(fromjson(Source(), streaming=True, header=['a'],
                         chunksize=64))
    assert next(rows) == ('a',)
    assert next(rows) == (1,)
    # the next record needs the tail, whose read blocks
    with pytest.raises(Blocked):
        next(rows)


# --------------------------------------------------------------------------
# Laziness and source lifecycle
# --------------------------------------------------------------------------

def test_view_construction_and_iter_do_not_open_source():
    source = ScriptedSource(lambda: chunked(RECORDS, 1))
    view = fromjson(source, streaming=True, chunksize=5)
    assert source.opens == 0
    iterator = iter(view)
    assert source.opens == 0
    assert next(iterator) == ('a',)
    assert source.opens == 1
    assert list(iterator) == EXPECTED[1:]
    assert source.opens == 1 and source.closes == 1


def test_repeated_traversal_reopens_source_once_each():
    source = ScriptedSource(lambda: chunked(RECORDS, 1))
    view = fromjson(source, streaming=True, chunksize=5)

    def materialize(table):
        rows = []
        for row in table:  # explicit loop: list(table) would add a __len__ pass
            rows.append(row)  # noqa: PERF402
        return rows

    for _ in range(3):
        assert materialize(view) == EXPECTED
    assert source.opens == 3 and source.closes == 3


def test_early_close_stops_reading_and_exits_context():
    data = b'[{"a": 1}, ' + b'{"a": 2}, ' * 100 + b'{"a": 99}]'
    source = ScriptedSource(lambda: chunked(data, 1))
    iterator = iter(fromjson(source, streaming=True, chunksize=3))
    next(iterator)
    next(iterator)
    next(iterator)
    iterator.close()
    assert source.closes == 1
    reads_at_close = source.streams[0].reads
    # the reader must not run on to EOF after close
    assert source.streams[0].reads == reads_at_close


def test_source_exception_still_closes_context():
    data = b'[{"a": 1}, ' + b'{"a": 2}, ' * 100 + b']'
    source = ScriptedSource(lambda: chunked(data, 1))
    original_open = source.open

    def failing_open(mode='rb'):
        context = original_open(mode)
        context.__enter__().boom_at = 25
        return context

    source.open = failing_open
    with pytest.raises(OSError, match='stream exploded'):
        list(fromjson(source, streaming=True, chunksize=4))
    assert source.closes == 1


# --------------------------------------------------------------------------
# Headers and sampling
# --------------------------------------------------------------------------

@pytest.mark.parametrize('header_type', [list, tuple, iter])
def test_explicit_header_types(tmpdir, header_type):
    path = write_bytes(tmpdir, RECORDS)
    actual = fromjson(path, streaming=True, header=header_type(['a']))
    ieq(EXPECTED, actual)
    ieq(EXPECTED, actual)  # reusable across passes


def test_explicit_empty_header(tmpdir):
    path = write_bytes(tmpdir, RECORDS)
    actual = fromjson(path, streaming=True, header=())
    assert list(actual) == [(), (), (), (), ()]


def test_header_iterator_consumed_only_once():
    consumed = []

    def header():
        for name in ('a',):
            consumed.append(name)
            yield name

    source = ScriptedSource(lambda: chunked(RECORDS, 1))
    view = fromjson(source, streaming=True, header=header())
    assert consumed == []
    for _ in view:
        pass
    assert consumed == ['a']
    for _ in view:
        pass
    assert consumed == ['a']


def test_header_is_lazy(tmpdir):
    calls = []

    def header():
        calls.append('header')
        yield 'a'

    path = write_bytes(tmpdir, RECORDS)
    view = fromjson(path, streaming=True, header=header())
    iterator = iter(view)
    assert calls == []
    assert next(iterator) == ('a',)
    assert calls == ['header']


@pytest.mark.parametrize('sample, expected', [
    (0, [(), (), ()]),
    (1, [('id',), ('001',), ('002',)]),
    (2, [('id', 'later'), ('001', None), ('002', 'v')]),
])
def test_sampling_semantics(sample, expected):
    data = b'[{"id": "001"}, {"id": "002", "later": "v"}]'
    source = ScriptedSource(lambda: chunked(data, 1))
    actual = fromjson(source, streaming=True, sample=sample)
    ieq(expected, actual)


def test_sampled_objects_are_not_decoded_twice(tmpdir):
    path = write_bytes(tmpdir, b'[{"x": 10}, {"x": 20}]')
    calls = []

    def parse_int(value):
        calls.append(value)
        return int(value)

    view = fromjson(path, streaming=True, sample=1, parse_int=parse_int,
                    chunksize=1)
    for _ in view:
        pass
    assert sorted(calls) == ['10', '20']


def test_empty_array_yields_empty_header(tmpdir):
    path = write_bytes(tmpdir, b'  [\n\t ]  ')
    assert list(fromjson(path, streaming=True)) == [()]
    assert list(fromjson(path, streaming=True, header=['ignored'])) == [
        ('ignored',)]


def test_missing_values_and_ignored_extra_keys(tmpdir):
    data = b'[{"a": 1, "b": 2}, {"a": 3},' \
           b' {"a": 4, "b": 5, "c": 6}]'
    path = write_bytes(tmpdir, data)
    actual = fromjson(path, streaming=True, header=['a', 'b'], missing='X')
    ieq([('a', 'b'), (1, 2), (3, 'X'), (4, 5)], actual)


def test_header_key_order_is_first_occurrence(tmpdir):
    data = b'[{"z": 1, "a": 2}, {"a": 3, "m": 4}]'
    path = write_bytes(tmpdir, data)
    assert next(iter(fromjson(path, streaming=True, chunksize=1))) == (
        'z', 'a', 'm')


# --------------------------------------------------------------------------
# UTF-8, escapes and nesting across chunk boundaries
# --------------------------------------------------------------------------

def test_multibyte_characters_split_at_every_boundary(tmpdir):
    text = u'[{"name": "章子怡 αβγ"},' \
           u' {"name": "Արաm"}]'
    data = text.encode('utf-8')
    path = write_bytes(tmpdir, data)
    for chunksize in (1, 2, 3, 4, 5, 7, 64):
        rows = list(fromjson(path, streaming=True, chunksize=chunksize))
        assert rows[1][0] == u'章子怡 αβγ'
        assert rows[2][0] == u'Արաm'


def test_escapes_and_unicode_escapes_split_across_chunks(tmpdir):
    text = '[{"s": "a\\nb\\tc\\"d\\\\e"},' \
           ' {"u": "x\\u00e9y\\u4e2dz"},' \
           ' {"raw": "é"}]'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    reference = json.loads(text)
    for chunksize in (1, 2, 3, 8):
        rows = list(fromjson(path, streaming=True, chunksize=chunksize))
        assert rows[1] == (reference[0]['s'], None, None)
        assert rows[2][:2] == (None, reference[1]['u'])
        assert rows[3][2] == reference[2]['raw']


def test_nested_objects_arrays_and_numbers_across_chunks(tmpdir):
    text = '[{"n": 1, "f": -3.5e2, "nest": [1, 2, {"k": true}],' \
           ' "nil": null}, {"n": 2}]'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    reference = json.loads(text)
    for chunksize in (1, 2, 5):
        rows = list(fromjson(path, streaming=True, chunksize=chunksize))
        assert rows[1][0] == reference[0]['n']
        assert rows[1][1] == reference[0]['f']
        assert rows[1][2] == reference[0]['nest']
        assert rows[1][3] is None


def test_utf8_decoding_failure_raises_unicode_decode_error(tmpdir):
    path = write_bytes(tmpdir, b'[{"a": "caf\xc3"}]')  # truncated sequence
    with pytest.raises(UnicodeDecodeError):
        list(fromjson(path, streaming=True, chunksize=1))


# --------------------------------------------------------------------------
# Grammar errors: type and position, invariant across chunk sizes
# --------------------------------------------------------------------------

MALFORMED = [
    '',
    '   ',
    '[',
    '[,',
    '[{"a":1},',
    '[{"a":1},\n {"b": }]',
    '[\n {"x": [1, 2,\n]}\n]',
    '[\n\n {"a": 1\n} x]',
    '[{"a": "unterminated}]',
    '[{"a": 1}, {"b": 2\n}',
    '[{"a":1},]',
    '[{"a":1} {"b":2}]',
    '[{"a":1}]x',
    '[ { "a" : 1 } \n extra ]',
    '[,]',
    '[]  \n\t x',
    '[ {"a":1} , ]',
    '[\n  {"a":1}\n, {"b": } ]',
    '[{"a":1}, ,]',
    '[{"a": 1',
    '[{"a":\n 1',
    '[{"a": [1, 2',
    '[{"a": "x',
    '[{"a": {"b": 1}',
    # errors on a later line of a multi-line record must keep document
    # line, column, offset and formatted message aligned with json.loads
    '[\n  {"a": 1},\n  {"b":\n    "x" 1\n  }\n]',
    '[{"a":\n 1}, {"c":\n  "d" x}]',
    '   \n  [\n {"a": 1},\n {"b": }\n  ]',
    '[\n\n{\n "a":\n [1,\n  2 bad]\n}\n]',
]


@pytest.mark.parametrize('text', MALFORMED)
@pytest.mark.parametrize('chunksize', [1, 3, 16, 4096])
def test_syntax_error_matches_stdlib_position(tmpdir, text, chunksize):
    with pytest.raises(JSONDecodeError) as expected:
        json.loads(text)
    path = write_bytes(tmpdir, text.encode('utf-8'))
    with pytest.raises(JSONDecodeError) as raised:
        list(fromjson(path, streaming=True, chunksize=chunksize))
    err = raised.value
    ref = expected.value
    assert (err.msg, err.lineno, err.colno, err.pos) == (
        ref.msg, ref.lineno, ref.colno, ref.pos)
    # the formatted message is rebuilt from the absolute coordinates
    assert str(err) == str(ref)


@pytest.mark.parametrize('text', [
    '{"a": 1}', '1', '"s"', 'true', 'null',
    '[1, 2]', '["x"]', '[true]', '[null]',
    '[{"a": 1}, 2]', '[{"a": 1}, "s"]',
])
def test_non_array_or_non_object_members_are_argument_errors(tmpdir, text):
    path = write_bytes(tmpdir, text.encode('utf-8'))
    with pytest.raises(ArgumentError):
        list(fromjson(path, streaming=True, chunksize=1))


def test_valid_records_survive_a_later_syntax_error(tmpdir):
    path = write_bytes(tmpdir, b'[{"a": 1}, {"a": 2}, BROKEN]')
    iterator = iter(fromjson(path, streaming=True, header=['a'],
                             chunksize=1))
    rows = []
    with pytest.raises(JSONDecodeError):
        for row in iterator:
            rows.append(row)  # noqa: PERF402 - loop must stop at the error
    assert rows == [('a',), (1,), (2,)]


# --------------------------------------------------------------------------
# max_record_chars
# --------------------------------------------------------------------------

def test_record_length_limit_counts_characters_not_bytes(tmpdir):
    record = u'{"": "é"}'  # more UTF-8 bytes than decoded characters
    n_chars = len(record)
    assert len(record.encode('utf-8')) > n_chars
    path = write_bytes(tmpdir, (u'[' + record + u']').encode('utf-8'))
    # exactly at the limit is accepted
    list(fromjson(path, streaming=True,
                  max_record_chars=n_chars, chunksize=1))
    # one character over fails
    with pytest.raises(ArgumentError):
        list(fromjson(path, streaming=True,
                      max_record_chars=n_chars - 1, chunksize=1))


def test_record_length_includes_braces_and_internal_whitespace(tmpdir):
    path = write_bytes(tmpdir, b'[{}]')
    list(fromjson(path, streaming=True, max_record_chars=2, chunksize=1))
    with pytest.raises(ArgumentError):
        list(fromjson(path, streaming=True, max_record_chars=1, chunksize=1))


def test_record_length_excludes_between_object_whitespace(tmpdir):
    obj = '{"k": 1}'
    assert len(obj) == 8
    text = '[' + obj + (' ' * 5000) + ',' + (' ' * 5000) + obj + ']'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    list(fromjson(path, streaming=True, max_record_chars=8, chunksize=1))


def test_limit_error_position_is_stable_across_chunk_sizes(tmpdir):
    record = '{"k": "' + ('x' * 50) + '"}'
    path = write_bytes(tmpdir, ('[' + record + ']').encode('utf-8'))
    positions = set()
    for chunksize in (1, 7, 64):
        with pytest.raises(ArgumentError) as raised:
            list(fromjson(path, streaming=True,
                          max_record_chars=10, chunksize=chunksize))
        message = str(raised.value)
        positions.add(message)
    assert len(positions) == 1  # identical message regardless of chunking


# --------------------------------------------------------------------------
# Option validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('option', ['chunksize', 'max_record_chars'])
@pytest.mark.parametrize('value', [0, -1, True, False, 1.0, '10', None])
def test_size_options_must_be_positive_ints_not_bools(option, value):
    with pytest.raises(ArgumentError):
        fromjson('unused.json', streaming=True, **{option: value})


def test_lines_and_streaming_rejected_before_opening():
    source = ScriptedSource(lambda: [b'[]'])
    with pytest.raises(ArgumentError):
        fromjson(source, streaming=True, lines=True)
    assert source.opens == 0


def test_custom_cls_rejected_in_streaming_mode(tmpdir):
    path = write_bytes(tmpdir, b'[]')
    with pytest.raises(ArgumentError):
        fromjson(path, streaming=True, cls=json.JSONDecoder)


def test_extra_positional_argument_rejected_in_streaming_mode(tmpdir):
    path = write_bytes(tmpdir, b'[]')
    with pytest.raises(ArgumentError):
        fromjson(path, object, streaming=True)


def test_legacy_mode_keeps_accepting_cls(tmpdir, monkeypatch):
    path = write_bytes(tmpdir, b'[{"a": 1}]')
    captured = {}
    original_load = json_module.json.load

    def spy_load(stream, *args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(json_module.json, 'load', spy_load)
    list(fromjson(path, cls=json.JSONDecoder,
                  chunksize=123, max_record_chars=456, streaming=False))
    # the new streaming controls must never leak into json.load
    assert 'chunksize' not in captured['kwargs']
    assert 'max_record_chars' not in captured['kwargs']
    assert 'streaming' not in captured['kwargs']
    assert captured['kwargs']['cls'] is json.JSONDecoder


# --------------------------------------------------------------------------
# Decoder options
# --------------------------------------------------------------------------

def test_parse_float_decimal_preserved(tmpdir):
    text = '[{"x": 0.1234567890123456789}, {"x": 1.5}]'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    rows = list(fromjson(path, streaming=True, parse_float=Decimal,
                         chunksize=1))
    assert rows[1][0] == Decimal('0.1234567890123456789')
    assert isinstance(rows[1][0], Decimal)
    assert rows[2][0] == Decimal('1.5')


def test_all_standard_decoder_options_reach_record_decoding(tmpdir):
    text = '[{"i": 9007199254740993, "f": 0.5, "c": NaN}]'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    rows = list(fromjson(
        path, streaming=True, chunksize=1,
        parse_int=str, parse_float=Decimal,
        parse_constant=lambda value: 'const:' + value))
    assert rows[1] == ('9007199254740993', Decimal('0.5'), 'const:NaN')


def test_object_hook_applies_to_sampled_and_data_records(tmpdir):
    path = write_bytes(tmpdir, b'[{"v": 1}, {"v": 2}]')
    seen = []

    def hook(record):
        seen.append(record['v'])
        return OrderedDict([('doubled', record['v'] * 2)])

    view = fromjson(path, streaming=True, object_hook=hook,
                    sample=1, chunksize=1)
    rows = []
    for row in view:  # exactly one traversal; list(view) would iterate twice
        rows.append(row)  # noqa: PERF402
    assert seen == [1, 2]
    assert rows == [('doubled',), (2,), (4,)]


def test_object_pairs_hook_must_still_return_dict(tmpdir):
    path = write_bytes(tmpdir, b'[{"v": 1}]')

    def pairs_hook(pairs):
        return OrderedDict(pairs)

    rows = list(fromjson(path, streaming=True,
                         object_pairs_hook=pairs_hook, chunksize=1))
    assert rows == [('v',), (1,)]

    with pytest.raises(ArgumentError):
        list(fromjson(path, streaming=True,
                      object_pairs_hook=lambda pairs: [pairs], chunksize=1))


def test_object_hook_returning_non_dict_is_argument_error(tmpdir):
    path = write_bytes(tmpdir, b'[{"v": 1}]')
    with pytest.raises(ArgumentError):
        list(fromjson(path, streaming=True,
                      object_hook=lambda record: list(record), chunksize=1))


def test_strict_option_is_passed_through(tmpdir):
    path = write_bytes(tmpdir, b'[{"a": "x\ty"}]')
    # strict=False allows the literal control character
    list(fromjson(path, streaming=True, strict=False, chunksize=1))
    with pytest.raises(JSONDecodeError):
        list(fromjson(path, streaming=True, strict=True, chunksize=1))


def test_unknown_decoder_option_raises_type_error(tmpdir):
    path = write_bytes(tmpdir, b'[{"a": 1}]')
    with pytest.raises(TypeError):
        list(fromjson(path, streaming=True, no_such_option=1))


# --------------------------------------------------------------------------
# Compression and source types
# --------------------------------------------------------------------------

@pytest.mark.parametrize('suffix, opener', [
    ('.gz', gzip.open),
    ('.bz2', bz2.open),
])
@pytest.mark.parametrize('chunksize', [1, 7, 1000])
def test_compressed_files(tmpdir, suffix, opener, chunksize):
    payload = json.dumps(
        [{'z': i, 'name': u'é %d' % i} for i in range(50)]
    ).encode('utf-8')
    path = tmpdir.join('data' + suffix)
    with opener(str(path), 'wb') as handle:
        handle.write(payload)
    rows = list(fromjson(str(path), streaming=True, chunksize=chunksize))
    assert len(rows) == 51
    assert rows[1][0] == 0
    assert rows[-1][0] == 49


def test_memory_source(tmpdir):
    payload = json.dumps([{'z': i} for i in range(20)]).encode('utf-8')
    source = etl.MemorySource(payload)
    rows = list(fromjson(source, streaming=True, chunksize=5))
    assert len(rows) == 21


# --------------------------------------------------------------------------
# Memory bounds and scaling
# --------------------------------------------------------------------------

def test_long_whitespace_does_not_accumulate_memory():
    data = b'[{"a": 1}' + b' ' * 200000 + b', {"a": 2}]'
    stream = _ChunkedByteStream(io.BytesIO(data), chunksize=64)
    records = _iter_array_records(stream, 8388608)
    # drive the scanner through the whole inter-record whitespace gap,
    # tracking the retained buffer at every step
    peak_buffered = 0
    for _ in records:
        peak_buffered = max(peak_buffered, len(stream._buf) - stream._head)
    assert peak_buffered < 5000


def test_processing_a_long_record_is_not_quadratic(tmpdir):
    import time
    text = '[' + ','.join(
        '{"id": %d, "name": "record-number-%d"}' % (i, i)
        for i in range(400)) + ']'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    start = time.perf_counter()
    big_rows = list(fromjson(path, streaming=True, chunksize=4096))
    big_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    one_rows = list(fromjson(path, streaming=True, chunksize=1))
    one_elapsed = time.perf_counter() - start
    assert big_rows == one_rows and len(one_rows) == 401
    # byte-at-a-time must not degrade quadratically; a generous guard keeps
    # the assertion stable across loaded CI machines
    assert one_elapsed < max(big_elapsed * 25, 0.5)


def test_empty_read_run_below_threshold_recovers(tmpdir):
    # isolated empty reads (each followed by data again) must never be
    # mistaken for EOF even when they recur throughout the document
    chunks = []
    for byte in RECORDS:
        chunks.append(b'')
        chunks.append(bytes([byte]))
    source = ScriptedSource(lambda: chunks)
    ieq(EXPECTED, fromjson(source, streaming=True, chunksize=64))


def test_persistent_empty_reads_are_accepted_as_eof():
    # a closed stream returns b'' on every read; beyond the retry
    # threshold the reader must terminate rather than loop forever
    class EndlessEmpty(io.RawIOBase):
        def __init__(self):
            self.reads = 0

        def readable(self):
            return True

        def read(self, n=-1):
            self.reads += 1
            return b''

    stream = EndlessEmpty()

    class Source(object):
        def open(self, mode='rb'):
            class Context(object):
                def __enter__(self):
                    return stream

                def __exit__(self, *exc):
                    return False

            return Context()

    with pytest.raises(JSONDecodeError):
        list(fromjson(Source(), streaming=True))
    # bounded: the reader gave up after roughly the empty-read threshold,
    # rather than spinning indefinitely
    assert stream.reads <= _EMPTY_READ_EOF + 2


# --------------------------------------------------------------------------
# Backwards compatibility of the default (non-streaming) paths
# --------------------------------------------------------------------------

def test_default_array_path_unchanged(tmpdir):
    path = write_bytes(tmpdir, RECORDS)
    ieq(EXPECTED, fromjson(path))
    ieq(EXPECTED, fromjson(path, streaming=False))


def test_default_lines_path_unchanged(tmpdir):
    lines = b'{"a": 1}\n{"a": 2}\n'
    path = write_bytes(tmpdir, lines, name='data.jsonl')
    ieq([('a',), (1,), (2,)], fromjson(path, lines=True))


def test_streaming_equivalent_to_default_on_normal_documents(tmpdir):
    text = '[{"foo": "a", "bar": 1}, {"foo": "b"},' \
           ' {"foo": "c", "bar": 2, "baz": true}]'
    path = write_bytes(tmpdir, text.encode('utf-8'))
    default_rows = list(fromjson(path))
    for chunksize in (1, 13, 4096):
        assert list(fromjson(path, streaming=True,
                             chunksize=chunksize)) == default_rows
