# -*- coding: utf-8 -*-
"""Chunk-invariance regression tests for the streaming JSON array reader.

The same byte string must deliver the same complete record prefix and the
same earliest error no matter how the binary stream is split into reads.
The tests treat the splitting strategy as a variable:

* every cut point of small inputs is tried one byte at a time;
* several fixed ``chunksize`` values and randomized short reads are added;
* delivered rows, the exception category and JSON character positions are
  compared for every strategy and against :func:`json.loads` where the
  whole document happens to be valid or to fail identically.
"""
from __future__ import absolute_import, print_function, division

import gzip
import io
import json
import random
from collections import OrderedDict
from json import JSONDecodeError

import pytest

from petl import fromjson
from petl.errors import ArgumentError
from petl.io.json_streaming import _ChunkedByteStream


# --------------------------------------------------------------------------
# Stream and Source test doubles
# --------------------------------------------------------------------------

class _ScriptedStream(io.RawIOBase):
    """Non-seekable binary stream driven by a queue of chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.reads = 0
        self.requests = []
        self.seek_calls = 0
        self.max_request = 0
        self.peak_buffer_seen = 0

    def readable(self):
        return True

    def seekable(self):
        return False

    def seek(self, *args, **kwargs):
        self.seek_calls += 1
        raise AssertionError('streaming reader must not seek')

    def read(self, n=-1):
        self.reads += 1
        assert n is not None and n > 0, 'every read needs a positive limit'
        self.requests.append(n)
        self.max_request = max(self.max_request, n)
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
        stream = _ScriptedStream(self.make_chunks())
        self.streams.append(stream)
        source = self

        class Context(object):
            def __enter__(self):
                return stream

            def __exit__(self, *exc):
                source.closes += 1
                return False

        return Context()


def byte_cuts(data):
    """Every possible single split point of ``data`` (two pieces)."""
    return [[data[:i], data[i:]] for i in range(len(data) + 1)]


def all_singletons(data):
    """One byte per read, including an empty read up front."""
    return [b''] + [bytes([b]) for b in data]


def random_short_reads(data, seed, cap=13):
    """Randomly bounded chunks, each strictly shorter than the request."""
    rng = random.Random(seed)
    chunks = []
    pos = 0
    while pos < len(data):
        step = rng.randint(1, cap)
        chunks.append(data[pos:pos + step])
        pos += step
        if rng.random() < 0.15:
            chunks.append(b'')  # occasional transient empty read
    return chunks


def splitting_strategies(data, chunksizes=(1, 2, 3, 5, 64)):
    """Yield (label, chunks, chunksize) variants covering every cut."""
    for cut in byte_cuts(data):
        yield 'cut', cut, 64
    for chunksize in chunksizes:
        step = chunksize
        chunks = [data[i:i + step]
                  for i in range(0, len(data), step)]
        yield 'fixed', chunks, chunksize
    for seed in range(5):
        yield 'random', random_short_reads(data, seed), 16
    yield 'singletons', all_singletons(data), 1


# --------------------------------------------------------------------------
# Outcome collection
# --------------------------------------------------------------------------

def collect_outcome(data, chunks, chunksize, header=('id',), **kwargs):
    """Run one traversal and return (rows, error_signature).

    The error signature is (category, pos, lineno, colno, message) for
    JSONDecodeError and (category, reason, illegal_bytes) for
    UnicodeDecodeError; anything else carries its type name and message.
    """
    source = ScriptedSource(lambda: list(chunks))
    rows = []
    signature = None
    try:
        for row in fromjson(source, streaming=True, header=list(header),
                            chunksize=chunksize, **kwargs):
            rows.append(row)
    except JSONDecodeError as error:
        signature = ('JSONDecodeError', error.pos, error.lineno,
                     error.colno, str(error))
    except UnicodeDecodeError as error:
        signature = ('UnicodeDecodeError', error.reason,
                     bytes(error.object[error.start:error.end]))
    except Exception as error:  # signature records the concrete type
        signature = (type(error).__name__, str(error))
    assert source.opens == 1 and source.closes == 1
    stream = source.streams[0]
    assert stream.seek_calls == 0
    assert stream.max_request <= chunksize
    return rows, signature


def assert_strategies_agree(data, header=('id',), **kwargs):
    outcomes = {}
    for label, chunks, chunksize in splitting_strategies(data):
        outcomes[(label, chunksize)] = collect_outcome(
            data, chunks, chunksize, header=header, **kwargs)
    distinct = set((tuple(map(tuple, rows)), sig)
                   for rows, sig in outcomes.values())
    assert len(distinct) == 1, (
        'chunk-dependent behavior for %r:\n%r' % (data, outcomes))
    return next(iter(outcomes.values()))


# --------------------------------------------------------------------------
# The two reported defects and the error-prefix delivery contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize('chunksize', [1, 2, 3, 7, 64])
def test_reported_case_bad_byte_after_two_records(chunksize):
    data = b'[{"id":1},{"id":2},\xff]'
    source = ScriptedSource(
        lambda: [data[i:i + chunksize]
                 for i in range(0, len(data), chunksize)])
    rows, signature = collect_outcome(
        data, source.make_chunks(), chunksize)
    assert rows == [('id',), (1,), (2,)]
    assert signature[0] == 'UnicodeDecodeError'
    assert signature[2] == b'\xff'


@pytest.mark.parametrize('chunksize', [1, 2, 3, 7, 64])
def test_reported_case_syntax_error_before_bad_byte(chunksize):
    data = b'[{"id":1},!\xff]'
    source = ScriptedSource(
        lambda: [data[i:i + chunksize]
                 for i in range(0, len(data), chunksize)])
    rows, signature = collect_outcome(
        data, source.make_chunks(), chunksize)
    assert rows == [('id',), (1,)]
    assert signature[0] == 'JSONDecodeError'
    # character position 10 in the original document, not a byte offset
    assert signature[1] == 10
    assert signature[2] == 1 and signature[3] == 11


def test_exhaustive_split_points_for_reported_cases():
    rows, sig = assert_strategies_agree(b'[{"id":1},{"id":2},\xff]')
    assert rows == [('id',), (1,), (2,)] and sig[0] == 'UnicodeDecodeError'
    rows, sig = assert_strategies_agree(b'[{"id":1},!\xff]')
    assert rows == [('id',), (1,)] and sig[0] == 'JSONDecodeError'
    assert sig[1] == 10


def test_syntax_error_inside_object_then_bad_byte_prefers_json_error():
    # the '!' is a definite grammar fault before the bad UTF-8 byte
    data = b'[{"a": 1}, {"b": !\xff}]'
    rows, sig = assert_strategies_agree(data, header=('a', 'b'))
    assert rows == [('a', 'b'), (1, None)]
    assert sig[0] == 'JSONDecodeError'


def test_bad_byte_inside_otherwise_valid_prefix_prefers_unicode_error():
    # nothing is grammatically wrong before the bad byte
    data = b'[{"a": 1}, {"b": "ok\xff"}]'
    rows, sig = assert_strategies_agree(data, header=('a', 'b'))
    assert rows == [('a', 'b'), (1, None)]
    assert sig[0] == 'UnicodeDecodeError'
    assert sig[2] == b'\xff'


def test_truncated_keyword_then_bad_byte_prefers_json_error():
    # 'tru' cannot become a value where one was required: definite grammar
    # fault at the keyword position, before the bad byte
    data = b'[{"a": 1}, {"b": tru\xff}]'
    rows, sig = assert_strategies_agree(data, header=('a', 'b'))
    assert rows == [('a', 'b'), (1, None)]
    assert sig[0] == 'JSONDecodeError'


def test_bad_unicode_escape_before_bad_byte_prefers_json_error():
    data = b'[{"a": 1}, {"b": "x\\u00\xff"}]'
    rows, sig = assert_strategies_agree(data, header=('a', 'b'))
    assert rows == [('a', 'b'), (1, None)]
    assert sig[0] == 'JSONDecodeError'
    assert sig[4].startswith('Invalid \\uXXXX escape')


def test_unterminated_string_then_bad_byte_prefers_unicode_error():
    data = b'[{"a": 1}, {"b": "open\xff'
    rows, sig = assert_strategies_agree(data, header=('a', 'b'))
    assert rows == [('a', 'b'), (1, None)]
    assert sig[0] == 'UnicodeDecodeError'


def test_nested_syntax_error_before_bad_byte_prefers_json_error():
    data = b'[{"a": [1, 2, ! \xff]}]'
    rows, sig = assert_strategies_agree(data, header=('a',))
    assert rows == [('a',)]
    assert sig[0] == 'JSONDecodeError'


def test_dangling_multibyte_sequence_is_unicode_error_only_at_eof():
    data = b'[{"a": "caf\xc3"}]'
    rows, sig = assert_strategies_agree(data, header=('a',))
    assert rows == [('a',)]
    assert sig[0] == 'UnicodeDecodeError'
    # the real illegal bytes are preserved rather than replaced/ignored
    assert sig[2] == b'\xc3'


def test_multibyte_sequence_completed_by_later_chunk_is_valid(tmpdir):
    # the split lead byte must wait instead of failing early
    pieces = [b'[{"a": "caf', b'\xc3', b'\xa9"}]']
    source = ScriptedSource(lambda: list(pieces))
    rows = list(fromjson(source, streaming=True, header=['a'],
                         chunksize=64))
    assert rows == [('a',), (u'café',)]


def test_bad_byte_between_records_keeps_earlier_records():
    data = b'[{"id": 1}, {"id": 2}, \xff, {"id": 3}]'
    rows, sig = assert_strategies_agree(data)
    assert rows == [('id',), (1,), (2,)]
    assert sig[0] == 'UnicodeDecodeError'
    # the record after the error point is never decoded
    assert all(r != (3,) for r in rows)


def test_bad_byte_after_closed_array_is_unicode_error():
    data = b'  [{"id": 1}]\xff'
    rows, sig = assert_strategies_agree(data)
    assert rows == [('id',), (1,)]
    assert sig[0] == 'UnicodeDecodeError'


# --------------------------------------------------------------------------
# Incomplete prefixes wait for more input; only EOF decides
# --------------------------------------------------------------------------

class _BlockingSource(object):
    """Hands the prefix out, then blocks until the suffix is released."""

    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def open(self, mode='rb'):
        head = io.BytesIO(self.prefix)
        tail = io.BytesIO(self.suffix)

        class Stream(io.RawIOBase):
            def readable(self):
                return True

            def seekable(self):
                return False

            def read(self, n=-1):
                piece = head.read(n)
                if piece:
                    return piece
                return tail.read(n)

        stream = Stream()

        class Context(object):
            def __enter__(self):
                return stream

            def __exit__(self, *exc):
                return False

        return Context()


def test_incomplete_json_prefix_is_not_an_error_before_eof():
    source = _BlockingSource(b'[{"id": 1}, {"id":', b' 2}]')
    rows = list(fromjson(source, streaming=True, header=['id'],
                         chunksize=64))
    assert rows == [('id',), (1,), (2,)]


def test_truncated_json_at_eof_is_json_error_across_splits():
    data = b'[{"id": 1}, {"id": 2'
    rows, sig = assert_strategies_agree(data)
    assert rows == [('id',), (1,)]
    assert sig[0] == 'JSONDecodeError'


def test_truncated_utf8_at_eof_is_unicode_error_across_splits():
    data = b'[{"id": 1}, {"name": "\xe4\xb8'  # 中 cut after two bytes
    rows, sig = assert_strategies_agree(data, header=('id', 'name'))
    assert rows == [('id', 'name'), (1, None)]
    assert sig[0] == 'UnicodeDecodeError'


def test_empty_input_behaves_the_same_for_every_split():
    for data in (b'', b'   ', b'  \n\t '):
        rows, sig = assert_strategies_agree(data, header=('x',))
        assert rows == [('x',)]
        assert sig[0] == 'JSONDecodeError'


# --------------------------------------------------------------------------
# Header sampling follows the same prefix contract
# --------------------------------------------------------------------------

def test_sampling_holds_header_back_until_sample_complete():
    # only one complete record exists, but three are requested: sampling
    # cannot complete, so the grammar error surfaces on the first next()
    data = b'[{"id": 1}, !]'
    split = data.index(b'!')
    chunks = [data[:split], data[split:]]
    source = ScriptedSource(lambda: list(chunks))
    iterator = iter(fromjson(source, streaming=True, sample=3,
                             chunksize=64))
    with pytest.raises(JSONDecodeError):
        next(iterator)
    assert source.closes == 1


def test_sampling_error_before_completion_delivers_nothing():
    data = b'[{"id": 1}, BROKEN]'
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        delivered = []
        with pytest.raises(JSONDecodeError):
            for row in fromjson(source, streaming=True, sample=5,
                                chunksize=chunksize):
                delivered.append(row)
        assert delivered == []


def test_sampling_unicode_error_before_completion_delivers_nothing():
    data = b'[{"id": 1}, \xff]'
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        delivered = []
        with pytest.raises(UnicodeDecodeError):
            for row in fromjson(source, streaming=True, sample=5,
                                chunksize=chunksize):
                delivered.append(row)
        assert delivered == []


def test_header_emitted_after_sample_then_later_error_keeps_rows():
    data = b'[{"id": 1}, {"id": 2}, !]'
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        rows = []
        with pytest.raises(JSONDecodeError):
            for row in fromjson(source, streaming=True, sample=1,
                                chunksize=chunksize):
                rows.append(row)
        assert rows == [('id',), (1,), (2,)]


def test_header_emitted_when_array_closes_before_sample_reached(tmpdir):
    data = b'[{"id": 1}]'
    path = tmpdir.join('data.json')
    path.write_binary(data)
    rows = list(fromjson(str(path), streaming=True, sample=100))
    assert rows[0] == ('id',) and rows[1] == (1,)


# --------------------------------------------------------------------------
# Valid nested data, escapes and multibyte text under every split
# --------------------------------------------------------------------------

VALID_DOCUMENTS = [
    b'[{"a": {"b": [1, 2, {"c": true}]}, "d": null}, {"a": {}}]',
    u'[{"s": "line\\n\\"quote\\\\end", "u": "x\\u00e9", "raw": "é"}]'
    .encode('utf-8'),
    u'[{"中": "文", "emoji": "\U0001f600", "mix": "a中b"}]'.encode('utf-8'),
    b'[{"a": -3.5e2, "b": 0.0001, "c": [true, false, null]}]',
    b'[ { "a" : 1 } ,\n\t { "a" : 2 } ]',
    b'[{"nested": [[], [[]], {"k": [{"j": []}]}]}]',
]


@pytest.mark.parametrize('data', VALID_DOCUMENTS)
def test_valid_documents_identical_under_all_splits(data):
    reference = json.loads(data.decode('utf-8'))
    seen = set()
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        # serialise rows so nested lists/dicts are comparable by value
        rows = json.dumps(list(fromjson(source, streaming=True,
                                        chunksize=chunksize)),
                          default=str, ensure_ascii=False, sort_keys=True)
        seen.add(rows)
    assert len(seen) == 1
    # the streaming rows describe the same records as the stdlib document
    rendered = json.loads(next(iter(seen)))
    assert len(rendered) == len(reference) + 1


def test_multibyte_text_survives_every_cut_point_and_empty_read():
    text = u'[{"é": "中文字符串"}, {"é": "😀emoji"}]'
    data = text.encode('utf-8')
    seen = set()
    for cut in range(1, len(data) + 1):
        rows, signature = collect_outcome(
            data, [data[:cut], data[cut:]], 64, header=(u'é',))
        assert signature is None
        seen.add(tuple(rows))
        # an isolated transient empty read between the two pieces
        rows, signature = collect_outcome(
            data, [data[:cut], b'', data[cut:]], 64, header=(u'é',))
        assert signature is None
        seen.add(tuple(rows))
    assert len(seen) == 1
    assert seen.pop() == ((u'é',), (u'中文字符串',), (u'😀emoji',))


# --------------------------------------------------------------------------
# Hook ordering, counts and pass-through of user exceptions
# --------------------------------------------------------------------------

def test_hooks_run_once_per_object_and_not_after_error_point():
    data = b'[{"v": 1}, {"v": 2}, \xff]'
    for label, chunks, chunksize in splitting_strategies(data):
        calls = []

        def hook(record):
            calls.append(record['v'])
            return record

        source = ScriptedSource(lambda: list(chunks))
        with pytest.raises(UnicodeDecodeError):
            list(fromjson(source, streaming=True, header=['v'],
                          object_hook=hook, chunksize=chunksize))
        assert calls == [1, 2]


def test_hook_runs_on_sampled_and_data_records_exactly_once():
    data = b'[{"v": 10}, {"v": 20}, {"v": 30}]'
    calls = []

    def hook(record):
        calls.append(record['v'])
        return OrderedDict([('v', record['v'])])

    # a single explicit traversal: list() would add a second pass and a
    # second source open, which is expected Table behaviour
    source = ScriptedSource(lambda: all_singletons(data))
    rows = []
    for row in fromjson(source, streaming=True, sample=2,
                        object_hook=hook, chunksize=1):
        rows.append(row)
    assert rows == [('v',), (10,), (20,), (30,)]
    assert calls == [10, 20, 30]
    assert source.opens == 1 and source.closes == 1


def test_hook_exception_propagates_unchanged_across_splits():
    class Boom(Exception):
        pass

    def hook(record):
        if record['v'] == 2:
            raise Boom('user failure')
        return record

    data = b'[{"v": 1}, {"v": 2}, {"v": 3}]'
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        with pytest.raises(Boom, match='user failure'):
            list(fromjson(source, streaming=True, header=['v'],
                          object_hook=hook, chunksize=chunksize))


def test_pairs_hook_and_strict_option_survive_splits():
    data = b'[{"a": 1, "a": 2}]'
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        rows = list(fromjson(source, streaming=True, header=['a'],
                             object_pairs_hook=OrderedDict,
                             chunksize=chunksize))
        assert rows[-1] == (2,)

    tabbed = b'[{"a": "x\ty"}]'
    for label, chunks, chunksize in splitting_strategies(tabbed):
        source = ScriptedSource(lambda: list(chunks))
        list(fromjson(source, streaming=True, header=['a'],
                      strict=False, chunksize=chunksize))
        source = ScriptedSource(lambda: list(chunks))
        with pytest.raises(JSONDecodeError):
            list(fromjson(source, streaming=True, header=['a'],
                          strict=True, chunksize=chunksize))


# --------------------------------------------------------------------------
# max_record_chars counts decoded characters
# --------------------------------------------------------------------------

def test_record_limit_exactly_at_limit_and_one_over():
    record = u'{"k": "é中"}'
    n_chars = len(record)
    data = (u'[' + record + u']').encode('utf-8')
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        list(fromjson(source, streaming=True, header=['k'],
                      max_record_chars=n_chars, chunksize=chunksize))
        source = ScriptedSource(lambda: list(chunks))
        with pytest.raises(ArgumentError):
            list(fromjson(source, streaming=True, header=['k'],
                          max_record_chars=n_chars - 1,
                          chunksize=chunksize))


def test_record_limit_counts_characters_not_bytes_across_splits():
    # 4 decoded characters but 10 UTF-8 bytes
    data = u'[{"": "é"}]'.encode('utf-8')
    for label, chunks, chunksize in splitting_strategies(data):
        source = ScriptedSource(lambda: list(chunks))
        with pytest.raises(ArgumentError):
            list(fromjson(source, streaming=True,
                          max_record_chars=3, chunksize=chunksize))


# --------------------------------------------------------------------------
# gzip and Source lifecycle
# --------------------------------------------------------------------------

def test_gzip_file_across_chunk_sizes(tmpdir):
    payload = json.dumps([{'id': i, u'name': u' item-%d é' % i}
                          for i in range(30)]).encode('utf-8')
    path = tmpdir.join('data.json.gz')
    with gzip.open(str(path), 'wb') as handle:
        handle.write(payload)
    reference = list(fromjson(str(path), streaming=False))
    for chunksize in (1, 4, 64, 4096):
        rows = list(fromjson(str(path), streaming=True,
                             chunksize=chunksize))
        assert rows == reference


def test_context_exits_on_hook_error_and_early_close():
    data = b'[{"v": 1}, {"v": 2}]'

    def hook(record):
        raise RuntimeError('boom')

    source = ScriptedSource(lambda: all_singletons(data))
    with pytest.raises(RuntimeError):
        list(fromjson(source, streaming=True, header=['v'],
                      object_hook=hook, chunksize=1))
    assert source.closes == 1

    source = ScriptedSource(lambda: all_singletons(data))
    iterator = iter(fromjson(source, streaming=True, header=['v'],
                             chunksize=1))
    next(iterator)  # header row: with an explicit header the source is
    # opened lazily only when the first data row is pulled
    next(iterator)
    assert source.opens == 1
    iterator.close()
    assert source.closes == 1
    reads_at_close = source.streams[0].reads
    # closing must not run on to EOF
    assert source.streams[0].reads == reads_at_close


# --------------------------------------------------------------------------
# On-demand Sources: long documents never materialised in the test
# --------------------------------------------------------------------------

class GeneratingSource(object):
    """Stream a large document from fixed-size pieces on demand.

    The full payload is never held as one bytes object: pieces are created
    lazily and each read returns an independent copy. Tracking the largest
    piece ever handed out lets a test assert the streaming memory bound
    without buffering the document itself.
    """

    def __init__(self, piece_builder, read_cap):
        self.piece_builder = piece_builder
        self.read_cap = read_cap
        self.largest_piece = 0
        self.opens = 0
        self.closes = 0

    def open(self, mode='rb'):
        self.opens += 1
        builder = self.piece_builder()
        source = self

        class Stream(io.RawIOBase):
            def readable(self):
                return True

            def seekable(self):
                return False

            def seek(self, *args, **kwargs):
                raise AssertionError('must not seek')

            def read(self, n=-1):
                assert 0 < n <= source.read_cap
                piece = next(builder, b'')
                source.largest_piece = max(source.largest_piece,
                                           len(piece))
                return piece

        stream = Stream()

        class Context(object):
            def __enter__(self):
                return stream

            def __exit__(self, *exc):
                source.closes += 1
                return False

        return Context()


def test_error_after_very_long_single_line_keeps_position_and_memory():
    target_pos = 200000  # decoded characters on one line

    def builder():
        yield b'['
        yield b'{"id": 1},'
        yield b'{"long": "'
        # generate the filler lazily in small pieces
        remaining = target_pos - 30
        while remaining > 0:
            step = min(4096, remaining)
            yield b'a' * step
            remaining -= step
        yield b'", "bad": !}'
        yield b']'

    source = GeneratingSource(builder, read_cap=4096)
    with pytest.raises(JSONDecodeError) as raised:
        for row in fromjson(source, streaming=True, header=['id', 'long'],
                            chunksize=4096):
            pass
    error = raised.value
    assert error.lineno == 1
    assert error.pos >= target_pos
    assert source.opens == 1 and source.closes == 1
    # the source never produced a document-sized piece
    assert source.largest_piece <= 4096


def test_error_after_many_newlines_uses_character_offset():
    newline_count = 50000

    def builder():
        yield b'[{"id": 1},\n'
        remaining = newline_count
        while remaining > 0:
            step = min(4096, remaining)
            yield b'\n' * step
            remaining -= step
        yield b' {"id": !}]'

    source = GeneratingSource(builder, read_cap=4096)
    with pytest.raises(JSONDecodeError) as raised:
        list(fromjson(source, streaming=True, header=['id'],
                      chunksize=4096))
    error = raised.value
    # the error sits on the line after all the newlines
    assert error.lineno == newline_count + 2
    assert source.largest_piece <= 4096
    assert source.closes == 1


def test_stream_buffer_does_not_grow_with_document_length():
    # drive the low-level stream over a long generated run of whitespace
    # and assert the retained buffer stays around one chunk
    def builder():
        yield b' '
        for _ in range(100):
            yield b' ' * 4096
        yield b'[{"a": 1}]'

    class Binary(object):
        def read(self, n):
            return next(gen, b'')

    gen = builder()
    stream = _ChunkedByteStream(Binary(), chunksize=4096)
    peak = 0
    unit = stream.skip_ws()
    while isinstance(unit, int) and unit == 0x20:
        stream.consume()
        peak = max(peak, len(stream._buf) - stream._head)
        unit = stream.peek_unit()
    assert unit == 0x5B
    assert peak <= 4096 + 4


# --------------------------------------------------------------------------
# Randomized fuzz: compare every split strategy on mixed documents
# --------------------------------------------------------------------------

FUZZ_DOCUMENTS = [
    b'[{"id": 1}, {"id": 2}, !, {"id": 3}]',
    b'[{"id": 1}, {"id": 2}, \xff]',
    b'[{"a": "ok"}, {"b": "truncated',
    b'[{"a": 1}, {"b": [1, \xff]}]',
    b'  [{"x": true}\n, {"x": false}]\n  ',
    b'[{"\xc3\xa9": 1}, {"\xc3\xa9": 2}, !]',
    b'[{"id":1},!]',
    b'[]',
    b'[{}]',
    b'[{"a": 1},]',
]


@pytest.mark.parametrize('data', FUZZ_DOCUMENTS)
def test_fuzz_invariance(data):
    rows, signature = assert_strategies_agree(data)
    # sanity: errors always carry a category, and rows start with header
    assert rows and rows[0] == ('id',)
