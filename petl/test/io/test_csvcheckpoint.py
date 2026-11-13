# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import builtins
import csv
import io
import json

import pytest

import petl as etl
from petl.compat import PY2
from petl.errors import ArgumentError
from petl.io import csvcheckpoint as checkpoint_mod
from petl.io.csvcheckpoint import fromcsvcheckpoint
from petl.test.helpers import eq_


pytestmark = pytest.mark.skipif(
    PY2, reason='fromcsvcheckpoint is a Python 3-only addition')


# ---------------------------------------------------------------------------
# Helpers


def write_bytes(tmp_path, raw, name='data.csv'):
    path = tmp_path / name
    path.write_bytes(raw)
    return str(path)


def baseline(path, **csvargs):
    """Read a file the way ``fromcsv`` does, as the comparison oracle."""
    with builtins.open(path, 'rb') as buf:
        text = io.TextIOWrapper(buf, encoding='utf-8', errors='strict',
                                newline='')
        try:
            rows = [tuple(row) for row in csv.reader(text, **csvargs)]
        finally:
            text.detach()
    if rows:
        first = list(rows[0])
        if first and isinstance(first[0], str) and first[0].startswith('﻿'):
            # fromcsvcheckpoint strips the file-start BOM; mirror that here.
            first[0] = first[0][1:]
            rows[0] = tuple(first)
    return rows


def read_all(path, checkpoint=None, chunksize=65536, **csvargs):
    return [tuple(row) for row in fromcsvcheckpoint(
        path, checkpoint=checkpoint, chunksize=chunksize, **csvargs)]


FIXTURES = [
    # plain, LF
    (b'foo,bar\n1,2\n3,4\n', {}),
    # CRLF and no final terminator
    (b'foo,bar\r\n1,2\r\n3,4', {}),
    # lone CR
    (b'foo,bar\r1,2\r3,4\r', {}),
    # mixed terminators
    (b'foo,bar\r\n1,2\n3,4\r5,6\r\n', {}),
    # quoted multi-line field with CRLF, embedded delimiter and doubled quote
    (b'foo,bar\r\n"a,1 ""x""\r\nline2",2\r\nb,3\r\n', {}),
    # unicode, BOM, quoted newline, no final newline
    ('﻿名称,编号\n"张三\n李四",1\n"€",2'.encode('utf-8'), {}),
    # escape char, quoted escape and escaped newline
    (b'foo,bar\n"x\\"y",2\n"p\\\nq",3\n', dict(escapechar='\\')),
    # QUOTE_NONNUMERIC with quoted header and mixed data
    (b'"foo","bar"\n1,2\n"x",3\n', dict(quoting=csv.QUOTE_NONNUMERIC)),
    # QUOTE_NONE
    (b'foo;bar\n1;2\n3;4\n', dict(delimiter=';', quoting=csv.QUOTE_NONE)),
    # custom dialect with skipinitialspace
    (b"foo|bar\n 'x' |2\n", dict(delimiter='|', quotechar="'",
                                 skipinitialspace=True)),
    # multi-line header
    (b'"fo\no",bar\n1,2\n', {}),
    # header only, with and without final terminator
    (b'foo,bar\n', {}),
    (b'foo,bar', {}),
    # single field rows
    (b'name\n' + b'\n'.join(b'n%d' % i for i in range(50)) + b'\n', {}),
    # a final row without a terminator after a terminated one
    (b'a,b\n1,2\n3,4', {}),
    # CRLF final row without a terminator
    (b'a,b\r\n1,2\r\n3,4', {}),
    # a large record (larger than any short chunk) plus many rows
    (b'foo,bar\n' + b'"' + b'x' * 5000 + b'\n' + b'y' * 5000 + b'",2\n'
     + b''.join(b'%d,%d\n' % (i, i) for i in range(30)), {}),
]

CHUNKSIZES = [1, 2, 3, 7, 64, 65536]


# ---------------------------------------------------------------------------
# Equivalence with a normal full read


@pytest.mark.parametrize('index', range(len(FIXTURES)))
def test_full_read_matches_baseline(tmp_path, index):
    raw, csvargs = FIXTURES[index]
    path = write_bytes(tmp_path, raw)
    expect = baseline(path, **csvargs)
    for chunksize in CHUNKSIZES:
        eq_(expect, read_all(path, chunksize=chunksize, **csvargs))


@pytest.mark.parametrize('index', range(len(FIXTURES)))
def test_resume_at_every_boundary_reconstructs_table(tmp_path, index):
    raw, csvargs = FIXTURES[index]
    path = write_bytes(tmp_path, raw)
    expect = baseline(path, **csvargs)
    header, data = expect[0], expect[1:]

    # A checkpoint after the header alone.
    cursor = iter(fromcsvcheckpoint(path, chunksize=3, **csvargs))
    eq_(header, next(cursor))
    header_checkpoint = cursor.checkpoint()
    cursor.close()
    eq_(expect,
        read_all(path, checkpoint=header_checkpoint, chunksize=2, **csvargs))

    # A checkpoint after every delivered data record.
    cursor = iter(fromcsvcheckpoint(path, chunksize=3, **csvargs))
    delivered = [next(cursor)]
    for _ in range(len(data)):
        delivered.append(next(cursor))
        saved = json.loads(json.dumps(cursor.checkpoint()))
        resumed = read_all(path, checkpoint=saved, chunksize=2, **csvargs)
        eq_(header, resumed[0])
        eq_(expect, delivered + resumed[1:])
    cursor.close()


def test_resume_after_every_row_with_one_byte_chunks(tmp_path):
    raw = (b'foo,bar\r\n"a,1 ""x""\r\nline2",2\r\n"p\\\nq",3\r\n'
           b'\xe4\xb8\xad,4')
    path = write_bytes(tmp_path, raw)
    expect = read_all(path)
    eq_([('foo', 'bar'), ('a,1 "x"\r\nline2', '2'), ('p\\\nq', '3'),
         ('中', '4')], expect)

    cursor = iter(fromcsvcheckpoint(path, chunksize=1))
    delivered = [next(cursor)]
    for _ in range(len(expect) - 1):
        delivered.append(next(cursor))
        saved = cursor.checkpoint()
        resumed = read_all(path, checkpoint=saved, chunksize=1)
        eq_(expect, delivered + resumed[1:])
    cursor.close()


def test_types_are_preserved_on_resume(tmp_path):
    raw = b'"foo","bar"\n1,2\n"x",3\n'
    path = write_bytes(tmp_path, raw)
    cursor = iter(fromcsvcheckpoint(path, quoting=csv.QUOTE_NONNUMERIC))
    next(cursor)
    eq_((1.0, 2.0), next(cursor))
    saved = cursor.checkpoint()
    cursor.close()
    resumed = read_all(path, checkpoint=saved,
                       quoting=csv.QUOTE_NONNUMERIC)
    eq_([('foo', 'bar'), ('x', 3.0)], resumed)
    eq_([str, float], [type(v) for v in resumed[1]])


# ---------------------------------------------------------------------------
# Cursor semantics


def test_checkpoint_before_header_raises(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n')
    view = fromcsvcheckpoint(path)
    cursor = iter(view)
    with pytest.raises(ArgumentError):
        cursor.checkpoint()
    next(cursor)
    cursor.checkpoint()
    cursor.close()


def test_view_and_cursor_are_lazy(tmp_path, monkeypatch):
    path = write_bytes(tmp_path, b'a,b\n1,2\n')
    opened = []

    def fail_open(*args, **kwargs):
        opened.append(args)
        raise AssertionError('file must not be opened here')

    monkeypatch.setattr(checkpoint_mod, '_open_binary', fail_open)
    view = fromcsvcheckpoint(path)
    cursor = iter(view)
    assert opened == []
    with pytest.raises(AssertionError):
        next(cursor)


def test_cursors_are_independent(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n5,6\n')
    view = fromcsvcheckpoint(path, chunksize=2)
    first = iter(view)
    second = iter(view)
    eq_(('a', 'b'), next(first))
    eq_(('a', 'b'), next(second))
    eq_(('1', '2'), next(first))
    eq_(('1', '2'), next(second))
    first.checkpoint()
    eq_(('3', '4'), next(second))
    eq_(('3', '4'), next(first))
    first.close()
    # closing one cursor does not affect the other
    eq_(('5', '6'), next(second))
    # a third cursor starts from the view's checkpoint, not from `saved`
    third = iter(view)
    eq_(('a', 'b'), next(third))
    eq_(('1', '2'), next(third))
    # the same view can be iterated again after full consumption
    eq_([('a', 'b'), ('1', '2'), ('3', '4'), ('5', '6')], list(view))


def test_checkpoint_is_detached_and_does_not_consume(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    first = cursor.checkpoint()
    second = cursor.checkpoint()
    eq_(first, second)
    first['position']['data_records'] = 99
    first['header'].append('zzz')
    eq_(second, cursor.checkpoint())
    # no record was consumed by either call
    eq_(('1', '2'), next(cursor))


def test_external_checkpoint_is_not_mutated(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    seed = iter(fromcsvcheckpoint(path))
    next(seed)
    next(seed)
    saved = json.loads(json.dumps(seed.checkpoint()))
    snapshot = json.loads(json.dumps(saved))
    seed.close()
    cursor = iter(fromcsvcheckpoint(path, checkpoint=saved))
    next(cursor)
    next(cursor)
    cursor.checkpoint()
    cursor.close()
    eq_(snapshot, saved)


def test_close_is_idempotent_and_next_does_not_reopen(tmp_path, monkeypatch):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    cursor.close()

    def fail_open(*args, **kwargs):
        raise AssertionError('must not reopen after close')

    monkeypatch.setattr(checkpoint_mod, '_open_binary', fail_open)
    with pytest.raises(StopIteration):
        next(cursor)
    # checkpoint remains available without touching the file
    eq_(saved, cursor.checkpoint())


def test_exhausted_cursor_reports_last_checkpoint(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n')
    cursor = iter(fromcsvcheckpoint(path))
    eq_(('a', 'b'), next(cursor))
    eq_(('1', '2'), next(cursor))
    with pytest.raises(StopIteration):
        next(cursor)
    saved = cursor.checkpoint()
    eq_(1, saved['position']['data_records'])
    with pytest.raises(StopIteration):
        next(cursor)
    eq_(saved, cursor.checkpoint())


def test_close_before_first_next_never_opens(tmp_path, monkeypatch):
    path = write_bytes(tmp_path, b'a,b\n1,2\n')

    def fail_open(*args, **kwargs):
        raise AssertionError('must not open a closed cursor')

    cursor = iter(fromcsvcheckpoint(path))
    cursor.close()
    monkeypatch.setattr(checkpoint_mod, '_open_binary', fail_open)
    with pytest.raises(StopIteration):
        next(cursor)
    # The header was never obtained, so no checkpoint can be produced.
    with pytest.raises(ArgumentError):
        cursor.checkpoint()


def test_table_api_works(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    table = fromcsvcheckpoint(path)
    eq_(('a', 'b'), table.header())
    eq_([('a', 'b'), ('1', '2'), ('3', '4')], list(table))


# ---------------------------------------------------------------------------
# Content binding and relocation


def test_identical_copy_at_another_path_resumes(tmp_path):
    raw = b'a,b\n1,2\n3,4\n'
    first = write_bytes(tmp_path, raw, 'one.csv')
    second = write_bytes(tmp_path, raw, 'two.csv')
    cursor = iter(fromcsvcheckpoint(first))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    eq_([('a', 'b'), ('3', '4')], read_all(second, checkpoint=saved))
    # the checkpoint does not record the path
    assert 'one.csv' not in json.dumps(saved)
    assert 'two.csv' not in json.dumps(saved)


def test_same_size_modified_file_is_rejected(tmp_path):
    raw = b'a,b\n1,2\n3,4\n'
    path = write_bytes(tmp_path, raw)
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    modified = bytearray(raw)
    modified[-2] = ord('9')
    eq_(len(raw), len(modified))
    write_bytes(tmp_path, bytes(modified))
    with pytest.raises(ArgumentError):
        read_all(path, checkpoint=saved)


def test_truncated_file_is_rejected(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    write_bytes(tmp_path, b'a,b\n1,2\n')
    with pytest.raises(ArgumentError):
        read_all(path, checkpoint=saved)


def test_appended_file_is_rejected(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n')
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    with pytest.raises(ArgumentError):
        read_all(path, checkpoint=saved)


def test_parse_configuration_change_is_rejected(tmp_path):
    path = write_bytes(tmp_path, b'a;b\n1;2\n')
    cursor = iter(fromcsvcheckpoint(path, delimiter=';'))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    with pytest.raises(ArgumentError):
        read_all(path, checkpoint=saved, delimiter=',')
    with pytest.raises(ArgumentError):
        read_all(path, checkpoint=saved, delimiter=';', strict=True)


# ---------------------------------------------------------------------------
# Checkpoint structure validation


def _checkpoint_after_one_row(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n')
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    return path, saved


@pytest.mark.parametrize('case', [
    'missing-file', 'missing-position', 'wrong-type', 'unknown-version',
    'unknown-format', 'checksum', 'unexpected-field', 'bad-json',
    'not-a-dict', 'empty-header', 'duplicate-header', 'bad-config',
    'missing-checksum', 'negative-offset', 'bad-sha', 'file-extra-field',
    'position-extra-field', 'nested-wrong-type',
])
def test_bad_checkpoints_are_rejected(tmp_path, case):
    path, saved = _checkpoint_after_one_row(tmp_path)
    if case == 'missing-file':
        del saved['file']
    elif case == 'missing-position':
        del saved['position']
    elif case == 'wrong-type':
        saved['position']['data_records'] = '1'
    elif case == 'unknown-version':
        saved['version'] = 99
    elif case == 'unknown-format':
        saved['format'] = 'something-else'
    elif case == 'checksum':
        saved['position']['data_records'] = 0
    elif case == 'unexpected-field':
        saved['extra'] = True
    elif case == 'bad-json':
        saved = '{not json'
    elif case == 'not-a-dict':
        saved = [1, 2, 3]
    elif case == 'empty-header':
        saved['header'] = []
    elif case == 'duplicate-header':
        saved['header'] = ['a', 'a']
    elif case == 'bad-config':
        saved['config']['delimiter'] = ',,'
    elif case == 'missing-checksum':
        del saved['checksum']
    elif case == 'negative-offset':
        saved['position']['byte_offset'] = -1
    elif case == 'bad-sha':
        saved['file']['sha256'] = 'xyz'
    elif case == 'file-extra-field':
        saved['file']['extra'] = 1
    elif case == 'position-extra-field':
        saved['position']['extra'] = 1
    elif case == 'nested-wrong-type':
        saved['config'] = 'not an object'
    with pytest.raises(ArgumentError):
        fromcsvcheckpoint(path, checkpoint=saved)


def test_checkpoint_claiming_a_mid_line_offset_is_rejected(tmp_path):
    # Build a checkpoint whose content checksum is valid but whose offset is
    # not at a record boundary.
    path, saved = _checkpoint_after_one_row(tmp_path)
    saved['position']['byte_offset'] = saved['position']['byte_offset'] - 1
    saved['checksum'] = checkpoint_mod._checkpoint_checksum(
        {k: v for k, v in saved.items() if k != 'checksum'})
    with pytest.raises(ArgumentError):
        read_all(path, checkpoint=saved)


def test_checkpoint_json_string_is_accepted(tmp_path):
    path, saved = _checkpoint_after_one_row(tmp_path)
    eq_([('a', 'b'), ('3', '4')],
        read_all(path, checkpoint=json.dumps(saved)))


# ---------------------------------------------------------------------------
# Errors, record numbering and failure semantics


def test_width_error_is_argument_error_with_position(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n1,2,3\n5,6\n')
    cursor = iter(fromcsvcheckpoint(path))
    eq_(('a', 'b'), next(cursor))
    eq_(('1', '2'), next(cursor))
    last = cursor.checkpoint()
    with pytest.raises(ArgumentError) as excinfo:
        next(cursor)
    message = str(excinfo.value)
    assert 'data record 2' in message
    assert 'physical line 3' in message
    # the failed record was not published
    eq_(last, cursor.checkpoint())
    with pytest.raises(StopIteration):
        next(cursor)
    # resuming from the last good position re-reports the same error
    resumed = iter(fromcsvcheckpoint(path, checkpoint=last))
    eq_(('a', 'b'), next(resumed))
    with pytest.raises(ArgumentError) as excinfo2:
        next(resumed)
    assert 'data record 2' in str(excinfo2.value)
    assert 'physical line 3' in str(excinfo2.value)


def test_csv_error_keeps_type_and_position(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n"x"y,2\n')
    cursor = iter(fromcsvcheckpoint(path, strict=True))
    next(cursor)
    with pytest.raises(csv.Error) as excinfo:
        next(cursor)
    message = str(excinfo.value)
    assert 'data record 1' in message
    assert 'physical line 2' in message


def test_decode_error_keeps_type_and_cursor_terminates(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n\xff,3\n')
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    eq_(('1', '2'), next(cursor))
    last = cursor.checkpoint()
    with pytest.raises(UnicodeDecodeError):
        next(cursor)
    eq_(last, cursor.checkpoint())
    with pytest.raises(StopIteration):
        next(cursor)


def test_error_numbering_continues_after_resume(tmp_path):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n5\n')
    cursor = iter(fromcsvcheckpoint(path, chunksize=2))
    next(cursor)
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    resumed = iter(fromcsvcheckpoint(path, checkpoint=saved))
    eq_(('a', 'b'), next(resumed))
    with pytest.raises(ArgumentError) as excinfo:
        next(resumed)
    message = str(excinfo.value)
    assert 'data record 3' in message
    assert 'physical line 4' in message


# ---------------------------------------------------------------------------
# Lazy decoding must not be confused with downstream progress


def test_probe_restore_parses_only_header(tmp_path, monkeypatch):
    path = write_bytes(tmp_path, b'a,b\n1,2\n3,4\n5,6\n7,8\n')

    created = []

    class CountingReader(object):

        def __init__(self, inner):
            self.inner = inner
            self.records = 0
            created.append(self)

        def __iter__(self):
            return self

        def __next__(self):
            row = next(self.inner)
            self.records += 1
            return row

    original = checkpoint_mod._csv_reader

    def factory(iterable, **kwargs):
        return CountingReader(original(iterable, **kwargs))

    monkeypatch.setattr(checkpoint_mod, '_csv_reader', factory)

    # Full read parses the header plus every data record.
    eq_(5, len(read_all(path)))
    eq_([5], [reader.records for reader in created])

    # Make a checkpoint after the second data record, then resume.
    saved = None
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()
    eq_(2, saved['position']['data_records'])

    created[:] = []
    resumed = read_all(path, checkpoint=saved)
    eq_([('a', 'b'), ('5', '6'), ('7', '8')], resumed)
    # One reader parses only the header, the other only the two remaining
    # records: the prefix's CSV records are never handed to the parser.
    eq_(sorted(reader.records for reader in created), [1, 2])


def test_probe_chunk_read_sizes_and_file_release(tmp_path, monkeypatch):
    raw = b'a,b\r\n"x\r\ny",2\r\n3,4'
    path = write_bytes(tmp_path, raw)

    handles = []
    original = checkpoint_mod._open_binary

    class RecordingHandle(object):

        def __init__(self, handle):
            self.handle = handle
            self.reads = []
            self.was_closed = False

        def read(self, size=-1):
            data = self.handle.read(size)
            self.reads.append((size, len(data)))
            return data

        def seek(self, offset, whence=0):
            return self.handle.seek(offset, whence)

        def close(self):
            self.was_closed = True
            self.handle.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.close()

    def factory(*args, **kwargs):
        handle = RecordingHandle(original(*args, **kwargs))
        handles.append(handle)
        return handle

    monkeypatch.setattr(checkpoint_mod, '_open_binary', factory)

    for chunksize in (1, 2, 5, 65536):
        handles[:] = []
        result = read_all(path, chunksize=chunksize)
        eq_([('a', 'b'), ('x\r\ny', '2'), ('3', '4')], result)
        assert handles
        for handle in handles:
            assert handle.was_closed, 'file handle was not released'
            for size, _ in handle.reads:
                assert 0 < size <= chunksize, (size, chunksize)


def test_probe_bounded_reads_while_validating(tmp_path, monkeypatch):
    # A resumed read must scan the whole file for the digest (bounded reads),
    # so this checks the largest single read never exceeds chunksize even for
    # a file much larger than the chunk size.
    raw = b'a,b\n' + b''.join(b'%d,%d\n' % (i, i) for i in range(2000))
    path = write_bytes(tmp_path, raw)
    cursor = iter(fromcsvcheckpoint(path))
    next(cursor)
    next(cursor)
    saved = cursor.checkpoint()
    cursor.close()

    sizes = []
    original = checkpoint_mod._open_binary

    class RecordingHandle(object):

        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            sizes.append(size)
            return self.handle.read(size)

        def seek(self, offset, whence=0):
            return self.handle.seek(offset, whence)

        def close(self):
            self.handle.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.close()

    monkeypatch.setattr(
        checkpoint_mod, '_open_binary',
        lambda *a, **k: RecordingHandle(original(*a, **k)))
    read_all(path, checkpoint=saved, chunksize=128)
    assert max(sizes) <= 128


# ---------------------------------------------------------------------------
# Argument validation


@pytest.mark.parametrize('path', [
    None, 1, object(), '/tmp/data.csv.gz',
    '/tmp/data.csv.bz2', '/tmp/data.csv.zip', 'http://example.com/a.csv',
    'ftp://example.com/a.csv',
])
def test_invalid_path_is_rejected(path):
    with pytest.raises(ArgumentError):
        fromcsvcheckpoint(path)


def test_path_must_exist_at_iteration(tmp_path):
    view = fromcsvcheckpoint(str(tmp_path / 'missing.csv'))
    with pytest.raises(ArgumentError):
        next(iter(view))


def test_directory_path_is_rejected(tmp_path):
    view = fromcsvcheckpoint(str(tmp_path))
    with pytest.raises(ArgumentError):
        next(iter(view))


@pytest.mark.parametrize('chunksize', [0, -1, True, False, 1.5, '64', None])
def test_invalid_chunksize_is_rejected(tmp_path, chunksize):
    path = write_bytes(tmp_path, b'a,b\n')
    with pytest.raises(ArgumentError):
        fromcsvcheckpoint(path, chunksize=chunksize)


@pytest.mark.parametrize('csvargs', [
    dict(dialect='excel'),
    dict(dialect='excel-tab'),
    dict(lineterminator='\n'),
    dict(quotechar=''),
    dict(delimiter=''),
    dict(delimiter=None),
    dict(quoting=99),
    dict(quoting='minimal'),
    dict(strict='yes'),
    dict(doublequote='yes'),
    dict(unknown_option=True),
    # explicit header/encoding/errors are not part of this interface
    dict(header=['a', 'b']),
    dict(encoding='utf-8'),
    dict(errors='strict'),
])
def test_unsupported_csv_options_are_rejected(tmp_path, csvargs):
    path = write_bytes(tmp_path, b'a,b\n')
    with pytest.raises(ArgumentError):
        fromcsvcheckpoint(path, **csvargs)


def test_header_must_be_nonempty_and_unique(tmp_path):
    for raw in (b'', b'\n', b'a,a\n', b'a,,c\n', b'a,\n',
                b'"a","a"\n'):
        path = write_bytes(tmp_path, raw)
        with pytest.raises(ArgumentError):
            next(iter(fromcsvcheckpoint(path)))


def test_file_with_no_data_rows_is_legal(tmp_path):
    for raw in (b'a,b\n', b'a,b', b'a,"b\nc"\n'):
        path = write_bytes(tmp_path, raw)
        rows = read_all(path)
        eq_(1, len(rows))
        cursor = iter(fromcsvcheckpoint(path))
        next(cursor)
        saved = cursor.checkpoint()
        cursor.close()
        eq_(rows, read_all(path, checkpoint=saved))


# ---------------------------------------------------------------------------
# Compatibility with the existing CSV and Source machinery


def test_existing_csv_entry_points_are_unaffected(tmp_path):
    path = write_bytes(tmp_path, b'foo,bar\r\na,1\r\nb,2\r\n')
    eq_([('foo', 'bar'), ('a', '1'), ('b', '2')], list(etl.fromcsv(path)))
    eq_([('foo', 'bar'), ('a', '1'), ('b', '2')], read_all(path))
    out = tmp_path / 'out.csv'
    etl.tocsv([('foo', 'bar'), ('a', 1)], str(out))
    eq_(b'foo,bar\r\na,1\r\n', out.read_bytes())
    tsvin = write_bytes(tmp_path, b'foo\tbar\n1\t2\n', 'in.tsv')
    eq_([('foo', 'bar'), ('1', '2')], list(etl.fromtsv(tsvin)))
