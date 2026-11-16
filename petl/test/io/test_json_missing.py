# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import json

import pytest

from petl import MISSING, fromdicts, fromjson, tojson, ArgumentError
from petl.io.json import JsonView, DictsGeneratorView


# --------------------------------------------------------------------------
# input: preserve_missing
# --------------------------------------------------------------------------

SPARSE_RECORDS = [
    {'id': 1, 'note': 'a'},
    {'id': 2, 'note': None},     # explicit null
    {'note': 'c'},               # missing id
    {'id': 4},                   # missing note
    {},                          # missing both
]
HEADER = ['id', 'note']
EXPECTED_PRESERVED = [
    ('id', 'note'),
    (1, 'a'),
    (2, None),
    (MISSING, 'c'),
    (4, MISSING),
    (MISSING, MISSING),
]


def test_fromdicts_preserve_missing():
    table = fromdicts(SPARSE_RECORDS, header=HEADER, preserve_missing=True)
    rows = [tuple(r) for r in table]
    assert rows == EXPECTED_PRESERVED
    # exact identity checks, not just equality
    data_rows = rows[1:]
    assert data_rows[2][0] is MISSING
    assert data_rows[3][1] is MISSING
    assert data_rows[1][1] is None
    assert data_rows[4][0] is MISSING and data_rows[4][1] is MISSING


def test_fromdicts_default_keeps_legacy_none_fill():
    table = fromdicts(SPARSE_RECORDS, header=HEADER)
    rows = [tuple(r) for r in table]
    # row for {'id': 4} (input index 3) keeps id 4 with note filled to None
    assert rows[4] == (4, None)
    assert all(cell is not MISSING for row in rows[1:] for cell in row)


def test_fromdicts_preserve_missing_with_generator():
    gen = (rec for rec in SPARSE_RECORDS)
    table = fromdicts(gen, header=HEADER, preserve_missing=True)
    assert isinstance(table, DictsGeneratorView)
    first_pass = [tuple(r) for r in table]
    assert first_pass == EXPECTED_PRESERVED
    # generator view is cached in a temp file; second pass must keep identity
    second_pass = [tuple(r) for r in table]
    assert second_pass == first_pass
    # data row for {'id': 4} is at output index 4 (header plus three rows)
    assert second_pass[4][0] == 4 and second_pass[4][1] is MISSING


def test_fromdicts_preserve_missing_sampled_header():
    records = [{'id': 1}, {'note': 'x'}, {'id': 3, 'note': 'y'}]
    table = fromdicts(records, sample=2, preserve_missing=True)
    rows = [tuple(r) for r in table]
    assert rows[0] == ('id', 'note')
    # no automatic header expansion after sampling
    assert rows[1] == (1, MISSING)
    assert rows[2] == (MISSING, 'x')
    assert rows[3] == (3, 'y')


def test_fromdicts_preserve_missing_empty_input():
    assert [tuple(r) for r in fromdicts([], preserve_missing=True)] == [()]
    assert [tuple(r) for r in fromdicts(iter([]), preserve_missing=True)] == [()]


def test_preserve_missing_conflicts_with_missing_value():
    with pytest.raises(ArgumentError):
        fromdicts(SPARSE_RECORDS, header=HEADER, missing=-1,
                  preserve_missing=True)
    # missing=None remains allowed
    table = fromdicts(SPARSE_RECORDS, header=HEADER, missing=None,
                      preserve_missing=True)
    assert list(table)[1] == (1, 'a')


@pytest.mark.parametrize('lines', [False, True])
def test_fromjson_preserve_missing(tmpdir, lines):
    if lines:
        content = '\n'.join(json.dumps(r) for r in SPARSE_RECORDS)
    else:
        content = json.dumps(SPARSE_RECORDS)
    path = tmpdir.join('sparse.json')
    path.write(content)
    table = fromjson(str(path), header=HEADER, lines=lines,
                     preserve_missing=True)
    assert isinstance(table, JsonView)
    rows = [tuple(r) for r in table]
    assert rows == EXPECTED_PRESERVED
    # repeat iteration reopens the source and preserves identity again
    rows2 = [tuple(r) for r in table]
    assert rows2 == EXPECTED_PRESERVED


@pytest.mark.parametrize('lines', [False, True])
def test_fromjson_preserve_missing_sampled_header(tmpdir, lines):
    records = [{'id': 1, 'note': 'a'}, {'id': 2}]
    content = ('\n'.join(json.dumps(r) for r in records) if lines
               else json.dumps(records))
    path = tmpdir.join('sampled.json')
    path.write(content)
    table = fromjson(str(path), lines=lines, preserve_missing=True)
    rows = [tuple(r) for r in table]
    assert rows[0] == ('id', 'note')
    assert rows[2] == (2, MISSING)


@pytest.mark.parametrize('lines', [False, True])
def test_fromjson_preserve_missing_empty(tmpdir, lines):
    path = tmpdir.join('empty.json')
    if lines:
        path.write('')
    else:
        path.write('[]')
    table = fromjson(str(path), lines=lines, preserve_missing=True)
    assert [tuple(r) for r in table] == [()]


@pytest.mark.parametrize('lines', [False, True])
def test_fromjson_argument_error_on_conflict(tmpdir, lines):
    path = tmpdir.join('x.json')
    path.write('[]' if not lines else '')
    with pytest.raises(ArgumentError):
        fromjson(str(path), lines=lines, missing=-1, preserve_missing=True)


def test_string_named_missing_is_not_marker():
    records = [{'id': 'MISSING'}, {'note': 'x'}]
    table = fromdicts(records, header=['id', 'note'], preserve_missing=True)
    rows = list(table)
    assert rows[1][0] == 'MISSING'
    assert rows[1][0] is not MISSING
    assert rows[2][0] is MISSING


# --------------------------------------------------------------------------
# output: omit_missing
# --------------------------------------------------------------------------

def _sparse_table():
    return [
        ('id', 'note'),
        (1, None),
        (MISSING, 'c'),
        (3, MISSING),
        (MISSING, MISSING),
    ]


@pytest.mark.parametrize('lines', [False, True])
def test_tojson_omit_missing(tmpdir, lines):
    path = tmpdir.join('out')
    tojson(_sparse_table(), str(path), omit_missing=True, lines=lines,
           sort_keys=True)
    text = path.read()
    if lines:
        decoded = [json.loads(line) for line in text.strip().split('\n')]
    else:
        decoded = json.loads(text)

    # full expected rows and exact key sets per row
    assert decoded[0] == {'id': 1, 'note': None}
    assert decoded[1] == {'note': 'c'}
    assert decoded[2] == {'id': 3}
    assert decoded[3] == {}
    assert [set(obj) for obj in decoded] == [
        {'id', 'note'}, {'note'}, {'id'}, set()
    ]
    # explicit None is still a null on the wire
    assert '"note": null' in text


@pytest.mark.parametrize('lines', [False, True])
def test_tojson_typeerror_without_omit(tmpdir, lines):
    path = tmpdir.join('bad')
    with pytest.raises(TypeError):
        tojson(_sparse_table(), str(path), lines=lines)
    # nothing was written for the failed encode
    assert not path.check()


def test_tojson_omit_roundtrip_array_and_lines(tmpdir):
    table = _sparse_table()
    p_arr = tmpdir.join('rt.json')
    p_lines = tmpdir.join('rt.jsonl')
    tojson(table, str(p_arr), omit_missing=True)
    tojson(table, str(p_lines), omit_missing=True, lines=True)
    back_arr = [tuple(r) for r in
                fromjson(str(p_arr), header=['id', 'note'],
                         preserve_missing=True)]
    back_lines = [tuple(r) for r in
                  fromjson(str(p_lines), lines=True, header=['id', 'note'],
                           preserve_missing=True)]
    expected = [tuple(r) for r in table]
    assert back_arr == expected
    assert back_lines == expected
    # identity survives the full sparse -> JSON -> read chain; the data row
    # (3, MISSING) is expected index 3 (header at 0, then three data rows)
    assert back_arr[3][0] == 3 and back_arr[3][1] is MISSING


def test_tojson_ordinary_table_unchanged(tmpdir):
    table = [('foo', 'bar'), ('a', 1), ('b', None)]
    path = tmpdir.join('ordinary.json')
    tojson(table, str(path), sort_keys=True)
    assert json.loads(path.read()) == [
        {'bar': 1, 'foo': 'a'}, {'bar': None, 'foo': 'b'}
    ]
    # default path must still tolerate None
    tojson(table, str(tmpdir.join('ordinary2.json')), omit_missing=True)


def test_tojson_short_rows_keep_legacy_padding(tmpdir):
    table = [('a', 'b'), (1,)]
    path = tmpdir.join('short.json')
    tojson(table, str(path))
    assert json.loads(path.read()) == [{'a': 1, 'b': None}]


if __name__ == '__main__':
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__]))
