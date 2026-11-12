from __future__ import absolute_import, print_function, division


import os

import pytest

import petl as etl
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.transform.setops import KEYDIFF_HEADER


RESULT_HEADER = ('op', 'key', 'before', 'after', 'changed')


def _changes(result):
    rows = [tuple(row) for row in result]
    assert KEYDIFF_HEADER == RESULT_HEADER
    assert rows[0] == RESULT_HEADER
    return rows[1:]


def _reorder(header, target_header, rows):
    indices = [list(header).index(f) for f in target_header]
    return [tuple(row[i] for i in indices) for row in rows]


def _apply(before_header, before_rows, changes, key_fields):
    """Apply the change stream to before and return a keyed mapping."""
    key_indices = [list(before_header).index(f) for f in key_fields]
    records = {}
    for row in before_rows:
        k = tuple(row[i] for i in key_indices)
        records[k] = tuple(row)
    for op, k, before_row, after_row, changed in changes:
        if op == 'delete':
            assert records[k] == before_row
            del records[k]
        elif op == 'insert':
            assert k not in records
            assert before_row is None
            records[k] = after_row
        elif op == 'update':
            assert records[k] == before_row
            records[k] = after_row
        else:
            raise AssertionError('unexpected op %r' % (op,))
    return records


def _after_mapping(after_header, after_rows, before_header, key_fields):
    reordered = _reorder(after_header, before_header, after_rows)
    key_indices = [list(before_header).index(f) for f in key_fields]
    return {tuple(row[i] for i in key_indices): row for row in reordered}


def _assert_reconstructs(before, after, changes, key_fields):
    applied = _apply(before[0], before[1:], changes, key_fields)
    expected = _after_mapping(after[0], after[1:], before[0], key_fields)
    assert applied == expected


class CountingSource(object):

    def __init__(self, rows):
        self.rows = rows
        self.iterations = 0
        self.consumed = 0
        self.closed = 0

    def __iter__(self):
        self.iterations += 1

        def gen():
            try:
                for row in self.rows:
                    self.consumed += 1
                    yield list(row)
            finally:
                self.closed += 1

        return gen()


class FailingSource(object):

    def __init__(self, rows, error):
        self.rows = rows
        self.error = error
        self.consumed = 0
        self.closed = 0

    def __iter__(self):
        def gen():
            try:
                for row in self.rows:
                    self.consumed += 1
                    yield list(row)
                raise self.error
            finally:
                self.closed += 1

        return gen()


class CloseFailingIterator(object):

    def __init__(self, error):
        self.error = error
        self.i = 0
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == 1:
            return ['id', 'v']
        raise self.error

    next = __next__

    def close(self):
        self.close_calls += 1
        raise RuntimeError('close failed')


def test_basic_change_stream_only():
    before = [['id', 'name', 'score'],
              [1, 'Alice', 10],
              [2, 'Bob', 20],
              [3, 'Carol', 30]]
    after = [['id', 'name', 'score'],
             [1, 'Alice', 11],
             [4, 'Dave', 40],
             [2, 'Bob', 20]]
    changes = _changes(etl.keydiff(before, after, 'id'))
    assert changes == [
        ('update', (1,), (1, 'Alice', 10), (1, 'Alice', 11), ('score',)),
        ('delete', (3,), (3, 'Carol', 30), None, ()),
        ('insert', (4,), None, (4, 'Dave', 40), ()),
    ]
    _assert_reconstructs(before, after, changes, ['id'])


def test_contract_key_always_tuple():
    before = [['id', 'v'], [1, 'a']]
    after = [['id', 'v'], [1, 'b']]
    change = _changes(etl.keydiff(before, after, 'id'))[0]
    assert change[0] == 'update'
    assert change[1] == (1,)
    assert change[4] == ('v',)


def test_composite_key():
    before = [['a', 'b', 'v'],
              [1, 1, 'x'],
              [1, 2, 'y'],
              [2, 1, 'z']]
    # after's header is (b, a, v); keys are collected in key-name order
    # (a, b) and rows are always rendered in before's (a, b, v) order.
    after = [['b', 'a', 'v'],
             [1, 1, 'x2'],
             [2, 1, 'z'],
             [1, 2, 'y'],
             [3, 1, 'w']]
    changes = _changes(etl.keydiff(before, after, ('a', 'b')))
    assert changes == [
        ('update', (1, 1), (1, 1, 'x'), (1, 1, 'x2'), ('v',)),
        ('update', (1, 2), (1, 2, 'y'), (1, 2, 'z'), ('v',)),
        ('insert', (1, 3), None, (1, 3, 'w'), ()),
        ('update', (2, 1), (2, 1, 'z'), (2, 1, 'y'), ('v',)),
    ]
    _assert_reconstructs(before, after, changes, ['a', 'b'])


def test_header_reordering_uses_before_order():
    before = [['id', 'name', 'score'],
              [2, 'Bob', 20]]
    after = [['score', 'name', 'id'],
             [21, 'Bob', 2],
             [40, 'Dave', 4]]
    changes = _changes(etl.keydiff(before, after, 'id'))
    assert changes == [
        ('update', (2,), (2, 'Bob', 20), (2, 'Bob', 21), ('score',)),
        ('insert', (4,), None, (4, 'Dave', 40), ()),
    ]
    _assert_reconstructs(before, after, changes, ['id'])


def test_changed_order_follows_before_header():
    before = [['id', 'z', 'a', 'm'],
              [1, 1, 2, 3]]
    after = [['a', 'm', 'id', 'z'],
             [20, 3, 1, 10]]
    change = _changes(etl.keydiff(before, after, 'id'))[0]
    assert change[4] == ('z', 'a')


def test_none_key_and_none_values():
    before = [['id', 'v'],
              [None, 'a'],
              [1, 'b'],
              [2, 'c']]
    after = [['v', 'id'],
             [None, None],
             ['b', 1],
             ['c', 2]]
    changes = _changes(etl.keydiff(before, after, 'id'))
    assert changes == [
        ('update', (None,), (None, 'a'), (None, None), ('v',)),
    ]
    _assert_reconstructs(before, after, changes, ['id'])


def test_unhashable_attributes():
    before = [['id', 'payload', 'tags'],
              [1, {'a': [1, 2]}, ['x', 'y']],
              [2, {'a': [3]}, ['z']]]
    after = [['id', 'tags', 'payload'],
             [1, ['x', 'y'], {'a': [1, 3]}],
             [2, ['z'], {'a': [3]}]]
    changes = _changes(etl.keydiff(before, after, 'id'))
    assert changes == [
        ('update', (1,), (1, {'a': [1, 2]}, ['x', 'y']),
         (1, {'a': [1, 3]}, ['x', 'y']), ('payload',)),
    ]
    _assert_reconstructs(before, after, changes, ['id'])


def test_unhashable_key_values():
    before = [['k', 'v'], [[1], 'a'], [[2], 'b']]
    after = [['v', 'k'], ['a2', [1]], ['b', [2]]]
    changes = _changes(etl.keydiff(before, after, 'k'))
    assert changes == [
        ('update', ([1],), ([1], 'a'), ([1], 'a2'), ('v',)),
    ]


def test_empty_tables():
    header = ['id', 'v']
    only_header = [header]
    assert list(etl.keydiff(only_header, only_header, 'id')) == \
        [RESULT_HEADER]

    filled = [header, [1, 'a']]
    assert _changes(etl.keydiff(only_header, filled, 'id')) == [
        ('insert', (1,), None, (1, 'a'), ()),
    ]
    assert _changes(etl.keydiff(filled, only_header, 'id')) == [
        ('delete', (1,), (1, 'a'), None, ()),
    ]


def test_no_header_errors():
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff([], [['id'], [1]], 'id'))
    assert 'before' in str(excinfo.value)

    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff([['id'], [1]], [], 'id'))
    assert 'after' in str(excinfo.value)


def test_header_validation_errors():
    with pytest.raises(ArgumentError):
        list(etl.keydiff([['id', 'id'], [1, 2]],
                         [['id', 'id'], [1, 2]], 'id'))
    with pytest.raises(ArgumentError):
        list(etl.keydiff([['id', 'v'], [1, 2]],
                         [['id', 'v', 'v'], [1, 2, 3]], 'id'))
    with pytest.raises(ArgumentError):
        list(etl.keydiff([['id', 'v'], [1, 2]],
                         [['id', 'w'], [1, 2]], 'id'))


def test_key_selection_errors():
    t1 = [['id', 'v'], [1, 'a']]
    t2 = [['v', 'id'], ['a', 1]]
    with pytest.raises(FieldSelectionError):
        list(etl.keydiff(t1, [['v', 'id'], ['a', 1]], 'x'))
    with pytest.raises(ArgumentError):
        list(etl.keydiff(t1, t2, []))
    with pytest.raises(ArgumentError):
        list(etl.keydiff(t1, t2, ['id', 'id']))
    with pytest.raises(ArgumentError):
        list(etl.keydiff([['1', 'id', 'v'], ['x', 1, 'a']],
                         [['id', '1', 'v'], [1, 'x', 'a']], ['1', 1]))
    with pytest.raises(ArgumentError):
        list(etl.keydiff(t1, t2, None))
    with pytest.raises(ArgumentError):
        list(etl.keydiff(t1, t2, 0))


def test_duplicate_keys_before_and_after():
    before = [['id', 'v'], [1, 'a'], [1, 'a']]
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id'))
    message = str(excinfo.value)
    assert 'before' in message and 'row 2' in message

    before = [['id', 'v'], [1, 'a']]
    after = [['id', 'v'], [1, 'a'], [1, 'a']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id'))
    message = str(excinfo.value)
    assert 'after' in message and 'row 2' in message


def test_duplicate_key_located_in_unsorted_input():
    before = [['id', 'v'], [2, 'b'], [1, 'a'], [2, 'b2'], [3, 'c']]
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id'))
    message = str(excinfo.value)
    assert 'before' in message and 'row 3' in message


def test_duplicate_composite_key():
    before = [['a', 'b', 'v'], [1, 1, 'x'], [1, 2, 'y']]
    after = [['b', 'a', 'v'], [1, 1, 'x'], [1, 1, 'z']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, ['a', 'b']))
    assert 'after' in str(excinfo.value)


def test_row_width_errors_located():
    before = [['id', 'v'], [1, 'a'], [2]]
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id'))
    message = str(excinfo.value)
    assert 'before' in message and 'row 2' in message

    before = [['id', 'v'], [1, 'a']]
    after = [['id', 'v'], [1, 'a'], [2, 'b', 'extra']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id'))
    message = str(excinfo.value)
    assert 'after' in message and 'row 2' in message


def test_presorted_validation():
    before = [['id', 'v'], [2, 'a'], [1, 'b']]
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', presorted=True))
    message = str(excinfo.value)
    assert 'before' in message and 'row 2' in message

    before = [['id', 'v'], [1, 'a']]
    after = [['id', 'v'], [2, 'b'], [1, 'c']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', presorted=True))
    message = str(excinfo.value)
    assert 'after' in message and 'row 2' in message


def test_presorted_duplicate_key():
    before = [['id', 'v'], [1, 'a'], [1, 'a']]
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', presorted=True))
    assert 'before' in str(excinfo.value)


def test_errors_in_multi_pass_merge_clean_up(tmp_path, monkeypatch):
    import petl.transform.setops as setops
    monkeypatch.setattr(setops, 'KEYDIFF_MAX_OPEN_CHUNKS', 2)
    before = [['id', 'v'],
              [4, 'a'], [1, 'b'], [4, 'c'], [2, 'd']]
    after = [['id', 'v'], [1, 'b']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', buffersize=2,
                         tempdir=str(tmp_path)))
    assert 'before' in str(excinfo.value)
    assert os.listdir(str(tmp_path)) == []


def test_tail_errors_are_not_skipped():
    # after is exhausted first; a bad tail row on before must still fail.
    before = [['id', 'v'], [1, 'a'], [2, 'b'], [3]]
    after = [['id', 'v'], [1, 'z']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', presorted=True))
    message = str(excinfo.value)
    assert 'before' in message and 'row 3' in message

    # before is exhausted first; duplicate tail key on after must still fail.
    before = [['id', 'v'], [1, 'a']]
    after = [['id', 'v'], [2, 'b'], [2, 'c']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', presorted=True))
    message = str(excinfo.value)
    assert 'after' in message and 'row 2' in message


def test_tail_errors_with_external_sort():
    before = [['id', 'v']] + [[i, 'v%d' % i] for i in range(10)] + [[99]]
    after = [['id', 'v'], [5, 'changed']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', buffersize=3))
    message = str(excinfo.value)
    assert 'before' in message and 'row 11' in message


def test_complete_stream_ordered_and_reconstructs_after():
    before = [['id', 'name', 'tags'],
              [4, 'D', ['d']],
              [1, 'A', ['a']],
              [3, 'C', ['c']],
              [2, 'B', ['b']]]
    after = [['tags', 'name', 'id'],
             [['b2'], 'B', 2],
             [['a'], 'A', 1],
             [['e'], 'E', 5],
             [['z'], 'C', 3]]
    changes = _changes(etl.keydiff(before, after, 'id'))
    assert changes == [
        ('update', (2,), (2, 'B', ['b']), (2, 'B', ['b2']), ('tags',)),
        ('update', (3,), (3, 'C', ['c']), (3, 'C', ['z']), ('tags',)),
        ('delete', (4,), (4, 'D', ['d']), None, ()),
        ('insert', (5,), None, (5, 'E', ['e']), ()),
    ]
    keys = [c[1][0] for c in changes]
    assert keys == sorted(keys)
    _assert_reconstructs(before, after, changes, ['id'])


def test_duplicate_key_detected_across_chunks(tmp_path):
    before = [['id', 'v'],
              [5, 'a'], [1, 'b'], [3, 'c'], [2, 'd'], [4, 'e'], [3, 'f']]
    after = [['id', 'v'], [1, 'b']]
    with pytest.raises(ArgumentError) as excinfo:
        list(etl.keydiff(before, after, 'id', buffersize=2,
                         tempdir=str(tmp_path)))
    message = str(excinfo.value)
    assert 'before' in message and 'row 6' in message
    assert os.listdir(str(tmp_path)) == []


def test_source_error_during_external_sort_cleans_temp_files(tmp_path):
    error = ValueError('source failed')
    before = FailingSource([['id', 'v']] + [[i, 'b'] for i in range(10)],
                           error)
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ValueError) as excinfo:
        list(etl.keydiff(before, after, 'id', buffersize=2,
                         tempdir=str(tmp_path)))
    assert excinfo.value is error
    assert before.closed == 1
    assert os.listdir(str(tmp_path)) == []


def test_duplicate_key_error_during_merge_cleans_temp_files(tmp_path):
    # A duplicate only becomes adjacent during the external merge.
    before = [['id', 'v'],
              [4, 'a'], [1, 'b'], [4, 'c']]
    after = [['id', 'v'], [1, 'b']]
    with pytest.raises(ArgumentError):
        list(etl.keydiff(before, after, 'id', buffersize=2,
                         tempdir=str(tmp_path)))
    assert os.listdir(str(tmp_path)) == []


def test_update_insert_delete_stream():
    before = [['k', 'x', 'y'],
              [1, 'same', 'old'],
              [2, 'same', 'same'],
              [3, 'only-before', 'z']]
    after = [['y', 'k', 'x'],
             [None, 1, None],
             ['same', 2, 'same'],
             ['z', 3, 'only-before'],
             ['new', 4, 'only-after']]
    changes = _changes(etl.keydiff(before, after, 'k'))
    assert changes == [
        ('update', (1,), (1, 'same', 'old'), (1, None, None),
         ('x', 'y')),
        ('insert', (4,), None, (4, 'only-after', 'new'), ()),
    ]
    _assert_reconstructs(before, after, changes, ['k'])


def test_multi_pass_merge_bounds_open_chunks(tmp_path, monkeypatch):
    import petl.transform.setops as setops
    monkeypatch.setattr(setops, 'KEYDIFF_MAX_OPEN_CHUNKS', 2)
    n = 30
    before = [['id', 'v']] + [[i, 'b'] for i in range(n)]
    after = [['v', 'id']] + [['a', i] for i in range(n)]
    changes = _changes(etl.keydiff(before, after, 'id', buffersize=2,
                                   tempdir=str(tmp_path)))
    assert changes == [
        ('update', (i,), (i, 'b'), (i, 'a'), ('v',)) for i in range(n)
    ]
    assert os.listdir(str(tmp_path)) == []


def test_over_sort_buffer_matches_presorted(tmp_path):
    n = 40
    before_rows = []
    after_rows = []
    for i in range(n):
        before_rows.append([i, 'b%d' % i, i * 2])
        if i % 5 == 0:
            after_rows.append([i, 'b%d' % i, i * 2])
        elif i % 3 == 0:
            after_rows.append([i, 'B%d' % i, i * 2])
    for i in range(n, n + 7):
        after_rows.append([i, 'new%d' % i, i])
    before = [['id', 'name', 'score']] + before_rows
    after = [['score', 'name', 'id']] + [[r[2], r[1], r[0]]
                                         for r in after_rows]

    chunked = _changes(etl.keydiff(before, after, 'id', buffersize=4,
                                   tempdir=str(tmp_path)))
    presorted_before = [before[0]] + sorted(before[1:],
                                            key=lambda row: row[0])
    presorted_after = [after[0]] + sorted(after[1:],
                                          key=lambda row: row[2])
    presorted = _changes(etl.keydiff(presorted_before, presorted_after,
                                     'id', presorted=True))
    assert chunked == presorted
    _assert_reconstructs(before, after, chunked, ['id'])
    # no temporary files are left behind after normal exhaustion
    assert os.listdir(str(tmp_path)) == []


def test_over_sort_buffer_creates_and_releases_temp_files(tmp_path):
    n = 20
    before = [['id', 'v']] + [[i, 'b'] for i in range(n)]
    after = [['id', 'v']] + [[i, 'a'] for i in range(n)]
    result = etl.keydiff(before, after, 'id', buffersize=3,
                         tempdir=str(tmp_path))
    it = iter(result)
    assert next(it) == RESULT_HEADER
    assert next(it)[0] == 'update'
    assert os.listdir(str(tmp_path)) != []
    it.close()
    assert os.listdir(str(tmp_path)) == []


def test_config_sort_buffer_default(tmp_path):
    before = [['id', 'v']] + [[i, 'b'] for i in range(12)]
    after = [['id', 'v']] + [[i, 'a'] for i in range(6, 18)]
    old = config.sort_buffersize
    try:
        config.sort_buffersize = 2
        changes = _changes(etl.keydiff(before, after, 'id',
                                       tempdir=str(tmp_path)))
        assert changes == [
            ('delete', (i,), (i, 'b'), None, ()) for i in range(6)
        ] + [
            ('update', (i,), (i, 'b'), (i, 'a'), ('v',))
            for i in range(6, 12)
        ] + [
            ('insert', (i,), None, (i, 'a'), ())
            for i in range(12, 18)
        ]
        assert os.listdir(str(tmp_path)) == []
    finally:
        config.sort_buffersize = old


def test_explicit_in_memory_buffer():
    before = [['id', 'v']] + [[i, 'a'] for i in range(10)]
    after = [['id', 'v']] + [[i, 'b'] for i in range(10)]
    changes = _changes(etl.keydiff(before, after, 'id',
                                   buffersize=None))
    assert len(changes) == 10
    assert all(c[0] == 'update' for c in changes)


def test_view_creation_does_not_read_input():
    before = CountingSource([['id', 'v'], [1, 'a']])
    after = CountingSource([['id', 'v'], [1, 'b']])
    view = etl.keydiff(before, after, 'id')
    assert before.iterations == 0 and before.consumed == 0
    assert after.iterations == 0 and after.consumed == 0
    assert view.before is before and view.after is after


def test_one_source_iterator_per_side_and_header_not_reopened():
    before = CountingSource([['id', 'v'], [2, 'b'], [1, 'a']])
    after = CountingSource([['id', 'v'], [1, 'a'], [2, 'c']])
    changes = _changes(etl.keydiff(before, after, 'id'))
    assert changes == [
        ('update', (2,), (2, 'b'), (2, 'c'), ('v',)),
    ]
    assert before.iterations == 1
    assert after.iterations == 1
    assert before.consumed == 3
    assert after.consumed == 3
    assert before.closed == 1 and after.closed == 1


def test_external_sort_consumes_each_source_once(tmp_path):
    before = CountingSource([['id', 'v']] + [[i, 'b'] for i in range(12)])
    after = CountingSource([['v', 'id']] + [['a', i] for i in range(6, 18)])
    changes = _changes(etl.keydiff(before, after, 'id', buffersize=3,
                                   tempdir=str(tmp_path)))
    assert len(changes) == 18
    assert before.iterations == 1 and after.iterations == 1
    assert before.consumed == 13  # header + 12 data rows
    assert after.consumed == 13
    assert before.closed == 1 and after.closed == 1
    assert os.listdir(str(tmp_path)) == []


def test_repeatable_source_is_rechecked_each_pass():
    before = [['id', 'v'], [1, 'a'], [2, 'b']]
    after = [['id', 'v'], [1, 'a'], [2, 'b']]
    view = etl.keydiff(before, after, 'id')
    assert _changes(view) == []
    assert _changes(view) == []
    # mutate the repeatable source and re-validate the new data
    after[2] = [2, 'b2']
    changes = _changes(view)
    assert changes == [('update', (2,), (2, 'b'), (2, 'b2'), ('v',))]


def test_source_exception_propagates_unchanged():
    error = ValueError('boom')
    before = FailingSource([['id', 'v'], [1, 'a']], error)
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ValueError) as excinfo:
        _changes(etl.keydiff(before, after, 'id'))
    assert excinfo.value is error
    assert before.closed == 1


def test_close_error_does_not_mask_source_error():
    error = ValueError('original')
    before = CloseFailingIterator(error)
    after = [['id', 'v'], [1, 'a']]
    with pytest.raises(ValueError) as excinfo:
        _changes(etl.keydiff(before, after, 'id'))
    assert excinfo.value is error
    assert before.close_calls == 1


def test_caller_close_releases_sources_and_temp_files(tmp_path):
    before = CountingSource([['id', 'v'], [1, 'a'], [2, 'b'], [3, 'c']])
    after = CountingSource([['id', 'v'], [1, 'a'], [2, 'b'], [3, 'c']])
    result = etl.keydiff(before, after, 'id', buffersize=2,
                         tempdir=str(tmp_path))
    it = iter(result)
    assert next(it) == RESULT_HEADER
    it.close()
    assert before.closed == 1 and after.closed == 1
    assert os.listdir(str(tmp_path)) == []
    consumed_before = before.consumed
    consumed_after = after.consumed
    assert before.consumed == consumed_before
    assert after.consumed == consumed_after


def test_caller_close_mid_stream_releases_resources(tmp_path):
    before = CountingSource([['id', 'v']] + [[i, 'b'] for i in range(10)])
    after = CountingSource([['id', 'v']] + [[i, 'a'] for i in range(10)])
    result = etl.keydiff(before, after, 'id', buffersize=2,
                         tempdir=str(tmp_path))
    it = iter(result)
    next(it)
    next(it)
    consumed_before = before.consumed
    consumed_after = after.consumed
    it.close()
    assert before.closed == 1 and after.closed == 1
    assert os.listdir(str(tmp_path)) == []
    assert before.consumed == consumed_before
    assert after.consumed == consumed_after


def test_presorted_streams_first_change_before_tail_consumed():
    before = CountingSource([['id', 'v']] + [[i, 'b'] for i in range(20)])
    after = CountingSource([['id', 'v']] + [[i, 'a'] for i in range(20)])
    result = etl.keydiff(before, after, 'id', presorted=True)
    it = iter(result)
    assert next(it) == RESULT_HEADER
    change = next(it)
    assert change[0] == 'update'
    assert before.consumed < 21
    assert after.consumed < 21

    # the caller may stop early; no further rows are consumed
    before_consumed = before.consumed
    after_consumed = after.consumed
    it.close()
    assert before.consumed == before_consumed
    assert after.consumed == after_consumed
    assert before.closed == 1 and after.closed == 1


def test_presorted_constant_lookahead():
    before = CountingSource([['id', 'v']] + [[i, 'b'] for i in range(100)])
    after = CountingSource([['id', 'v']])
    result = etl.keydiff(before, after, 'id', presorted=True)
    it = iter(result)
    next(it)
    next(it)
    assert before.consumed == 2
    assert after.consumed == 1


def test_table_chaining():
    before = etl.wrap([['id', 'v'], [1, 'a'], [2, 'b']])
    after = [['id', 'v'], [1, 'a2'], [3, 'c']]
    changes = _changes(before.keydiff(after, 'id'))
    assert changes == [
        ('update', (1,), (1, 'a'), (1, 'a2'), ('v',)),
        ('delete', (2,), (2, 'b'), None, ()),
        ('insert', (3,), None, (3, 'c'), ()),
    ]


def test_original_setops_unchanged():
    a = [['foo', 'bar'], ['A', 1], ['B', 2]]
    b = [['foo', 'bar'], ['A', 9], ['B', 2]]
    added, subtracted = etl.diff(a, b)
    assert list(added) == [('foo', 'bar'), ('A', 9)]
    assert list(subtracted) == [('foo', 'bar'), ('A', 1)]
    rc = etl.recordcomplement(a, b)
    assert list(rc) == [('foo', 'bar'), ('A', 1)]
