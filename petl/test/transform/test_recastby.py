from __future__ import absolute_import, print_function, division

import itertools
import os

import pytest

import petl as etl
import petl.config
from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl.transform.reshape import recastby, RecastByView
from petl.util.base import Table


def _rows(table):
    return [tuple(row) for row in table]


def test_recastby_basic():
    table = (('id', 'variable', 'value'),
             (1, 'age', 12),
             (1, 'gender', 'F'),
             (2, 'age', 17),
             (2, 'gender', 'M'))

    expect = (('id', 'age', 'gender'),
              (1, 12, 'F'),
              (2, 17, 'M'))

    result = recastby(table, 'id', 'variable', 'value', ['age', 'gender'])
    ieq(expect, result)
    ieq(expect, result)


def test_recastby_table_method_and_export():
    table = [('id', 'variable', 'value'),
             (1, 'a', 10),
             (2, 'a', 20)]
    expect = [('id', 'a'), (1, 10), (2, 20)]

    assert hasattr(etl, 'recastby')
    assert etl.recastby is recastby
    assert isinstance(recastby(table, 'id', 'variable', 'value', ['a']),
                      RecastByView)
    # bound Table method, via etl.wrap() and via Table.unwrap
    assert _rows(etl.wrap(table).recastby('id', 'variable', 'value',
                                          ['a'])) == expect
    assert isinstance(Table.recastby, type(recastby))


def test_recastby_does_not_shadow_table_duplicates_method():
    table = [('id', 'variable', 'value'), (1, 'a', 10)]
    view = recastby(table, 'id', 'variable', 'value', ['a'])
    # Instance attributes must not hide the fluent API method.
    assert callable(view.duplicates)
    assert callable(view.recastby)


def test_recastby_output_order_and_full_header():
    table = (('id', 'variable', 'value'),
             (1, 'c', 3),
             (1, 'a', 1),
             (1, 'b', 2),
             (2, 'b', 20),
             (2, 'c', 30))

    expect = (('id', 'b', 'c', 'a'),
              (1, 2, 3, 1),
              (2, 20, 30, None))

    ieq(expect, recastby(table, 'id', 'variable', 'value', ['b', 'c', 'a']))
    # row ordering: keys sorted ascending
    assert _rows(recastby(table, 'id', 'variable', 'value',
                          ['b', 'c', 'a']))[1][0] == 1
    assert _rows(recastby(table, 'id', 'variable', 'value',
                          ['b', 'c', 'a']))[2][0] == 2


def test_recastby_compound_key():
    table = (('id', 'time', 'variable', 'value'),
             (2, 11, 'a', 21),
             (1, 12, 'a', 11),
             (1, 11, 'a', 10),
             (2, 11, 'b', 22),
             (1, 11, 'b', 12))

    expect = (('id', 'time', 'a', 'b'),
              (1, 11, 10, 12),
              (1, 12, 11, None),
              (2, 11, 21, 22))

    ieq(expect, recastby(table, ['id', 'time'], 'variable', 'value',
                         ['a', 'b']))
    ieq(expect, recastby(table, ('id', 'time'), 'variable', 'value',
                         ['a', 'b']))


def test_recastby_none_key():
    table = (('id', 'variable', 'value'),
             (None, 'a', 1),
             (2, 'a', 3),
             (1, 'a', 2))

    expect = (('id', 'a'),
              (None, 1),
              (1, 2),
              (2, 3))

    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a']))


def test_recastby_none_value_is_real_value():
    table = (('id', 'variable', 'value'),
             (1, 'a', None),
             (2, 'a', 20))

    # missing defaults to None but an explicit None value must be preserved
    expect = (('id', 'a'), (1, None), (2, 20))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         missing=0))


def test_recastby_missing_fill():
    table = (('id', 'variable', 'value'),
             (1, 'a', 10),
             (2, 'b', 20))

    expect = (('id', 'a', 'b'),
              (1, 10, 'M'),
              (2, 'M', 20))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a', 'b'],
                         missing='M'))


@pytest.mark.parametrize('duplicates', ['first', 'last'])
@pytest.mark.parametrize('buffersize', [None, 1, 2, 3])
def test_recastby_duplicates_first_last_stable(buffersize, duplicates):
    table = (('id', 'variable', 'value'),
             (2, 'a', 20),
             (1, 'a', 10),
             (1, 'a', 11),
             (1, 'a', 12),
             (3, 'a', 30),
             (3, 'a', 31))

    if duplicates == 'first':
        expect = (('id', 'a'), (1, 10), (2, 20), (3, 30))
    else:
        expect = (('id', 'a'), (1, 12), (2, 20), (3, 31))

    view = recastby(table, 'id', 'variable', 'value', ['a'],
                    duplicates=duplicates, buffersize=buffersize,
                    tempdir=None)
    ieq(expect, view)
    ieq(expect, view)


def test_recastby_duplicates_first_keeps_none():
    table = (('id', 'variable', 'value'),
             (1, 'a', None),
             (1, 'a', 11))

    expect = (('id', 'a'), (1, None))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         duplicates='first'))

    expect = (('id', 'a'), (1, 11))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         duplicates='last'))


def test_recastby_duplicates_error_reports_row_key_indicator():
    table = (('id', 'variable', 'value'),
             (2, 'a', 20),
             (1, 'a', 10),
             (1, 'a', 11))

    with pytest.raises(ArgumentError) as exc:
        _rows(recastby(table, 'id', 'variable', 'value', ['a']))
    msg = str(exc.value)
    # original data row number (data rows counted from 1), key and indicator
    assert 'data row 3' in msg
    assert "'a'" in msg
    assert '1' in msg
    assert 'data row 2' in msg  # first seen


def test_recastby_duplicates_error_compound_key_message():
    table = (('id', 'time', 'variable', 'value'),
             (1, 11, 'a', 10),
             (1, 11, 'a', 11))

    with pytest.raises(ArgumentError) as exc:
        _rows(recastby(table, ['id', 'time'], 'variable', 'value', ['a']))
    msg = str(exc.value)
    assert 'data row 2' in msg
    assert '(1, 11)' in msg


@pytest.mark.parametrize('presorted', [False, True])
@pytest.mark.parametrize('buffersize', [None, 2])
def test_recastby_duplicates_with_unhashable_values(presorted, buffersize):
    table = (('id', 'variable', 'value'),
             (1, 'a', [1, 2]),
             (1, 'a', {'x': 1}),
             (2, 'a', [3]))

    kwargs = dict(presorted=presorted)
    if not presorted:
        kwargs['buffersize'] = buffersize

    expect = (('id', 'a'), (1, [1, 2]), (2, [3]))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         duplicates='first', **kwargs))

    expect = (('id', 'a'), (1, {'x': 1}), (2, [3]))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         duplicates='last', **kwargs))


def test_recastby_unknown_error_reports_row_key_indicator():
    table = (('id', 'variable', 'value'),
             (1, 'a', 10),
             (1, 'zzz', 20))

    with pytest.raises(ArgumentError) as exc:
        _rows(recastby(table, 'id', 'variable', 'value', ['a']))
    msg = str(exc.value)
    assert 'unknown indicator' in msg
    assert "'zzz'" in msg
    assert 'data row 2' in msg
    assert '1' in msg


def test_recastby_unknown_ignore_group_with_only_unknown():
    table = (('id', 'variable', 'value'),
             (1, 'zzz', 10),
             (2, 'a', 20),
             (2, 'zzz', 21),
             (3, 'zzz', 30),
             (3, 'yyy', 31))

    expect = (('id', 'a'),
              (1, None),
              (2, 20),
              (3, None))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         unknown='ignore'))

    expect = (('id', 'a'),
              (1, 'M'),
              (2, 20),
              (3, 'M'))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a'],
                         unknown='ignore', missing='M'))


@pytest.mark.parametrize('kwargs, message', [
    ({'duplicates': 'bad'}, 'invalid duplicates mode: '),
    ({'unknown': 'bad'}, 'invalid unknown mode: '),
    ({'duplicates': None}, 'invalid duplicates mode: '),
    ({'buffersize': 0}, 'buffersize must be a positive integer'),
    ({'buffersize': -1}, 'buffersize must be a positive integer'),
    ({'buffersize': True}, 'buffersize must be a positive integer'),
    ({'buffersize': 'x'}, 'buffersize must be a positive integer'),
])
def test_recastby_invalid_modes_rejected_eagerly(kwargs, message):
    table = [('id', 'variable', 'value'), (1, 'a', 10)]
    with pytest.raises(ArgumentError) as exc:
        recastby(table, 'id', 'variable', 'value', ['a'], **kwargs)
    assert message in str(exc.value)


@pytest.mark.parametrize('key, fields, message', [
    ((), ['a'], 'key must contain at least one field name'),
    (['id', 'id'], ['a'], 'duplicate key field'),
    ([1], ['a'], 'invalid key field'),
    ('missing', ['a'], 'invalid key field: missing'),
    ('id', [], 'fields must contain at least one field name'),
    ('id', ['a', 'a'], 'duplicate field in fields'),
    ('id', [1], 'fields must be field names'),
    ('id', ['id'], 'conflicts with a key field'),
])
def test_recastby_invalid_key_and_fields(key, fields, message):
    table = [('id', 'variable', 'value'), (1, 'a', 10)]
    with pytest.raises(ArgumentError) as exc:
        list(recastby(table, key, 'variable', 'value', fields))
    assert message in str(exc.value)


@pytest.mark.parametrize('variablefield, valuefield, message', [
    ('missing', 'value', 'invalid variable field'),
    ('variable', 'missing', 'invalid value field'),
    (None, 'value', 'variablefield must be a single field name'),
    ('variable', None, 'valuefield must be a single field name'),
    ('id', 'value', 'variable field cannot be a key field'),
    ('variable', 'id', 'value field cannot be a key field'),
    ('variable', 'variable', 'must be different'),
])
def test_recastby_invalid_variable_and_value_fields(variablefield,
                                                    valuefield, message):
    table = [('id', 'variable', 'value'), (1, 'a', 10)]
    with pytest.raises(ArgumentError) as exc:
        list(recastby(table, 'id', variablefield, valuefield, ['a']))
    assert message in str(exc.value)


def test_recastby_duplicate_header_field():
    table = [('id', 'variable', 'variable'), (1, 'a', 10)]
    with pytest.raises(ArgumentError) as exc:
        list(recastby(table, 'id', 'variable', 'variable', ['a']))
    assert 'duplicate header field' in str(exc.value)


def test_recastby_headerless_errors():
    with pytest.raises(ArgumentError) as exc:
        list(recastby([], 'id', 'variable', 'value', ['a']))
    assert 'no header row' in str(exc.value)


def test_recastby_header_only_outputs_header():
    table = [('id', 'variable', 'value')]
    expect = [('id', 'a', 'b')]
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a', 'b']))


def test_recastby_row_width_mismatch():
    table = [('id', 'variable', 'value'), (1, 'a', 10, 'extra')]
    with pytest.raises(ArgumentError) as exc:
        list(recastby(table, 'id', 'variable', 'value', ['a']))
    assert 'data row 1' in str(exc.value)
    assert 'header has 3 fields' in str(exc.value)


def test_recastby_presorted_backwards_reports_row():
    table = (('id', 'variable', 'value'),
             (2, 'a', 20),
             (1, 'a', 10),
             (3, 'a', 30))
    with pytest.raises(ArgumentError) as exc:
        _rows(recastby(table, 'id', 'variable', 'value', ['a'],
                       presorted=True))
    msg = str(exc.value)
    assert 'out of order' in msg
    assert 'data row 2' in msg
    assert '1' in msg and '2' in msg


def test_recastby_presorted_accepts_equal_and_none_keys():
    table = (('id', 'variable', 'value'),
             (None, 'a', 1),
             (None, 'b', 2),
             (1, 'a', 10),
             (1, 'b', 11),
             (2, 'a', 20))
    expect = (('id', 'a', 'b'),
              (None, 1, 2),
              (1, 10, 11),
              (2, 20, None))
    ieq(expect, recastby(table, 'id', 'variable', 'value', ['a', 'b'],
                         presorted=True))


def test_recastby_presorted_delivers_group_early():
    # records one row at a time and tracks how many have been pulled
    pulled = [0]

    class Source(object):
        def __init__(self):
            self.rows = [('id', 'variable', 'value'),
                         (1, 'a', 10),
                         (1, 'b', 11),
                         (1, 'a', 12),  # duplicate inside group
                         (2, 'a', 20),
                         (2, 'b', 21)]

        def __iter__(self):
            def gen():
                for row in self.rows:
                    pulled[0] += 1
                    yield row
            return gen()

    view = recastby(Source(), 'id', 'variable', 'value', ['a', 'b'],
                    duplicates='first', presorted=True)
    it = iter(view)
    assert next(it) == ('id', 'a', 'b')
    # first group: pull only header + first row of second group
    assert next(it) == (1, 10, 11)
    assert pulled[0] == 5  # header + 3 rows of group 1 + first row of group 2


def test_recastby_presorted_memory_does_not_grow_with_group():
    # A group of size N should not require retaining N records; only the
    # first row of the next group is read to deliver the previous group.
    class Source(object):
        def __init__(self, ngroups, group_size):
            self.ngroups = ngroups
            self.group_size = group_size

        def __iter__(self):
            yield ('id', 'variable', 'value')
            for g in range(self.ngroups):
                for i in range(self.group_size):
                    yield (g, 'a', i)

    view = recastby(Source(2, 100), 'id', 'variable', 'value', ['a'],
                    duplicates='last', presorted=True)
    it = iter(view)
    assert next(it) == ('id', 'a')
    assert next(it) == (0, 99)
    # only header plus first row of group 1 is requested here -> bounded
    assert next(it) == (1, 99)


def test_recastby_presorted_bounded_row_retention():
    # Verify with reference counting that the number of live cell payloads
    # does not grow with the size of a presorted group.
    class Payload(object):
        live = 0
        peak = 0

        def __init__(self, value):
            self.value = value
            Payload.live += 1
            Payload.peak = max(Payload.peak, Payload.live)

        def __del__(self):
            Payload.live -= 1

    def peak_for(group_size):
        # Payloads are created lazily by the source generator, so the live
        # count reflects what the transform retains rather than the source
        # data structure.
        def source():
            yield ('id', 'variable', 'value')
            for i in range(group_size):
                yield (0, 'a', Payload(i))
            yield (1, 'a', Payload(-1))

        Payload.live = 0
        Payload.peak = 0
        view = recastby(source(), 'id', 'variable', 'value', ['a'],
                        duplicates='last', presorted=True)
        it = iter(view)
        assert next(it) == ('id', 'a')
        row = next(it)
        assert row[0] == 0 and row[1].value == group_size - 1
        peak = Payload.peak
        it.close()
        return peak

    small = peak_for(10)
    large = peak_for(5000)
    # A constant number of payloads is retained regardless of group size.
    assert large <= small + 3


def test_recastby_fields_iterator_not_consumed_lazily():
    consumed = []

    def fields():
        for f in ['a', 'b']:
            consumed.append(f)
            yield f

    table = [('id', 'variable', 'value'), (1, 'a', 10), (1, 'b', 11)]
    view = recastby(table, 'id', 'variable', 'value', fields())
    assert consumed == []
    it = iter(view)
    assert consumed == []
    assert next(it) == ('id', 'a', 'b')
    assert consumed == ['a', 'b']
    assert next(it) == (1, 10, 11)
    assert consumed == ['a', 'b']


def test_recastby_fields_iterator_reused_across_iterations():
    consumed = []

    def fields():
        for f in ['a', 'b']:
            consumed.append(f)
            yield f

    table = [('id', 'variable', 'value'), (1, 'b', 11), (1, 'a', 10)]
    view = recastby(table, 'id', 'variable', 'value', fields())

    expect = [('id', 'a', 'b'), (1, 10, 11)]
    first = _rows(view)
    # The one-shot iterator can only be consumed once; the view remembers
    # the resolved order.
    second = _rows(view)
    assert first == second == expect
    assert consumed == ['a', 'b']


def test_recastby_does_not_mutate_caller_fields():
    fields = ['b', 'a']
    view = recastby([('id', 'variable', 'value'), (1, 'a', 1), (1, 'b', 2)],
                    'id', 'variable', 'value', fields)
    assert list(view) == [('id', 'b', 'a'), (1, 2, 1)]
    assert fields == ['b', 'a']


def test_recastby_one_source_iterator_per_iteration():
    sources = []

    class Source(object):
        def __iter__(self):
            sources.append(self)
            return iter([('id', 'variable', 'value'),
                         (2, 'a', 20), (1, 'a', 10)])

    view = recastby(Source(), 'id', 'variable', 'value', ['a'])
    expect = [('id', 'a'), (1, 10), (2, 20)]
    assert _rows(view) == expect
    assert len(sources) == 1
    assert _rows(view) == expect
    assert len(sources) == 2


def test_recastby_repeats_consistent_and_no_reuse_leak():
    table = [('id', 'variable', 'value'), (2, 'a', 20), (1, 'a', 10)]
    view = recastby(table, 'id', 'variable', 'value', ['a'])
    assert _rows(view) == [('id', 'a'), (1, 10), (2, 20)]

    # partially consume one iterator, then start ge
    it1 = iter(view)
    assert next(it1) == ('id', 'a')
    # a second iterator must be independent (own source traversal)
    it2 = iter(view)
    assert next(it2) == ('id', 'a')
    assert next(it2) == (1, 10)
    assert next(it1) == (1, 10)
    it1.close()
    it2.close()
    # a fresh pass is still consistent
    assert _rows(view) == [('id', 'a'), (1, 10), (2, 20)]


def test_recastby_interleaved_iterators_independent():
    table = [('id', 'variable', 'value'),
             (3, 'a', 30), (1, 'a', 10), (2, 'a', 20)]
    view = recastby(table, 'id', 'variable', 'value', ['a'])
    it1 = iter(view)
    it2 = iter(view)
    assert next(it1) == ('id', 'a')
    assert next(it2) == ('id', 'a')
    assert next(it1) == (1, 10)
    assert next(it2) == (1, 10)
    assert next(it1) == (2, 20)
    assert next(it2) == (2, 20)
    assert next(it1) == (3, 30)
    assert next(it2) == (3, 30)
    with pytest.raises(StopIteration):
        next(it1)
    with pytest.raises(StopIteration):
        next(it2)


def test_recastby_presorted_interleaved_iterators_independent():
    table = [('id', 'variable', 'value'),
             (1, 'a', 10), (2, 'a', 20), (3, 'a', 30)]
    view = recastby(table, 'id', 'variable', 'value', ['a'],
                    presorted=True)
    it1 = iter(view)
    it2 = iter(view)
    assert next(it1) == ('id', 'a')
    assert next(it2) == ('id', 'a')
    assert next(it1) == (1, 10)
    assert next(it1) == (2, 20)
    assert next(it2) == (1, 10)
    assert next(it2) == (2, 20)
    assert next(it1) == (3, 30)
    assert next(it2) == (3, 30)


def test_recastby_failure_does_not_leak_group_values():
    # A failed merge due to duplicate must not poison a later fresh pass.
    table = [('id', 'variable', 'value'),
             (1, 'a', 10), (1, 'a', 11), (2, 'a', 20)]
    view = recastby(table, 'id', 'variable', 'value', ['a'])
    with pytest.raises(ArgumentError):
        _rows(view)
    # second pass is clean (and still errors, but not with stale values)
    with pytest.raises(ArgumentError):
        _rows(view)

    ok = recastby(table, 'id', 'variable', 'value', ['a'],
                  duplicates='first')
    # a short read of a different view must not affect this view
    it = iter(recastby(table, 'id', 'variable', 'value', ['a']))
    next(it)
    it.close()
    assert _rows(ok) == [('id', 'a'), (1, 10), (2, 20)]


def test_recastby_temp_files_cleaned_on_explicit_close_after_error(tmpdir):
    # An error raised while consuming the sorted output must still remove the
    # spill files made for this iteration.
    table = [('id', 'variable', 'value'),
             (1, 'a', 10), (1, 'a', 11),
             (2, 'a', 20), (3, 'a', 30), (4, 'a', 40)]
    view = recastby(table, 'id', 'variable', 'value', ['a'],
                    buffersize=1, tempdir=str(tmpdir))
    with pytest.raises(ArgumentError):
        _rows(view)
    assert os.listdir(str(tmpdir)) == []


def test_recastby_temp_files_cleaned_when_source_errors_during_spill(tmpdir):
    class Source(object):
        def __iter__(self):
            yield ('id', 'variable', 'value')
            for i in range(8):
                yield (i, 'a', i)
            raise RuntimeError('late source failure')

    view = recastby(Source(), 'id', 'variable', 'value', ['a'],
                    buffersize=2, tempdir=str(tmpdir))
    with pytest.raises(RuntimeError, match='late source failure'):
        _rows(view)
    assert os.listdir(str(tmpdir)) == []


def test_recastby_duplicate_diagnostics_after_spilling(tmpdir):
    # The duplicate is only noticed after the sorted merge across spill
    # chunks, yet the reported data row is the original input row.
    table = [('id', 'variable', 'value'),
             (3, 'a', 30), (2, 'a', 20), (1, 'a', 10), (1, 'a', 11)]
    view = recastby(table, 'id', 'variable', 'value', ['a'],
                    buffersize=1, tempdir=str(tmpdir))
    with pytest.raises(ArgumentError) as exc:
        _rows(view)
    msg = str(exc.value)
    assert 'data row 4' in msg
    assert 'data row 3' in msg  # first occurrence, before the sort
    assert os.listdir(str(tmpdir)) == []


def test_recastby_closes_source_on_exhaustion_and_error():
    events = []

    class Source(object):
        def __init__(self, dup=False):
            self.dup = dup

        def __iter__(self):
            events.append('open')
            gen = self._gen()
            return gen

        def _gen(self):
            try:
                yield ('id', 'variable', 'value')
                yield (1, 'a', 10)
                yield (1, 'a', 11) if self.dup else (2, 'a', 20)
            finally:
                events.append('closed')

    view = recastby(Source(), 'id', 'variable', 'value', ['a'],
                    duplicates='last')
    assert _rows(view) == [('id', 'a'), (1, 10), (2, 20)]
    assert events == ['open', 'closed']

    events[:] = []
    bad = recastby(Source(dup=True), 'id', 'variable', 'value', ['a'])
    with pytest.raises(ArgumentError):
        _rows(bad)
    assert events == ['open', 'closed']


def test_recastby_closes_source_on_explicit_close():
    events = []

    class Source(object):
        def __iter__(self):
            events.append('open')
            try:
                yield ('id', 'variable', 'value')
                for i in range(10):
                    yield (i, 'a', i)
            finally:
                events.append('closed')

    view = recastby(Source(), 'id', 'variable', 'value', ['a'],
                    buffersize=2, tempdir=None, presorted=True)
    it = iter(view)
    assert next(it) == ('id', 'a')
    assert next(it) == (0, 0)
    assert events == ['open']
    it.close()
    assert events == ['open', 'closed']


def test_recastby_source_error_propagates_and_cleanup_does_not_mask(tmpdir):
    class Source(object):
        def __iter__(self):
            yield ('id', 'variable', 'value')
            yield (2, 'a', 20)
            yield (1, 'a', 10)
            raise ValueError('boom')

    view = recastby(Source(), 'id', 'variable', 'value', ['a'],
                    buffersize=2, tempdir=str(tmpdir))
    with pytest.raises(ValueError, match='boom'):
        _rows(view)
    # temp files from this run are removed
    assert os.listdir(str(tmpdir)) == []


def test_recastby_temp_files_are_created_in_tempdir(tmpdir):
    table = [('id', 'variable', 'value'),
             (3, 'a', 30), (1, 'a', 10), (2, 'a', 20), (4, 'a', 40)]
    view = recastby(table, 'id', 'variable', 'value', ['a'],
                    buffersize=1, tempdir=str(tmpdir))
    it = iter(view)
    assert next(it) == ('id', 'a')
    assert next(it) == (1, 10)
    # the spill files really live in the requested temp directory
    names = os.listdir(str(tmpdir))
    assert names
    for name in names:
        assert os.path.exists(os.path.join(str(tmpdir), name))
    it.close()
    assert os.listdir(str(tmpdir)) == []


def test_recastby_temp_files_cleaned_on_close(tmpdir):
    table = [('id', 'variable', 'value'),
             (3, 'a', 30), (1, 'a', 10), (2, 'a', 20), (4, 'a', 40)]
    view = recastby(table, 'id', 'variable', 'value', ['a'],
                    buffersize=1, tempdir=str(tmpdir))
    it = iter(view)
    assert next(it) == ('id', 'a')
    assert next(it) == (1, 10)
    assert os.listdir(str(tmpdir)) != []
    it.close()
    assert os.listdir(str(tmpdir)) == []


def test_recastby_buffersize_spill_consistency():
    # Force multi-chunk merge sort and verify first/last agree with an
    # in-memory run.
    table = [('id', 'variable', 'value')]
    for i in range(20):
        table.append((i % 5, 'a', i))

    for duplicates in ('first', 'last'):
        expect = _rows(recastby(table, 'id', 'variable', 'value', ['a'],
                                duplicates=duplicates))
        actual = _rows(recastby(table, 'id', 'variable', 'value', ['a'],
                                duplicates=duplicates, buffersize=2,
                                tempdir=None))
        assert actual == expect


def test_recastby_buffersize_uses_config(monkeypatch):
    monkeypatch.setattr(petl.config, 'sort_buffersize', 1)
    table = [('id', 'variable', 'value'),
             (3, 'a', 30), (1, 'a', 10), (2, 'a', 20)]
    expect = [('id', 'a'), (1, 10), (2, 20), (3, 30)]
    assert _rows(recastby(table, 'id', 'variable', 'value', ['a'])) == expect


def test_recastby_single_shot_source_not_replayable():
    one_shot = itertools.chain([('id', 'variable', 'value'),
                                (1, 'a', 10)])
    view = recastby(one_shot, 'id', 'variable', 'value', ['a'])
    assert _rows(view) == [('id', 'a'), (1, 10)]
    # the underlying iterator is exhausted and is not cached/replayed
    with pytest.raises(ArgumentError, match='no header row'):
        _rows(view)


def test_recastby_source_not_consumed_by_construction_or_iter():
    reads = []

    class Source(object):
        def __iter__(self):
            reads.append(True)
            return iter([('id', 'variable', 'value'), (1, 'a', 10)])

    view = recastby(Source(), 'id', 'variable', 'value', ['a'])
    assert reads == []
    it = iter(view)
    assert reads == []
    assert next(it) == ('id', 'a')
    assert len(reads) == 1


def test_recastby_partial_read_failure_does_not_replay_stale_group():
    # closing a partially consumed result must not leave the current group's
    # values visible in a later pass
    table = [('id', 'variable', 'value'),
             (1, 'a', 10), (1, 'b', 11), (2, 'a', 20), (2, 'b', 21)]
    view = recastby(table, 'id', 'variable', 'value', ['a', 'b'])
    it = iter(view)
    assert next(it) == ('id', 'a', 'b')
    assert next(it) == (1, 10, 11)
    it.close()
    assert _rows(view) == [('id', 'a', 'b'),
                           (1, 10, 11),
                           (2, 20, 21)]


def test_recastby_fields_single_string():
    table = [('id', 'variable', 'value'), (1, 'a', 10), (2, 'a', 20)]
    expect = [('id', 'a'), (1, 10), (2, 20)]
    ieq(expect, recastby(table, 'id', 'variable', 'value', 'a'))


def test_recastby_fields_error_cached_on_one_shot():
    consumed = []

    def fields():
        for f in ['a', 'a']:
            consumed.append(f)
            yield f

    view = recastby([('id', 'variable', 'value'), (1, 'a', 10)],
                    'id', 'variable', 'value', fields())
    with pytest.raises(ArgumentError, match='duplicate field'):
        _rows(view)
    # the error is reused, but the one-shot iterator is not consumed again
    with pytest.raises(ArgumentError, match='duplicate field'):
        _rows(view)


def test_recastby_presorted_one_row_per_input_row():
    # Each row is consumed exactly once even for large groups.
    table = [('id', 'variable', 'value')] + \
            [(1, 'a', i) for i in range(50)] + \
            [(2, 'a', 100)]
    it = iter(recastby(table, 'id', 'variable', 'value', ['a'],
                       duplicates='last', presorted=True))
    expect = [('id', 'a'), (1, 49), (2, 100)]
    assert _rows(it) == expect


# ---------------------------------------------------------------------------
# External sort with many spill chunks: bounded file descriptors, bounded
# retained rows, and complete results verified independently.
#
# Small `buffersize` values force many spill chunks, well past the merge
# fan-in, so the multi-pass reduction is exercised. These tests do not
# depend on the fan-in constant's value.
# ---------------------------------------------------------------------------


def _collect(table):
    return [tuple(r) for r in table]


def _spill_source(nrows, key_mod=None):
    """Header plus `nrows` molten rows in descending key order.

    When `key_mod` is given, keys cycle through ``range(key_mod)`` so many
    rows share a key and land in different spill chunks.
    """
    def source():
        yield ('k', 'v', 'x')
        for i in range(nrows, 0, -1):
            yield (i % key_mod if key_mod else i, 'a', i)
    return source()


def _expected_recast(nrows, key_mod=None, keep='last'):
    """Independently computed expected rows for `_spill_source`.

    With `key_mod` set, several rows share a key. The source counts down,
    so for ``keep='last'`` the latest input row is the smallest `i` of the
    key; for ``keep='first'`` it is the largest `i`.
    """
    keys = sorted(set(i % key_mod if key_mod else i
                      for i in range(1, nrows + 1)))
    out = []
    for key in keys:
        rows = [i for i in range(1, nrows + 1)
                if (i % key_mod if key_mod else i) == key]
        value = min(rows) if keep == 'last' else max(rows)
        out.append((key, value))
    return out


class _TrackedFile(object):
    """Wraps a real run-file object to count concurrently open files."""

    def __init__(self, f, counters):
        self._file = f
        self._counters = counters
        counters['live'] += 1
        counters['peak'] = max(counters['peak'], counters['live'])

    def __getattr__(self, name):
        return getattr(self._file, name)

    def close(self):
        f = self._file
        if f is not None:
            self._file = None
            self._counters['live'] -= 1
            f.close()


def _tracked_open_probe(monkeypatch):
    """Instrument the single place run files are opened."""
    import petl.transform.reshape as rm
    counters = {'live': 0, 'peak': 0, 'opens': 0, 'closes': 0}
    real_open = rm._open_recastby_chunk_file

    def tracking_open(fn):
        counters['opens'] += 1
        return _TrackedFile(real_open(fn), counters)

    monkeypatch.setattr(rm, '_open_recastby_chunk_file', tracking_open)
    return counters


def test_recastby_subprocess_fd_limit_reproduction(tmpdir):
    """Original reproduction: 600 segments, buffersize=2, 256 soft FD cap."""
    import subprocess
    import sys
    import textwrap

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    script = textwrap.dedent('''
        import os, sys
        sys.path.insert(0, %r)
        import petl as etl

        tmpdir = sys.argv[1]
        src = [('k', 'v', 'x')] + [(k, 'a', k) for k in range(600, 0, -1)]
        rows = [tuple(r) for r in etl.recastby(
            iter(src), 'k', 'v', 'x', ['a'], buffersize=2, tempdir=tmpdir)]
        expected = [('k', 'a')] + [(k, k) for k in range(1, 601)]
        assert rows == expected, (len(rows), rows[:3], rows[-3:])
        # the script itself lives in the spill directory; only run files
        # created by recastby must be gone
        spills = [n for n in os.listdir(tmpdir) if n != 'repro.py']
        assert spills == [], spills
        print('OK')
    ''' % (root,))
    script_path = str(tmpdir.join('repro.py'))
    with open(script_path, 'w') as f:
        f.write(script)

    prelude = (
        'import resource\n'
        'soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n'
        'resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))\n'
        'import runpy\n'
        'runpy.run_path(%r, run_name="__main__")\n' % (script_path,)
    )
    result = subprocess.run([sys.executable, '-c', prelude, str(tmpdir)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip().endswith('OK')


@pytest.mark.parametrize('nrows', [50, 400, 1500])
@pytest.mark.parametrize('buffersize', [2, 3])
def test_recastby_many_segments_full_result(nrows, buffersize, tmpdir):
    """Multi-pass external processing produces the complete result."""
    actual = _collect(recastby(_spill_source(nrows), 'k', 'v', 'x', ['a'],
                               buffersize=buffersize, tempdir=str(tmpdir)))
    assert actual[0] == ('k', 'a')
    assert actual[1:] == _expected_recast(nrows)
    assert os.listdir(str(tmpdir)) == []


@pytest.mark.parametrize('keep', ['first', 'last'])
def test_recastby_many_segments_shared_keys_original_order(keep, tmpdir):
    """first/last across many chunks keep original input order."""
    nrows = 400
    actual = _collect(recastby(_spill_source(nrows, key_mod=5), 'k', 'v', 'x',
                               ['a'], duplicates=keep, buffersize=2,
                               tempdir=str(tmpdir)))
    assert actual[0] == ('k', 'a')
    assert actual[1:] == _expected_recast(nrows, key_mod=5, keep=keep)
    assert os.listdir(str(tmpdir)) == []


def test_recastby_many_segments_duplicate_row_number(tmpdir):
    """A duplicate across a chunk boundary reports original row numbers."""
    nrows = 200
    table = [('k', 'v', 'x')]
    for i in range(nrows, 0, -1):
        table.append((i, 'a', i))
    # add a duplicate of key 7 far from its first occurrence
    table.append((7, 'a', 999))
    # data row numbers count from 1 after the header; key 7 first appears
    # at data row nrows - 7 + 1 == 194
    view = recastby(table, 'k', 'v', 'x', ['a'], buffersize=3,
                    tempdir=str(tmpdir))
    with pytest.raises(ArgumentError) as exc:
        _collect(view)
    msg = str(exc.value)
    assert 'data row %d' % (nrows - 7 + 1) in msg
    assert 'data row %d' % (nrows + 1) in msg
    assert "'a'" in msg
    assert os.listdir(str(tmpdir)) == []


@pytest.mark.parametrize('buffersize, max_peak', [(2, 16), (4, 16), (16, 64)])
def test_recastby_many_segments_open_files_bounded(buffersize, max_peak,
                                                   monkeypatch, tmpdir):
    """Peak open run files is independent of the number of chunks."""
    peaks = []
    for nrows in (40, 400, 1600):
        counters = _tracked_open_probe(monkeypatch)
        actual = _collect(recastby(_spill_source(nrows), 'k', 'v', 'x',
                                   ['a'], buffersize=buffersize,
                                   tempdir=str(tmpdir)))
        assert actual[1:] == _expected_recast(nrows)
        assert counters['live'] == 0
        assert counters['opens'] >= 2  # multi-pass merging really happened
        assert os.listdir(str(tmpdir)) == []
        peaks.append(counters['peak'])
    # Both larger inputs produce more chunks than a single merge pass can
    # consume, so the peak cannot grow with the segment count any further
    # (a 4x larger input produces 4x more chunks), and it never exceeds the
    # fixed fan-in.
    assert peaks[1] == peaks[2]
    assert max(peaks) <= max_peak


class _RetainedPayload(object):
    """Module-level (picklable) payload with live/peak reference counting."""

    live = 0
    peak = 0

    def __init__(self, value):
        self.value = value
        _RetainedPayload.live += 1
        _RetainedPayload.peak = max(_RetainedPayload.peak,
                                    _RetainedPayload.live)

    def __del__(self):
        _RetainedPayload.live -= 1

    def __eq__(self, other):
        return (isinstance(other, _RetainedPayload)
                and self.value == other.value)

    def __reduce__(self):
        return (_RetainedPayload, (self.value,))


@pytest.mark.parametrize('nrows', [30, 400, 1500])
def test_recastby_many_segments_retained_rows_bounded(nrows, tmpdir):
    """The number of live value payloads stays bounded as chunks grow."""
    def run(rows):
        _RetainedPayload.live = _RetainedPayload.peak = 0

        def source():
            yield ('k', 'v', 'x')
            for i in range(rows, 0, -1):
                yield (i % 7, 'a', _RetainedPayload(i))

        out = _collect(recastby(source(), 'k', 'v', 'x', ['a'],
                                duplicates='last', buffersize=2,
                                tempdir=str(tmpdir)))
        assert out[0] == ('k', 'a')
        return _RetainedPayload.peak

    small = run(10)
    probe = run(nrows)
    # A constant number of records is live regardless of chunk count: the
    # sort buffer plus one buffered record per merged run, not one per
    # spilled chunk or one per group held for the whole merge.
    assert probe <= small + 8
    assert os.listdir(str(tmpdir)) == []


def test_recastby_many_segments_early_close_cleanup(monkeypatch, tmpdir):
    """Many-segment merge: early close frees files and the tracked handles."""
    counters = _tracked_open_probe(monkeypatch)
    table = [('k', 'v', 'x')] + [(i, 'a', i)
                                 for i in range(500, 0, -1)]
    it = iter(recastby(table, 'k', 'v', 'x', ['a'], buffersize=2,
                       tempdir=str(tmpdir)))
    assert next(it) == ('k', 'a')
    assert next(it) == (1, 1)
    assert os.listdir(str(tmpdir)) != []
    it.close()
    assert os.listdir(str(tmpdir)) == []
    assert counters['live'] == 0


def test_recastby_many_segments_source_error_cleanup(tmpdir):
    """A source failure during spilling removes all chunk files."""
    class Source(object):
        def __iter__(self):
            yield ('k', 'v', 'x')
            for i in range(300, 0, -1):
                yield (i, 'a', i)
            raise RuntimeError('source boom')

    with pytest.raises(RuntimeError, match='source boom'):
        _collect(recastby(Source(), 'k', 'v', 'x', ['a'], buffersize=2,
                          tempdir=str(tmpdir)))
    assert os.listdir(str(tmpdir)) == []


def test_recastby_many_segments_write_failure_cleanup(monkeypatch, tmpdir):
    """An injected pickle failure while writing a chunk is cleaned up."""
    import petl.transform.reshape as rm

    calls = [0]
    real_dump = rm._pickle_dump_recastby_record

    def failing_dump(record, f):
        calls[0] += 1
        if calls[0] == 5:
            raise OSError('disk full (injected)')
        return real_dump(record, f)

    monkeypatch.setattr(rm, '_pickle_dump_recastby_record', failing_dump)
    table = [('k', 'v', 'x')] + [(i, 'a', i) for i in range(300, 0, -1)]
    with pytest.raises(OSError, match='disk full'):
        _collect(recastby(table, 'k', 'v', 'x', ['a'], buffersize=2,
                          tempdir=str(tmpdir)))
    assert os.listdir(str(tmpdir)) == []


def test_recastby_merge_read_failure_cleanup(monkeypatch, tmpdir):
    """A failure opening a run file mid-merge is cleaned up."""
    import petl.transform.reshape as rm

    table = [('k', 'v', 'x')] + [(i, 'a', i) for i in range(300, 0, -1)]

    # First run clean to learn how many run files are opened in total.
    counters = _tracked_open_probe(monkeypatch)
    _collect(recastby(table, 'k', 'v', 'x', ['a'], buffersize=2,
                      tempdir=str(tmpdir)))
    total_opens = counters['opens']
    assert total_opens > 2

    real_open = rm._open_recastby_chunk_file
    for fail_at in (2, 3, total_opens // 2, total_opens):
        calls = [0]

        def flaky_open(fn, fail_at=fail_at, calls=calls):
            calls[0] += 1
            if calls[0] == fail_at:
                raise OSError('read failed (injected at %d)' % fail_at)
            return real_open(fn)

        monkeypatch.setattr(rm, '_open_recastby_chunk_file', flaky_open)
        with pytest.raises(OSError, match='read failed'):
            _collect(recastby(table, 'k', 'v', 'x', ['a'], buffersize=2,
                              tempdir=str(tmpdir)))
        assert os.listdir(str(tmpdir)) == []


def test_recastby_merge_pickle_failure_cleanup(monkeypatch, tmpdir):
    """A corrupt run file (pickle read failure) is cleaned up."""
    import pickle as pickle_module
    import petl.transform.reshape as rm

    real_load = pickle_module.load
    calls = [0]

    def failing_load(f):
        calls[0] += 1
        if calls[0] > 2:
            raise ValueError('corrupt chunk (injected)')
        return real_load(f)

    # _iter_recastby_chunk_file uses the module-level pickle.load imported
    # into reshape, so patch there.
    monkeypatch.setattr(rm.pickle, 'load', failing_load)
    table = [('k', 'v', 'x')] + [(i, 'a', i) for i in range(300, 0, -1)]
    with pytest.raises((ValueError, pickle_module.UnpicklingError)):
        _collect(recastby(table, 'k', 'v', 'x', ['a'], buffersize=2,
                          tempdir=str(tmpdir)))
    assert os.listdir(str(tmpdir)) == []


def test_recastby_many_segments_interleaved_iterators_independent(
        monkeypatch, tmpdir):
    """Interleaved result iterators have separate files and cleanup."""
    d1 = str(tmpdir.join('a'))
    d2 = str(tmpdir.join('b'))
    os.mkdir(d1)
    os.mkdir(d2)
    table = [('k', 'v', 'x')] + [(i, 'a', i) for i in range(200, 0, -1)]
    view = recastby(table, 'k', 'v', 'x', ['a'], buffersize=2, tempdir=d1)
    it1 = iter(view)
    it2 = iter(view)
    assert next(it1) == ('k', 'a')
    assert next(it2) == ('k', 'a')
    assert next(it1) == (1, 1)
    # a second view shares no state with the first
    view2 = recastby(_spill_source(150), 'k', 'v', 'x', ['a'],
                     buffersize=2, tempdir=d2)
    it3 = iter(view2)
    assert next(it3) == ('k', 'a')
    assert next(it3) == (1, 1)
    # closing one must not disturb the others or their directories
    it1.close()
    assert os.listdir(d1) == []
    assert next(it2) == (1, 1)  # it2 had only consumed its header
    assert next(it3) == (2, 2)
    assert os.listdir(d2) != []
    it2.close()
    it3.close()
    assert os.listdir(d1) == []
    assert os.listdir(d2) == []


def test_recastby_many_segments_source_closed_after_early_close(tmpdir):
    """The source iterator is closed even when the result is closed early
    in a multi-pass external sort."""
    events = []

    class Source(object):
        def __iter__(self):
            events.append('open')
            return self._gen()

        def _gen(self):
            try:
                yield ('k', 'v', 'x')
                for i in range(200, 0, -1):
                    yield (i, 'a', i)
            finally:
                events.append('closed')

    it = iter(recastby(Source(), 'k', 'v', 'x', ['a'], buffersize=2,
                       tempdir=str(tmpdir)))
    assert next(it) == ('k', 'a')
    assert events == ['open']
    it.close()
    assert events == ['open', 'closed']
    assert os.listdir(str(tmpdir)) == []


def test_recastby_many_segments_compound_and_none_keys(tmpdir):
    """Compound keys and None keys survive multi-pass external sorting."""
    table = [('a', 'b', 'v', 'x')]
    for i in range(200, 0, -1):
        table.append((None if i % 10 == 0 else i % 4, i % 3, 'v', i))
    out = _collect(recastby(table, ['a', 'b'], 'v', 'x', ['v'],
                            duplicates='last', buffersize=2,
                            tempdir=str(tmpdir)))

    # independent expectation: group by (a, b), keys in PETL order with
    # None first, and keep the latest input row (smallest i, reverse source)
    groups = {}
    for i in range(200, 0, -1):
        key = (None if i % 10 == 0 else i % 4, i % 3)
        groups.setdefault(key, []).append(i)
    keys = sorted(groups, key=lambda k: (
        (0, 0) if k[0] is None else (1, k[0]), k[1]))
    assert out[0] == ('a', 'b', 'v')
    assert out[1:] == [(k[0], k[1], min(groups[k])) for k in keys]
    assert os.listdir(str(tmpdir)) == []


def test_recastby_many_segments_unknown_ignore_only_unknown(tmpdir):
    """unknown='ignore' keeps keys that have only unknown indicators even
    when the keys are spread over many chunks."""
    table = [('k', 'v', 'x')]
    for i in range(400, 0, -1):
        table.append((i, 'zzz', i))          # only unknown
        table.append((i, 'a', i * 1000))     # known
    out = _collect(recastby(table, 'k', 'v', 'x', ['a'], unknown='ignore',
                            buffersize=2, tempdir=str(tmpdir)))
    assert out == [('k', 'a')] + [(i, i * 1000) for i in range(1, 401)]
    assert os.listdir(str(tmpdir)) == []

    # a table where every key is only unknown still emits one row per key
    table2 = [('k', 'v', 'x')] + [(i, 'zzz', i)
                                  for i in range(50, 0, -1)]
    out2 = _collect(recastby(table2, 'k', 'v', 'x', ['a'],
                             unknown='ignore', missing='M', buffersize=2,
                             tempdir=str(tmpdir)))
    assert out2 == [('k', 'a')] + [(i, 'M') for i in range(1, 51)]
    assert os.listdir(str(tmpdir)) == []


def test_recastby_header_only_and_small_sort_consistent(tmpdir):
    """Header-only, in-memory, and spilled runs agree on header/order."""
    header_only = [('k', 'v', 'x')]
    for kwargs in ({}, {'buffersize': 1}):
        assert _collect(recastby(header_only, 'k', 'v', 'x', ['a'],
                                 tempdir=str(tmpdir), **kwargs)) == \
            [('k', 'a')]

    small = [('k', 'v', 'x'), (3, 'a', 30), (1, 'a', 10), (2, 'a', 20)]
    expect = [('k', 'a'), (1, 10), (2, 20), (3, 30)]
    assert _collect(recastby(small, 'k', 'v', 'x', ['a'])) == expect
    assert _collect(recastby(small, 'k', 'v', 'x', ['a'], buffersize=1,
                             tempdir=str(tmpdir))) == expect
    assert os.listdir(str(tmpdir)) == []
