from __future__ import absolute_import, print_function, division

import os
import tempfile

import pytest

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError, RowLengthError
from petl.transform.convertcapture import BAD_HEADER, ConvertCapture, \
    _SpooledQueue, convertcapture


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class CountingSource(object):
    """Wraps an in-memory table and counts iterators taken / closes made."""

    def __init__(self, rows):
        self.rows = rows
        self.iterators = 0
        self.closes = 0

    def __iter__(self):
        self.iterators += 1
        return CountingSource.Iter(self.rows, self)

    class Iter(object):

        def __init__(self, rows, owner):
            self._it = iter(rows)
            self._owner = owner

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)

        # Python 2 name
        next = __next__

        def close(self):
            self._owner.closes += 1


class FailingSource(object):
    """Raises ``exc`` from the data iterator after ``n_good`` data rows."""

    def __init__(self, header, rows, exc, n_good=None):
        self.header = header
        self.rows = rows
        self.exc = exc
        self.n_good = len(rows) if n_good is None else n_good
        self.closes = 0

    def __iter__(self):
        return FailingSource.Iter(self)

    class Iter(object):

        def __init__(self, owner):
            self._owner = owner
            self._step = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._step == 0:
                self._step += 1
                return self._owner.header
            if self._step - 1 >= self._owner.n_good:
                raise self._owner.exc
            row = self._owner.rows[self._step - 1]
            self._step += 1
            return row

        next = __next__

        def close(self):
            self._owner.closes += 1


def odd_good_table(n=30):
    rows = [['n', 'v']]
    for i in range(1, n + 1):
        rows.append((i, 'ok' if i % 2 else 'bad'))
    return rows


def odd_good_converter():
    def conv(v):
        if v == 'bad':
            raise ValueError('nope')
        return v.upper()
    return conv


def materialise(branch):
    return list(branch)


# ---------------------------------------------------------------------------
# construction and configuration
# ---------------------------------------------------------------------------

def test_function_and_table_method_exist():
    assert etl.convertcapture is convertcapture
    assert etl.Table.convertcapture is convertcapture
    table = etl.wrap([['a'], ['1']])
    cap = table.convertcapture([('a', int)])
    assert isinstance(cap, ConvertCapture)
    assert cap.good is not cap.bad
    cap.close()


def test_construction_does_not_touch_source():
    source = CountingSource([['a'], ['1']])
    cap = convertcapture(source, [('a', int)])
    good_iter = iter(cap.good)
    bad_iter = iter(cap.bad)
    assert source.iterators == 0
    # first actual consumption opens exactly one iterator
    assert next(good_iter) == ('a',)
    assert source.iterators == 1
    next(bad_iter)
    assert source.iterators == 1
    list(good_iter)
    list(bad_iter)
    assert source.iterators == 1
    cap.close()
    assert source.closes == 1


def test_invalid_converters_at_construction():
    table = [['a', 'b']]
    bad_specs = [
        None, [], {}, 'x', ('x', int),
        [('a',)],                       # pair too short
        [('a', int, 1)],               # pair too long
        [(1, int)],                     # non-string field
        [('a', int), ('a', str)],       # duplicate field
        [('a', 1)],                     # non-callable
        [['a', int]],                   # list pair instead of tuple
        [('a', int), ('b', None)],      # non-callable
    ]
    for spec in bad_specs:
        with pytest.raises(ArgumentError):
            convertcapture(table, spec)


def test_invalid_buffersize_at_construction():
    table = [['a']]
    for size in (0, -1, 1.5, True, False, '2', 2.0):
        with pytest.raises(ArgumentError):
            convertcapture(table, [('a', int)], buffersize=size)
    # valid values accepted
    convertcapture(table, [('a', int)], buffersize=1).close()
    convertcapture(table, [('a', int)], buffersize=100000).close()


def test_deferred_header_and_field_validation():
    # missing converter field
    cap = convertcapture([['a']], [('b', int)])
    with pytest.raises(FieldSelectionError):
        next(iter(cap.good))
    cap.close()

    # duplicate header names
    cap = convertcapture([['a', 'a'], [1, 2]], [('a', int)])
    with pytest.raises(ArgumentError):
        next(iter(cap.good))
    cap.close()

    # non-string header (converter field names themselves stay strings)
    cap = convertcapture([[1]], [('1', int)])
    with pytest.raises(ArgumentError):
        next(iter(cap.good))
    cap.close()

    # completely empty source (no header)
    cap = convertcapture([], [('a', int)])
    with pytest.raises(ArgumentError):
        next(iter(cap.good))
    cap.close()


# ---------------------------------------------------------------------------
# good / bad content and ordering
# ---------------------------------------------------------------------------

def test_basic_good_and_bad_content():
    table = [['foo', 'bar'],
             ['1', '2'],
             ['3', 'x'],
             ['5', '6'],
             ['7', '8', '9']]
    cap = convertcapture(table, [('foo', int), ('bar', int)])
    good = materialise(cap.good)
    bad = materialise(cap.bad)
    assert good == [('foo', 'bar'), (1, 2), (5, 6)]
    assert bad[0] == BAD_HEADER
    assert bad[1] == (2, ('3', 'x'), 'bar', 'x',
                      'builtins.ValueError',
                      "invalid literal for int() with base 10: 'x'")
    assert bad[2] == (4, ('7', '8', '9'), None, 3,
                      'petl.RowLengthError',
                      str(RowLengthError(2, 3)))
    cap.close()


def test_row_length_error_uses_actual_width_and_skips_converters():
    calls = []

    def conv(v):
        calls.append(v)
        return int(v)

    table = [['a', 'b'],
             ['1', '2'],
             ['3'],
             ['4', '5', '6']]
    cap = convertcapture(table, [('a', conv), ('b', conv)])
    bad = materialise(cap.bad)
    rows = bad[1:]
    assert rows[0][0] == 2 and rows[0][2] is None and rows[0][3] == 1
    assert rows[0][4] == 'petl.RowLengthError'
    assert rows[1][0] == 3 and rows[1][3] == 3
    # no converter call for the wrong-width rows
    assert calls == ['1', '2']
    cap.close()


def test_converter_order_original_values_and_short_circuit():
    calls = []

    def make(tag):
        def func(v):
            calls.append((tag, v))
            if v == 'boom':
                raise RuntimeError('boom-' + tag)
            return tag + ':' + str(v)
        return func

    table = [['a', 'b', 'c'],
             ['x', 'y', 'z'],
             ['boom', 'y2', 'z2'],
             ['x2', 'boom', 'z3'],
             [None, 'y4', None]]
    cap = convertcapture(
        table, [('a', make('A')), ('b', make('B')), ('c', make('C'))])
    good = materialise(cap.good)
    bad = materialise(cap.bad)

    assert good == [
        ('a', 'b', 'c'),
        ('A:x', 'B:y', 'C:z'),
        ('A:None', 'B:y4', 'C:None'),
    ]
    # short circuit: converters after the failing field never run
    assert ('B', 'y2') not in calls and ('C', 'z2') not in calls
    assert ('C', 'z3') not in calls
    # each selected field converter runs at most once per processed row
    assert calls.count(('A', 'x')) == 1
    assert calls.count(('B', 'y')) == 1
    assert calls.count(('C', 'z')) == 1
    # first failing field / original value reported
    assert bad[1][:4] == (2, ('boom', 'y2', 'z2'), 'a', 'boom')
    assert bad[1][4] == 'builtins.RuntimeError'
    assert bad[1][5] == 'boom-A'
    assert bad[2][:4] == (3, ('x2', 'boom', 'z3'), 'b', 'boom')
    cap.close()


def test_none_results_and_inputs_are_successes():
    table = [['a'], ['1'], ['2']]
    cap = convertcapture(table, [('a', lambda v: None)])
    good = materialise(cap.good)
    bad = materialise(cap.bad)
    assert good == [('a',), (None,), (None,)]
    assert bad == [BAD_HEADER]
    cap.close()


def test_unhashable_values_round_trip_through_spool():
    table = [['k', 'v'],
             [1, {'x': [1, 2, {}]}],
             [2, ['a', 'b', set()]],
             [3, {'bad': True}]]

    def conv(v):
        if isinstance(v, dict) and v.get('bad'):
            raise ValueError('dict bad')
        return v

    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(table, [('v', conv)], buffersize=1,
                             tempdir=tmp)
        good = materialise(cap.good)
        bad = materialise(cap.bad)
        cap.close()
    assert good[1] == (1, {'x': [1, 2, {}]})
    assert good[2] == (2, ['a', 'b', set()])
    assert bad[1][0] == 3 and bad[1][3] == {'bad': True}


def test_all_success_all_failure_and_empty():
    # all success
    cap = convertcapture([['a'], ['1'], ['2']], [('a', int)])
    assert materialise(cap.good) == [('a',), (1,), (2,)]
    assert materialise(cap.bad) == [BAD_HEADER]
    cap.close()

    # all failure
    cap = convertcapture([['a'], ['x'], ['y']], [('a', int)])
    assert materialise(cap.good) == [('a',)]
    bad = materialise(cap.bad)
    assert [r[0] for r in bad[1:]] == [1, 2]
    cap.close()

    # empty data section
    cap = convertcapture([['a', 'b']], [('a', int)])
    assert materialise(cap.good) == [('a', 'b')]
    assert materialise(cap.bad) == [BAD_HEADER]
    cap.close()


def test_branch_order_preserved_in_each_branch():
    rows = odd_good_table(20)
    cap = convertcapture(rows, [('v', odd_good_converter())], buffersize=3)
    good = materialise(cap.good)
    bad = materialise(cap.bad)
    assert [r[0] for r in good[1:]] == list(range(1, 21, 2))
    assert [r[0] for r in bad[1:]] == list(range(2, 21, 2))
    cap.close()


# ---------------------------------------------------------------------------
# single consumption
# ---------------------------------------------------------------------------

def test_branches_are_single_consumption():
    cap = convertcapture(odd_good_table(6), [('v', odd_good_converter())])
    iter(cap.good)
    with pytest.raises(ArgumentError):
        iter(cap.good)
    iter(cap.bad)
    with pytest.raises(ArgumentError):
        iter(cap.bad)
    cap.close()


def test_iterating_closed_branch_rejected():
    cap = convertcapture(odd_good_table(6), [('v', odd_good_converter())])
    cap.good.close()
    with pytest.raises(ArgumentError):
        iter(cap.good)
    # the other branch is still usable
    bad = materialise(cap.bad)
    assert [r[0] for r in bad[1:]] == list(range(2, 7, 2))
    cap.close()


def test_len_not_supported():
    cap = convertcapture([['a'], ['1']], [('a', int)])
    with pytest.raises(TypeError):
        len(cap.good)
    cap.close()


# ---------------------------------------------------------------------------
# consumption orders and interleaving
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('buffersize', [1, 2, 3, 5])
def test_good_first_bad_first_and_interleaved(buffersize):
    rows = odd_good_table(30)
    expected_good = list(range(1, 31, 2))
    expected_bad = list(range(2, 31, 2))

    with tempfile.TemporaryDirectory() as tmp:
        # good fully first
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=buffersize, tempdir=tmp)
        good = materialise(cap.good)
        bad = materialise(cap.bad)
        assert [r[0] for r in good[1:]] == expected_good
        assert [r[0] for r in bad[1:]] == expected_bad
        cap.close()
        assert os.listdir(tmp) == []

        # bad fully first
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=buffersize, tempdir=tmp)
        bad = materialise(cap.bad)
        good = materialise(cap.good)
        assert [r[0] for r in good[1:]] == expected_good
        assert [r[0] for r in bad[1:]] == expected_bad
        cap.close()
        assert os.listdir(tmp) == []

        # interleaved
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=buffersize, tempdir=tmp)
        gi = iter(cap.good)
        bi = iter(cap.bad)
        assert next(gi) == ('n', 'v')
        assert next(bi) == BAD_HEADER
        got_g, got_b = [], []
        for step in range(120):
            active = gi if step % 2 == 0 else bi
            target = got_g if step % 2 == 0 else got_b
            try:
                target.append(next(active))
            except StopIteration:
                pass
        while True:
            try:
                got_g.append(next(gi))
            except StopIteration:
                break
        while True:
            try:
                got_b.append(next(bi))
            except StopIteration:
                break
        assert [r[0] for r in got_g] == expected_good
        assert [r[0] for r in got_b] == expected_bad
        cap.close()
        assert os.listdir(tmp) == []


def test_unconsumed_side_does_not_block_other_side_eof():
    # every row goes bad; good must still reach a clean EOF on its own
    cap = convertcapture([['a'], ['x'], ['y'], ['z']], [('a', int)],
                         buffersize=1)
    good = materialise(cap.good)
    assert good == [('a',)]
    cap.close()

    # mirror: every row goes good; bad reaches EOF with just its header
    cap = convertcapture([['a'], [1], [2]], [('a', lambda v: v)],
                         buffersize=1)
    bad = materialise(cap.bad)
    assert bad == [BAD_HEADER]
    cap.close()


# ---------------------------------------------------------------------------
# buffering / spooling resource management
# ---------------------------------------------------------------------------

def test_spooled_queue_bounds_memory_and_keeps_fifo():
    with tempfile.TemporaryDirectory() as tmp:
        q = _SpooledQueue(3, tmp)
        for i in range(100):
            q.put(i)
            assert len(q._mem) <= 3
        assert [q.get() for _ in range(100)] == list(range(100))
        q.close()
        assert os.listdir(tmp) == []


def test_disk_usage_tracks_backlog_not_history():
    with tempfile.TemporaryDirectory() as tmp:
        q = _SpooledQueue(2, tmp)
        for i in range(20):
            q.put(i)
        for _ in range(10):
            q.get()
        # only the unread backlog may remain on disk
        assert len(os.listdir(tmp)) <= 5
        for i in range(20, 40):
            q.put(i)
        for _ in range(29):
            q.get()
        assert len(os.listdir(tmp)) <= 1
        q.get()
        q.close()
        assert os.listdir(tmp) == []


def test_temp_files_created_under_tempdir_and_reclaimed():
    rows = odd_good_table(40)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=2, tempdir=tmp)
        good = materialise(cap.good)
        assert len(os.listdir(tmp)) > 0  # backlog really did spill
        bad = materialise(cap.bad)
        assert len(good) == 21 and len(bad) == 21
        cap.close()
        assert os.listdir(tmp) == []


def test_branch_abandonment_reclaims_its_files():
    rows = odd_good_table(20)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=1, tempdir=tmp)
        gi = iter(cap.good)
        bi = iter(cap.bad)
        next(gi)
        next(bi)
        bi.close()  # abandon bad, release its spool files
        good = list(gi)
        assert [r[0] for r in good] == list(range(1, 21, 2))
        cap.close()
        assert os.listdir(tmp) == []


def test_unpicklable_value_fails_capture_with_prefix():
    class NotPicklable(object):
        def __reduce__(self):
            raise TypeError('cannot pickle me')

    table = [['a'],
             [NotPicklable()],   # row 1 good
             ['x'],              # row 2 bad
             [NotPicklable()],   # row 3 good -> forces spool of row 1
             [NotPicklable()]]
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(
            table,
            [('a', lambda v: (_ for _ in ()).throw(ValueError('bad'))
              if v == 'x' else v)],
            buffersize=1, tempdir=tmp)
        bad_it = iter(cap.bad)
        assert next(bad_it) == BAD_HEADER
        assert next(bad_it)[0] == 2
        with pytest.raises(TypeError):
            next(bad_it)
        # good branch delivers its in-memory prefix, then the same failure
        good_it = iter(cap.good)
        assert next(good_it) == ('a',)
        next(good_it)
        with pytest.raises(TypeError):
            next(good_it)
        cap.close()
        assert os.listdir(tmp) == []


def test_bad_tempdir_surfaces_io_error():
    # good rows accumulate while the bad branch drives processing; with
    # buffersize=1 the second buffered good row must spill into the bogus
    # temp directory and the I/O error fails the whole capture.
    table = [['a'], [1], ['x'], [2], [3]]

    def conv(v):
        if v == 'x':
            raise ValueError('bad')
        return v

    cap = convertcapture(table, [('a', conv)], buffersize=1,
                         tempdir='/no/such/petl/dir')
    it = iter(cap.bad)
    assert next(it) == BAD_HEADER
    assert next(it)[0] == 2
    with pytest.raises(OSError):
        next(it)
    cap.close()


# ---------------------------------------------------------------------------
# failure propagation
# ---------------------------------------------------------------------------

def test_source_failure_delivers_prefixes_then_same_exception():
    class Boom(Exception):
        pass

    rows = [(i, 'ok' if i % 2 else 'bad') for i in range(1, 9)]
    source = FailingSource(('k', 'v'), rows, Boom('source exploded'))
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', odd_good_converter())],
                             buffersize=2, tempdir=tmp)
        gi = iter(cap.good)
        bi = iter(cap.bad)
        assert next(gi) == ('k', 'v')
        assert next(bi) == BAD_HEADER
        # take a small prefix from each side
        good_prefix = [next(gi), next(gi)]
        bad_prefix = [next(bi), next(bi)]
        assert [r[0] for r in good_prefix] == [1, 3]
        assert [r[0] for r in bad_prefix] == [2, 4]

        def drain(it):
            out, err = [], None
            while True:
                try:
                    out.append(next(it))
                except StopIteration:
                    break
                except BaseException as e:
                    err = e
                    break
            return out, err

        g_rest, e1 = drain(gi)
        b_rest, e2 = drain(bi)
        assert isinstance(e1, Boom) and e1 is e2
        assert [r[0] for r in g_rest] == [5, 7]
        assert [r[0] for r in b_rest] == [6, 8]
        # the source stops advancing and keeps reporting the same exception
        with pytest.raises(Boom) as info:
            next(gi)
        assert info.value is e1
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_source_failure_before_any_prefix_raises_on_both_branches():
    class Boom(Exception):
        pass

    source = FailingSource(('a',), [], Boom('early'))
    cap = convertcapture(source, [('a', lambda v: v)])
    gi = iter(cap.good)
    assert next(gi) == ('a',)
    with pytest.raises(Boom):
        next(gi)
    bi = iter(cap.bad)
    assert next(bi) == BAD_HEADER
    with pytest.raises(Boom):
        next(bi)
    cap.close()


def test_keyboard_interrupt_terminates_capture():
    class InterruptingSource(object):
        def __iter__(self):
            yield ('a',)
            yield (1,)
            raise KeyboardInterrupt()

    cap = convertcapture(InterruptingSource(), [('a', lambda v: v)])
    it = iter(cap.good)
    next(it)
    next(it)
    with pytest.raises(KeyboardInterrupt):
        next(it)
    with pytest.raises(KeyboardInterrupt):
        next(it)
    cap.close()

    # BaseException raised inside a converter also propagates
    class InterruptingConverter(object):
        def __call__(self, v):
            raise SystemExit(1)

    cap = convertcapture([['a'], [1]], [('a', InterruptingConverter())])
    it = iter(cap.good)
    next(it)
    with pytest.raises(SystemExit):
        next(it)
    cap.close()


def test_failing_source_close_does_not_mask_original_failure():
    class Boom(Exception):
        pass

    class Source(object):
        def __iter__(self):
            return Source.Iter()

        class Iter(object):
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.n += 1
                if self.n == 1:
                    return ('a',)  # header delivered fine
                raise Boom('source boom')

            def close(self):
                raise OSError('close boom')

    cap = convertcapture(Source(), [('a', lambda v: v)])
    it = iter(cap.good)
    next(it)  # header
    with pytest.raises(Boom):
        next(it)
    with pytest.raises(Boom):
        list(cap.bad)
    cap.close()  # must not raise the OSError


def test_failing_close_on_clean_eof_becomes_capture_failure():
    class Source(object):
        def __iter__(self):
            return Source.Iter()

        class Iter(object):
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.n += 1
                if self.n == 1:
                    return ('a',)
                if self.n == 2:
                    return (1,)
                raise StopIteration

            def close(self):
                raise OSError('only close fails')

    cap = convertcapture(Source(), [('a', lambda v: v)])
    it = iter(cap.good)
    assert next(it) == ('a',)
    assert next(it) == (1,)
    with pytest.raises(OSError):
        next(it)
    cap.close()


# ---------------------------------------------------------------------------
# lifecycle: context manager, close, abandonment
# ---------------------------------------------------------------------------

def test_context_manager_closes_source_and_tempfiles():
    source = CountingSource([['a'], [1], [2]])
    with tempfile.TemporaryDirectory() as tmp:
        with convertcapture(source, [('a', lambda v: v)],
                            buffersize=1, tempdir=tmp) as cap:
            materialise(cap.good)
            materialise(cap.bad)
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_context_manager_preserves_user_exception():
    source = CountingSource([['a'], [1]])
    try:
        with convertcapture(source, [('a', lambda v: v)]) as cap:
            it = iter(cap.good)
            next(it)  # first next() starts the capture and opens the source
            raise ValueError('user')
    except ValueError as e:
        assert str(e) == 'user'
    else:  # pragma: no cover
        pytest.fail('user exception was swallowed')
    assert source.closes == 1


def test_close_is_idempotent_and_safe_before_start():
    source = CountingSource([['a'], [1]])
    cap = convertcapture(source, [('a', int)])
    cap.close()
    cap.close()
    assert source.iterators == 0
    assert source.closes == 0


def test_consumption_after_close_rejected():
    cap = convertcapture([['a'], [1]], [('a', lambda v: v)])
    it = iter(cap.good)
    next(it)
    cap.close()
    with pytest.raises(ArgumentError):
        next(it)


def test_abandon_one_branch_keeps_the_other_usable():
    rows = odd_good_table(12)
    # abandon good before any consumption
    cap = convertcapture(rows, [('v', odd_good_converter())], buffersize=2)
    cap.good.close()
    bad = materialise(cap.bad)
    assert [r[0] for r in bad[1:]] == list(range(2, 13, 2))
    cap.close()

    # abandon bad mid-stream; good still complete, later bad rows dropped
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=2, tempdir=tmp)
        gi = iter(cap.good)
        bi = iter(cap.bad)
        next(gi)
        next(bi)
        next(bi)
        bi.close()
        good = list(gi)
        assert [r[0] for r in good] == list(range(1, 13, 2))
        cap.close()
        assert os.listdir(tmp) == []


def test_abandon_both_branches_stops_source():
    source = CountingSource(odd_good_table(20))
    cap = convertcapture(source, [('v', odd_good_converter())], buffersize=2)
    gi = iter(cap.good)
    bi = iter(cap.bad)
    next(gi)
    next(bi)
    gi.close()
    bi.close()
    cap.close()
    assert source.closes == 1
    # source was closed exactly once
    assert source.iterators == 1


def test_branch_close_before_iteration():
    cap = convertcapture(odd_good_table(8), [('v', odd_good_converter())])
    cap.bad.close()
    good = materialise(cap.good)
    assert [r[0] for r in good[1:]] == list(range(1, 9, 2))
    cap.close()


def test_inflight_row_still_converted_once_after_branch_close():
    calls = []

    def conv(v):
        calls.append(v)
        if v == 'bad':
            raise ValueError('bad')
        return v

    cap = convertcapture([['a'], [1], [2]], [('a', conv)], buffersize=1)
    it = iter(cap.good)
    next(it)  # header
    it.close()
    cap.close()
    # closing without processing data rows ran no converters
    assert calls == []


def test_generator_close_abandons_branch():
    rows = odd_good_table(12)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=1, tempdir=tmp)

        def consume_few(branch):
            it = iter(branch)
            yield next(it)
            yield next(it)

        gen = consume_few(cap.good)
        assert next(gen) == ('n', 'v')
        first_good = next(gen)
        assert first_good[0] == 1 and first_good[1] == 'OK'
        gen.close()
        # the branch must be abandoned even though only its generator closed
        assert cap._abandoned['good']
        # the other branch stays fully usable
        bad = materialise(cap.bad)
        assert [r[0] for r in bad[1:]] == list(range(2, 13, 2))
        cap.close()
        assert os.listdir(tmp) == []


def test_dropped_iterator_is_reclaimed_on_gc():
    import gc
    import weakref

    rows = odd_good_table(12)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(rows, [('v', odd_good_converter())],
                             buffersize=1, tempdir=tmp)
        it = iter(cap.good)
        next(it)
        ref = weakref.ref(it)
        del it
        gc.collect()
        assert ref() is None
        assert cap._abandoned['good']
        cap.close()
        assert os.listdir(tmp) == []


def test_failure_re_raised_with_same_identity_after_teardown():
    class Boom(Exception):
        pass

    class Source(object):
        def __iter__(self):
            yield ('a',)
            raise Boom('boom')

    cap = convertcapture(Source(), [('a', lambda v: v)])
    it = iter(cap.good)
    next(it)
    errors = []
    for _ in range(3):
        try:
            next(it)
        except Boom as exc:
            errors.append(exc)
    assert len(errors) == 3
    assert errors[0] is errors[1] is errors[2]
    cap.close()


def test_default_tempdir_used_when_not_given():
    # no tempdir argument: spooling still works and leaves nothing behind
    cap = convertcapture(odd_good_table(20),
                         [('v', odd_good_converter())], buffersize=1)
    good = materialise(cap.good)
    bad = materialise(cap.bad)
    assert [r[0] for r in good[1:]] == list(range(1, 21, 2))
    assert [r[0] for r in bad[1:]] == list(range(2, 21, 2))
    cap.close()


def test_bad_header_constant():
    assert BAD_HEADER == ('row', 'record', 'field', 'value',
                          'error_type', 'error')


def test_non_iterable_row_fails_capture_not_bad():
    class WeirdSource(object):
        def __iter__(self):
            yield ('a', 'b')
            yield 42  # not a row at all

    cap = convertcapture(WeirdSource(), [('a', lambda v: v)])
    it = iter(cap.good)
    assert next(it) == ('a', 'b')
    with pytest.raises(TypeError):
        next(it)
    # the bad branch must report the same failure, not a normal EOF
    bad_it = iter(cap.bad)
    assert next(bad_it) == BAD_HEADER
    with pytest.raises(TypeError):
        next(bad_it)
    cap.close()


def test_start_failure_single_iterator_and_bad_header_then_error():
    class EmptySource(object):
        iterator_count = 0

        def __iter__(self):
            EmptySource.iterator_count += 1
            return iter(())

    cap = convertcapture(EmptySource(), [('a', int)])
    good_it = iter(cap.good)
    with pytest.raises(ArgumentError):
        next(good_it)
    bad_it = iter(cap.bad)
    # the fixed bad header is independent of the source and still delivered
    assert next(bad_it) == BAD_HEADER
    with pytest.raises(ArgumentError):
        next(bad_it)
    # only one iterator was ever taken, even across two branches
    assert EmptySource.iterator_count == 1
    cap.close()
