from __future__ import absolute_import, print_function, division

import builtins
import os
import pickle
import sys
import tempfile

import pytest

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError, RowLengthError
from petl.transform.convertcapture import BAD_HEADER, ConvertCapture, \
    _SpooledQueue, convertcapture

# the module object itself, for monkeypatching its file operations
cc_module = sys.modules['petl.transform.convertcapture']
_REAL_NAMED_TEMPORARY_FILE = cc_module.NamedTemporaryFile
_REAL_OPEN = builtins.open


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


# ---------------------------------------------------------------------------
# spool failure injection: helpers
# ---------------------------------------------------------------------------

class RaiseOnPickle(object):
    """An object whose ``__reduce__`` raises a fixed exception object."""

    def __init__(self, exc):
        self.exc = exc

    def __reduce__(self):
        raise self.exc


class InstrumentedSource(object):
    """In-memory table source counting iterators, reads and closes."""

    def __init__(self, rows, close_exc=None):
        self.rows = rows
        self.close_exc = close_exc
        self.iterators = 0
        self.reads = 0
        self.closes = 0

    def __iter__(self):
        self.iterators += 1
        owner = self

        class It(object):

            def __init__(self):
                self._it = iter(owner.rows)

            def __iter__(self):
                return self

            def __next__(self):
                row = next(self._it)
                owner.reads += 1
                return row

            next = __next__

            def close(self):
                owner.closes += 1
                if owner.close_exc is not None:
                    raise owner.close_exc

        return It()


class CountingConverter(object):
    """Records every value it is called with."""

    def __init__(self, func=None):
        self.func = func if func is not None else (lambda v: v)
        self.calls = []

    def __call__(self, v):
        self.calls.append(v)
        return self.func(v)


class SpoolFaults(object):
    """Fault-injecting replacements for the module's file operations.

    ``faults`` is a list of dicts with keys ``method`` (``'tempfile'``,
    ``'open'``, ``'write'``, ``'flush'``, ``'close'`` or ``'read'``),
    ``at`` (the 1-based global call number of that method to fail) and
    ``exc`` (the fixed exception object to raise). A ``'write'`` fault may
    set ``partial`` to write half of the record bytes before raising. Each
    fault fires at most once. Every created path is recorded in
    ``self.paths`` in creation order.
    """

    def __init__(self, faults=()):
        self.faults = [dict(fault, fired=False) for fault in faults]
        self.counts = {}
        self.paths = []

    def install(self, monkeypatch):
        monkeypatch.setattr(cc_module, 'NamedTemporaryFile', self.temp_file)
        monkeypatch.setattr(cc_module, 'open', self.open, raising=False)

    def temp_file(self, mode, dir=None, delete=False, prefix='', suffix=''):
        self._check('tempfile')
        f = _REAL_NAMED_TEMPORARY_FILE(mode=mode, dir=dir, delete=delete,
                                       prefix=prefix, suffix=suffix)
        return self._record(f, f.name)

    def open(self, path, mode='r', *args, **kwargs):
        self._check('open')
        return self._record(_REAL_OPEN(path, mode, *args, **kwargs), path)

    def _record(self, f, path):
        self.paths.append(path)
        return _FaultyFile(self, f, path)

    def _matching_fault(self, method):
        self.counts[method] = self.counts.get(method, 0) + 1
        for fault in self.faults:
            if fault['method'] == method and not fault['fired'] \
                    and fault['at'] == self.counts[method]:
                fault['fired'] = True
                return fault
        return None

    def _check(self, method):
        fault = self._matching_fault(method)
        if fault is not None:
            raise fault['exc']

    def fired(self, method):
        return any(fault['method'] == method and fault['fired']
                   for fault in self.faults)


class _FaultyFile(object):
    """Wraps a real binary file, injecting faults on demand."""

    def __init__(self, faults, real, path):
        self._faults = faults
        self._real = real
        self.name = getattr(real, 'name', path)

    def __getattr__(self, item):
        return getattr(self._real, item)

    def write(self, data):
        fault = self._faults._matching_fault('write')
        if fault is not None:
            if fault.get('partial'):
                # simulate a write that lands only some bytes before failing
                self._real.write(data[:len(data) // 2])
            raise fault['exc']
        return self._real.write(data)

    def flush(self):
        self._faults._check('flush')
        return self._real.flush()

    def close(self):
        self._faults._check('close')
        return self._real.close()

    def read(self, *args):
        self._faults._check('read')
        return self._real.read(*args)

    def readline(self, *args):
        self._faults._check('read')
        return self._real.readline(*args)


def drain(it):
    """Pull every remaining row; return (rows, terminal exception or None)."""
    rows, err = [], None
    while True:
        try:
            rows.append(next(it))
        except StopIteration:
            break
        except BaseException as e:
            err = e
            break
    return rows, err


def scenario1_source(exc):
    """The first reported reproduction: an unpicklable fifth row."""
    bomb = RaiseOnPickle(exc)
    rows = [('v',), (1,), (2,), (3,), (4,), (bomb,), (6,)]
    return InstrumentedSource(rows), bomb


# ---------------------------------------------------------------------------
# spool staging failures preserve the queued prefix
# ---------------------------------------------------------------------------

def test_staging_failure_bad_first_preserves_queued_prefix():
    # the first reported reproduction, exactly as specified
    type_error = TypeError('fixed TypeError object')
    source, bomb = scenario1_source(type_error)
    conv = CountingConverter()
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=3, tempdir=tmp)
        # read bad up to the error first
        bad_rows, bad_err = drain(iter(cap.bad))
        assert bad_rows == [BAD_HEADER]
        assert bad_err is type_error
        # good must still deliver everything queued before the failed row
        good_rows, good_err = drain(iter(cap.good))
        assert good_rows == [('v',), (1,), (2,), (3,), (4,)]
        assert good_err is type_error
        # row 6 was never read or converted; the source was closed once
        assert source.iterators == 1
        assert source.reads == 6
        assert conv.calls == [1, 2, 3, 4, bomb]
        assert source.closes == 1
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_staging_failure_interleaved_consumption():
    type_error = TypeError('fixed TypeError object')
    source, bomb = scenario1_source(type_error)
    conv = CountingConverter()
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=3, tempdir=tmp)
        gi, bi = iter(cap.good), iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(gi) == ('v',)
        # bad drives the pump until staging row 5 fails
        with pytest.raises(TypeError) as info:
            next(bi)
        assert info.value is type_error
        # good still collects its whole valid prefix, one record at a time
        assert next(gi) == (1,)
        assert next(gi) == (2,)
        assert next(gi) == (3,)
        assert next(gi) == (4,)
        with pytest.raises(TypeError) as info:
            next(gi)
        assert info.value is type_error
        # both branches keep raising the same object
        with pytest.raises(TypeError) as info:
            next(bi)
        assert info.value is type_error
        assert source.reads == 6
        assert conv.calls == [1, 2, 3, 4, bomb]
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_unpicklable_row_never_spooled_when_good_is_read_first():
    # spooling is lazy: consumed promptly, the unpicklable row stays in
    # memory and is delivered as-is; no failure occurs at all
    type_error = TypeError('fixed TypeError object')
    source, bomb = scenario1_source(type_error)
    conv = CountingConverter()
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=3, tempdir=tmp)
        good_rows, good_err = drain(iter(cap.good))
        assert good_rows == [('v',), (1,), (2,), (3,), (4,), (bomb,), (6,)]
        assert good_err is None
        bad_rows, bad_err = drain(iter(cap.bad))
        assert bad_rows == [BAD_HEADER]
        assert bad_err is None
        assert source.reads == 7
        assert conv.calls == [1, 2, 3, 4, bomb, 6]
        cap.close()
        assert os.listdir(tmp) == []


def test_partial_serialisation_failure_preserves_prefix():
    # pickling the row starts (first field) and then fails; with dumps-first
    # serialisation no partial bytes ever reach the spool file
    type_error = TypeError('fixed TypeError object')
    bomb = RaiseOnPickle(type_error)
    rows = [('a', 'b'),
            (1, 'x'), (2, 'x'), (3, 'x'), (4, 'x'),
            (5, bomb),   # row 5: serialisation fails after field 'a'
            (6, 'x')]
    source = InstrumentedSource(rows)
    conv = CountingConverter()
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('a', conv)], buffersize=2, tempdir=tmp)
        bad_rows, bad_err = drain(iter(cap.bad))
        assert bad_rows == [BAD_HEADER]
        assert bad_err is type_error
        good_rows, good_err = drain(iter(cap.good))
        assert good_rows == [('a', 'b'), (1, 'x'), (2, 'x'),
                             (3, 'x'), (4, 'x')]
        assert good_err is type_error
        assert source.reads == 6
        assert conv.calls == [1, 2, 3, 4, 5]
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_failed_bad_record_is_not_delivered_either():
    # a bad record whose original row cannot be pickled fails staging on the
    # bad side: earlier bad rows are still delivered, the failed one is not
    type_error = TypeError('fixed TypeError object')
    bomb = RaiseOnPickle(type_error)

    def func(v):
        if v == 'ok':
            return v
        raise ValueError('bad value')

    rows = [('v',), ('x',), (bomb,), ('ok',)]
    source = InstrumentedSource(rows)
    conv = CountingConverter(func)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        # drive the pump through good; row 2's bad record spills and fails
        good_rows, good_err = drain(iter(cap.good))
        assert good_rows == [('v',)]
        assert good_err is type_error
        bad_rows, bad_err = drain(iter(cap.bad))
        assert [r[0] for r in bad_rows[1:]] == [1]
        assert bad_rows[1][4] == 'builtins.ValueError'
        assert bad_err is type_error
        # row 3 was read and converted (its routing failed); row 'ok' was not
        assert source.reads == 3
        assert conv.calls == ['x', bomb]
        cap.close()
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# spool write / flush / open failures at different backlog positions
# ---------------------------------------------------------------------------

def _write_failure_case(monkeypatch, tmp, faults, buffersize, n_rows,
                        expected_prefix, expected_reads):
    os_error = OSError('fixed OSError object')
    for fault in faults:
        fault.setdefault('exc', os_error)
    injector = SpoolFaults(faults)
    injector.install(monkeypatch)
    rows = [('v',)] + [(i,) for i in range(1, n_rows + 1)]
    source = InstrumentedSource(rows)
    conv = CountingConverter()
    cap = convertcapture(source, [('v', conv)], buffersize=buffersize,
                         tempdir=tmp)
    bad_rows, bad_err = drain(iter(cap.bad))
    assert bad_rows == [BAD_HEADER]
    assert bad_err is os_error
    good_rows, good_err = drain(iter(cap.good))
    assert good_rows == [('v',)] + [(i,) for i in expected_prefix]
    assert good_err is os_error
    assert source.reads == expected_reads
    assert conv.calls == list(range(1, expected_reads))
    assert source.closes == 1
    cap.close()
    assert source.closes == 1
    assert os.listdir(tmp) == []
    return injector


def test_write_failure_at_first_spill(monkeypatch):
    # the buffer just filled and the very first spilled record cannot be
    # written: only the in-memory prefix survives
    with tempfile.TemporaryDirectory() as tmp:
        _write_failure_case(
            monkeypatch, tmp,
            faults=[dict(method='write', at=1)],
            buffersize=2, n_rows=6,
            expected_prefix=[1, 2], expected_reads=4)


def test_write_failure_partial_write_after_spills(monkeypatch):
    # the fourth write lands half of its bytes before failing; the three
    # sealed one-record segments and the memory record are unaffected, and
    # the half-written record is never read back
    with tempfile.TemporaryDirectory() as tmp:
        _write_failure_case(
            monkeypatch, tmp,
            faults=[dict(method='write', at=4, partial=True)],
            buffersize=1, n_rows=7,
            expected_prefix=[1, 2, 3, 4], expected_reads=6)


def test_write_failure_inside_later_segment(monkeypatch):
    # one sealed segment plus one complete record in the active segment
    # survive a write failure on the following record
    with tempfile.TemporaryDirectory() as tmp:
        _write_failure_case(
            monkeypatch, tmp,
            faults=[dict(method='write', at=3)],
            buffersize=2, n_rows=6,
            expected_prefix=[1, 2, 3, 4], expected_reads=6)


def test_partial_write_into_nonempty_segment(monkeypatch):
    # the failed record's partial bytes land behind two complete records in
    # the same still-open segment; the reader stops at the segment record
    # count and never touches the garbage bytes
    with tempfile.TemporaryDirectory() as tmp:
        _write_failure_case(
            monkeypatch, tmp,
            faults=[dict(method='write', at=3, partial=True)],
            buffersize=3, n_rows=8,
            expected_prefix=[1, 2, 3, 4, 5], expected_reads=7)


def test_flush_failure_preserves_written_records(monkeypatch):
    # the second record's flush fails; the first record of the active
    # segment was already flushed and stays deliverable
    with tempfile.TemporaryDirectory() as tmp:
        _write_failure_case(
            monkeypatch, tmp,
            faults=[dict(method='flush', at=2)],
            buffersize=2, n_rows=6,
            expected_prefix=[1, 2, 3], expected_reads=5)


def test_tempfile_creation_failure_after_spill(monkeypatch):
    # opening the second spool file fails; the memory record and the first
    # sealed segment survive
    with tempfile.TemporaryDirectory() as tmp:
        _write_failure_case(
            monkeypatch, tmp,
            faults=[dict(method='tempfile', at=2)],
            buffersize=1, n_rows=5,
            expected_prefix=[1, 2], expected_reads=4)


def test_close_failure_during_seal_is_cleanup(monkeypatch):
    # a close error on a fully flushed segment loses nothing and does not
    # fail the capture
    os_error = OSError('fixed OSError object')
    injector = SpoolFaults([dict(method='close', at=1, exc=os_error)])
    injector.install(monkeypatch)
    rows = [('v',)] + [(i,) for i in range(1, 5)]
    source = InstrumentedSource(rows)
    conv = CountingConverter()
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        # reading the (empty) bad branch first pumps everything and makes
        # the good backlog spill; the first seal's close() then fails
        bad_rows, bad_err = drain(iter(cap.bad))
        assert bad_rows == [BAD_HEADER]
        assert bad_err is None
        assert injector.fired('close')
        good_rows, good_err = drain(iter(cap.good))
        assert good_rows == [('v',), (1,), (2,), (3,), (4,)]
        assert good_err is None
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_spooled_queue_put_failure_keeps_earlier_records(monkeypatch):
    # queue-level durability contract: a failed put() drops only the record
    # being staged
    os_error = OSError('fixed OSError object')
    injector = SpoolFaults([dict(method='write', at=2, exc=os_error)])
    injector.install(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        q = _SpooledQueue(2, tmp)
        q.put(1)
        q.put(2)
        q.put(3)  # first spill, written fine
        with pytest.raises(OSError) as info:
            q.put(4)
        assert info.value is os_error
        assert len(q) == 3
        assert [q.get(), q.get(), q.get()] == [1, 2, 3]
        with pytest.raises(IndexError):
            q.get()
        q.close()
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# spool read failures terminate the whole capture
# ---------------------------------------------------------------------------

def test_read_failure_terminates_capture_bad_first(monkeypatch):
    # the second reported reproduction, exactly as specified
    os_error = OSError('fixed OSError object')
    rows = [('v',)] + [(i,) for i in range(1, 8)] + [('x',), ('y',)]
    source = InstrumentedSource(rows)

    def func(v):
        if v == 'x':
            raise ValueError('bad-x')
        return v

    conv = CountingConverter(func)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        bad8 = next(bi)  # drives the pump; good spills across memory and disk
        assert bad8[0] == 8 and bad8[3] == 'x'
        assert len(os.listdir(tmp)) == 6
        # good's next spool read now fails with the fixed OSError
        SpoolFaults([dict(method='open', at=1, exc=os_error)]).install(
            monkeypatch)
        gi = iter(cap.good)
        assert next(gi) == ('v',)
        assert next(gi) == (1,)  # memory-resident prefix
        with pytest.raises(OSError) as info:
            next(gi)
        assert info.value is os_error
        # the failure terminated the capture immediately
        assert source.closes == 1
        assert source.reads == 9
        assert conv.calls == [1, 2, 3, 4, 5, 6, 7, 'x']
        # bad is still in use: same failure, no normal EOF, no more pumping
        bad_rest, bad_err = drain(bi)
        assert bad_rest == []
        assert bad_err is os_error
        assert source.reads == 9
        assert conv.calls == [1, 2, 3, 4, 5, 6, 7, 'x']
        # repeated raises keep the same identity on both branches
        with pytest.raises(OSError) as info:
            next(gi)
        assert info.value is os_error
        with pytest.raises(OSError) as info:
            next(bi)
        assert info.value is os_error
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_read_failure_after_segments_already_read(monkeypatch):
    # the second spool open fails: the first sealed segment is still
    # delivered before the capture-wide failure
    os_error = OSError('fixed OSError object')
    rows = [('v',)] + [(i,) for i in range(1, 8)] + [('x',)]
    source = InstrumentedSource(rows)

    def func(v):
        if v == 'x':
            raise ValueError('bad-x')
        return v

    conv = CountingConverter(func)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(bi)[0] == 8
        SpoolFaults([dict(method='open', at=2, exc=os_error)]).install(
            monkeypatch)
        gi = iter(cap.good)
        assert next(gi) == ('v',)
        assert next(gi) == (1,)  # from memory
        assert next(gi) == (2,)  # first sealed segment reads fine
        with pytest.raises(OSError) as info:
            next(gi)
        assert info.value is os_error
        bad_rest, bad_err = drain(bi)
        assert bad_rest == []
        assert bad_err is os_error
        assert source.reads == 9
        assert source.closes == 1
        cap.close()
        assert os.listdir(tmp) == []


def test_read_method_failure_inside_first_segment(monkeypatch):
    # the spool file opens but reading it fails: only the memory prefix is
    # delivered before the same failure reaches both branches
    os_error = OSError('fixed OSError object')
    rows = [('v',)] + [(i,) for i in range(1, 6)] + [('x',)]
    source = InstrumentedSource(rows)

    def func(v):
        if v == 'x':
            raise ValueError('bad-x')
        return v

    conv = CountingConverter(func)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(bi)[0] == 6
        SpoolFaults([dict(method='read', at=1, exc=os_error)]).install(
            monkeypatch)
        gi = iter(cap.good)
        assert next(gi) == ('v',)
        assert next(gi) == (1,)
        with pytest.raises(OSError) as info:
            next(gi)
        assert info.value is os_error
        bad_rest, bad_err = drain(bi)
        assert bad_rest == []
        assert bad_err is os_error
        assert source.closes == 1
        cap.close()
        assert os.listdir(tmp) == []


def test_read_failure_good_backlog_then_bad_consumed(monkeypatch):
    # mirror order: bad backlog spills, good is consumed first and the
    # failure surfaces when bad reads its spool
    os_error = OSError('fixed OSError object')
    rows = [('n', 'v')]
    for i in range(1, 8):
        rows.append((i, 'bad'))  # all rows fail conversion -> bad branch
    rows.append((8, 'ok'))
    source = InstrumentedSource(rows)
    conv = CountingConverter(odd_good_converter())
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        gi = iter(cap.good)
        assert next(gi) == ('n', 'v')
        assert next(gi) == (8, 'OK')  # pumps everything; bad backlog spills
        assert len(os.listdir(tmp)) == 6
        SpoolFaults([dict(method='open', at=1, exc=os_error)]).install(
            monkeypatch)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(bi)[0] == 1  # memory-resident bad prefix
        with pytest.raises(OSError) as info:
            next(bi)
        assert info.value is os_error
        # good is exhausted but the capture failed: no clean EOF either
        with pytest.raises(OSError) as info:
            next(gi)
        assert info.value is os_error
        assert source.closes == 1
        cap.close()
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# truncated spool content is a hard failure, never a clean EOF
# ---------------------------------------------------------------------------

def test_truncated_sealed_segment_fails_both_branches(monkeypatch):
    rows = [('v',)] + [(i,) for i in range(1, 8)] + [('x',), ('y',)]
    source = InstrumentedSource(rows)

    def func(v):
        if v in ('x', 'y'):
            raise ValueError('bad-' + v)
        return v

    conv = CountingConverter(func)
    injector = SpoolFaults()  # no faults; only records spool paths
    injector.install(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(bi)[0] == 8
        assert next(bi)[0] == 9
        # good holds (1,) in memory and rows 2..7 in six sealed files;
        # corrupt the third file (row 4) so it ends before its record
        assert len(injector.paths) == 6
        os.truncate(injector.paths[2], 0)
        gi = iter(cap.good)
        assert next(gi) == ('v',)
        assert next(gi) == (1,)
        assert next(gi) == (2,)
        assert next(gi) == (3,)
        with pytest.raises(EOFError) as info:
            next(gi)
        err = info.value
        assert 'still unread' in str(err)
        # bad is still in use and gets the same object, not a normal EOF
        bad_rest, bad_err = drain(bi)
        assert bad_rest == []
        assert bad_err is err
        assert source.reads == 10
        assert source.closes == 1
        cap.close()
        assert os.listdir(tmp) == []


def test_truncated_active_segment_fails_instead_of_eof(monkeypatch):
    rows = [('v',)] + [(i,) for i in range(1, 6)] + [('x',)]
    source = InstrumentedSource(rows)

    def func(v):
        if v == 'x':
            raise ValueError('bad-x')
        return v

    conv = CountingConverter(func)
    injector = SpoolFaults()
    injector.install(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=3, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(bi)[0] == 6
        # rows 1..3 wait in memory, rows 4..5 in the still-open segment
        assert len(injector.paths) == 1
        os.truncate(injector.paths[0], 2)
        gi = iter(cap.good)
        assert next(gi) == ('v',)
        assert [next(gi), next(gi), next(gi)] == [(1,), (2,), (3,)]
        with pytest.raises((EOFError, pickle.UnpicklingError)) as info:
            next(gi)
        err = info.value
        bad_rest, bad_err = drain(bi)
        assert bad_rest == []
        assert bad_err is err
        assert source.closes == 1
        cap.close()
        assert os.listdir(tmp) == []


def test_truncated_spool_queue_level(monkeypatch):
    injector = SpoolFaults()
    injector.install(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        q = _SpooledQueue(2, tmp)
        for i in range(4):
            q.put(i)
        # rows 0..1 in memory, 2..3 sealed on disk; cut the segment short
        os.truncate(injector.paths[0], 0)
        assert q.get() == 0
        assert q.get() == 1
        with pytest.raises(EOFError):
            q.get()
        q.close()
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# source failures with backlog on both sides
# ---------------------------------------------------------------------------

class Boom(Exception):
    pass


@pytest.mark.parametrize('buffersize', [50, 1])
def test_source_failure_with_backlog_both_sides(buffersize):
    # buffersize=50 keeps the backlog in memory; buffersize=1 spills both
    # branches to disk before the source fails
    boom = Boom('source exploded')
    rows = [(i, 'ok' if i % 2 else 'bad') for i in range(1, 9)]
    source = FailingSource(('k', 'v'), rows, boom)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', odd_good_converter())],
                             buffersize=buffersize, tempdir=tmp)
        gi, bi = iter(cap.good), iter(cap.bad)
        assert next(gi) == ('k', 'v')
        assert next(bi) == BAD_HEADER
        good_rest, good_err = drain(gi)
        bad_rest, bad_err = drain(bi)
        # every queued record was delivered before the failure, on both sides
        assert [r[0] for r in good_rest] == [1, 3, 5, 7]
        assert [r[0] for r in bad_rest] == [2, 4, 6, 8]
        assert good_err is boom
        assert bad_err is boom
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# original failure vs cleanup errors
# ---------------------------------------------------------------------------

def test_read_failure_wins_over_cleanup_errors(monkeypatch):
    read_error = OSError('fixed read failure')
    close_error = OSError('fixed source close failure')
    spool_close_error = OSError('fixed spool close failure')
    rows = [('v',)] + [(i,) for i in range(1, 8)] + [('x',)]
    source = InstrumentedSource(rows, close_exc=close_error)

    def func(v):
        if v == 'x':
            raise ValueError('bad-x')
        return v

    conv = CountingConverter(func)
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        assert next(bi)[0] == 8
        faults = SpoolFaults(
            [dict(method='open', at=1, exc=read_error)]
            + [dict(method='close', at=i, exc=spool_close_error)
               for i in range(1, 6)])
        faults.install(monkeypatch)
        gi = iter(cap.good)
        assert next(gi) == ('v',)
        assert next(gi) == (1,)
        with pytest.raises(OSError) as info:
            next(gi)
        # the original read failure, not any cleanup error
        assert info.value is read_error
        bad_rest, bad_err = drain(bi)
        assert bad_rest == []
        assert bad_err is read_error
        # the source close was attempted exactly once (and failed quietly)
        assert source.closes == 1
        # explicit close() must not raise any of the cleanup errors
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []


def test_write_failure_wins_over_cleanup_errors(monkeypatch):
    write_error = OSError('fixed write failure')
    close_error = OSError('fixed source close failure')
    rows = [('v',)] + [(i,) for i in range(1, 6)]
    source = InstrumentedSource(rows, close_exc=close_error)
    conv = CountingConverter()
    with tempfile.TemporaryDirectory() as tmp:
        faults = SpoolFaults([dict(method='write', at=1, exc=write_error)])
        faults.install(monkeypatch)
        cap = convertcapture(source, [('v', conv)], buffersize=2, tempdir=tmp)
        bad_rows, bad_err = drain(iter(cap.bad))
        assert bad_rows == [BAD_HEADER]
        assert bad_err is write_error
        good_rows, good_err = drain(iter(cap.good))
        assert good_rows == [('v',), (1,), (2,)]
        assert good_err is write_error
        assert source.closes == 1
        cap.close()
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# no-fault equivalence, early close, object isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', ['good-first', 'bad-first', 'interleaved'])
def test_no_fault_small_and_large_buffers_match(order):
    rows = odd_good_table(30)
    expected_good = [('n', 'v')] + [(i, 'OK') for i in range(1, 31, 2)]
    results = {}
    for size in (1, 3, 1000):
        with tempfile.TemporaryDirectory() as tmp:
            cap = convertcapture(rows, [('v', odd_good_converter())],
                                 buffersize=size, tempdir=tmp)
            if order == 'good-first':
                good = materialise(cap.good)
                bad = materialise(cap.bad)
            elif order == 'bad-first':
                bad = materialise(cap.bad)
                good = materialise(cap.good)
            else:
                gi, bi = iter(cap.good), iter(cap.bad)
                good, bad = [next(gi)], [next(bi)]
                for _ in range(200):
                    g_row, g_err = drain_once(gi)
                    b_row, b_err = drain_once(bi)
                    if g_err is not None or b_err is not None:
                        break
                    if g_row is not None:
                        good.append(g_row)
                    if b_row is not None:
                        bad.append(b_row)
                    if g_row is None and b_row is None:
                        break
                good += drain(gi)[0]
                bad += drain(bi)[0]
            cap.close()
            assert os.listdir(tmp) == []
            results[size] = (good, bad)
    for size in (1, 3, 1000):
        assert results[size][0] == expected_good
        assert [r[0] for r in results[size][1][1:]] == list(range(2, 31, 2))
    assert results[1] == results[3] == results[1000]


def drain_once(it):
    try:
        return next(it), None
    except StopIteration:
        return None, None
    except BaseException as e:
        return None, e


def test_early_close_releases_spool_and_stops_source():
    source = InstrumentedSource(odd_good_table(30))
    conv = CountingConverter(odd_good_converter())
    with tempfile.TemporaryDirectory() as tmp:
        cap = convertcapture(source, [('v', conv)], buffersize=1, tempdir=tmp)
        bi = iter(cap.bad)
        assert next(bi) == BAD_HEADER
        # each bad row read pumps two rows; the good backlog spills to disk
        bad_prefix = [next(bi) for _ in range(5)]
        assert [r[0] for r in bad_prefix] == [2, 4, 6, 8, 10]
        assert len(os.listdir(tmp)) > 0
        gi = iter(cap.good)
        assert next(gi) == ('n', 'v')
        reads_before = source.reads
        calls_before = len(conv.calls)
        cap.close()
        assert source.closes == 1
        assert os.listdir(tmp) == []
        # nothing advances after close, on either branch
        with pytest.raises(ArgumentError):
            next(gi)
        with pytest.raises(ArgumentError):
            next(bi)
        with pytest.raises(ArgumentError):
            iter(cap.bad)
        assert source.reads == reads_before
        assert len(conv.calls) == calls_before
        # idempotent, and the source is still closed exactly once
        cap.close()
        assert source.closes == 1
        assert source.iterators == 1


def test_two_captures_isolate_their_spool_files(monkeypatch):
    injector1 = SpoolFaults()
    injector2 = SpoolFaults()
    rows1 = odd_good_table(12)
    rows2 = odd_good_table(12)
    with tempfile.TemporaryDirectory() as tmp:
        injector1.install(monkeypatch)
        cap1 = convertcapture(InstrumentedSource(rows1),
                              [('v', odd_good_converter())],
                              buffersize=1, tempdir=tmp)
        b1 = iter(cap1.bad)
        assert next(b1) == BAD_HEADER
        assert [next(b1)[0] for _ in range(3)] == [2, 4, 6]  # good spills
        paths1 = list(injector1.paths)
        monkeypatch.undo()

        injector2.install(monkeypatch)
        cap2 = convertcapture(InstrumentedSource(rows2),
                              [('v', odd_good_converter())],
                              buffersize=1, tempdir=tmp)
        b2 = iter(cap2.bad)
        assert next(b2) == BAD_HEADER
        assert [next(b2)[0] for _ in range(3)] == [2, 4, 6]
        paths2 = list(injector2.paths)
        monkeypatch.undo()

        # disjoint file sets, both present on disk
        assert paths1 and paths2
        assert set(paths1).isdisjoint(paths2)
        assert all(os.path.exists(p) for p in paths1 + paths2)

        # closing the first capture removes only its own files
        cap1.close()
        assert not any(os.path.exists(p) for p in paths1)
        assert all(os.path.exists(p) for p in paths2)

        # the second capture is fully usable afterwards
        good2 = materialise(cap2.good)
        bad2 = [next(b2)] + list(b2)
        assert [r[0] for r in good2[1:]] == list(range(1, 13, 2))
        assert [r[0] for r in bad2] == [8, 10, 12]
        cap2.close()
        assert os.listdir(tmp) == []
