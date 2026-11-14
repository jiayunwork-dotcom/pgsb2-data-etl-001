from __future__ import absolute_import, print_function, division

import gc
import threading
import time
import weakref

import pytest

from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl.transform.maps import rowmapconcurrent


class RetryableError(Exception):
    pass


class OtherError(Exception):
    pass


class Watchdog(object):
    """Fail a test if a barrier wait does not complete in time, so timing
    problems surface as assertion errors instead of hangs."""

    def __init__(self, timeout=10):
        self.timeout = timeout

    def __enter__(self):
        return self

    def wait(self, event):
        if not event.wait(self.timeout):
            raise AssertionError('timed out waiting for event')

    def barrier(self, barrier):
        try:
            barrier.wait(self.timeout)
        except threading.BrokenBarrierError:
            raise AssertionError('timed out waiting on barrier')

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False
        return False


class Consumer(threading.Thread):
    """Drive an iterator in a separate thread so the test thread stays free
    to coordinate events while a next() call is in flight."""

    def __init__(self, it, on_row=None):
        super(Consumer, self).__init__()
        self.it = it
        self.rows = []
        self.error = None
        self.on_row = on_row
        self.done = threading.Event()
        self.first_row = threading.Event()

    def run(self):
        first = True
        try:
            for row in self.it:
                self.rows.append(row)
                if self.on_row is not None:
                    self.on_row(row)
                if first:
                    self.first_row.set()
                    first = False
        except StopIteration:
            pass
        except BaseException as e:  # noqa: B036 - re-raised by the test
            self.error = e
        finally:
            self.done.set()


def _counting_source(rows):
    """A source table that counts how often next() was called and records
    the thread doing the calls."""
    header = rows[0]
    data = list(rows[1:])
    state = {'next_calls': 0, 'threads': set(), 'closed': False,
             'close_calls': 0}

    class CountingIterator(object):

        def __init__(self):
            self._i = 0

        def __iter__(self):
            return self

        def __next__(self):
            state['threads'].add(threading.get_ident())
            if self._i >= len(data) + 1:
                raise StopIteration()
            row = header if self._i == 0 else data[self._i - 1]
            self._i += 1
            state['next_calls'] += 1
            return row

        def close(self):
            state['closed'] = True
            state['close_calls'] += 1

    class CountingTable(object):

        def __iter__(self):
            return CountingIterator()

    return CountingTable(), state


def _petl_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith('petl-rowmapconcurrent')]


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------

def test_rowmapconcurrent_basic_ordered():
    table = (('id', 'sex', 'age', 'height', 'weight'),
             (1, 'male', 16, 1.45, 62.0),
             (2, 'female', 19, 1.34, 55.4),
             (3, 'female', 17, 1.78, 74.4),
             (4, 'male', 21, 1.33, 45.2),
             (5, '-', 25, 1.65, 51.9))

    def rowmapper(row):
        return [row[0], row.sex.upper(), row.age * 12,
                row.height / row.weight ** 2]

    actual = rowmapconcurrent(table, rowmapper,
                              header=['subject_id', 'gender', 'age_months',
                                      'bmi'], workers=3)
    expect = (('subject_id', 'gender', 'age_months', 'bmi'),
              (1, 'MALE', 16 * 12, 1.45 / 62.0 ** 2),
              (2, 'FEMALE', 19 * 12, 1.34 / 55.4 ** 2),
              (3, 'FEMALE', 17 * 12, 1.78 / 74.4 ** 2),
              (4, 'MALE', 21 * 12, 1.33 / 45.2 ** 2),
              (5, '-', 25 * 12, 1.65 / 51.9 ** 2))
    ieq(expect, actual)
    ieq(expect, actual)  # a view can be iterated repeatedly and independently


def test_rowmapconcurrent_table_method_chaining():
    import petl as etl
    table = etl.wrap((('id', 'value'), (1, 'a'), (2, 'b')))

    def rowmapper(row):
        return [row[0], row[1].upper()]

    actual = table.rowmapconcurrent(rowmapper, header=['id', 'value'],
                                    workers=2)
    ieq((('id', 'value'), (1, 'A'), (2, 'B')), actual)


def test_rowmapconcurrent_empty_and_headerless():
    def rowmapper(row):
        return [row[0]]

    # header only
    actual = rowmapconcurrent((('foo',),), rowmapper, header=('bar',))
    ieq((('bar',),), actual)

    # no rows at all
    actual = rowmapconcurrent([], rowmapper, header=('bar',))
    ieq([], actual)


def test_rowmapconcurrent_unordered_output():
    # completion order differs from input order: gate the earlier rows so a
    # later row provably finishes first, then check every row appears once
    release_early = threading.Event()
    later_finished = threading.Event()
    later_count = [0]
    count_lock = threading.Lock()

    def rowmapper(row):
        value = int(row[0])
        if value < 3:
            later_finished.wait(10)
            release_early.wait(10)
        else:
            with count_lock:
                later_count[0] += 1
                if later_count[0] == 3:
                    later_finished.set()
        return [value]

    table = (('foo',),) + tuple((i,) for i in range(6))
    it = iter(rowmapconcurrent(table, rowmapper, header=('foo',),
                               workers=4, ordered=False))
    assert next(it) == ('foo',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(later_finished)
    # rows 3..5 have already completed while rows 0..2 are gated
    assert consumer.first_row.wait(10)
    release_early.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is None
    values = sorted(r[0] for r in consumer.rows)
    assert values == list(range(6))  # every successful row exactly once
    # the first delivered row is one of the later input rows
    assert consumer.rows[0][0] >= 3


def test_rowmapconcurrent_unordered_fast_first_delivered_first():
    started = threading.Event()
    release_slow = threading.Event()
    done_fast = threading.Event()

    def rowmapper(row):
        if row[0] == 'slow':
            started.set()
            release_slow.wait(10)
            return ['slow-done']
        done_fast.set()
        return ['fast-done']

    table = (('foo',), ('slow',), ('fast',))
    it = iter(rowmapconcurrent(table, rowmapper, header=('out',),
                               workers=2, ordered=False))
    assert next(it) == ('out',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(started)
        # the fast row (second input row) completes while the first is blocked
        wd.wait(done_fast)
        wd.wait(consumer.first_row)
    assert consumer.rows[0] == ('fast-done',)
    release_slow.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is None
    assert consumer.rows == [('fast-done',), ('slow-done',)]


# --------------------------------------------------------------------------
# ordering, window and source thread affinity
# --------------------------------------------------------------------------

def test_rowmapconcurrent_first_row_delivered_before_window_fills():
    # ordered mode: row 0 finishes immediately while the remaining rows fill
    # the window behind it. The first result must be delivered without the
    # slow rows completing.
    release = threading.Event()
    first_finished = threading.Event()

    def rowmapper(row):
        if row[0] == 0:
            first_finished.set()
            return [0]
        release.wait(10)
        return [row[0]]

    table = (('foo',),) + tuple((i,) for i in range(6))
    it = iter(rowmapconcurrent(table, rowmapper, header=('foo',),
                               workers=2, maxpending=4, ordered=True))
    assert next(it) == ('foo',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(first_finished)
        wd.wait(consumer.first_row)
    # delivered as soon as row 0 and its predecessors are done, not after the
    # whole window fills
    assert consumer.rows[0] == (0,)
    release.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is None
    assert [r[0] for r in consumer.rows] == [0, 1, 2, 3, 4, 5]


def test_rowmapconcurrent_window_is_bounded():
    # with workers=2 and maxpending=4, once two mappers are running the
    # source must have supplied at most 4 data rows (2 running + 2 queued)
    release = threading.Event()
    inside = [0]
    inside_lock = threading.Lock()
    two_running = threading.Event()

    table, state = _counting_source(
        (('foo',),) + tuple((i,) for i in range(10)))

    def rowmapper(row):
        with inside_lock:
            inside[0] += 1
            if inside[0] == 2:
                two_running.set()
        release.wait(10)
        return [row[0]]

    it = iter(rowmapconcurrent(table, rowmapper, header=('foo',),
                               workers=2, maxpending=4, ordered=True))
    assert next(it) == ('foo',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(two_running)
    # both mappers are blocked; the pump cannot pull beyond the window
    assert state['next_calls'] <= 1 + 4
    assert state['next_calls'] >= 1 + 2
    release.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is None
    assert [r[0] for r in consumer.rows] == list(range(10))


def test_rowmapconcurrent_default_maxpending():
    view = rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                            workers=3)
    assert view.maxpending == 6


def test_rowmapconcurrent_source_only_read_in_caller_thread():
    table, state = _counting_source(
        (('foo',),) + tuple((i,) for i in range(12)))
    caller_ident = threading.get_ident()

    seen_threads = set()

    def rowmapper(row):
        seen_threads.add(threading.get_ident())
        return [row[0]]

    result = rowmapconcurrent(table, rowmapper, header=('foo',), workers=4)
    # iterate manually: list() would call len() first and iterate the view
    # twice (standard petl container behaviour)
    rows = []
    for row in result:
        rows.append(row)
    assert rows[1:] == [(i,) for i in range(12)]
    assert state['threads'] == {caller_ident}
    assert state['next_calls'] == 13  # header, twelve rows, one StopIteration
    # and the rowmapper really did run on other threads
    assert seen_threads and caller_ident not in seen_threads


def test_rowmapconcurrent_no_source_or_threads_on_create_and_iter():
    table, state = _counting_source((('foo',), (1,), (2,)))
    view = rowmapconcurrent(table, lambda r: [r[0]], header=('foo',),
                            workers=2)
    assert state['next_calls'] == 0
    it = iter(view)
    assert state['next_calls'] == 0
    assert _petl_threads() == []
    # even pulling the header starts nothing
    assert next(it) == ('foo',)
    assert state['next_calls'] == 1
    assert _petl_threads() == []
    it.close()


def test_rowmapconcurrent_header_failure_closes_source_without_threads():
    class BoomIterator(object):

        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError('cannot read header')

        def close(self):
            state['closed'] = True

    class BoomTable(object):

        def __iter__(self):
            return BoomIterator()

    state = {'closed': False}
    it = iter(rowmapconcurrent(BoomTable(), lambda r: [r[0]],
                               header=('a',), workers=2))
    with pytest.raises(ValueError):
        next(it)
    assert state['closed'] is True
    assert _petl_threads() == []
    with pytest.raises(StopIteration):
        next(it)


def test_rowmapconcurrent_zero_delay_is_accepted_and_retries():
    attempts = []

    def rowmapper(row):
        attempts.append(1)
        if len(attempts) == 1:
            raise RetryableError('once')
        return ['ok']

    actual = rowmapconcurrent((('foo',), (1,)), rowmapper, header=('bar',),
                              workers=1, retry_on=(RetryableError,),
                              retry_delays=(0,))
    ieq((('bar',), ('ok',)), actual)
    assert len(attempts) == 2


# --------------------------------------------------------------------------
# retries
# --------------------------------------------------------------------------

def test_rowmapconcurrent_retry_then_success_retains_position():
    attempts = []
    lock = threading.Lock()

    def rowmapper(row):
        with lock:
            n = len([a for a in attempts if a == row[0]])
            attempts.append(row[0])
        if row[0] == 2 and n == 0:
            raise RetryableError('boom')
        return [row[0] * 10]

    table = (('foo',), (1,), (2,), (3,))
    actual = rowmapconcurrent(table, rowmapper, header=('bar',),
                              workers=2, retry_on=(RetryableError,),
                              retry_delays=(0,))
    ieq((('bar',), (10,), (20,), (30,)), actual)
    assert attempts.count(2) == 2  # the failing row was retried once


def test_rowmapconcurrent_retry_does_not_hold_worker():
    # workers=1: while row 0 waits for its retry, row 1 must still run
    allow_failure = threading.Event()
    row1_done = threading.Event()
    order = []
    order_lock = threading.Lock()

    def rowmapper(row):
        if row[0] == 0:
            with order_lock:
                first = not order
                order.append(0)
            if first:
                allow_failure.wait(10)
                raise RetryableError('retry me')
            with order_lock:
                order.append('0-retry')
            return ['zero']
        with order_lock:
            order.append(1)
        row1_done.set()
        return ['one']

    table = (('foo',), (0,), (1,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(RetryableError,),
                               retry_delays=(0,), ordered=True))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    # let row 0 fail; its zero-delay retry is scheduled without occupying the
    # single worker, which must pick up row 1 in the meantime
    allow_failure.set()
    with Watchdog() as wd:
        wd.wait(row1_done)
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is None
    assert consumer.rows == [('zero',), ('one',)]
    assert order[:2] == [0, 1]


def test_rowmapconcurrent_retry_delays_are_waited():
    first_attempt = threading.Event()

    def rowmapper(row):
        if not first_attempt.is_set():
            first_attempt.set()
            raise RetryableError('not yet')
        return ['ok']

    table = (('foo',), (1,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(RetryableError,),
                               retry_delays=(3600,)))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(first_attempt)
    # the retry is now waiting on a long delay; close must cancel it at once
    it.close()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert isinstance(consumer.error, StopIteration) or consumer.error is None
    assert _petl_threads() == []


def test_rowmapconcurrent_retries_exhausted_raises_original():
    err = RetryableError('persistent')

    def rowmapper(row):
        raise err

    table = (('foo',), (1,), (2,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=2, retry_on=(RetryableError,),
                               retry_delays=(0, 0)))
    assert next(it) == ('bar',)
    with pytest.raises(RetryableError) as exc:
        next(it)
    assert exc.value is err  # the original exception object
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_rowmapconcurrent_non_retryable_exception_fails_row():
    err = OtherError('no retries for me')

    def rowmapper(row):
        raise err

    table = (('foo',), (1,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(RetryableError,),
                               retry_delays=(0, 0)))
    assert next(it) == ('bar',)
    with pytest.raises(OtherError) as exc:
        next(it)
    assert exc.value is err


# --------------------------------------------------------------------------
# failure semantics
# --------------------------------------------------------------------------

def test_rowmapconcurrent_failure_position_ordered_and_identity():
    err = OtherError('row 2 fails')
    source, state = _counting_source(
        (('foo',), (1,), (2,), (3,), (4,)))

    def rowmapper(row):
        if row[0] == 2:
            raise err
        return [row[0]]

    it = iter(rowmapconcurrent(source, rowmapper, header=('bar',),
                               workers=4, ordered=True))
    assert next(it) == ('bar',)
    assert next(it) == (1,)  # every result before the failure is delivered
    next_calls_at_failure = state['next_calls']
    with pytest.raises(OtherError) as exc:
        next(it)  # failure raised at the failing row's position
    assert exc.value is err  # the original exception object
    # after failure no more rows are pulled from the source
    assert state['next_calls'] == next_calls_at_failure
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_rowmapconcurrent_failure_unordered_discards_later_results():
    err = OtherError('middle fails')
    release_slow = threading.Event()
    slow_started = threading.Event()
    ok_finished = threading.Event()
    bad_raised = threading.Event()

    def rowmapper(row):
        if row[0] == 'bad':
            # let every other admitted call start and the good row finish
            # before the terminal failure fixes the completion boundary
            assert ok_finished.wait(10)
            assert slow_started.wait(10)
            bad_raised.set()
            raise err
        if row[0] == 'slow':
            slow_started.set()
            release_slow.wait(10)
            return ['slow']
        ok_finished.set()
        return [row[0]]

    table = (('foo',), ('ok',), ('bad',), ('slow',))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=3, ordered=False))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    # release the still-running call only once the failure boundary has been
    # recorded, so the slow row provably completes after it; teardown can then
    # join it without waiting out its blocking timeout

    def release_after_boundary():
        assert bad_raised.wait(10)
        assert it._state.failed.wait(10)
        release_slow.set()

    watcher = threading.Thread(target=release_after_boundary)
    watcher.start()
    with Watchdog() as wd:
        # the good row finishes before the failure is observed
        wd.wait(ok_finished)
        wd.wait(slow_started)
        wd.wait(consumer.done)
    watcher.join(10)
    assert consumer.error is err  # original exception object propagated
    assert ('ok',) in consumer.rows
    assert ('slow',) not in consumer.rows  # finished after failure: dropped
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_rowmapconcurrent_base_exception_not_retried():
    class Fatal(BaseException):
        pass

    fatal = Fatal('fatal')
    attempts = []

    def rowmapper(row):
        attempts.append(1)
        raise fatal

    table = (('foo',), (1,))
    # BaseException subclasses cannot be registered in retry_on
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(),
                               retry_delays=(0, 0)))
    assert next(it) == ('bar',)
    with pytest.raises(Fatal) as exc:
        next(it)
    assert exc.value is fatal
    assert len(attempts) == 1  # never retried
    assert _petl_threads() == []


def test_rowmapconcurrent_keyboard_interrupt_in_mapper():
    def rowmapper(row):
        raise KeyboardInterrupt()

    table = (('foo',), (1,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1))
    assert next(it) == ('bar',)
    with pytest.raises(KeyboardInterrupt):
        next(it)
    assert _petl_threads() == []


def test_rowmapconcurrent_keyboard_interrupt_in_caller_next():
    release = threading.Event()

    def rowmapper(row):
        release.wait(10)
        return [row[0]]

    table = (('foo',), (1,), (2,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1))
    assert next(it) == ('bar',)
    # simulate Ctrl-C landing while the caller waits inside next()

    def interrupt():
        release.wait(2)
        import _thread
        _thread.interrupt_main()

    t = threading.Thread(target=interrupt)
    t.start()
    with pytest.raises(KeyboardInterrupt):
        next(it)
    t.join()
    release.set()
    assert _petl_threads() == []


# --------------------------------------------------------------------------
# length errors
# --------------------------------------------------------------------------

def test_rowmapconcurrent_wrong_length_fails_with_row_number():
    def rowmapper(row):
        if row[0] == 3:
            return [1, 2]  # header expects one field
        return [row[0]]

    table = (('foo',), (1,), (2,), (3,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=2, retry_on=(ArgumentError,),
                               retry_delays=(0, 0, 0)))
    assert next(it) == ('bar',)
    assert next(it) == (1,)
    assert next(it) == (2,)
    with pytest.raises(ArgumentError) as exc:
        next(it)
    assert 'row 3' in str(exc.value)
    assert _petl_threads() == []


def test_rowmapconcurrent_non_sequence_return_fails_with_row_number():
    def rowmapper(row):
        if row[0] == 2:
            return 42
        return [row[0]]

    table = (('foo',), (1,), (2,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1))
    assert next(it) == ('bar',)
    assert next(it) == (1,)
    with pytest.raises(ArgumentError) as exc:
        next(it)
    assert 'row 2' in str(exc.value)


# --------------------------------------------------------------------------
# teardown
# --------------------------------------------------------------------------

def test_rowmapconcurrent_close_waits_for_running_calls():
    entered = threading.Event()
    release = threading.Event()
    mapper_returned = threading.Event()

    def rowmapper(row):
        entered.set()
        release.wait(10)
        mapper_returned.set()
        return [row[0]]

    table = (('foo',), (1,), (2,), (3,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=2))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(entered)
    release.set()  # let the in-flight call finish; close must wait for it
    it.close()
    # close() returned only after the in-flight call had finished
    assert mapper_returned.is_set()
    assert _petl_threads() == []
    with Watchdog() as wd:
        wd.wait(consumer.done)
    with pytest.raises(StopIteration):
        next(it)


def test_rowmapconcurrent_close_cancels_waiting_retry():
    first_attempt = threading.Event()

    def rowmapper(row):
        if not first_attempt.is_set():
            first_attempt.set()
            raise RetryableError('later')
        return ['should-never-run']

    table = (('foo',), (1,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(RetryableError,),
                               retry_delays=(3600,)))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(first_attempt)
    it.close()  # must not wait for the hour-long delay
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert _petl_threads() == []


def test_rowmapconcurrent_close_closes_source_iterator():
    source, state = _counting_source((('foo',), (1,), (2,)))
    it = iter(rowmapconcurrent(source, lambda r: [r[0]], header=('bar',),
                               workers=1))
    assert next(it) == ('bar',)
    it.close()
    assert state['closed'] is True
    assert _petl_threads() == []


def test_rowmapconcurrent_exhaustion_closes_source_and_threads():
    source, state = _counting_source((('foo',), (1,), (2,)))
    result = rowmapconcurrent(source, lambda r: [r[0]], header=('bar',),
                              workers=2)
    ieq((('bar',), (1,), (2,)), result)
    assert state['closed'] is True
    assert _petl_threads() == []


def test_rowmapconcurrent_garbage_collection_cleans_up():
    entered = threading.Event()
    release = threading.Event()

    def rowmapper(row):
        entered.set()
        release.wait(10)
        return [row[0]]

    table = (('foo',),) + tuple((i,) for i in range(8))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=3))
    assert next(it) == ('bar',)
    # hold the iterator in a container so the helper thread does not keep a
    # direct closure reference that would survive `del it`
    holder = [it]

    # start the workers and leave the first call blocked inside rowmapper
    def pull_first_row():
        try:
            next(holder[0])
        except BaseException:
            pass

    puller = threading.Thread(target=pull_first_row)
    puller.start()
    with Watchdog() as wd:
        wd.wait(entered)
    release.set()  # unblock the in-flight calls so __del__ can join them
    ref = weakref.ref(it)
    holder.clear()
    del it
    with Watchdog() as wd:
        puller.join(wd.timeout)
    assert not puller.is_alive()
    gc.collect()
    assert ref() is None  # the abandoned iterator is collectable
    assert _petl_threads() == []  # __del__ joined every worker thread


def test_rowmapconcurrent_iterators_are_independent():
    table = (('foo',),) + tuple((i,) for i in range(6))

    def rowmapper(row):
        return [row[0] + 1]

    view = rowmapconcurrent(table, rowmapper, header=('bar',), workers=2)
    it1 = iter(view)
    it2 = iter(view)
    assert next(it1) == ('bar',)
    assert next(it2) == ('bar',)
    # close one iterator; the other must keep working with its own threads
    it1.close()
    assert list(it2) == [(i + 1,) for i in range(6)]
    assert _petl_threads() == []


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [0, -1, 1.5, True, False, '2'])
def test_rowmapconcurrent_invalid_workers(bad):
    with pytest.raises(ArgumentError):
        rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                         workers=bad)


@pytest.mark.parametrize('bad', [0, -1, 1.5, True, False, '2'])
def test_rowmapconcurrent_invalid_maxpending(bad):
    with pytest.raises(ArgumentError):
        rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                         maxpending=bad)


def test_rowmapconcurrent_invalid_retry_on():
    with pytest.raises(ArgumentError):
        rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                         retry_on=(ValueError, 42))
    with pytest.raises(ArgumentError):
        rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                         retry_on=(BaseException,))
    with pytest.raises(ArgumentError):
        rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                         retry_on=[ValueError])  # must be a tuple


@pytest.mark.parametrize('bad', [-0.1, float('inf'), float('nan'), '1', True])
def test_rowmapconcurrent_invalid_retry_delays(bad):
    with pytest.raises(ArgumentError):
        rowmapconcurrent((('a',),), lambda r: [r[0]], header=('a',),
                         retry_delays=(bad,))


def test_rowmapconcurrent_validation_happens_without_source_read():
    source, state = _counting_source((('foo',), (1,)))
    with pytest.raises(ArgumentError):
        rowmapconcurrent(source, lambda r: [r[0]], header=('a',),
                         workers=True)
    assert state['next_calls'] == 0


# --------------------------------------------------------------------------
# failure boundaries: a terminal failure must stop intake immediately
# --------------------------------------------------------------------------

class GatedSource(object):
    """Source whose next() for data row n waits until data row n-1 has
    entered the mapper (or a failure has been announced). This bounds how far
    the pump can have read ahead when a row fails, and lets tests prove that
    later rows never enter rowmapper."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0
        self.entered = {}
        self.stop_wait = threading.Event()
        self.lock = threading.Lock()
        self.pulled = []
        self.threads = set()
        self.close_calls = 0

    def entered_event(self, n):
        with self.lock:
            return self.entered.setdefault(n, threading.Event())

    def __iter__(self):
        return self

    def __next__(self):
        self.threads.add(threading.get_ident())
        if self.i == 0:
            self.i = 1
            return ('foo',)
        n = self.i - 1
        if n >= 1:
            ev = self.entered_event(n - 1)
            deadline = time.monotonic() + 10
            while not ev.is_set() and not self.stop_wait.is_set():
                if time.monotonic() > deadline:
                    raise AssertionError(
                        'source next for row %d timed out waiting for row %d'
                        % (n, n - 1))
                ev.wait(0.01)
        if n >= len(self.values):
            raise StopIteration()
        self.i += 1
        self.pulled.append(n)
        return (self.values[n],)

    def close(self):
        self.close_calls += 1


def _drive_failing_repro(ordered, exc):
    source = GatedSource(range(6))
    calls = []
    lock = threading.Lock()
    caller_ident = threading.get_ident()

    def rowmapper(row):
        n = row[0]
        with lock:
            calls.append(n)
        source.entered_event(n).set()
        if n == 0:
            source.stop_wait.set()
            raise exc
        return [n]

    it = iter(rowmapconcurrent(source, rowmapper, header=('bar',),
                               workers=1, maxpending=4, ordered=ordered))
    assert next(it) == ('bar',)
    with pytest.raises(type(exc)) as raised:
        next(it)
    assert raised.value is exc
    time.sleep(0.1)
    pulled_at_error = len(source.pulled)
    assert calls == [0]
    with pytest.raises(StopIteration):
        next(it)
    assert len(source.pulled) == pulled_at_error  # no pulls after the error
    assert source.close_calls == 1
    assert source.threads == {caller_ident}
    assert _petl_threads() == []


@pytest.mark.parametrize('ordered', [True, False])
def test_failstop_reported_repro_row0(ordered):
    _drive_failing_repro(ordered, OtherError('row 0 boom'))


@pytest.mark.parametrize('ordered', [True, False])
def test_failstop_reported_repro_keyboard_interrupt(ordered):
    _drive_failing_repro(ordered, KeyboardInterrupt())


def test_failstop_earlier_blocked_later_fails_ordered():
    release_row0 = threading.Event()
    row1_failed = threading.Event()
    calls = []
    lock = threading.Lock()
    err = OtherError('row 1')

    def rowmapper(row):
        n = row[0]
        with lock:
            calls.append(n)
        if n == 0:
            assert release_row0.wait(10)
            return ['zero']
        if n == 1:
            row1_failed.set()
            raise err
        return [str(n)]

    table = (('foo',),) + tuple((i,) for i in range(8))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=2, maxpending=6, ordered=True))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(row1_failed)
        time.sleep(0.05)  # give a (buggy) free worker a chance to start row 2
    assert calls == [0, 1]  # the boundary at row 1 stopped rows 2+
    release_row0.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.rows == [('zero',)]   # earlier success delivered first
    assert consumer.error is err          # then the failure at row 1
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_earlier_row_waiting_retry_is_delivered():
    # Row 0's retry is pending when row 1 fails terminally. The caller must
    # not return the row-1 error early: row 0's retry still runs, its result
    # is delivered, and only then is the error raised at row 1.
    row0_first_attempt = threading.Event()
    row1_failed = threading.Event()
    retry_started = threading.Event()
    allow_retry = threading.Event()
    attempts = []
    lock = threading.Lock()
    err = OtherError('row 1 terminal')

    def rowmapper(row):
        n = row[0]
        with lock:
            attempts.append(n)
            count0 = attempts.count(0)
        if n == 0:
            if count0 == 1:
                row0_first_attempt.set()
                raise RetryableError('retry row 0')
            retry_started.set()
            assert allow_retry.wait(10)
            return ['zero']
        if n == 1:
            row1_failed.set()
            raise err
        return [str(n)]

    table = (('foo',),) + tuple((i,) for i in range(6))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=2, maxpending=6, ordered=True,
                               retry_on=(RetryableError,),
                               retry_delays=(0.2,)))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(row0_first_attempt)
        wd.wait(row1_failed)
        wd.wait(retry_started)
        # row 1 has already failed, but the caller still waits for row 0
        assert consumer.rows == []
        assert not consumer.done.is_set()
    allow_retry.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.rows == [('zero',)]
    assert consumer.error is err
    assert attempts.count(0) == 2      # the retry really did run
    assert 2 not in attempts and 3 not in attempts
    assert _petl_threads() == []


def test_failstop_cancels_delayed_retry_beyond_boundary():
    # Row 2 is waiting on a one-hour retry when row 1 fails terminally. The
    # out-of-boundary retry must be cancelled at once and never call
    # rowmapper again.
    row2_first = threading.Event()
    attempts = []
    lock = threading.Lock()
    err = OtherError('row 1 terminal')

    def rowmapper(row):
        n = row[0]
        with lock:
            attempts.append(n)
            count2 = attempts.count(2)
        if n == 0:
            return ['zero']
        if n == 1:
            assert row2_first.wait(10)
            raise err
        if n == 2:
            if count2 == 1:
                row2_first.set()
                raise RetryableError('row 2 retry')
            return ['two']
        return [str(n)]

    table = (('foo',),) + tuple((i,) for i in range(6))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=3, maxpending=3, ordered=True,
                               retry_on=(RetryableError,),
                               retry_delays=(3600,)))
    assert next(it) == ('bar',)
    assert next(it) == ('zero',)       # row 0's success is delivered first
    with pytest.raises(OtherError) as raised:
        next(it)  # then the row-1 failure
    assert raised.value is err
    snapshot = list(attempts)
    time.sleep(0.2)
    # no new call of any kind after the boundary was reported, and in
    # particular the hour-long retry for row 2 was cancelled rather than run
    assert attempts == snapshot
    assert attempts.count(2) == 1
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_earlier_boundary_wins_when_it_fails_later_ordered():
    # Row 2 fails first (completion order), then row 1 fails. The boundary
    # must retreat to the earlier input row, whose exception object wins.
    row2_failed = threading.Event()
    calls = []
    lock = threading.Lock()
    err1 = OtherError('row 1')
    err2 = OtherError('row 2')

    def rowmapper(row):
        n = row[0]
        with lock:
            calls.append(n)
        if n == 0:
            return ['zero']
        if n == 1:
            assert row2_failed.wait(10)
            raise err1
        if n == 2:
            row2_failed.set()
            raise err2
        return [str(n)]

    table = (('foo',),) + tuple((i,) for i in range(3))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=3, maxpending=3, ordered=True))
    assert next(it) == ('bar',)
    assert next(it) == ('zero',)   # row 0 delivered, then the boundary
    with pytest.raises(OtherError) as raised:
        next(it)
    assert raised.value is err1
    assert sorted(calls) == [0, 1, 2]
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_first_boundary_stays_when_later_row_fails_ordered():
    row1_failed = threading.Event()
    row2_failed = threading.Event()
    err1 = OtherError('row 1')
    err2 = OtherError('row 2')

    def rowmapper(row):
        n = row[0]
        if n == 0:
            return ['zero']
        if n == 1:
            row1_failed.set()
            raise err1
        if n == 2:
            assert row1_failed.wait(10)
            row2_failed.set()
            raise err2
        return [str(n)]

    table = (('foo',),) + tuple((i,) for i in range(3))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=3, maxpending=3, ordered=True))
    assert next(it) == ('bar',)
    assert next(it) == ('zero',)   # row 0 is delivered before the boundary
    with pytest.raises(OtherError) as raised:
        next(it)
    assert raised.value is err1
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_unordered_delivers_completed_prefix_then_raises():
    # Row 0 and row 1 complete before row 2 fails; row 3 completes after the
    # boundary. The two prefix rows are delivered in completion order once
    # each, row 3 is never delivered, then the original error is raised.
    a_delivered = threading.Event()
    b_done = threading.Event()
    bad_failed = threading.Event()
    release_late = threading.Event()
    late_done = threading.Event()
    err = OtherError('bad')

    def rowmapper(row):
        key = row[0]
        if key == 'a':
            return ['a-done']
        if key == 'b':
            # wait until 'a' was actually delivered, fixing completion order
            assert a_delivered.wait(10)
            b_done.set()
            return ['b-done']
        if key == 'bad':
            assert b_done.wait(10)
            bad_failed.set()
            raise err
        assert bad_failed.wait(10)
        assert release_late.wait(10)
        late_done.set()
        return ['late-done']

    def on_row(row):
        if row == ('a-done',):
            a_delivered.set()

    table = (('foo',), ('a',), ('b',), ('bad',), ('late',))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=4, maxpending=4, ordered=False))
    assert next(it) == ('bar',)
    consumer = Consumer(it, on_row=on_row)
    consumer.start()

    def release_after_failure():
        # once the failure boundary is recorded, let 'late' complete so
        # teardown can join it; its result must still be discarded
        bad_failed.wait(10)
        assert it._state.failed.wait(10)
        release_late.set()

    watcher = threading.Thread(target=release_after_failure)
    watcher.start()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    watcher.join(10)
    # 'a' and 'b' completed before the boundary and are delivered once each
    # in completion order; 'late' is discarded; then the original error.
    assert consumer.rows == [('a-done',), ('b-done',)]
    assert consumer.error is err
    assert _petl_threads() == []
    with pytest.raises(StopIteration):
        next(it)


def test_failstop_unordered_first_completed_failure_wins():
    # Row 1 fails first in completion order even though row 2 has the larger
    # input index; in unordered mode row 1's exception object is the one
    # raised. (In ordered mode the earlier index governs instead.)
    may_fail = threading.Event()
    row1_failed = threading.Event()
    err1 = OtherError('row 1')
    err2 = OtherError('row 2')

    def rowmapper(row):
        n = row[0]
        if n == 1:
            assert may_fail.wait(10)
            row1_failed.set()
            raise err1
        if n == 2:
            assert may_fail.wait(10)
            assert row1_failed.wait(10)   # row 1's failure completes first
            raise err2
        return [n]

    # only rows 0, 1 and 2 exist, so no ungated later success can be
    # published ahead of the failures
    table = (('foo',),) + tuple((i,) for i in range(3))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=3, maxpending=3, ordered=False))
    assert next(it) == ('bar',)
    # row 0 has completed and is delivered before either failure is allowed
    assert next(it) == (0,)
    may_fail.set()
    with pytest.raises(OtherError) as raised:
        next(it)
    # in unordered mode the boundary is the first failure to complete; both
    # rows fail only after release, and row 1 won the completion race by
    # ordering row 2 behind row 1's failure below
    assert raised.value is err1
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_output_len_raises_same_object_not_retried():
    class BadLen(object):
        def __init__(self, exc):
            self.exc = exc

        def __len__(self):
            raise self.exc

    exc = RuntimeError('len exploded')
    bad = BadLen(exc)
    attempts = []

    def rowmapper(row):
        attempts.append(row[0])
        if row[0] == 2:
            return bad
        return [row[0]]

    table = (('foo',), (1,), (2,), (3,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(RuntimeError,),
                               retry_delays=(0, 0)))
    assert next(it) == ('bar',)
    assert next(it) == (1,)
    with pytest.raises(RuntimeError) as raised:
        next(it)
    assert raised.value is exc             # the sequence's own object
    assert attempts.count(2) == 1          # output handling is never retried
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_output_iter_raises_same_object_not_retried():
    class BadIter(object):
        def __init__(self, exc):
            self.exc = exc

        def __len__(self):
            return 1

        def __iter__(self):
            raise self.exc

    exc = OtherError('iter exploded')
    bad = BadIter(exc)
    attempts = []

    def rowmapper(row):
        attempts.append(row[0])
        if row[0] == 2:
            return bad
        return [row[0]]

    table = (('foo',), (1,), (2,), (3,))
    it = iter(rowmapconcurrent(table, rowmapper, header=('bar',),
                               workers=1, retry_on=(OtherError,),
                               retry_delays=(0, 0)))
    assert next(it) == ('bar',)
    assert next(it) == (1,)
    with pytest.raises(OtherError) as raised:
        next(it)
    assert raised.value is exc
    assert attempts.count(2) == 1
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_base_exception_stops_intake():
    class Fatal(BaseException):
        pass

    fatal = Fatal('fatal')
    source = GatedSource(range(6))
    calls = []
    lock = threading.Lock()

    def rowmapper(row):
        n = row[0]
        with lock:
            calls.append(n)
        source.entered_event(n).set()
        if n == 0:
            source.stop_wait.set()
            raise fatal
        return [n]

    it = iter(rowmapconcurrent(source, rowmapper, header=('bar',),
                               workers=1, maxpending=4, ordered=True))
    assert next(it) == ('bar',)
    with pytest.raises(Fatal) as raised:
        next(it)
    assert raised.value is fatal
    time.sleep(0.1)
    assert calls == [0]
    assert source.close_calls == 1
    assert _petl_threads() == []


def test_failstop_source_read_crossing_failure_unordered():
    # A source next() for row 1 is already blocked when row 0 fails. That
    # read is allowed to finish, but the returned row must not be mapped, and
    # no further source reads may start.
    read_in_flight = threading.Event()
    allow_read_return = threading.Event()
    row0_failed = threading.Event()
    calls = []
    err = OtherError('row 0')

    class SlowSource(object):
        def __init__(self):
            self.i = 0
            self.pulled = []
            self.close_calls = 0
            self.threads = set()

        def __iter__(self):
            return self

        def __next__(self):
            self.threads.add(threading.get_ident())
            if self.i == 0:
                self.i = 1
                return ('foo',)
            n = self.i - 1
            self.i += 1
            if n == 1:
                read_in_flight.set()
                assert allow_read_return.wait(10)
            self.pulled.append(n)
            return (n,)

        def close(self):
            self.close_calls += 1

    source = SlowSource()

    def rowmapper(row):
        calls.append(row[0])
        if row[0] == 0:
            row0_failed.set()
            raise err
        return [row[0]]

    it = iter(rowmapconcurrent(source, rowmapper, header=('foo',),
                               workers=1, maxpending=4, ordered=False))
    assert next(it) == ('foo',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(row0_failed)
        wd.wait(read_in_flight)
    allow_read_return.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is err
    assert calls == [0]             # the crossed read's row was not mapped
    assert source.pulled == [0, 1]  # the in-flight read finished
    assert source.close_calls == 1
    assert _petl_threads() == []


def test_failstop_source_read_crossing_failure_ordered():
    # Ordered variant: row 1 is the boundary, and a read for row 2 is in
    # flight. Row 2 lies at the boundary, so it must not be mapped.
    row1_failed = threading.Event()
    read2_in_flight = threading.Event()
    allow_read2 = threading.Event()
    calls = []
    pulled = []
    err = OtherError('row 1')

    class SlowSource(object):
        def __init__(self):
            self.i = 0
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.i == 0:
                self.i = 1
                return ('foo',)
            n = self.i - 1
            self.i += 1
            if n == 2:
                assert row1_failed.wait(10)
                read2_in_flight.set()
                assert allow_read2.wait(10)
            if n >= 4:
                raise StopIteration()
            pulled.append(n)
            return (n,)

        def close(self):
            self.close_calls += 1

    def rowmapper(row):
        calls.append(row[0])
        if row[0] == 0:
            return [0]
        if row[0] == 1:
            row1_failed.set()
            raise err
        return [row[0]]

    source = SlowSource()
    it = iter(rowmapconcurrent(source, rowmapper, header=('foo',),
                               workers=1, maxpending=4, ordered=True))
    assert next(it) == ('foo',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(read2_in_flight)
    allow_read2.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.rows == [(0,)]
    assert consumer.error is err
    assert 2 not in calls and 3 not in calls
    assert source.close_calls == 1
    assert _petl_threads() == []


def test_failstop_cleanup_does_not_mask_original_exception():
    err = OtherError('boom')

    class RaisingCloseSource(object):
        def __init__(self):
            self.i = 0
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.i == 0:
                self.i = 1
                return ('foo',)
            if self.i == 1:
                self.i = 2
                return (1,)
            raise StopIteration()

        def close(self):
            self.close_calls += 1
            raise RuntimeError('close blew up')

    def rowmapper(row):
        raise err

    source = RaisingCloseSource()
    it = iter(rowmapconcurrent(source, rowmapper, header=('foo',),
                               workers=1))
    assert next(it) == ('foo',)
    with pytest.raises(OtherError) as raised:
        next(it)
    assert raised.value is err         # close's error did not replace it
    assert source.close_calls == 1
    with pytest.raises(StopIteration):
        next(it)
    assert _petl_threads() == []


def test_failstop_worker_internal_error_reaches_caller(monkeypatch):
    # If a worker hits an unexpected internal error outside the rowmapper and
    # output handlers, it must surface it instead of dying silently and
    # leaving the caller waiting forever.
    import petl.transform.maps as maps

    def boom(state, job):
        raise RuntimeError('internal worker error')

    monkeypatch.setattr(maps, '_process_job', boom)
    it = iter(rowmapconcurrent((('foo',), (1,), (2,)), lambda r: [r[0]],
                               header=('bar',), workers=1))
    assert next(it) == ('bar',)
    consumer = Consumer(it)
    consumer.daemon = True
    consumer.start()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert isinstance(consumer.error, RuntimeError)
    assert 'internal worker error' in str(consumer.error)
    assert _petl_threads() == []


# --------------------------------------------------------------------------
# normal-path regression alongside the new failure machinery
# --------------------------------------------------------------------------

def test_failstop_normal_path_still_ordered_and_bounded():
    release = threading.Event()
    two_running = threading.Event()
    inside = []
    lock = threading.Lock()
    table, state = _counting_source(
        (('foo',),) + tuple((i,) for i in range(10)))

    def rowmapper(row):
        with lock:
            inside.append(row[0])
            if len(inside) == 2:
                two_running.set()
        assert release.wait(10)
        return [row[0]]

    it = iter(rowmapconcurrent(table, rowmapper, header=('foo',),
                               workers=2, maxpending=4, ordered=True))
    assert next(it) == ('foo',)
    consumer = Consumer(it)
    consumer.start()
    with Watchdog() as wd:
        wd.wait(two_running)
    assert state['next_calls'] <= 1 + 4
    release.set()
    with Watchdog() as wd:
        wd.wait(consumer.done)
    assert consumer.error is None
    assert [r[0] for r in consumer.rows] == list(range(10))
    assert state['closed'] is True
    assert _petl_threads() == []
