from __future__ import absolute_import, print_function, division

import gc
import threading
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

    def __init__(self, it):
        super(Consumer, self).__init__()
        self.it = it
        self.rows = []
        self.error = None
        self.done = threading.Event()
        self.first_row = threading.Event()

    def run(self):
        first = True
        try:
            for row in self.it:
                self.rows.append(row)
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
    state = {'next_calls': 0, 'threads': set(), 'closed': False}

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

    def rowmapper(row):
        if row[0] == 'bad':
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
    # release the still-running call as soon as the good row has been
    # delivered, so teardown can join it without waiting out its blocking
    # timeout; the slow row still finishes after the failure is observed

    def release_on_first_row():
        consumer.first_row.wait(10)
        release_slow.set()

    watcher = threading.Thread(target=release_on_first_row)
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
