from __future__ import absolute_import, print_function, division


import heapq
import math
import numbers
import operator
import queue
import threading
import time
import weakref
from collections import OrderedDict, namedtuple
from petl.compat import next, string_types, text_type


import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort


def fieldmap(table, mappings=None, failonerror=None, errorvalue=None, trusted=True):
    """
    Transform a table, mapping fields arbitrarily between input and output.
    E.g.::

        >>> import petl as etl
        >>> from collections import OrderedDict
        >>> table1 = [['id', 'sex', 'age', 'height', 'weight'],
        ...           [1, 'male', 16, 1.45, 62.0],
        ...           [2, 'female', 19, 1.34, 55.4],
        ...           [3, 'female', 17, 1.78, 74.4],
        ...           [4, 'male', 21, 1.33, 45.2],
        ...           [5, '-', 25, 1.65, 51.9]]
        >>> mappings = OrderedDict()
        >>> # rename a field
        ... mappings['subject_id'] = 'id'
        >>> # translate a field
        ... mappings['gender'] = 'sex', {'male': 'M', 'female': 'F'}
        >>> # apply a calculation to a field
        ... mappings['age_months'] = 'age', lambda v: v * 12
        >>> # apply a calculation to a combination of fields
        ... mappings['bmi'] = lambda rec: rec['weight'] / rec['height']**2
        >>> # transform and inspect the output
        ... table2 = etl.fieldmap(table1, mappings)
        >>> table2
        +------------+--------+------------+--------------------+
        | subject_id | gender | age_months | bmi                |
        +============+========+============+====================+
        |          1 | 'M'    |        192 |  29.48870392390012 |
        +------------+--------+------------+--------------------+
        |          2 | 'F'    |        228 |   30.8531967030519 |
        +------------+--------+------------+--------------------+
        |          3 | 'F'    |        204 | 23.481883600555488 |
        +------------+--------+------------+--------------------+
        |          4 | 'M'    |        252 |  25.55260331279326 |
        +------------+--------+------------+--------------------+
        |          5 | '-'    |        300 |   19.0633608815427 |
        +------------+--------+------------+--------------------+

    Note also that the mapping value can be an expression string, which will be
    converted to a lambda function via :func:`petl.util.base.expr`.

    The ``trusted`` keyword argument can be used to specify whether the
    expression is trusted. See `:func:`petl.util.base.expr` for details.

    The `failonerror` and `errorvalue` keyword arguments are documented
    under :func:`petl.config.failonerror`
    """

    return FieldMapView(table, mappings=mappings, failonerror=failonerror,
                        errorvalue=errorvalue, trusted=trusted)


Table.fieldmap = fieldmap


class FieldMapView(Table):

    def __init__(self, source, mappings=None, failonerror=None,
                 errorvalue=None, trusted=True):
        self.source = source
        if mappings is None:
            self.mappings = OrderedDict()
        else:
            self.mappings = mappings
        self.failonerror = (config.failonerror if failonerror is None
                                else failonerror)
        self.errorvalue = errorvalue
        self.trusted = trusted

    def __setitem__(self, key, value):
        self.mappings[key] = value

    def __iter__(self):
        return iterfieldmap(self.source, self.mappings, self.failonerror,
                            self.errorvalue, self.trusted)


def iterfieldmap(source, mappings, failonerror, errorvalue, trusted):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    outhdr = mappings.keys()
    yield tuple(outhdr)

    mapfuns = dict()
    for outfld, m in mappings.items():
        if m in hdr:
            mapfuns[outfld] = operator.itemgetter(m)
        elif isinstance(m, int) and m < len(hdr):
            mapfuns[outfld] = operator.itemgetter(m)
        elif isinstance(m, string_types):
            mapfuns[outfld] = expr(m, trusted=trusted)
        elif callable(m):
            mapfuns[outfld] = m
        elif isinstance(m, (tuple, list)) and len(m) == 2:
            srcfld = m[0]
            fm = m[1]
            if callable(fm):
                mapfuns[outfld] = composefun(fm, srcfld)
            elif isinstance(fm, dict):
                mapfuns[outfld] = composedict(fm, srcfld)
            else:
                raise ArgumentError('expected callable or dict')
        else:
            raise ArgumentError('invalid mapping %r: %r' % (outfld, m))

    # wrap rows as records
    it = (Record(row, flds) for row in it)
    for row in it:
        outrow = list()
        for outfld in outhdr:
            try:
                val = mapfuns[outfld](row)
            except Exception as e:
                if failonerror == 'inline':
                    val = e
                elif failonerror:
                    raise e
                else:
                    val = errorvalue
            outrow.append(val)
        yield tuple(outrow)


def composefun(f, srcfld):
    def g(rec):
        return f(rec[srcfld])
    return g


def composedict(d, srcfld):
    def g(rec):
        k = rec[srcfld]
        if k in d:
            return d[k]
        else:
            return k
    return g


def rowmap(table, rowmapper, header, failonerror=None):
    """
    Transform rows via an arbitrary function. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'sex', 'age', 'height', 'weight'],
        ...           [1, 'male', 16, 1.45, 62.0],
        ...           [2, 'female', 19, 1.34, 55.4],
        ...           [3, 'female', 17, 1.78, 74.4],
        ...           [4, 'male', 21, 1.33, 45.2],
        ...           [5, '-', 25, 1.65, 51.9]]
        >>> def rowmapper(row):
        ...     transmf = {'male': 'M', 'female': 'F'}
        ...     return [row[0],
        ...             transmf[row['sex']] if row['sex'] in transmf else None,
        ...             row.age * 12,
        ...             row.height / row.weight ** 2]
        ...
        >>> table2 = etl.rowmap(table1, rowmapper,
        ...                     header=['subject_id', 'gender', 'age_months',
        ...                             'bmi'])
        >>> table2
        +------------+--------+------------+-----------------------+
        | subject_id | gender | age_months | bmi                   |
        +============+========+============+=======================+
        |          1 | 'M'    |        192 | 0.0003772112382934443 |
        +------------+--------+------------+-----------------------+
        |          2 | 'F'    |        228 | 0.0004366015456998006 |
        +------------+--------+------------+-----------------------+
        |          3 | 'F'    |        204 | 0.0003215689675106949 |
        +------------+--------+------------+-----------------------+
        |          4 | 'M'    |        252 | 0.0006509906805544679 |
        +------------+--------+------------+-----------------------+
        |          5 | None   |        300 | 0.0006125608384287258 |
        +------------+--------+------------+-----------------------+

    The `rowmapper` function should accept a single row and return a single
    row (list or tuple).

    The `failonerror` keyword argument is documented under
    :func:`petl.config.failonerror`
    """

    return RowMapView(table, rowmapper, header, failonerror=failonerror)


Table.rowmap = rowmap


class RowMapView(Table):

    def __init__(self, source, rowmapper, header, failonerror=None):
        self.source = source
        self.rowmapper = rowmapper
        self._header = header
        self.failonerror = (config.failonerror if failonerror is None
                                else failonerror)

    def __iter__(self):
        return iterrowmap(self.source, self.rowmapper, self._header,
                          self.failonerror)


def iterrowmap(source, rowmapper, header, failonerror):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    yield tuple(header)
    it = (Record(row, flds) for row in it)
    for row in it:
        try:
            outrow = rowmapper(row)
            yield tuple(outrow)
        except Exception as e:
            if failonerror == 'inline':
                yield tuple([e])
            elif failonerror:
                raise e


def rowmapmany(table, rowgenerator, header, failonerror=None):
    """
    Map each input row to any number of output rows via an arbitrary
    function. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'sex', 'age', 'height', 'weight'],
        ...           [1, 'male', 16, 1.45, 62.0],
        ...           [2, 'female', 19, 1.34, 55.4],
        ...           [3, '-', 17, 1.78, 74.4],
        ...           [4, 'male', 21, 1.33]]
        >>> def rowgenerator(row):
        ...     transmf = {'male': 'M', 'female': 'F'}
        ...     yield [row[0], 'gender',
        ...            transmf[row['sex']] if row['sex'] in transmf else None]
        ...     yield [row[0], 'age_months', row.age * 12]
        ...     yield [row[0], 'bmi', row.height / row.weight ** 2]
        ...
        >>> table2 = etl.rowmapmany(table1, rowgenerator,
        ...                         header=['subject_id', 'variable', 'value'])
        >>> table2.lookall()
        +------------+--------------+-----------------------+
        | subject_id | variable     | value                 |
        +============+==============+=======================+
        |          1 | 'gender'     | 'M'                   |
        +------------+--------------+-----------------------+
        |          1 | 'age_months' |                   192 |
        +------------+--------------+-----------------------+
        |          1 | 'bmi'        | 0.0003772112382934443 |
        +------------+--------------+-----------------------+
        |          2 | 'gender'     | 'F'                   |
        +------------+--------------+-----------------------+
        |          2 | 'age_months' |                   228 |
        +------------+--------------+-----------------------+
        |          2 | 'bmi'        | 0.0004366015456998006 |
        +------------+--------------+-----------------------+
        |          3 | 'gender'     | None                  |
        +------------+--------------+-----------------------+
        |          3 | 'age_months' |                   204 |
        +------------+--------------+-----------------------+
        |          3 | 'bmi'        | 0.0003215689675106949 |
        +------------+--------------+-----------------------+
        |          4 | 'gender'     | 'M'                   |
        +------------+--------------+-----------------------+
        |          4 | 'age_months' |                   252 |
        +------------+--------------+-----------------------+

    The `rowgenerator` function should accept a single row and yield zero or
    more rows (lists or tuples).

    The `failonerror` keyword argument is documented under
    :func:`petl.config.failonerror`

    See also the :func:`petl.transform.reshape.melt` function.

    """

    return RowMapManyView(table, rowgenerator, header, failonerror=failonerror)


Table.rowmapmany = rowmapmany


class RowMapManyView(Table):

    def __init__(self, source, rowgenerator, header, failonerror=None):
        self.source = source
        self.rowgenerator = rowgenerator
        self._header = header
        self.failonerror = (config.failonerror if failonerror is None
                                else failonerror)

    def __iter__(self):
        return iterrowmapmany(self.source, self.rowgenerator, self._header,
                              self.failonerror)


def iterrowmapmany(source, rowgenerator, header, failonerror):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    yield tuple(header)
    it = (Record(row, flds) for row in it)
    for row in it:
        try:
            for outrow in rowgenerator(row):
                yield tuple(outrow)
        except Exception as e:
            if failonerror == 'inline':
                yield tuple([e])
            elif failonerror:
                raise e
            else:
                pass


def rowgroupmap(table, key, mapper, header=None, presorted=False,
                buffersize=None, tempdir=None, cache=True):
    """
    Group rows under the given key then apply `mapper` to yield zero or more
    output rows for each input group of rows.

    """

    return RowGroupMapView(table, key, mapper, header=header,
                           presorted=presorted,
                           buffersize=buffersize, tempdir=tempdir, cache=cache)


Table.rowgroupmap = rowgroupmap


class RowGroupMapView(Table):

    def __init__(self, source, key, mapper, header=None,
                 presorted=False, buffersize=None, tempdir=None, cache=True):
        if presorted:
            self.source = source
        else:
            self.source = sort(source, key, buffersize=buffersize,
                               tempdir=tempdir, cache=cache)
        self.key = key
        self._header = header
        self.mapper = mapper

    def __iter__(self):
        return iterrowgroupmap(self.source, self.key, self.mapper, self._header)


def iterrowgroupmap(source, key, mapper, header):
    yield tuple(header)
    for key, rows in rowgroupby(source, key):
        for row in mapper(key, rows):
            yield row


def rowmapconcurrent(table, rowmapper, header, workers=4, maxpending=None,
                     ordered=True, retry_on=(), retry_delays=()):
    """
    Transform rows via an arbitrary function, running the function
    concurrently in worker threads. Useful when `rowmapper` spends most of
    its time waiting on external services (e.g. HTTP APIs or database
    lookups) and would otherwise process rows one at a time. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'sex', 'age'],
        ...           [1, 'male', 16],
        ...           [2, 'female', 19],
        ...           [3, 'female', 17],
        ...           [4, 'male', 21],
        ...           [5, '-', 25]]
        >>> def rowmapper(row):
        ...     transmf = {'male': 'M', 'female': 'F'}
        ...     return [row[0],
        ...             transmf[row['sex']] if row['sex'] in transmf else None,
        ...             row.age * 12]
        ...
        >>> table2 = etl.rowmapconcurrent(table1, rowmapper,
        ...                               header=['subject_id', 'gender',
        ...                                       'age_months'],
        ...                               workers=4)
        >>> table2
        +------------+--------+------------+
        | subject_id | gender | age_months |
        +============+========+============+
        |          1 | 'M'    |        192 |
        +------------+--------+------------+
        |          2 | 'F'    |        228 |
        +------------+--------+------------+
        |          3 | 'F'    |        204 |
        +------------+--------+------------+
        |          4 | 'M'    |        252 |
        +------------+--------+------------+
        |          5 | None   |        300 |
        +------------+--------+------------+

    The `rowmapper` function is used in the same way as with
    :func:`petl.transform.maps.rowmap`: it receives a single row as a
    :class:`petl.util.base.Record` and must return a sequence (list or
    tuple) whose length equals the length of `header`. If the returned
    value is not a sequence or its length does not match, the row fails
    with an :class:`petl.errors.ArgumentError` carrying the one-based data
    row number (the first row after the table header is row 1); this
    failure is never retried.

    The output header is ``tuple(header)``.

    `workers` is the maximum number of `rowmapper` calls executing at the
    same time. It must be a positive integer (booleans are rejected) and
    the work is carried out by threads, never by separate processes.

    `maxpending` bounds the number of rows that have been pulled from the
    source table but not yet handed to the caller, including rows currently
    being mapped, rows waiting for a retry, and rows whose result is ready
    but waiting for earlier rows (in ordered mode). If ``None``, it
    defaults to ``2 * workers``. It must be a positive integer (booleans
    are rejected). Because the window is bounded, memory does not grow with
    the total number of input rows.

    If `ordered` is true (the default), output rows are delivered in the
    same order as the input rows: a row is delivered as soon as it and all
    rows before it have finished, without waiting for the whole window to
    fill. If `ordered` is false, rows are delivered in completion order and
    every successful row appears exactly once.

    `retry_on` is a tuple of exception classes (subclasses of
    :class:`Exception`); when `rowmapper` raises an instance of one of
    them, the row is retried. `retry_delays` is a sequence of non-negative
    finite numbers; its length is the maximum number of retries and the
    k-th retry (one-based) waits ``retry_delays[k - 1]`` seconds before
    running again. Retry waits do not occupy a worker thread, so with
    ``workers=1`` other rows can still be processed while a row is waiting
    to be retried. In ordered mode, a retried row keeps its original
    position. Any other exception, or exhausting the retries, makes the
    row fail.

    Failure semantics differ by mode. A *terminal* failure is a non-retryable
    exception, exhausted retries, an invalid `rowmapper` result, or any
    :class:`BaseException`; a retryable exception with attempts remaining
    is not terminal.

    * with `ordered` true, the failure position is the input row index. Once
      row *j* has failed terminally, no further rows are pulled from the
      source and no new calls (or retries) are started for rows at or after
      *j*; calls already running may finish, but their results are not
      delivered. Rows before *j* still follow the normal retry rules, their
      successful prefix is delivered in order, and then the original
      exception object is raised at *j*'s position. If an earlier row later
      fails terminally too, the boundary moves back to that earlier row.
      A retry that is waiting when the boundary is set is not abandoned for
      rows before the boundary (the prefix is still owed), while retries at
      or after the boundary are cancelled immediately.
    * with `ordered` false, results are delivered in completion order. The
      first terminal failure to complete fixes the boundary: successful
      results that completed before it are each delivered once in completion
      order, results that complete afterwards are discarded, and then that
      boundary failure's original exception object is raised. No source
      reads and no new calls or retries start once the boundary is fixed.
      Simultaneous completions may be ordered either way, but delivery and
      the stop-intake decision both follow that single chosen order.

    A source read already in flight when a failure is determined is allowed
    to finish, but the row it returns only enters `rowmapper` if it still
    falls within the boundary described above. Calls already running are
    not forcibly killed; before the exception (or a ``close()``) returns to
    the caller, those calls are waited for, all threads of this iteration
    have exited, and the source iterator is closed if it supports ``close()``.
    Cleanup never replaces the caller's original exception, and excluded
    delayed retries are cancelled without waiting out their delays. After
    an error, further ``next()`` calls end the iteration.

    If the object returned by `rowmapper` raises from its own ``__len__`` or
    when it is iterated to build the final row, that same exception object
    is handed to the caller; such output-handling errors are never retried.

    :class:`BaseException` raised by `rowmapper` (e.g.
    :class:`KeyboardInterrupt`, :class:`SystemExit`) is never retried and
    is re-raised in the caller thread.

    The source table is only advanced in the thread that calls ``next()``
    on the result iterator, so thread-affine sources such as
    :mod:`sqlite3` cursors are safe; worker threads only execute
    `rowmapper`.

    Iteration cleans up (worker threads exit, pending retries are
    cancelled and the source iterator is closed if it supports ``close()``)
    on normal exhaustion, on failure, when the result iterator's
    ``close()`` is called, when the caller thread receives a
    :class:`KeyboardInterrupt` inside ``next()``, and when an unread
    iterator is garbage collected. Iterators created from the same view
    are independent and each creates its own threads.

    Invalid parameters raise :class:`petl.errors.ArgumentError` when the
    view is created, before the source table is touched.

    """

    return RowMapConcurrentView(table, rowmapper, header, workers=workers,
                                maxpending=maxpending, ordered=ordered,
                                retry_on=retry_on, retry_delays=retry_delays)


Table.rowmapconcurrent = rowmapconcurrent


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArgumentError(
            '%s must be a positive integer, got %r' % (name, value))
    return value


class RowMapConcurrentView(Table):

    def __init__(self, source, rowmapper, header, workers=4, maxpending=None,
                 ordered=True, retry_on=(), retry_delays=()):
        # all parameter errors are raised here, without touching the source
        workers = _positive_int(workers, 'workers')
        if maxpending is None:
            maxpending = 2 * workers
        else:
            maxpending = _positive_int(maxpending, 'maxpending')
        if not isinstance(retry_on, tuple):
            raise ArgumentError(
                'retry_on must be a tuple of Exception subclasses, got %r'
                % (retry_on,))
        for exc_type in retry_on:
            if not isinstance(exc_type, type) or not issubclass(exc_type,
                                                                Exception):
                raise ArgumentError(
                    'retry_on must contain only Exception subclasses, got %r'
                    % (exc_type,))
        delays = tuple(retry_delays)
        for delay in delays:
            if isinstance(delay, bool) or not isinstance(delay, numbers.Real) \
                    or delay < 0 or not math.isfinite(delay):
                raise ArgumentError(
                    'retry_delays must be a sequence of non-negative finite '
                    'numbers, got %r' % (delay,))

        self.source = source
        self.rowmapper = rowmapper
        self._header = tuple(header)
        self.workers = workers
        self.maxpending = maxpending
        self.ordered = ordered
        self.retry_on = retry_on
        self.retry_delays = delays

    def __iter__(self):
        return _RowMapConcurrentIterator(self)


_Success = namedtuple('_Success', ('index', 'row'))
_Failure = namedtuple('_Failure', ('index', 'error'))
# job queue sentinel: a worker that pulls this exits
_SENTINEL = object()
# result queue sentinel: the iterator has been closed
_CLOSED = object()


class _ConcurrentState(object):
    """All mutable runtime state of a concurrent iteration. Kept separate from
    the iterator object so the garbage-collection finalizer only needs to
    reference this state, never the iterator itself: an abandoned iterator
    with running worker threads can then still be collected."""

    def __init__(self, view):
        self.view = view
        self.rowmapper = view.rowmapper
        self.header = view._header
        self.workers = view.workers
        self.maxpending = view.maxpending
        self.ordered = view.ordered
        self.retry_on = view.retry_on
        self.retry_delays = view.retry_delays
        self.nfields = len(view._header)

        self.source = None
        self.flds = None
        self.header_emitted = False
        self.started = False
        self.closed = False
        self.source_exhausted = False
        self.row_number = 0        # one-based data row number
        self.next_index = 0        # index of the next row to pull
        self.inflight = 0          # rows pulled but not yet delivered

        self.job_queue = queue.Queue()
        self.results = queue.Queue()

        # ordered mode: results waiting for earlier rows, keyed by index
        self.buffer = {}
        self.deliver_index = 0

        # retry scheduling
        self.timer_cond = threading.Condition(threading.Lock())
        self.retry_heap = []       # (due_time, sequence, job)
        self.retry_seq = 0

        self.closing = threading.Event()
        self.threads = []

        # Terminal-failure boundary. ``failed`` is set as soon as a terminal
        # failure is determined; workers consult it before starting any new
        # call, and the caller stops pulling from the source.
        # ``boundary_index`` is the input row index of the boundary failure in
        # ordered mode and the input row index of the first failure to complete
        # in unordered mode. The original exception object to re-raise travels
        # in the published ``_Failure`` entry. ``completion_lock`` serialises
        # boundary claims and result publication, so completion order and the
        # delivery decision agree on one ordering.
        self.failed = threading.Event()
        self.boundary_index = None
        self.completion_lock = threading.Lock()


def _snapshot_boundary(state):
    """Return ``(failed, boundary_index)`` under the completion lock so the
    pair is always observed consistently."""
    with state.completion_lock:
        return state.failed.is_set(), state.boundary_index


def _job_admitted(state, index):
    """Decide whether a not-yet-started job may call `rowmapper` now.

    In ordered mode rows before the failure boundary are still part of the
    deliverable prefix and keep running (including their retries); rows at or
    after the boundary must not start. In unordered mode the first completed
    failure ends the intake, so no new call may start at all.
    """
    if state.closing.is_set():
        return False
    with state.completion_lock:
        if not state.failed.is_set():
            return True
        if state.ordered:
            boundary = state.boundary_index
            return boundary is not None and index < boundary
        return False


def _prune_retries(state, keep_before):
    """Drop scheduled retries for rows at or after the failure boundary.
    ``keep_before`` is an exclusive row-index bound. Excluded delayed retries
    are removed from the heap right away instead of waiting for their due
    time; the timer is woken so it re-evaluates its next deadline."""
    with state.timer_cond:
        kept = [item for item in state.retry_heap
                if item[2][0] < keep_before]
        if len(kept) != len(state.retry_heap):
            state.retry_heap = kept
            heapq.heapify(state.retry_heap)
        state.timer_cond.notify_all()


def _publish_success(state, index, row):
    """Publish a successful result unless it lies beyond the failure
    boundary. Publication and boundary claims are serialised by the same
    lock, which defines one consistent completion order."""
    with state.completion_lock:
        if state.failed.is_set():
            if not state.ordered:
                return
            boundary = state.boundary_index
            if boundary is None or index >= boundary:
                return
        state.results.put(_Success(index, row))


def _publish_failure(state, index, error):
    """Record a terminal failure and (the first time, or when an earlier row
    fails) the boundary it establishes. The original exception object is
    carried through unchanged."""
    if state.ordered:
        with state.completion_lock:
            failed = state.failed.is_set()
            if not failed:
                state.boundary_index = index
                state.failed.set()
                promote = True
            elif index < state.boundary_index:
                # an earlier row failed afterwards: move the boundary back
                state.boundary_index = index
                promote = True
            else:
                # failure at or after the boundary: nothing to deliver or say
                promote = False
            if promote:
                state.results.put(_Failure(index, error))
            boundary = state.boundary_index
        if promote:
            _prune_retries(state, boundary)
    else:
        with state.completion_lock:
            if state.failed.is_set():
                return
            state.boundary_index = index
            state.failed.set()
            state.results.put(_Failure(index, error))
        _prune_retries(state, -1)


def _teardown_state(state):
    if state.closed:
        return
    state.closed = True
    state.closing.set()
    try:
        _teardown_impl(state)
    except BaseException:
        # Cleanup must never mask the exception the caller is about to see
        # (most importantly BaseException such as KeyboardInterrupt). The
        # joining of workers and closing of the source are each defensive
        # below; this is just the last line of defence.
        pass


def _teardown_impl(state):
    try:
        if state.started:
            # wake the retry timer so any waiting retry is cancelled at once;
            # workers discard queued jobs without calling rowmapper once
            # closing is set, then exit on their sentinel
            try:
                with state.timer_cond:
                    state.timer_cond.notify_all()
            except Exception:
                pass
            for _ in range(state.workers):
                state.job_queue.put(_SENTINEL)
            # join without a timeout: before this returns every in-flight
            # rowmapper call must have finished and every thread must exit
            current = threading.current_thread()
            for t in state.threads:
                if t is current:
                    continue
                try:
                    t.join()
                except Exception:
                    pass
    finally:
        source = state.source
        if source is not None and hasattr(source, 'close'):
            try:
                source.close()
            except Exception:
                pass
        # unblock any consumer still waiting on results (defensive)
        state.results.put(_CLOSED)


class _RowMapConcurrentIterator(object):
    """Iterator driving the worker pool. The source table is only ever
    advanced in the thread that owns this object."""

    def __init__(self, view):
        self._state = _ConcurrentState(view)
        # the finalizer captures only the state, never `self`, so garbage
        # collection of an abandoned iterator still triggers cleanup
        self._finalizer = weakref.finalize(
            self, _teardown_state, self._state)

    # ------------------------------------------------------------------
    # iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        state = self._state
        if state.closed:
            raise StopIteration
        if not state.header_emitted:
            # first next(): read the header in the caller thread; no threads
            # are started and the source is not touched before this point
            try:
                state.source = iter(state.view.source)
                hdr = next(state.source)
            except StopIteration:
                self._finalizer()
                raise
            except BaseException:
                self._finalizer()
                raise
            state.flds = list(map(text_type, hdr))
            state.header_emitted = True
            return state.header
        if not state.started:
            self._start()
        try:
            return self._next_row()
        except BaseException:
            # includes failure exceptions and KeyboardInterrupt in the caller
            self._finalizer()
            raise

    # python 2 compatibility shim (harmless on python 3)
    next = __next__

    def close(self):
        self._finalizer()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _start(self):
        state = self._state
        state.started = True
        # module-level targets hold only a weak reference to the state, so a
        # running thread never keeps the iterator (or anything reachable from
        # it) alive on its own
        state_ref = weakref.ref(state)
        for i in range(state.workers):
            t = threading.Thread(target=_rowmap_worker, args=(state_ref,),
                                 name='petl-rowmapconcurrent-%d' % i)
            t.daemon = True
            state.threads.append(t)
            t.start()
        timer = threading.Thread(target=_retry_timer, args=(state_ref,),
                                 name='petl-rowmapconcurrent-timer')
        timer.daemon = True
        state.threads.append(timer)
        timer.start()

    # ------------------------------------------------------------------
    # caller-side scheduling: source is only advanced here
    # ------------------------------------------------------------------

    def _pump(self):
        """Pull rows from the source and submit them until the pending
        window is full, the source is exhausted, a terminal failure has
        established a boundary, or the iterator is closing.
        """
        state = self._state
        while state.inflight < state.maxpending \
                and not state.source_exhausted \
                and not state.closing.is_set() \
                and not state.failed.is_set():
            try:
                raw_row = next(state.source)
            except StopIteration:
                state.source_exhausted = True
                return
            # a failure may have been determined while the source read was in
            # flight; the row just read must not be admitted past the boundary
            failed, boundary = _snapshot_boundary(state)
            if failed and (not state.ordered or boundary is None
                           or state.next_index >= boundary):
                return
            state.row_number += 1
            record = Record(raw_row, state.flds)
            index = state.next_index
            state.next_index += 1
            state.inflight += 1
            state.job_queue.put((index, state.row_number, record, 0))

    def _next_row(self):
        if self._state.ordered:
            return self._next_ordered()
        return self._next_unordered()

    def _next_unordered(self):
        state = self._state
        self._pump()
        if state.source_exhausted and state.inflight == 0:
            raise StopIteration
        outcome = state.results.get()
        if outcome is _CLOSED:
            raise StopIteration
        state.inflight -= 1
        if isinstance(outcome, _Failure):
            # everything completed before the boundary failure was already
            # published (and consumed) ahead of it in the queue; drop the
            # rest, tear down, then raise the original exception object
            self._finalizer()
            raise outcome.error
        self._pump()
        return outcome.row

    def _next_ordered(self):
        state = self._state
        while True:
            self._pump()
            ready = state.buffer.pop(state.deliver_index, None)
            if ready is not None:
                state.deliver_index += 1
                state.inflight -= 1
                if isinstance(ready, _Failure):
                    self._finalizer()
                    raise ready.error
                return ready.row
            if state.source_exhausted and state.inflight == 0:
                raise StopIteration
            outcome = state.results.get()
            if outcome is _CLOSED:
                raise StopIteration
            state.buffer[outcome.index] = outcome


def _rowmap_worker(state_ref):
    """Worker thread entry point. Holds only a weak reference to the runtime
    state, so abandoning the iterator lets garbage collection run its
    finalizer even while a call is still in flight."""
    while True:
        state = state_ref()
        if state is None:
            # the state was garbage collected; nothing else can enqueue work
            return
        job = state.job_queue.get()
        if job is _SENTINEL:
            return
        try:
            _process_job(state, job)
        except BaseException as error:
            # The per-job handlers cover rowmapper and all output handling;
            # this is a last-resort guard against an unexpected internal
            # error. Surface it to the caller instead of letting this thread
            # die silently while the caller waits forever, then keep serving
            # the queue (the next terminal state will wake the caller).
            try:
                index = job[0]
            except Exception:
                index = -1
            _publish_failure(state, index, error)


def _process_job(state, job):
    index, row_number, record, attempt = job
    if not _job_admitted(state, index):
        # teardown in progress, or a terminal failure was determined
        # before this call could start: never enter rowmapper
        return
    try:
        outrow = state.rowmapper(record)
    except BaseException as error:
        _handle_error(state, index, row_number, record, attempt, error)
        return
    # Output handling is never retried. If the returned sequence's own
    # ``__len__`` or iteration raises, the very same exception object is
    # handed to the caller; only a genuinely non-sequence return value is
    # reported as an ArgumentError carrying the row number.
    try:
        length = len(outrow)
    except TypeError as error:
        if hasattr(type(outrow), '__len__'):
            # the sequence itself raised TypeError from ``__len__``:
            # propagate that same exception object
            _publish_failure(state, index, error)
        else:
            error = ArgumentError(
                'rowmapper returned a non-sequence value for row %d: '
                '%r' % (row_number, outrow))
            error._petl_no_retry = True
            _publish_failure(state, index, error)
        return
    except BaseException as error:
        _publish_failure(state, index, error)
        return
    if length != state.nfields:
        error = ArgumentError(
            'rowmapper returned %d values for row %d but '
            'header has %d fields'
            % (length, row_number, state.nfields))
        error._petl_no_retry = True
        _publish_failure(state, index, error)
        return
    try:
        final_row = tuple(outrow)
    except BaseException as error:
        _publish_failure(state, index, error)
        return
    _publish_success(state, index, final_row)


def _handle_error(state, index, row_number, record, attempt, error):
    # BaseException (KeyboardInterrupt, SystemExit, ...) never retries,
    # and wrong-length/non-sequence return values fail the row outright
    if isinstance(error, Exception) \
            and not getattr(error, '_petl_no_retry', False) \
            and isinstance(error, state.retry_on) \
            and attempt < len(state.retry_delays) \
            and not state.closing.is_set() \
            and not _is_retry_excluded(state, index):
        delay = state.retry_delays[attempt]
        _schedule_retry(state, delay,
                        (index, row_number, record, attempt + 1))
    else:
        _publish_failure(state, index, error)


def _is_retry_excluded(state, index):
    """A row at or after a determined failure boundary is never retried:
    its retry is cancelled on the spot rather than waiting for its delay."""
    with state.completion_lock:
        if not state.failed.is_set():
            return False
        if state.ordered:
            return state.boundary_index is not None \
                and index >= state.boundary_index
        return True


def _schedule_retry(state, delay, job):
    due = time.monotonic() + delay
    with state.timer_cond:
        state.retry_seq += 1
        heapq.heappush(state.retry_heap, (due, state.retry_seq, job))
        state.timer_cond.notify_all()
    # A boundary may have been determined while this retry was being
    # scheduled; if so it is excluded and is dropped immediately rather than
    # waiting out its delay.
    with state.completion_lock:
        excluded = state.failed.is_set() and (
            not state.ordered
            or (state.boundary_index is not None
                and job[0] >= state.boundary_index))
        boundary = state.boundary_index
    if excluded:
        _prune_retries(state, -1 if not state.ordered else boundary)


def _retry_timer(state_ref):
    """Timer thread entry point, holding only a weak reference. Retry delays
    are waited here rather than inside a worker thread."""
    while True:
        state = state_ref()
        if state is None:
            return
        if state.closing.is_set():
            with state.timer_cond:
                state.retry_heap.clear()
            return
        with state.timer_cond:
            while not state.retry_heap and not state.closing.is_set():
                state.timer_cond.wait()
            if state.closing.is_set():
                state.retry_heap.clear()
                return
            due, seq, job = state.retry_heap[0]
            now = time.monotonic()
            if now < due:
                state.timer_cond.wait(due - now)
                continue
            heapq.heappop(state.retry_heap)
        if state.closing.is_set():
            # cancelled while waiting: do not start the retry
            return
        if not _job_admitted(state, job[0]):
            # a boundary excluded this retry while it was waiting
            continue
        state.job_queue.put(job)
