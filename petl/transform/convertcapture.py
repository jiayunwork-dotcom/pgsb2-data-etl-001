"""Single-pass conversion with good/bad row capture.

This module provides :func:`convertcapture`, which applies a set of
converters to an input table during a single read of the source and routes
each data row to one of two output tables:

* ``good`` holds the rows for which every converter succeeded;
* ``bad`` holds one record per row that could not be converted.

Both output tables share the same single iterator over the input. Rows that
the slower consumer has not yet read are buffered, with the in-memory part of
each buffer bounded by ``buffersize``; larger backlogs spill to temporary
pickle files under ``tempdir``. Each branch table may be iterated only once.

"""

from __future__ import absolute_import, print_function, division

import os
from collections import deque
from tempfile import NamedTemporaryFile

from petl.compat import next, pickle, string_types, text_type
from petl.errors import ArgumentError, FieldSelectionError, RowLengthError
from petl.util.base import Table


BAD_HEADER = ('row', 'record', 'field', 'value', 'error_type', 'error')
_ROW_LENGTH_ERROR_TYPE = 'petl.RowLengthError'


def convertcapture(table, converters, buffersize=1000, tempdir=None):
    """Apply converters to a table during a single read, capturing failures.

    The input table is read at most once (only one iterator is ever taken on
    it). Every data row is converted exactly once and then routed to one of
    two branch tables exposed on the returned object::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['1', '2'],
        ...           ['3', 'x'],
        ...           ['5', '6']]
        >>> with etl.convertcapture(table1, [('foo', int), ('bar', int)]) as cap:
        ...     good = list(cap.good)
        ...     bad = list(cap.bad)
        >>> good
        [('foo', 'bar'), (1, 2), (5, 6)]
        >>> bad  # doctest: +NORMALIZE_WHITESPACE
        [('row', 'record', 'field', 'value', 'error_type', 'error'),
         (2, ('3', 'x'), 'bar', 'x', 'builtins.ValueError',
          "invalid literal for int() with base 10: 'x'")]

    Parameters
    ----------

    table : Table
        Input table. Its header must consist of unique string field names and
        every data row must have the same width as the header (rows of the
        wrong width are reported through the ``bad`` branch rather than
        raising).
    converters : sequence of (field name, callable) pairs
        A non-empty ordered sequence (e.g. a list of tuples). Field names must
        not be repeated and every named field must exist in the input header.
        The converters are applied in the given order for each row and each
        converter receives the *original* value of its field; a selected
        field's converter is called at most once per row. If one converter
        raises an :class:`Exception`, the remaining converters for that row
        are skipped and the whole row goes to ``bad``; partially converted
        values are never emitted.
    buffersize : int
        Positive integer (booleans are not accepted). At most this many
        undelivered records per branch wait in memory; any further backlog is
        spooled to temporary disk files.
    tempdir : str, optional
        Directory in which temporary spool files are created. Defaults to the
        system temporary directory.

    Returns
    -------

    ConvertCapture
        A context manager with ``good`` and ``bad`` branch tables and an
        idempotent :meth:`~ConvertCapture.close`.

    The ``good`` branch keeps the original header and contains the fully
    converted rows. The fixed header of the ``bad`` branch is
    ``('row', 'record', 'field', 'value', 'error_type', 'error')`` where:

    * ``row`` is the one-based data row number (the first row after the
      header is row 1);
    * ``record`` is the complete original row as a tuple;
    * ``field`` and ``value`` are the first failing field name and its
      original value (both ``None``/the actual column count for rows of the
      wrong width, whose ``error_type`` is the fixed string
      ``'petl.RowLengthError'`` and for which no converter is called);
    * ``error_type`` is the module-qualified class name of the exception
      (``type(e).__module__ + '.' + type(e).__qualname__``);
    * ``error`` is ``str(exception)``.

    Both branches preserve the relative order of the input rows. A legitimate
    ``None`` converter result is a success.

    Branch tables are **single-consumption**: each may have only one
    iterator; asking for a second raises :class:`petl.errors.ArgumentError`.
    The two branches may be consumed in any order or interleaved from the
    same thread; thread-safe concurrent consumption is not supported. Creating
    the capture, accessing a branch and calling ``iter(branch)`` do not read
    the input or run any converter; the header and field configuration are
    validated when the first branch is actually consumed.

    Source iterator failures, spool I/O or serialisation failures and
    configuration errors fail the whole capture rather than producing bad
    rows. Each branch first yields any valid records already assigned to it
    and then raises the same original exception at its end; the source is not
    advanced afterwards. ``KeyboardInterrupt`` and other ``BaseException``
    subclasses propagate immediately and terminate the capture.

    Closing a branch iterator (or calling ``branch.close()`` before
    iterating) permanently abandons that branch: its buffered records and
    spool files are released, later rows belonging to it are discarded, and
    the other branch stays usable. Closing the capture (also via the context
    manager) closes the source iterator if it supports ``close()`` and
    removes every temporary file.

    Values and conversion results must be round-trippable with the standard
    :mod:`pickle` module.

    """

    return ConvertCapture(table, converters, buffersize=buffersize,
                          tempdir=tempdir)


Table.convertcapture = convertcapture


def _validate_converters(converters):
    # construction-time checks that do not need the input header
    if isinstance(converters, string_types) or not isinstance(
            converters, (list, tuple)):
        raise ArgumentError(
            'converters must be a non-empty ordered sequence of '
            '(field name, callable) pairs, found %r' % (converters,))
    if len(converters) == 0:
        raise ArgumentError('converters must be a non-empty sequence')
    seen = set()
    normalized = []
    for pair in converters:
        if (not isinstance(pair, tuple) or len(pair) != 2
                or isinstance(pair, string_types)):
            raise ArgumentError(
                'each converter must be a (field name, callable) pair, '
                'found %r' % (pair,))
        field, func = pair
        if not isinstance(field, string_types):
            raise ArgumentError(
                'converter field names must be strings, found %r' % (field,))
        if field in seen:
            raise ArgumentError('duplicate converter field name: %r' % field)
        if not callable(func):
            raise ArgumentError(
                'converter for field %r must be callable, found %r'
                % (field, func))
        seen.add(field)
        normalized.append((field, func))
    return normalized


def _validate_buffersize(buffersize):
    # bool is a subclass of int and must be rejected explicitly
    if isinstance(buffersize, bool) or not isinstance(buffersize, int) \
            or buffersize <= 0:
        raise ArgumentError(
            'buffersize must be a positive integer, found %r'
            % (buffersize,))
    return buffersize


def _qualified_exc_name(exc):
    cls = type(exc)
    qualname = getattr(cls, '__qualname__', None)
    if qualname is None:  # pragma: no cover - Python 2 fallback
        qualname = cls.__name__
    module = cls.__module__ or 'builtins'
    return module + '.' + qualname


class _SpooledQueue(object):
    """FIFO record queue with a bounded in-memory portion.

    Up to ``buffersize`` records are held in memory; once the queue grows
    beyond that, records are appended to temporary pickle files. A file is
    deleted as soon as its records have been delivered, so disk usage tracks
    the current unread backlog rather than the number of records ever seen.
    At most one spool file is open for reading at a time, alongside the
    (optional) file being written.

    """

    def __init__(self, buffersize, tempdir):
        self.buffersize = buffersize
        self.tempdir = tempdir
        self._mem = deque()
        # sealed, unread segments as [path, remaining record count], in order
        self._disk = deque()
        # open reader on the oldest sealed segment, if any
        self._reader = None
        self._reader_path = None
        # open file currently being appended to; created lazily on overflow
        self._active = None
        self._active_name = None
        self._active_count = 0
        self.closed = False

    def __len__(self):
        total = len(self._mem) + self._active_count
        for _path, remaining in self._disk:
            total += remaining
        return total

    def put(self, record):
        if self.closed:
            return
        if len(self._mem) < self.buffersize and self._active is None \
                and not self._disk and self._reader is None:
            # fast path: the whole queue still fits in the memory buffer
            self._mem.append(record)
            return
        if self._active is None:
            # the memory buffer is full: further records go to disk
            self._open_active()
        try:
            pickle.dump(record, self._active,
                        protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            # never leave a half-written record behind
            self._close_active(discard=True)
            raise
        self._active_count += 1
        if self._active_count >= self.buffersize:
            self._seal_active()

    def get(self):
        if self._mem:
            return self._mem.popleft()
        # drain sealed segments in order, one open reader at a time
        while self._disk:
            if self._reader is None:
                self._reader_path = self._disk[0][0]
                self._reader = open(self._reader_path, 'rb')
            try:
                record = pickle.load(self._reader)
            except EOFError:
                self._close_reader()
                self._disk.popleft()
                _unlink_quiet(self._reader_path)
                self._reader_path = None
                continue
            self._disk[0][1] -= 1
            if self._disk[0][1] == 0:
                self._close_reader()
                self._disk.popleft()
                _unlink_quiet(self._reader_path)
                self._reader_path = None
            return record
        if self._active is not None and self._active_count:
            # nothing left in memory or sealed files: finish the open segment
            # and read it through the same sealed-segment path
            self._seal_active()
            return self.get()
        raise IndexError('get from empty queue')

    def _open_active(self):
        f = NamedTemporaryFile(mode='wb', dir=self.tempdir, delete=False,
                               prefix='petl-convertcapture-', suffix='.pkl')
        self._active = f
        self._active_name = f.name
        self._active_count = 0

    def _seal_active(self):
        self._active.flush()
        self._active.close()
        self._disk.append([self._active_name, self._active_count])
        self._active = None
        self._active_name = None
        self._active_count = 0

    def _close_active(self, discard):
        try:
            self._active.close()
        finally:
            if discard and self._active_name is not None:
                _unlink_quiet(self._active_name)
            self._active = None
            self._active_name = None
            self._active_count = 0

    def _close_reader(self):
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self._reader = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._mem.clear()
        if self._active is not None:
            self._close_active(discard=True)
        self._close_reader()
        while self._disk:
            path, _remaining = self._disk.popleft()
            _unlink_quiet(path)
        self._reader_path = None


def _unlink_quiet(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _branch_iterate(capture, side, state):
    """Generator backing :class:`_BranchIterator`.

    It deliberately closes over no branch-iterator instance, so closing an
    enclosing consumer generator (or dropping the iterator) runs the
    ``finally`` block without a reference cycle. ``state`` is a two-element
    list: ``state[0]`` is flipped to True when the branch is abandoned, and
    ``state[1]`` records that the generator reached a clean end of input.

    """
    try:
        # first delivery: make sure the shared pipeline is started (this is
        # where header/field validation happens) and emit the header
        capture._ensure_started()
        yield capture.good._header if side == 'good' else BAD_HEADER
        while True:
            try:
                yield capture._next_branch(side)
            except StopIteration:
                # clean end of this branch; break so the generator finishes
                # normally (PEP 479 forbids StopIteration escaping a generator)
                state[1] = True
                return
    finally:
        state[0] = True
        capture._abandon(side)


class _BranchIterator(object):
    """Single-use iterator over a :class:`ConvertCapture` branch.

    The iteration state lives in a real generator (``_gen``) so that closing
    an enclosing consumer generator (or simply dropping it) triggers the
    generator's ``close()`` and runs the abandonment logic in
    :func:`_branch_iterate`.

    """

    def __init__(self, capture, side):
        self._capture = capture
        self._side = side
        # mutable flags shared with the generator's finally-block; kept out of
        # the generator closure to avoid an iterator <-> generator cycle.
        # state[0]: abandoned/closed; state[1]: generator finished normally
        self._state = [False, False]
        self._gen = _branch_iterate(capture, side, self._state)

    @property
    def _closed(self):
        return self._state[0]

    def __iter__(self):
        return self

    def __length_hint__(self):
        # list() and friends pre-size via this hint; the inherited container
        # __len__ would open a second, forbidden, iterator. The length is
        # unknown for a single-consumption streaming branch.
        return NotImplemented

    def __next__(self):
        if self._state[0] and not self._state[1]:
            # the generator was torn down by an exception or an explicit
            # close(): re-surface the recorded capture failure (the same
            # object for both branches) or end the abandoned branch quietly
            failure = self._capture._failure
            if failure is not None:
                raise failure
            raise StopIteration
        return next(self._gen)

    # Python 2 name
    next = __next__

    def close(self):
        if self._closed:
            return
        # closing the generator runs its finally-block, which performs the
        # actual queue/source teardown for the branch
        self._gen.close()

    def __del__(self):
        # safety net for consumers that simply drop the iterator
        try:
            self._gen.close()
        except Exception:
            pass


class _CaptureBranch(Table):
    """One of the good/bad output tables of a :class:`ConvertCapture`."""

    def __init__(self, capture, side, header):
        self._capture = capture
        self._side = side
        self._header = header
        # True once an iterator has been handed out; the branch must never
        # issue a second one. The iterator itself is not retained here, so
        # that dropping it (or generator-close) can release the branch.
        self._iterated = False
        self.abandoned = False

    def __iter__(self):
        if self.abandoned:
            raise ArgumentError(
                'this convertcapture branch has been closed and can no '
                'longer be iterated')
        if self._iterated:
            raise ArgumentError(
                'convertcapture branches can only be consumed once; this '
                'branch has already been iterated')
        self._iterated = True
        return _BranchIterator(self._capture, self._side)

    def close(self):
        """Permanently abandon this branch and release its buffers."""
        if self.abandoned:
            return
        self.abandoned = True
        # no iterator is retained here; if one is alive its own close()/GC
        # performs the queue teardown, and this covers the never-iterated case
        self._capture._abandon(self._side)

    def __len__(self):
        # the inherited implementation would open a second (forbidden)
        # iterator; length is unknown until the single consumption completes
        raise TypeError(
            'convertcapture branches are single-consumption tables and do '
            'not support len(); materialise the iteration yourself, e.g. '
            'with list()')

    def __length_hint__(self):
        # stop list() and friends from probing the inherited __len__
        return NotImplemented

    def __repr__(self):
        return '<convertcapture %s branch>' % self._side


class ConvertCapture(object):
    """Single-pass good/bad split produced by :func:`convertcapture`.

    Instances are context managers. The :attr:`good` and :attr:`bad`
    attributes are single-consumption :class:`petl.util.base.Table` objects.

    """

    def __init__(self, source, converters, buffersize=1000, tempdir=None):
        self._source = source
        self._converter_spec = _validate_converters(converters)
        self._buffersize = _validate_buffersize(buffersize)
        self._tempdir = tempdir

        self._source_iter = None
        self._started = False
        self._finished = False
        self._closed = False
        self._failure = None
        self._row_number = 0
        self._width = None
        # list of (field index, callable), in converters order
        self._indexed_converters = None

        self._queues = {
            'good': _SpooledQueue(self._buffersize, self._tempdir),
            'bad': _SpooledQueue(self._buffersize, self._tempdir),
        }
        self._abandoned = {'good': False, 'bad': False}
        self.good = _CaptureBranch(self, 'good', None)
        self.bad = _CaptureBranch(self, 'bad', BAD_HEADER)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException:
            # cleanup errors must not cover an exception from the with-block
            if exc_value is None:
                raise
        # never swallow exceptions raised inside the with-block
        return False

    def close(self):
        """Close the source iterator and release all buffered data.

        Idempotent. If the capture already failed, the original failure is not
        replaced by any error raised while cleaning up.

        """
        if self._closed:
            return
        self._closed = True
        self.good.abandoned = True
        self.bad.abandoned = True
        cleanup_error = None
        if self._started and not self._finished:
            try:
                self._close_source()
            except BaseException as e:
                cleanup_error = e
        for side in ('good', 'bad'):
            try:
                self._queues[side].close()
            except BaseException as e:
                if cleanup_error is None:
                    cleanup_error = e
        if self._failure is not None:
            # the original failure always wins
            return
        if cleanup_error is not None:
            raise cleanup_error

    # -- branch plumbing ---------------------------------------------------

    def _abandon(self, side):
        if self._abandoned[side]:
            return
        self._abandoned[side] = True
        self._queues[side].close()
        if self._abandoned['good'] and self._abandoned['bad']:
            # nobody left to consume: stop pulling and release everything
            self._terminate()

    def _terminate(self):
        if not self._finished:
            self._finished = True
            try:
                self._close_source()
            except BaseException:
                # abandonment is a user-driven teardown; a failing close is
                # surfaced by the explicit close() path instead
                pass
        for side in ('good', 'bad'):
            self._queues[side].close()

    def _next_branch(self, side):
        if self._closed:
            raise ArgumentError(
                'this convertcapture has been closed; its branches can no '
                'longer be consumed')
        self._ensure_started()
        queue = self._queues[side]
        while True:
            if len(queue) > 0:
                return queue.get()
            # nothing waiting for this branch: drive the shared pipeline,
            # unless everything has already been processed
            if self._finished:
                if self._failure is not None:
                    raise self._failure
                raise StopIteration
            self._pump_once(side)

    def _ensure_started(self):
        if self._started:
            return
        self._started = True
        try:
            self._source_iter = iter(self._source)
            try:
                header = next(self._source_iter)
            except StopIteration:
                raise ArgumentError(
                    'input table has no header row; convertcapture requires '
                    'a header of unique string field names')
            self._validate_header(header)
            self._width = len(header)
            self.good._header = tuple(header)
            flds = [text_type(h) for h in header]
            indexed = []
            for field, func in self._converter_spec:
                try:
                    index = flds.index(field)
                except ValueError:
                    raise FieldSelectionError(field)
                indexed.append((index, func))
            self._indexed_converters = indexed
        except BaseException as e:
            # nothing deliverable yet; make sure the source is released and
            # remember the failure for the other branch
            self._fail(e)
            raise

    def _validate_header(self, header):
        if not isinstance(header, (tuple, list)):
            raise ArgumentError(
                'input header must be a sequence of unique string field '
                'names, found %r' % (header,))
        names = list(header)
        for name in names:
            if not isinstance(name, string_types):
                raise ArgumentError(
                    'input header field names must be strings, found %r'
                    % (name,))
        if len(set(names)) != len(names):
            raise ArgumentError(
                'input header field names must be unique, found %r'
                % (header,))

    def _pump_once(self, requesting_side):
        """Process exactly one source row and route it."""
        try:
            try:
                row = next(self._source_iter)
            except StopIteration:
                # clean end of input; a failing close becomes the capture
                # failure rather than a spurious normal EOF
                self._finished = True
                try:
                    self._close_source()
                except BaseException as close_exc:
                    self._failure = close_exc
                return
            self._row_number += 1
            record = tuple(row)
            if len(record) != self._width:
                exc = RowLengthError(self._width, len(record))
                bad_record = (
                    self._row_number,
                    record,
                    None,
                    len(record),
                    _ROW_LENGTH_ERROR_TYPE,
                    str(exc),
                )
                self._route('bad', bad_record)
                return
            converted = list(record)
            for index, func in self._indexed_converters:
                original = record[index]
                try:
                    converted[index] = func(original)
                except Exception as exc:
                    # later converters are skipped; the half-converted row is
                    # never emitted through good
                    field_name = self.good._header[index]
                    bad_record = (
                        self._row_number,
                        record,
                        field_name,
                        original,
                        _qualified_exc_name(exc),
                        str(exc),
                    )
                    self._route('bad', bad_record)
                    return
            self._route('good', tuple(converted))
        except BaseException as exc:
            # source errors and spool/serialisation errors fail the capture;
            # BaseException (e.g. KeyboardInterrupt) terminates it as well
            self._fail(exc)
            raise

    def _route(self, side, record):
        if self._abandoned[side]:
            # consumer gave up: drop the record without touching any buffer
            return
        self._queues[side].put(record)

    def _fail(self, exc):
        if self._failure is None:
            self._failure = exc
        self._finished = True
        # releasing the source must not replace the original failure
        try:
            self._close_source()
        except BaseException:
            pass

    def _close_source(self):
        it = self._source_iter
        if it is None:
            return
        # never invoke the source close twice
        self._source_iter = None
        closer = getattr(it, 'close', None)
        if callable(closer):
            closer()
