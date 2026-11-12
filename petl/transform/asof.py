from __future__ import absolute_import, print_function, division


import math
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter

from petl.compat import next, text_type, string_types, integer_types
from petl.comparison import Comparable
from petl.errors import ArgumentError, FieldSelectionError
from petl.transform.sorts import sort
from petl.util.base import Table, asindices


_DIRECTIONS = ('backward', 'forward', 'nearest')


def asofjoin(left, right, key=None, lkey=None, rkey=None, by=None,
             direction='backward', tolerance=None,
             allow_exact_matches=True, presorted=False, buffersize=None,
             tempdir=None, missing=None, lprefix=None, rprefix='right_'):
    """
    Join each row of the `left` table with at most one row of the `right`
    table based on an as-of match between time fields, optionally within one
    or more business grouping fields. Each right table row may be reused by
    any number of left table rows. E.g.::

        >>> import petl as etl
        >>> left = [('asset', 't', 'event'),
        ...         ('B', 7, 'eB'),
        ...         ('A', 4, 'eA'),
        ...         ('A', 6, 'eC')]
        >>> right = [('asset', 't', 'value'),
        ...          ('A', 5, 'later'),
        ...          ('A', 3, 'old'),
        ...          ('A', 3, 'revised'),
        ...          ('B', 10, 'future')]
        >>> table = etl.asofjoin(left, right, key='t', by='asset')
        >>> table.lookall()
        +-------+---+-------+---------+-------------+
        | asset | t | event | right_t | right_value |
        +=======+===+=======+=========+=============+
        | 'A'   | 4 | 'eA'  |       3 | 'revised'   |
        +-------+---+-------+---------+-------------+
        | 'A'   | 6 | 'eC'  |       5 | 'later'     |
        +-------+---+-------+---------+-------------+
        | 'B'   | 7 | 'eB'  | None    | None        |
        +-------+---+-------+---------+-------------+

    The time field must be given explicitly, either via `key` when both
    tables use the same field name, or via both `lkey` and `rkey`; the two
    forms are mutually exclusive. Field names and integer positions
    (including negative positions) are accepted.

    `by` names one or more fields present in both tables that identify the
    business group within which matches are made. A time field cannot also be
    a grouping field. If `by` is `None`, matching is performed across the
    whole tables.

    `direction` selects how the right table time is positioned relative to
    the left table time:

    - ``'backward'`` (the default): the most recent right row with time less
      than or equal to the event time;
    - ``'forward'``: the earliest right row with time greater than or equal
      to the event time;
    - ``'nearest'``: the right row with the smallest absolute time distance;
      when the backward and forward distances are equal the earlier
      (backward) row is chosen.

    If `allow_exact_matches` is `False`, right rows whose time equals the
    event time are excluded before selecting a candidate. When several right
    rows share one time within a group, backward and nearest select the last
    such row in right table order and forward selects the first.

    `tolerance` is the maximum permitted time distance, inclusive. It must be
    a non-negative finite number when times are numeric and a non-negative
    :class:`datetime.timedelta` when times are
    :class:`datetime.datetime` values. Left rows without a candidate are
    still emitted, with right fields filled with `missing`.

    Times must be finite ``int`` or ``float`` values (``bool`` is rejected)
    or naive (timezone-free) :class:`datetime.datetime` values. Both tables
    of one call share a single time domain; ``None``, ``NaN``, infinities,
    strings, timezone-aware datetimes and mixed numeric/datetime domains are
    rejected explicitly rather than sorted as strings.

    Rows are emitted ordered by grouping fields and then left table time,
    ascending; left rows that share a group and time keep their original
    order. Without grouping fields, rows are ordered by left table time.

    Unless `presorted` is `True`, both inputs are stably sorted with
    :func:`petl.transform.sorts.sort`, honouring `buffersize` and
    `tempdir`; tables that do not fit the buffer are externally sorted and
    are never held in memory as a whole. When `presorted` is `True`, both
    inputs are assumed to already be ordered by their grouping fields and
    time, and the join performs a single streaming pass that only retains a
    constant number of right rows regardless of group size or the number of
    right rows sharing one time. Rows found to be out of order while actually
    consumed raise an :class:`petl.errors.ArgumentError` naming the side and
    data row numbers.

    The output header contains all left fields followed by the non-grouping
    right fields; the right time field is retained. `lprefix` and `rprefix`
    are added to the corresponding output field names, where `None` means no
    prefix; the default right prefix is ``'right_'``.

    The view is lazy: creating it reads neither input, and each iteration
    requests at most one iterator from each input.

    .. versionadded:: 1.7.32

    """

    return AsofJoinView(left, right, key=key, lkey=lkey, rkey=rkey, by=by,
                        direction=direction, tolerance=tolerance,
                        allow_exact_matches=allow_exact_matches,
                        presorted=presorted, buffersize=buffersize,
                        tempdir=tempdir, missing=missing, lprefix=lprefix,
                        rprefix=rprefix)


Table.asofjoin = asofjoin


def _normalise_by(by):
    """Return a tuple of grouping field names, or an empty tuple."""
    if by is None:
        return ()
    if isinstance(by, string_types):
        return (by,)
    if isinstance(by, (list, tuple)) and len(by) > 0:
        names = []
        for f in by:
            if not isinstance(f, string_types):
                raise ArgumentError(
                    'grouping fields must be field names, got %r' % (f,))
            if f in names:
                raise ArgumentError(
                    'duplicate grouping field: %r' % (f,))
            names.append(f)
        return tuple(names)
    raise ArgumentError(
        'by must be None, a field name or a non-empty sequence of field '
        'names, got %r' % (by,))


def _resolve_index(hdr, spec):
    """Resolve one time field spec (name or integer index) to a position."""
    if isinstance(spec, bool):
        # bool is an int subclass but never a valid field position
        raise FieldSelectionError(spec)
    if isinstance(spec, integer_types):
        n = len(hdr)
        if not (-n <= spec < n):
            raise FieldSelectionError(spec)
        return spec if spec >= 0 else n + spec
    flds = list(map(text_type, hdr))
    if spec in flds:
        return flds.index(spec)
    raise FieldSelectionError(spec)


def _validate_tolerance(tolerance):
    if tolerance is None:
        return
    if isinstance(tolerance, bool):
        raise ArgumentError(
            'tolerance must be a non-negative finite number or timedelta, '
            'got %r' % (tolerance,))
    if isinstance(tolerance, integer_types + (float,)):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ArgumentError(
                'numeric tolerance must be a non-negative finite number, '
                'got %r' % (tolerance,))
        return
    if isinstance(tolerance, timedelta):
        if tolerance < timedelta(0):
            raise ArgumentError(
                'timedelta tolerance must be non-negative, got %r'
                % (tolerance,))
        return
    raise ArgumentError(
        'tolerance must be a non-negative finite number or timedelta, '
        'got %r' % (tolerance,))


class AsofJoinView(Table):

    def __init__(self, left, right, key=None, lkey=None, rkey=None, by=None,
                 direction='backward', tolerance=None,
                 allow_exact_matches=True, presorted=False, buffersize=None,
                 tempdir=None, missing=None, lprefix=None, rprefix='right_'):

        # time field argument must be given in exactly one of the two forms
        if key is not None:
            if lkey is not None or rkey is not None:
                raise ArgumentError(
                    'key cannot be combined with lkey/rkey; give either key '
                    'or both lkey and rkey')
            lkey = rkey = key
        else:
            if lkey is None or rkey is None:
                raise ArgumentError(
                    'a time field is required: give key, or both lkey and '
                    'rkey')

        if direction not in _DIRECTIONS:
            raise ArgumentError(
                "direction must be one of 'backward', 'forward' or "
                "'nearest', got %r" % (direction,))

        if not isinstance(allow_exact_matches, bool):
            raise ArgumentError(
                'allow_exact_matches must be True or False, got %r'
                % (allow_exact_matches,))

        if not isinstance(presorted, bool):
            raise ArgumentError(
                'presorted must be True or False, got %r' % (presorted,))

        _validate_tolerance(tolerance)

        self.by = _normalise_by(by)
        self.left = left
        self.right = right
        self.lkey = lkey
        self.rkey = rkey
        self.direction = direction
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.presorted = presorted
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterasofjoin(
            self.left, self.right, self.lkey, self.rkey, self.by,
            self.direction, self.tolerance, self.allow_exact_matches,
            self.presorted, self.buffersize, self.tempdir, self.missing,
            self.lprefix, self.rprefix)


def _check_header_names(hdr, side):
    names = list(map(text_type, hdr))
    if len(names) == 0:
        raise ArgumentError('%s table has no header row' % side)
    seen = set()
    for name in names:
        if name in seen:
            raise ArgumentError(
                '%s table header has duplicate field name: %r'
                % (side, name))
        seen.add(name)


def _build_output_header(lhdr, rhdr, rgroup_idx, rprefix, lprefix):
    if lprefix is None:
        outhdr = [text_type(f) for f in lhdr]
    else:
        outhdr = [text_type(lprefix) + text_type(f) for f in lhdr]
    skip = set(rgroup_idx)
    for i, f in enumerate(rhdr):
        if i in skip:
            continue
        name = text_type(f)
        if rprefix is not None:
            name = text_type(rprefix) + name
        outhdr.append(name)
    seen = set()
    for name in outhdr:
        if name in seen:
            raise ArgumentError(
                'output header has duplicate field name: %r' % (name,))
        seen.add(name)
    return tuple(outhdr)


def _check_time_value(value, side, rownum):
    """Validate one time value and return its domain tag."""
    if value is None:
        raise ArgumentError(
            '%s table row %d has None in time field' % (side, rownum))
    if isinstance(value, bool):
        raise ArgumentError(
            '%s table row %d has boolean time value: %r'
            % (side, rownum, value))
    if isinstance(value, integer_types + (float,)):
        if not math.isfinite(value):
            raise ArgumentError(
                '%s table row %d has non-finite time value: %r'
                % (side, rownum, value))
        return 'number'
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ArgumentError(
                '%s table row %d has a timezone-aware datetime: %r'
                % (side, rownum, value))
        return 'datetime'
    raise ArgumentError(
        '%s table row %d has unsupported time value %r of type %s; times '
        'must be finite int/float or naive datetime'
        % (side, rownum, value, type(value).__name__))


def _check_tolerance_domain(tolerance, domain):
    if tolerance is None:
        return
    if domain == 'number':
        if isinstance(tolerance, bool) or not isinstance(
                tolerance, integer_types + (float,)):
            raise ArgumentError(
                'numeric time values require a numeric tolerance, got %r'
                % (tolerance,))
    else:
        if not isinstance(tolerance, timedelta):
            raise ArgumentError(
                'datetime time values require a timedelta tolerance, '
                'got %r' % (tolerance,))


def _validated_rows(it, width, side, check_order, gettime, getgroup,
                    domain_holder, tolerance):
    """Yield rows from `it` after width/time/order validation.

    `domain_holder` is a one-element list carrying the time domain tag shared
    between the left and right streams of one iteration.
    """
    rownum = 0
    prev_key = None
    prev_row = None
    for row in it:
        rownum += 1
        if len(row) != width:
            raise ArgumentError(
                '%s table data row %d has %d values but the header has %d '
                'fields' % (side, rownum, len(row), width))
        tval = gettime(row)
        domain = _check_time_value(tval, side, rownum)
        if domain_holder[0] is None:
            domain_holder[0] = domain
            _check_tolerance_domain(tolerance, domain)
        elif domain_holder[0] != domain:
            raise ArgumentError(
                '%s table row %d mixes time domains: expected %s time '
                'values but found %s (%r)'
                % (side, rownum, domain_holder[0], domain, tval))
        if check_order:
            key = (Comparable(getgroup(row)), Comparable(tval))
            if prev_key is not None and key < prev_key:
                raise ArgumentError(
                    '%s table rows are not ordered by grouping fields and '
                    'time: data row %d %r is followed by data row %d %r'
                    % (side, rownum - 1, prev_row, rownum, row))
            prev_key = key
            prev_row = row
        yield row


def _close_quiet(resource):
    close = getattr(resource, 'close', None)
    if close is not None:
        close()


class _HeaderDataTable(Table):
    """A table over one already-open data iterator.

    The header is cached and may be read on every iteration; the data rows
    are delivered exactly once, which is all the enclosing join consumes via
    its single sort pass.
    """

    def __init__(self, hdr, data_it):
        self._hdr = tuple(hdr)
        self._data_it = data_it
        self._used = False

    def __iter__(self):
        yield self._hdr
        if self._used:
            # sort views only re-enter after a full pass when caching data;
            # data rows are not available a second time
            return
        self._used = True
        for row in self._data_it:
            yield row


def iterasofjoin(left, right, lkey, rkey, by_fields, direction, tolerance,
                 allow_exact_matches, presorted, buffersize, tempdir,
                 missing, lprefix, rprefix):

    # Exactly one iterator is requested from each input. When sorting is
    # needed, the header is read from that iterator first and the remaining
    # data iterator is wrapped for the sort view, so no source is read twice.
    closeables = []
    try:
        lit = iter(left)
        rit = iter(right)
        closeables = [lit, rit]
        try:
            lhdr = next(lit)
        except StopIteration:
            raise ArgumentError('left table has no header row')
        try:
            rhdr = next(rit)
        except StopIteration:
            raise ArgumentError('right table has no header row')

        _check_header_names(lhdr, 'left')
        _check_header_names(rhdr, 'right')

        lt_idx = _resolve_index(lhdr, lkey)
        rt_idx = _resolve_index(rhdr, rkey)
        lgroup_idx = asindices(lhdr, by_fields) if by_fields else []
        rgroup_idx = asindices(rhdr, by_fields) if by_fields else []
        if lt_idx in lgroup_idx:
            raise ArgumentError(
                'left time field %r is also a grouping field' % (lkey,))
        if rt_idx in rgroup_idx:
            raise ArgumentError(
                'right time field %r is also a grouping field' % (rkey,))

        if not presorted:
            # stable external sorts consume the wrapped single iterator.
            # cache=True lets sort hold the (bounded) in-memory chunk or the
            # external chunk files itself, so its internal re-iteration of
            # the source header never asks for data rows twice.
            ltable = _HeaderDataTable(lhdr, lit)
            rtable = _HeaderDataTable(rhdr, rit)
            left_view = sort(ltable, key=tuple(lgroup_idx) + (lt_idx,),
                             buffersize=buffersize, tempdir=tempdir,
                             cache=True)
            right_view = sort(rtable, key=tuple(rgroup_idx) + (rt_idx,),
                              buffersize=buffersize, tempdir=tempdir,
                              cache=True)
            # the sorted generators drive the merge; the raw source
            # iterators stay reachable through their frames but are closed
            # explicitly as well so closeable sources are always released
            ls_it = iter(left_view)
            rs_it = iter(right_view)
            closeables = [ls_it, rs_it, lit, rit]
            next(ls_it)  # discard the sorted header row
            next(rs_it)
            ldata = ls_it
            rdata = rs_it
        else:
            ldata = lit
            rdata = rit

        outhdr = _build_output_header(lhdr, rhdr, rgroup_idx, rprefix,
                                      lprefix)
        yield outhdr

        rout_idx = [i for i in range(len(rhdr)) if i not in set(rgroup_idx)]
        get_rout = itemgetter(*rout_idx) if rout_idx else lambda row: ()
        get_ltime = itemgetter(lt_idx)
        get_rtime = itemgetter(rt_idx)
        if lgroup_idx:
            # always return tuples so single-field and compound groups share
            # one comparison shape
            get_lgroup = itemgetter(*lgroup_idx)
            get_rgroup = itemgetter(*rgroup_idx)
            if len(lgroup_idx) == 1:
                _gl = get_lgroup
                _gr = get_rgroup
                get_lgroup = lambda row, _g=_gl: (_g(row),)
                get_rgroup = lambda row, _g=_gr: (_g(row),)
        else:
            get_lgroup = lambda row: ()
            get_rgroup = lambda row: ()

        domain_holder = [None]
        lrows = _validated_rows(
            ldata, len(lhdr), 'left', presorted, get_ltime, get_lgroup,
            domain_holder, tolerance)
        rrows = _validated_rows(
            rdata, len(rhdr), 'right', presorted, get_rtime, get_rgroup,
            domain_holder, tolerance)

        n_rout = len(rout_idx)

        def missing_row(lrow):
            return tuple(lrow) + (missing,) * n_rout

        # Streaming right-scan state. Right rows are grouped into
        # (group, time) blocks lazily. A block only pulls its rows when its
        # last row is actually required (backward/nearest exact match); when
        # only its first row is needed (forward) or it is being skipped
        # (exact matches disabled), rows are advanced one at a time without
        # retaining them. No collection grows with group size or duplicate
        # count: only one predecessor block and one current block are held.
        class _Block(object):
            __slots__ = ('group', 'time', 'first', 'last', 'finished',
                         '_feed', 'boundary')

            def __init__(self, group, time, first, feed):
                self.group = group
                self.time = time
                self.first = first
                self.last = first
                self.finished = False
                # feed() returns the next validated (row, group, time) or
                # raises StopIteration
                self._feed = feed
                # the first (row, group, time) beyond this block, or None
                self.boundary = None

            def resolve(self):
                """Walk rows until the block boundary, retaining the last.

                The boundary is cached so repeated calls return the same
                pre-read next block without consuming another source row.
                """
                if self.finished:
                    return self.boundary
                while True:
                    try:
                        nxt, ng, nt = self._feed()
                    except StopIteration:
                        self.finished = True
                        self.boundary = None
                        return None
                    if ng == self.group and nt == self.time:
                        self.last = nxt
                        continue
                    self.finished = True
                    self.boundary = (nxt, ng, nt)
                    return self.boundary

        def _feed():
            row = next(rrows)
            return row, Comparable(get_rgroup(row)), get_rtime(row)

        cur = None       # current block (may be unfinished)

        def open_next(pending=None):
            try:
                if pending is None:
                    row, group, tval = _feed()
                else:
                    row, group, tval = pending
            except StopIteration:
                return None
            return _Block(group, tval, row, _feed)

        cur = open_next()
        # `strict_back` is the last block whose time is strictly less than
        # the current event time. The equal-time block stays on `cur` until
        # the event time advances. `skipped_eq` holds an equal block ignored
        # because exact matches are disabled; it becomes a strict
        # predecessor for later events with a greater time.
        strict_back = None
        skipped_eq = None

        def left_group_key(row):
            return Comparable(get_lgroup(row))

        for lgroup_cmp, lgroup_rows in groupby(lrows, key=left_group_key):

            # skip blocks belonging to earlier groups without resolving their
            # full contents beyond what was already read
            while cur is not None and cur.group < lgroup_cmp:
                strict_back = None
                skipped_eq = None
                boundary = cur.resolve()
                if boundary is None:
                    cur = None
                else:
                    cur = open_next(pending=boundary)
            strict_back = None
            skipped_eq = None

            for lrow in lgroup_rows:
                ltime = get_ltime(lrow)

                def in_tolerance(row, _lt=ltime):
                    if tolerance is None or row is None:
                        return True
                    return abs(get_rtime(row) - _lt) <= tolerance

                # a block skipped at an earlier event time because exact
                # matches were disabled becomes a strict predecessor now
                if skipped_eq is not None and skipped_eq.time < ltime:
                    strict_back = skipped_eq
                    skipped_eq = None

                # move blocks strictly before the event into `strict_back`;
                # the equal-time block, if any, stays parked on `cur`
                while cur is not None and cur.group == lgroup_cmp \
                        and cur.time < ltime:
                    boundary = cur.resolve()
                    strict_back = cur
                    if boundary is None:
                        cur = None
                    else:
                        cur = open_next(pending=boundary)

                in_group = cur is not None and cur.group == lgroup_cmp
                on_equal = in_group and cur.time == ltime

                # when exact matches are disabled, an equal-time block is not
                # a candidate at this event time: skip it now. The genuine
                # strict predecessor in `strict_back` is preserved; the
                # skipped block is promoted to strict predecessor for any
                # later event whose time is strictly greater.
                if on_equal and not allow_exact_matches:
                    skipped_eq = cur
                    boundary = cur.resolve()
                    if boundary is None:
                        cur = None
                    else:
                        cur = open_next(pending=boundary)
                    in_group = cur is not None and cur.group == lgroup_cmp
                    on_equal = False

                candidate = None

                if on_equal:
                    if direction == 'forward':
                        # only the block's first row is needed
                        candidate = cur.first
                    else:
                        # backward/nearest need the block's last row
                        cur.resolve()
                        candidate = cur.last
                    # The equal block stays on `cur`: repeated left events
                    # sharing this time must all match it. It moves into
                    # `strict_back` lazily when a later event's predecessor
                    # loop advances past it.

                # effective backward candidate: an allowed equal block wins,
                # otherwise the most recent block that is strictly before the
                # event time
                if on_equal:
                    bwd_block = cur
                elif strict_back is not None and \
                        strict_back.group == lgroup_cmp and \
                        strict_back.time < ltime:
                    bwd_block = strict_back
                else:
                    bwd_block = None

                if candidate is None:
                    if direction == 'backward':
                        if bwd_block is not None:
                            cand = bwd_block.last
                            if in_tolerance(cand):
                                candidate = cand

                    elif direction == 'forward':
                        if on_equal:
                            # the equal block is the forward candidate too
                            cand = cur.first
                            if in_tolerance(cand):
                                candidate = cand
                        elif cur is not None and cur.group == lgroup_cmp \
                                and cur.time > ltime:
                            cand = cur.first
                            if in_tolerance(cand):
                                candidate = cand

                    else:  # nearest
                        bwd = bwd_block.last if bwd_block is not None \
                            else None
                        fwd = cur.first if on_equal or (
                            cur is not None and cur.group == lgroup_cmp
                            and cur.time > ltime) else None
                        bwd_ok = bwd is not None and in_tolerance(bwd)
                        fwd_ok = fwd is not None and in_tolerance(fwd)
                        if bwd_ok and fwd_ok:
                            db = ltime - get_rtime(bwd)
                            df = get_rtime(fwd) - ltime
                            if db <= df:
                                candidate = bwd
                            else:
                                candidate = fwd
                        elif bwd_ok:
                            candidate = bwd
                        elif fwd_ok:
                            candidate = fwd

                # exact matches always satisfy any non-negative tolerance;
                # this guards any candidate selected without a local check
                if candidate is not None and tolerance is not None \
                        and not in_tolerance(candidate):
                    candidate = None

                if candidate is None:
                    yield missing_row(lrow)
                else:
                    yield tuple(lrow) + tuple(get_rout(candidate))

    finally:
        for resource in closeables:
            _close_quiet(resource)
