from __future__ import absolute_import, division, print_function

import itertools

from petl.comparison import Comparable, comparable_itemgetter
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.missing import key_contains_missing
from petl.transform.basics import cut, cutout, stack
from petl.transform.dedup import distinct
from petl.transform.sorts import sort
from petl.util.base import Table, asindices, data, header, rowgetter, rowgroupby

def natural_key(left, right):
    # determine key field or fields
    lhdr = header(left)
    lflds = list(map(str, lhdr))
    rhdr = header(right)
    rflds = list(map(str, rhdr))
    key = [f for f in lflds if f in rflds]
    assert len(key) > 0, 'no fields in common'
    if len(key) == 1:
        key = key[0]  # deal with singletons
    return key


def keys_from_args(left, right, key, lkey, rkey):

    if key is lkey is rkey is None:
        # no keys specified, attempt natural join
        lkey = rkey = natural_key(left, right)
    elif key is not None and lkey is rkey is None:
        # common key specified
        lkey = rkey = key
    elif key is None and lkey is not None and rkey is not None:
        # left and right keys specified
        pass
    else:
        raise ArgumentError(
            'bad key arguments: either specify key, or specify both lkey and '
            'rkey, or provide no key/lkey/rkey arguments at all (natural join)'
        )
    return lkey, rkey


def _nonkey_indices(hdr, indices):
    """Return non-key positions, including when keys use negative indexes.

    >>> _nonkey_indices(('id', 'value'), (-1,))
    [0]
    """
    key_indices = {i if i >= 0 else len(hdr) + i for i in indices}
    return [i for i in range(len(hdr)) if i not in key_indices]


def join(left, right, key=None, lkey=None, rkey=None, presorted=False,
         buffersize=None, tempdir=None, cache=True, lprefix=None,
         rprefix=None, match_missing=False):
    """
    Perform an equi-join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.join(table1, table2, key='id')
        >>> table3
        +----+----------+----------+
        | id | colour   | shape    |
        +====+==========+==========+
        |  1 | 'blue'   | 'circle' |
        +----+----------+----------+
        |  3 | 'purple' | 'square' |
        +----+----------+----------+

        >>> # if no key is given, a natural join is tried
        ... table4 = etl.join(table1, table2)
        >>> table4
        +----+----------+----------+
        | id | colour   | shape    |
        +====+==========+==========+
        |  1 | 'blue'   | 'circle' |
        +----+----------+----------+
        |  3 | 'purple' | 'square' |
        +----+----------+----------+

        >>> # note behaviour if the key is not unique in either or both tables
        ... table5 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [1, 'red'],
        ...           [2, 'purple']]
        >>> table6 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [1, 'square'],
        ...           [2, 'ellipse']]
        >>> table7 = etl.join(table5, table6, key='id')
        >>> table7
        +----+----------+-----------+
        | id | colour   | shape     |
        +====+==========+===========+
        |  1 | 'blue'   | 'circle'  |
        +----+----------+-----------+
        |  1 | 'blue'   | 'square'  |
        +----+----------+-----------+
        |  1 | 'red'    | 'circle'  |
        +----+----------+-----------+
        |  1 | 'red'    | 'square'  |
        +----+----------+-----------+
        |  2 | 'purple' | 'ellipse' |
        +----+----------+-----------+

        >>> # compound keys are supported
        ... table8 = [['id', 'time', 'height'],
        ...           [1, 1, 12.3],
        ...           [1, 2, 34.5],
        ...           [2, 1, 56.7]]
        >>> table9 = [['id', 'time', 'weight'],
        ...           [1, 2, 4.5],
        ...           [2, 1, 6.7],
        ...           [2, 2, 8.9]]
        >>> table10 = etl.join(table8, table9, key=['id', 'time'])
        >>> table10
        +----+------+--------+--------+
        | id | time | height | weight |
        +====+======+========+========+
        |  1 |    2 |   34.5 |    4.5 |
        +----+------+--------+--------+
        |  2 |    1 |   56.7 |    6.7 |
        +----+------+--------+--------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    If either side was read with ``preserve_missing=True``, key cells may
    hold the :data:`petl.MISSING` marker for fields absent from the source.
    By default (`match_missing=False`) any row whose key contains a MISSING
    component can never match another row - not even a row whose key is
    missing in the same positions - and is handled according to the outer
    join semantics. Set `match_missing=True` to treat MISSING as equal only
    to MISSING in the same key position; all other components keep their
    normal equality rules and ``None`` keys are unaffected.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, buffersize=buffersize, tempdir=tempdir,
                    cache=cache, lprefix=lprefix, rprefix=rprefix,
                    match_missing=match_missing)


Table.join = join


class JoinView(Table):

    def __init__(self, left, right, lkey, rkey,
                 presorted=False, leftouter=False, rightouter=False,
                 missing=None, buffersize=None, tempdir=None, cache=True,
                 lprefix=None, rprefix=None, match_missing=False):
        self.lkey = lkey
        self.rkey = rkey
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        if not presorted:
            self.left = sort(self.left, lkey, buffersize=buffersize,
                             tempdir=tempdir, cache=cache)
            self.right = sort(self.right, rkey, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        self.leftouter = leftouter
        self.rightouter = rightouter
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_missing = match_missing

    def __iter__(self):
        return iterjoin(self.left, self.right, self.lkey, self.rkey,
                        leftouter=self.leftouter, rightouter=self.rightouter,
                        missing=self.missing, lprefix=self.lprefix,
                        rprefix=self.rprefix,
                        match_missing=self.match_missing)


def leftjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
             presorted=False, buffersize=None, tempdir=None, cache=True,
             lprefix=None, rprefix=None, match_missing=False):
    """
    Perform a left outer join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.leftjoin(table1, table2, key='id')
        >>> table3
        +----+----------+----------+
        | id | colour   | shape    |
        +====+==========+==========+
        |  1 | 'blue'   | 'circle' |
        +----+----------+----------+
        |  2 | 'red'    | None     |
        +----+----------+----------+
        |  3 | 'purple' | 'square' |
        +----+----------+----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, leftouter=True, rightouter=False,
                    missing=missing, buffersize=buffersize, tempdir=tempdir,
                    cache=cache, lprefix=lprefix, rprefix=rprefix,
                    match_missing=match_missing)


Table.leftjoin = leftjoin


def rightjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
              presorted=False, buffersize=None, tempdir=None, cache=True,
              lprefix=None, rprefix=None, match_missing=False):
    """
    Perform a right outer join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.rightjoin(table1, table2, key='id')
        >>> table3
        +----+----------+-----------+
        | id | colour   | shape     |
        +====+==========+===========+
        |  1 | 'blue'   | 'circle'  |
        +----+----------+-----------+
        |  3 | 'purple' | 'square'  |
        +----+----------+-----------+
        |  4 | None     | 'ellipse' |
        +----+----------+-----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, leftouter=False, rightouter=True,
                    missing=missing, buffersize=buffersize,
                    tempdir=tempdir, cache=cache, lprefix=lprefix,
                    rprefix=rprefix, match_missing=match_missing)


Table.rightjoin = rightjoin


def outerjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
              presorted=False, buffersize=None, tempdir=None, cache=True,
              lprefix=None, rprefix=None, match_missing=False):
    """
    Perform a full outer join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.outerjoin(table1, table2, key='id')
        >>> table3
        +----+----------+-----------+
        | id | colour   | shape     |
        +====+==========+===========+
        |  1 | 'blue'   | 'circle'  |
        +----+----------+-----------+
        |  2 | 'red'    | None      |
        +----+----------+-----------+
        |  3 | 'purple' | 'square'  |
        +----+----------+-----------+
        |  4 | None     | 'ellipse' |
        +----+----------+-----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, leftouter=True, rightouter=True,
                    missing=missing, buffersize=buffersize, tempdir=tempdir,
                    cache=cache, lprefix=lprefix, rprefix=rprefix,
                    match_missing=match_missing)


Table.outerjoin = outerjoin


def iterjoin(left, right, lkey, rkey, leftouter=False, rightouter=False,
             missing=None, lprefix=None, rprefix=None, match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)

    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)

    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f)) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f)) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join two groups of rows
    def joinrows(_lrowgrp, _rrowgrp):
        if _rrowgrp is None:
            for lrow in _lrowgrp:
                outrow = list(lrow)  # start with the left row
                # extend with missing values in place of the right row
                outrow.extend([missing] * len(rvind))
                yield tuple(outrow)
        elif _lrowgrp is None:
            for rrow in _rrowgrp:
                # start with missing values in place of the left row
                outrow = [missing] * len(lhdr)
                # set key values
                for li, ri in zip(lkind, rkind):
                    outrow[li] = rrow[ri]
                # extend with non-key values from the right row
                outrow.extend(rgetv(rrow))
                yield tuple(outrow)
        else:
            _rrowgrp = list(_rrowgrp)  # may need to iterate more than once
            for lrow in _lrowgrp:
                for rrow in _rrowgrp:
                    # start with the left row
                    outrow = list(lrow)
                    # extend with non-key values from the right row
                    outrow.extend(rgetv(rrow))
                    yield tuple(outrow)

    # construct group iterators for both tables
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)
    lrowgrp = []
    rrowgrp = []

    # sentinel used only to distinguish "iterator exhausted" from a genuine
    # key, since MISSING and None are both real key values
    _END = object()

    def _next_group(group_iter):
        try:
            return next(group_iter)
        except StopIteration:
            return _END, None

    # loop until *both* of the iterators are exhausted
    # initialise here to handle empty tables
    lkval, rkval = Comparable(None), Comparable(None)
    lkval, lrowgrp = _next_group(lgit)
    rkval, rrowgrp = _next_group(rgit)

    while lkval is not _END and rkval is not _END:
        # When match_missing is off, a key containing any MISSING
        # component is unmatchable: do not even compare it against the
        # other side (no cartesian enumeration, no cross-side probing).
        # Such a group is consumed once and emitted or dropped purely by
        # the outer-join semantics, so work stays linear in the rows seen.
        if not match_missing and key_contains_missing(lkval.inner):
            if leftouter:
                for row in joinrows(list(lrowgrp), None):
                    yield tuple(row)
            lkval, lrowgrp = _next_group(lgit)
            continue
        if not match_missing and key_contains_missing(rkval.inner):
            if rightouter:
                for row in joinrows(None, list(rrowgrp)):
                    yield tuple(row)
            rkval, rrowgrp = _next_group(rgit)
            continue

        if lkval < rkval:
            if leftouter:
                for row in joinrows(list(lrowgrp), None):
                    yield tuple(row)
            # advance left
            lkval, lrowgrp = _next_group(lgit)
        elif lkval > rkval:
            if rightouter:
                for row in joinrows(None, list(rrowgrp)):
                    yield tuple(row)
            # advance right
            rkval, rrowgrp = _next_group(rgit)
        else:
            for row in joinrows(list(lrowgrp), rrowgrp):
                yield tuple(row)
            # advance both
            lkval, lrowgrp = _next_group(lgit)
            rkval, rrowgrp = _next_group(rgit)

    # make sure any left rows remaining are yielded
    if leftouter:
        if lkval is not _END:
            # yield the group that got left hanging plus the rest
            for row in joinrows(list(lrowgrp), None):
                yield tuple(row)
        for lkval, lrowgrp in lgit:
            for row in joinrows(list(lrowgrp), None):
                yield tuple(row)

    # make sure any right rows remaining are yielded
    if rightouter:
        if rkval is not _END:
            for row in joinrows(None, list(rrowgrp)):
                yield tuple(row)
        for rkval, rrowgrp in rgit:
            for row in joinrows(None, list(rrowgrp)):
                yield tuple(row)


def crossjoin(*tables, **kwargs):
    """
    Form the cartesian product of the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square']]
        >>> table3 = etl.crossjoin(table1, table2)
        >>> table3
        +----+--------+----+----------+
        | id | colour | id | shape    |
        +====+========+====+==========+
        |  1 | 'blue' |  1 | 'circle' |
        +----+--------+----+----------+
        |  1 | 'blue' |  3 | 'square' |
        +----+--------+----+----------+
        |  2 | 'red'  |  1 | 'circle' |
        +----+--------+----+----------+
        |  2 | 'red'  |  3 | 'square' |
        +----+--------+----+----------+

    If `prefix` is `True` then field names in the output table header will be
    prefixed by the index of the input table.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    return CrossJoinView(*tables, **kwargs)


Table.crossjoin = crossjoin


class CrossJoinView(Table):

    def __init__(self, *sources, **kwargs):
        self.prefix = kwargs.get('prefix', False)
        self.missing = kwargs.get('missing', None)
        self.sources = [stack(source, missing=self.missing) for source in sources]

    def __iter__(self):
        return itercrossjoin(self.sources, self.prefix)


def itercrossjoin(sources, prefix):

    # construct fields
    outhdr = list()
    for i, s in enumerate(sources):
        if prefix:
            # use one-based numbering
            outhdr.extend([text_type(i+1) + '_' + text_type(f) for f in header(s)])
        else:
            outhdr.extend(header(s))
    yield tuple(outhdr)

    datasrcs = [data(src) for src in sources]
    for prod in itertools.product(*datasrcs):
        outrow = list()
        for row in prod:
            outrow.extend(row)
        yield tuple(outrow)


def antijoin(left, right, key=None, lkey=None, rkey=None, presorted=False,
             buffersize=None, tempdir=None, cache=True, match_missing=False):
    """
    Return rows from the `left` table where the key value does not occur in
    the `right` table. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [0, 'black'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [4, 'yellow'],
        ...           [5, 'white']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square']]
        >>> table3 = etl.antijoin(table1, table2, key='id')
        >>> table3
        +----+----------+
        | id | colour   |
        +====+==========+
        |  0 | 'black'  |
        +----+----------+
        |  2 | 'red'    |
        +----+----------+
        |  4 | 'yellow' |
        +----+----------+
        |  5 | 'white'  |
        +----+----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return AntiJoinView(left=left, right=right, lkey=lkey, rkey=rkey,
                        presorted=presorted, buffersize=buffersize,
                        tempdir=tempdir, cache=cache,
                        match_missing=match_missing)


Table.antijoin = antijoin


class AntiJoinView(Table):

    def __init__(self, left, right, lkey, rkey, presorted=False,
                 buffersize=None, tempdir=None, cache=True,
                 match_missing=False):
        if presorted:
            self.left = left
            self.right = right
        else:
            self.left = sort(left, lkey, buffersize=buffersize,
                             tempdir=tempdir, cache=cache)
            self.right = sort(right, rkey, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        self.lkey = lkey
        self.rkey = rkey
        self.match_missing = match_missing

    def __iter__(self):
        return iterantijoin(self.left, self.right, self.lkey, self.rkey,
                            match_missing=self.match_missing)


def iterantijoin(left, right, lkey, rkey, match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)
    yield tuple(lhdr)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)

    # construct group iterators for both tables
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)
    lrowgrp = []

    _END = object()

    def _next_group(group_iter):
        try:
            return next(group_iter)
        except StopIteration:
            return _END, None

    # loop until *either* of the iterators is exhausted
    lkval, rkval = Comparable(None), Comparable(None)
    lkval, lrowgrp = _next_group(lgit)
    rkval, _ = _next_group(rgit)

    while lkval is not _END and rkval is not _END:
        # a MISSING-bearing key never matches unless match_missing is on,
        # so the left group is always retained by the anti-join
        if not match_missing and key_contains_missing(lkval.inner):
            for row in lrowgrp:
                yield tuple(row)
            lkval, lrowgrp = _next_group(lgit)
            continue
        if lkval < rkval:
            for row in lrowgrp:
                yield tuple(row)
            # advance left
            lkval, lrowgrp = _next_group(lgit)
        elif lkval > rkval:
            # advance right
            rkval, _ = _next_group(rgit)
        else:
            # advance both
            lkval, lrowgrp = _next_group(lgit)
            rkval, _ = _next_group(rgit)

    # any left over (including groups the right side never reached)?
    if lkval is not _END:
        # yield the group that got left hanging plus the remainder
        for row in lrowgrp:
            yield tuple(row)
    for lkval, lrowgrp in lgit:
        for row in lrowgrp:
            yield tuple(row)


def lookupjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
               presorted=False, buffersize=None, tempdir=None, cache=True,
               lprefix=None, rprefix=None, match_missing=False):
    """
    Perform a left join, but where the key is not unique in the right-hand
    table, arbitrarily choose the first row and ignore others. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'color', 'cost'],
        ...           [1, 'blue', 12],
        ...           [2, 'red', 8],
        ...           [3, 'purple', 4]]
        >>> table2 = [['id', 'shape', 'size'],
        ...           [1, 'circle', 'big'],
        ...           [1, 'circle', 'small'],
        ...           [2, 'square', 'tiny'],
        ...           [2, 'square', 'big'],
        ...           [3, 'ellipse', 'small'],
        ...           [3, 'ellipse', 'tiny']]
        >>> table3 = etl.lookupjoin(table1, table2, key='id')
        >>> table3
        +----+----------+------+-----------+---------+
        | id | color    | cost | shape     | size    |
        +====+==========+======+===========+=========+
        |  1 | 'blue'   |   12 | 'circle'  | 'big'   |
        +----+----------+------+-----------+---------+
        |  2 | 'red'    |    8 | 'square'  | 'tiny'  |
        +----+----------+------+-----------+---------+
        |  3 | 'purple' |    4 | 'ellipse' | 'small' |
        +----+----------+------+-----------+---------+

    See also :func:`petl.transform.joins.leftjoin`.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return LookupJoinView(left, right, lkey, rkey, presorted=presorted,
                          missing=missing, buffersize=buffersize,
                          tempdir=tempdir, cache=cache,
                          lprefix=lprefix, rprefix=rprefix,
                          match_missing=match_missing)


Table.lookupjoin = lookupjoin


class LookupJoinView(Table):

    def __init__(self, left, right, lkey, rkey, presorted=False, missing=None,
                 buffersize=None, tempdir=None, cache=True,
                 lprefix=None, rprefix=None, match_missing=False):
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        if not presorted:
            self.left = sort(self.left, lkey, buffersize=buffersize,
                             tempdir=tempdir, cache=cache)
            self.right = sort(self.right, rkey, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_missing = match_missing

    def __iter__(self):
        return iterlookupjoin(self.left, self.right, self.lkey, self.rkey,
                              missing=self.missing, lprefix=self.lprefix,
                              rprefix=self.rprefix,
                              match_missing=self.match_missing)


def iterlookupjoin(left, right, lkey, rkey, missing=None, lprefix=None,
                   rprefix=None, match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)

    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)

    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f)) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f)) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join two groups of rows
    def joinrows(_lrowgrp, _rrowgrp):
        if _rrowgrp is None:
            for lrow in _lrowgrp:
                outrow = list(lrow)  # start with the left row
                # extend with missing values in place of the right row
                outrow.extend([missing] * len(rvind))
                yield tuple(outrow)
        else:
            rrow = next(iter(_rrowgrp))  # pick first arbitrarily
            for lrow in _lrowgrp:
                # start with the left row
                outrow = list(lrow)
                # extend with non-key values from the right row
                outrow.extend(rgetv(rrow))
                yield tuple(outrow)

    # construct group iterators for both tables
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)
    lrowgrp = []

    _END = object()

    def _next_group(group_iter):
        try:
            return next(group_iter)
        except StopIteration:
            return _END, None

    # loop until *either* of the iterators is exhausted
    lkval, lrowgrp = _next_group(lgit)
    rkval, rrowgrp = _next_group(rgit)

    while lkval is not _END and rkval is not _END:
        # MISSING-bearing keys never find a right row unless
        # match_missing is enabled; emit the left-padded row once.
        if not match_missing and key_contains_missing(lkval.inner):
            for row in joinrows(list(lrowgrp), None):
                yield tuple(row)
            lkval, lrowgrp = _next_group(lgit)
            continue
        if lkval < rkval:
            for row in joinrows(list(lrowgrp), None):
                yield tuple(row)
            # advance left
            lkval, lrowgrp = _next_group(lgit)
        elif lkval > rkval:
            # advance right
            rkval, rrowgrp = _next_group(rgit)
        else:
            for row in joinrows(list(lrowgrp), rrowgrp):
                yield tuple(row)
            # advance both
            lkval, lrowgrp = _next_group(lgit)
            rkval, rrowgrp = _next_group(rgit)

    # make sure any left rows remaining are yielded (lookup join is always
    # left-outer); the last picked-off group plus anything left in the pipe
    if lkval is not _END:
        for row in joinrows(list(lrowgrp), None):
            yield tuple(row)
    for lkval, lrowgrp in lgit:
        for row in joinrows(list(lrowgrp), None):
            yield tuple(row)


def unjoin(table, value, key=None, autoincrement=(1, 1), presorted=False,
           buffersize=None, tempdir=None, cache=True):
    """
    Split a table into two tables by reversing an inner join. E.g.::

        >>> import petl as etl
        >>> # join key is present in the table
        ... table1 = (('foo', 'bar', 'baz'),
        ...           ('A', 1, 'apple'),
        ...           ('B', 1, 'apple'),
        ...           ('C', 2, 'orange'))
        >>> table2, table3 = etl.unjoin(table1, 'baz', key='bar')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'B' |   1 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+

        >>> table3
        +-----+----------+
        | bar | baz      |
        +=====+==========+
        |   1 | 'apple'  |
        +-----+----------+
        |   2 | 'orange' |
        +-----+----------+

        >>> # an integer join key can also be reconstructed
        ... table4 = (('foo', 'bar'),
        ...           ('A', 'apple'),
        ...           ('B', 'apple'),
        ...           ('C', 'orange'))
        >>> table5, table6 = etl.unjoin(table4, 'bar')
        >>> table5
        +-----+--------+
        | foo | bar_id |
        +=====+========+
        | 'A' |      1 |
        +-----+--------+
        | 'B' |      1 |
        +-----+--------+
        | 'C' |      2 |
        +-----+--------+

        >>> table6
        +----+----------+
        | id | bar      |
        +====+==========+
        |  1 | 'apple'  |
        +----+----------+
        |  2 | 'orange' |
        +----+----------+

    The `autoincrement` parameter controls how an integer join key is
    reconstructed, and should be a tuple of (`start`, `step`).

    """

    if key is None:
        # first sort the table by the value field
        if presorted:
            tbl_sorted = table
        else:
            tbl_sorted = sort(table, value, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        # on the left, return the original table but with the value field
        # replaced by an incrementing integer
        left = ConvertToIncrementingCounterView(tbl_sorted, value,
                                                autoincrement)
        # on the right, return a new table with distinct values from the
        # given field
        right = EnumerateDistinctView(tbl_sorted, value, autoincrement)
    else:
        # on the left, return distinct rows from the original table
        # with the value field cut out
        left = distinct(cutout(table, value))
        # on the right, return distinct rows from the original table
        # with all fields but the key and value cut out
        right = distinct(cut(table, key, value))
    return left, right


class ConvertToIncrementingCounterView(Table):

    def __init__(self, tbl, value, autoincrement):
        self.table = tbl
        self.value = value
        self.autoincrement = autoincrement

    def __iter__(self):
        it = iter(self.table)
        hdr = next(it)
        table = itertools.chain([hdr], it)
        value = self.value
        vidx = hdr.index(value)
        outhdr = list(hdr)
        outhdr[vidx] = '%s_id' % value
        yield tuple(outhdr)
        offset, multiplier = self.autoincrement
        for n, (_, group) in enumerate(rowgroupby(table, value)):
            for row in group:
                outrow = list(row)
                outrow[vidx] = (n * multiplier) + offset
                yield tuple(outrow)


Table.unjoin = unjoin


class EnumerateDistinctView(Table):

    def __init__(self, tbl, value, autoincrement):
        self.table = tbl
        self.value = value
        self.autoincrement = autoincrement

    def __iter__(self):
        offset, multiplier = self.autoincrement
        yield ('id', self.value)
        for n, (v, _) in enumerate(rowgroupby(self.table, self.value)):
            yield ((n * multiplier) + offset, v)
