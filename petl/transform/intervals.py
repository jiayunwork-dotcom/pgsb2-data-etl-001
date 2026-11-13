from __future__ import absolute_import, print_function, division


import math
import os
import sqlite3
import tempfile
from decimal import Decimal, localcontext
from operator import itemgetter, attrgetter
from petl.compat import text_type, string_types, pickle


from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.comparison import Comparable
from petl.errors import ArgumentError, DuplicateKeyError, FieldSelectionError
from petl.transform.basics import addfield
from petl.transform.sorts import sort


def tupletree(table, start='start', stop='stop', value=None):
    """
    Construct an interval tree for the given table, where each node in the tree
    is a row of the table.

    """

    import intervaltree
    tree = intervaltree.IntervalTree()
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    assert start in flds, 'start field not recognised'
    assert stop in flds, 'stop field not recognised'
    getstart = itemgetter(flds.index(start))
    getstop = itemgetter(flds.index(stop))
    if value is None:
        getvalue = tuple
    else:
        valueindices = asindices(hdr, value)
        assert len(valueindices) > 0, 'invalid value field specification'
        getvalue = itemgetter(*valueindices)
    for row in it:
        tree.addi(getstart(row), getstop(row), getvalue(row))
    return tree


def facettupletrees(table, key, start='start', stop='stop', value=None):
    """
    Construct faceted interval trees for the given table, where each node in
    the tree is a row of the table.

    """

    import intervaltree
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    assert start in flds, 'start field not recognised'
    assert stop in flds, 'stop field not recognised'
    getstart = itemgetter(flds.index(start))
    getstop = itemgetter(flds.index(stop))
    if value is None:
        getvalue = tuple
    else:
        valueindices = asindices(hdr, value)
        assert len(valueindices) > 0, 'invalid value field specification'
        getvalue = itemgetter(*valueindices)
    keyindices = asindices(hdr, key)
    assert len(keyindices) > 0, 'invalid key'
    getkey = itemgetter(*keyindices)

    trees = dict()
    for row in it:
        k = getkey(row)
        if k not in trees:
            trees[k] = intervaltree.IntervalTree()
        trees[k].addi(getstart(row), getstop(row), getvalue(row))
    return trees


def recordtree(table, start='start', stop='stop'):
    """
    Construct an interval tree for the given table, where each node in the
    tree is a row of the table represented as a record object.

    """

    import intervaltree
    getstart = attrgetter(start)
    getstop = attrgetter(stop)
    tree = intervaltree.IntervalTree()
    for rec in records(table):
        tree.addi(getstart(rec), getstop(rec), rec)
    return tree


def facetrecordtrees(table, key, start='start', stop='stop'):
    """
    Construct faceted interval trees for the given table, where each node in 
    the tree is a record.

    """

    import intervaltree
    getstart = attrgetter(start)
    getstop = attrgetter(stop)
    getkey = attrgetter(key)
    trees = dict()
    for rec in records(table):
        k = getkey(rec)
        if k not in trees:
            trees[k] = intervaltree.IntervalTree()
        trees[k].addi(getstart(rec), getstop(rec), rec)
    return trees


def intervallookup(table, start='start', stop='stop', value=None,
                   include_stop=False):
    """
    Construct an interval lookup for the given table. E.g.::

        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookup(table, 'start', 'stop')
        >>> lkp.search(0, 1)
        []
        >>> lkp.search(1, 2)
        [(1, 4, 'foo')]
        >>> lkp.search(2, 4)
        [(1, 4, 'foo'), (3, 7, 'bar')]
        >>> lkp.search(2, 5)
        [(1, 4, 'foo'), (3, 7, 'bar'), (4, 9, 'baz')]
        >>> lkp.search(9, 14)
        []
        >>> lkp.search(19, 140)
        []
        >>> lkp.search(0)
        []
        >>> lkp.search(1)
        [(1, 4, 'foo')]
        >>> lkp.search(2)
        [(1, 4, 'foo')]
        >>> lkp.search(4)
        [(3, 7, 'bar'), (4, 9, 'baz')]
        >>> lkp.search(5)
        [(3, 7, 'bar'), (4, 9, 'baz')]

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    Some examples using the `include_stop` and `value` keyword arguments::
    
        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookup(table, 'start', 'stop', include_stop=True,
        ...                          value='value')
        >>> lkp.search(0, 1)
        ['foo']
        >>> lkp.search(1, 2)
        ['foo']
        >>> lkp.search(2, 4)
        ['foo', 'bar', 'baz']
        >>> lkp.search(2, 5)
        ['foo', 'bar', 'baz']
        >>> lkp.search(9, 14)
        ['baz']
        >>> lkp.search(19, 140)
        []
        >>> lkp.search(0)
        []
        >>> lkp.search(1)
        ['foo']
        >>> lkp.search(2)
        ['foo']
        >>> lkp.search(4)
        ['foo', 'bar', 'baz']
        >>> lkp.search(5)
        ['bar', 'baz']

    """

    tree = tupletree(table, start=start, stop=stop, value=value)
    return IntervalTreeLookup(tree, include_stop=include_stop)


Table.intervallookup = intervallookup


def _search_tree(tree, start, stop, include_stop):
    if stop is None:
        if include_stop:
            stop = start + 1
            start -= 1
            args = (start, stop)
        else:
            args = (start,)
    else:
        if include_stop:
            stop += 1
            start -= 1
        args = (start, stop)
    if len(args) == 2:
        results = sorted(tree.overlap(*args))
    else:
        results = sorted(tree.at(*args))
    return results


class IntervalTreeLookup(object):
    
    def __init__(self, tree, include_stop=False):
        self.tree = tree
        self.include_stop = include_stop

    def search(self, start, stop=None):
        results = _search_tree(self.tree, start, stop, self.include_stop)
        return [r.data for r in results]

    find = search


def intervallookupone(table, start='start', stop='stop', value=None,
                      include_stop=False, strict=True):
    """
    Construct an interval lookup for the given table, returning at most one
    result for each query. E.g.::

        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookupone(table, 'start', 'stop', strict=False)
        >>> lkp.search(0, 1)
        >>> lkp.search(1, 2)
        (1, 4, 'foo')
        >>> lkp.search(2, 4)
        (1, 4, 'foo')
        >>> lkp.search(2, 5)
        (1, 4, 'foo')
        >>> lkp.search(9, 14)
        >>> lkp.search(19, 140)
        >>> lkp.search(0)
        >>> lkp.search(1)
        (1, 4, 'foo')
        >>> lkp.search(2)
        (1, 4, 'foo')
        >>> lkp.search(4)
        (3, 7, 'bar')
        >>> lkp.search(5)
        (3, 7, 'bar')

    If ``strict=True``, queries returning more than one result will
    raise a `DuplicateKeyError`. If ``strict=False`` and there is
    more than one result, the first result is returned.

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """

    tree = tupletree(table, start=start, stop=stop, value=value)
    return IntervalTreeLookupOne(tree, strict=strict, include_stop=include_stop)


Table.intervallookupone = intervallookupone


class IntervalTreeLookupOne(object):
    
    def __init__(self, tree, strict=True, include_stop=False):
        self.tree = tree
        self.strict = strict
        self.include_stop = include_stop
        
    def search(self, start, stop=None):
        results = _search_tree(self.tree, start, stop, self.include_stop)
        if len(results) == 0:
            return None
        elif len(results) > 1 and self.strict:
            raise DuplicateKeyError((start, stop))
        else:
            return results[0].data

    find = search


def intervalrecordlookup(table, start='start', stop='stop', include_stop=False):
    """
    As :func:`petl.transform.intervals.intervallookup` but return records
    instead of tuples.

    """

    tree = recordtree(table, start=start, stop=stop)
    return IntervalTreeLookup(tree, include_stop=include_stop)


Table.intervalrecordlookup = intervalrecordlookup


def intervalrecordlookupone(table, start='start', stop='stop',
                            include_stop=False, strict=True):
    """
    As :func:`petl.transform.intervals.intervallookupone` but return records
    instead of tuples.

    """

    tree = recordtree(table, start=start, stop=stop)
    return IntervalTreeLookupOne(tree, include_stop=include_stop, strict=strict)


Table.intervalrecordlookupone = intervalrecordlookupone


def facetintervallookup(table, key, start='start', stop='stop',
                        value=None, include_stop=False):
    """

    Construct a faceted interval lookup for the given table. E.g.::

        >>> import petl as etl
        >>> table = (('type', 'start', 'stop', 'value'),
        ...          ('apple', 1, 4, 'foo'),
        ...          ('apple', 3, 7, 'bar'),
        ...          ('orange', 4, 9, 'baz'))
        >>> lkp = etl.facetintervallookup(table, key='type', start='start', stop='stop')
        >>> lkp['apple'].search(1, 2)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(2, 4)
        [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
        >>> lkp['apple'].search(2, 5)
        [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
        >>> lkp['orange'].search(2, 5)
        [('orange', 4, 9, 'baz')]
        >>> lkp['orange'].search(9, 14)
        []
        >>> lkp['orange'].search(19, 140)
        []
        >>> lkp['apple'].search(1)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(2)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(4)
        [('apple', 3, 7, 'bar')]
        >>> lkp['apple'].search(5)
        [('apple', 3, 7, 'bar')]
        >>> lkp['orange'].search(5)
        [('orange', 4, 9, 'baz')]

    """

    trees = facettupletrees(table, key, start=start, stop=stop, value=value)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookup(trees[k], include_stop=include_stop)
    return out


Table.facetintervallookup = facetintervallookup


def facetintervallookupone(table, key, start='start', stop='stop',
                           value=None, include_stop=False, strict=True):
    """
    Construct a faceted interval lookup for the given table, returning at most
    one result for each query.
    
    If ``strict=True``, queries returning more than one result will
    raise a `DuplicateKeyError`. If ``strict=False`` and there is
    more than one result, the first result is returned.

    """
    
    trees = facettupletrees(table, key, start=start, stop=stop, value=value)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookupOne(trees[k], include_stop=include_stop,
                                       strict=strict)
    return out


Table.facetintervallookupone = facetintervallookupone


def facetintervalrecordlookup(table, key, start='start', stop='stop',
                              include_stop=False):
    """
    As :func:`petl.transform.intervals.facetintervallookup` but return records.
    
    """

    trees = facetrecordtrees(table, key, start=start, stop=stop)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookup(trees[k], include_stop=include_stop)
    return out


Table.facetintervalrecordlookup = facetintervalrecordlookup


def facetintervalrecordlookupone(table, key, start, stop, include_stop=False,
                                 strict=True):
    """
    As :func:`petl.transform.intervals.facetintervallookupone` but return
    records.

    """
    
    trees = facetrecordtrees(table, key, start=start, stop=stop)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookupOne(trees[k], include_stop=include_stop,
                                       strict=strict)
    return out


Table.facetintervalrecordlookupone = facetintervalrecordlookupone


def intervaljoin(left, right, lstart='start', lstop='stop', rstart='start',
                 rstop='stop', lkey=None, rkey=None, include_stop=False,
                 lprefix=None, rprefix=None):
    """
    Join two tables by overlapping intervals. E.g.::

        >>> import petl as etl
        >>> left = [['begin', 'end', 'quux'],
        ...         [1, 2, 'a'],
        ...         [2, 4, 'b'],
        ...         [2, 5, 'c'],
        ...         [9, 14, 'd'],
        ...         [1, 1, 'e'],
        ...         [10, 10, 'f']]
        >>> right = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> table1 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end',
        ...                           rstart='start', rstop='stop')
        >>> table1.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+

        >>> # include stop coordinate in intervals
        ... table2 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end',
        ...                           rstart='start', rstop='stop',
        ...                           include_stop=True)
        >>> table2.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     9 |  14 | 'd'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     1 |   1 | 'e'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    An additional key comparison can be made, e.g.::
    
        >>> import petl as etl
        >>> left = (('fruit', 'begin', 'end'),
        ...         ('apple', 1, 2),
        ...         ('apple', 2, 4),
        ...         ('apple', 2, 5),
        ...         ('orange', 2, 5),
        ...         ('orange', 9, 14),
        ...         ('orange', 19, 140),
        ...         ('apple', 1, 1))
        >>> right = (('type', 'start', 'stop', 'value'),
        ...          ('apple', 1, 4, 'foo'),
        ...          ('apple', 3, 7, 'bar'),
        ...          ('orange', 4, 9, 'baz'))
        >>> table3 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end', lkey='fruit',
        ...                           rstart='start', rstop='stop', rkey='type')
        >>> table3.lookall()
        +----------+-------+-----+----------+-------+------+-------+
        | fruit    | begin | end | type     | start | stop | value |
        +==========+=======+=====+==========+=======+======+=======+
        | 'apple'  |     1 |   2 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   4 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   4 | 'apple'  |     3 |    7 | 'bar' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   5 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   5 | 'apple'  |     3 |    7 | 'bar' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'orange' |     2 |   5 | 'orange' |     4 |    9 | 'baz' |
        +----------+-------+-----+----------+-------+------+-------+

    """
    
    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalJoinView(left, right, lstart=lstart, lstop=lstop,
                            rstart=rstart, rstop=rstop, lkey=lkey,
                            rkey=rkey, include_stop=include_stop,
                            lprefix=lprefix, rprefix=rprefix)


Table.intervaljoin = intervaljoin


class IntervalJoinView(Table):
    
    def __init__(self, left, right, lstart='start', lstop='stop', 
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 include_stop=False, lprefix=None, rprefix=None):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.include_stop = include_stop
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterintervaljoin(
            left=self.left,
            right=self.right,
            lstart=self.lstart,
            lstop=self.lstop,
            rstart=self.rstart,
            rstop=self.rstop,
            lkey=self.lkey,
            rkey=self.rkey,
            include_stop=self.include_stop,
            missing=None,
            lprefix=self.lprefix,
            rprefix=self.rprefix,
            leftouter=False
        )
        

def intervalleftjoin(left, right, lstart='start', lstop='stop', rstart='start',
                     rstop='stop', lkey=None, rkey=None, include_stop=False,
                     missing=None, lprefix=None, rprefix=None):
    """
    Like :func:`petl.transform.intervals.intervaljoin` but rows from the left 
    table without a match in the right table are also included. E.g.::

        >>> import petl as etl
        >>> left = [['begin', 'end', 'quux'],
        ...         [1, 2, 'a'],
        ...         [2, 4, 'b'],
        ...         [2, 5, 'c'],
        ...         [9, 14, 'd'],
        ...         [1, 1, 'e'],
        ...         [10, 10, 'f']]
        >>> right = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> table1 = etl.intervalleftjoin(left, right,
        ...                               lstart='begin', lstop='end',
        ...                               rstart='start', rstop='stop')
        >>> table1.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     9 |  14 | 'd'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+
        |     1 |   1 | 'e'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+
        |    10 |  10 | 'f'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    
    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalLeftJoinView(left, right, lstart=lstart, lstop=lstop,
                                rstart=rstart, rstop=rstop, lkey=lkey,
                                rkey=rkey, include_stop=include_stop,
                                missing=missing, lprefix=lprefix,
                                rprefix=rprefix)


Table.intervalleftjoin = intervalleftjoin


class IntervalLeftJoinView(Table):
    
    def __init__(self, left, right, lstart='start', lstop='stop', 
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 missing=None, include_stop=False, lprefix=None, rprefix=None):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.missing = missing
        self.include_stop = include_stop
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterintervaljoin(
            left=self.left,
            right=self.right,
            lstart=self.lstart,
            lstop=self.lstop,
            rstart=self.rstart,
            rstop=self.rstop,
            lkey=self.lkey,
            rkey=self.rkey,
            include_stop=self.include_stop,
            missing=self.missing,
            lprefix=self.lprefix,
            rprefix=self.rprefix,
            leftouter=True
        )
        

def intervaljoinstream(left, right, lstart='start', lstop='stop',
                       rstart='start', rstop='stop', by=None, how='inner',
                       presorted=False, buffersize=1000, tempdir=None,
                       lprefix=None, rprefix='right_', missing=None):
    """
    Join two tables by overlapping intervals, with bounded in-memory state.

    Intervals use half-open semantics: start coordinates are included and stop
    coordinates are excluded.  Touching endpoints therefore do not overlap.
    Rows are paired only when their group values are equal and their intervals
    have a non-empty intersection.  Every pair is emitted, including pairs
    produced by duplicated rows.

    Unlike :func:`intervaljoin`, this transform does not build an in-memory
    interval tree for the whole right table.  When the inputs are not already
    sorted, rows are staged in a temporary on-disk SQLite database and merged
    with a bounded number of records retained in memory.  When
    ``presorted=True`` the inputs are consumed in their supplied order and
    matches are emitted as soon as the relevant interval boundaries have been
    confirmed.

    The output contains all left fields followed by the right non-group
    fields.  Rows are ordered by the group fields and left start coordinate;
    matches belonging to one left row are contiguous and ordered by right
    start, retaining original input order for equal starts.  With
    ``how='left'``, an unmatched left row is emitted once with right fields
    set to `missing`.

    The view is lazy: constructing it and calling :func:`iter` do not read
    either source.  Each traversal requests one iterator from each source and
    uses independent join state.  Unsorted inputs are staged in a temporary
    SQLite database with bounded in-memory cache, spilling to `tempdir` when
    needed; with ``presorted=True`` no whole-table sort is performed and
    matches are emitted after only the lookahead needed to establish interval
    boundaries.

    """

    return IntervalJoinStreamView(
        left, right, lstart=lstart, lstop=lstop, rstart=rstart,
        rstop=rstop, by=by, how=how, presorted=presorted,
        buffersize=buffersize, tempdir=tempdir, lprefix=lprefix,
        rprefix=rprefix, missing=missing)


Table.intervaljoinstream = intervaljoinstream


class IntervalJoinStreamView(Table):

    def __init__(self, left, right, lstart='start', lstop='stop',
                 rstart='start', rstop='stop', by=None, how='inner',
                 presorted=False, buffersize=1000, tempdir=None,
                 lprefix=None, rprefix='right_', missing=None):
        self.left = left
        self.right = right
        self.lstart = lstart
        self.lstop = lstop
        self.rstart = rstart
        self.rstop = rstop
        self.by = by
        self.how = how
        self.presorted = presorted
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.missing = missing

    def __iter__(self):
        return iterintervaljoinstream(
            self.left, self.right, self.lstart, self.lstop, self.rstart,
            self.rstop, self.by, self.how, self.presorted, self.buffersize,
            self.tempdir, self.lprefix, self.rprefix, self.missing)

    def __len__(self):
        # ``list(view)`` otherwise uses the inherited length hint and performs
        # a complete extra source traversal before allocation.
        raise TypeError('length of a streaming interval join is not known '
                        'without iterating it')


def _stream_check_buffersize(buffersize):
    if not isinstance(buffersize, int) or isinstance(buffersize, bool) \
            or buffersize < 1:
        raise ArgumentError(
            'buffersize must be a positive integer, got %r'
            % (buffersize,))


def _stream_group_fields(by):
    if by is None:
        return ()
    if isinstance(by, string_types):
        fields = (by,)
    else:
        try:
            fields = tuple(by)
        except TypeError:
            raise ArgumentError(
                'by must be a field name or non-empty sequence of field names')
    if not fields:
        raise ArgumentError(
            'by must be a field name or non-empty sequence of field names')
    for i, field in enumerate(fields):
        if not isinstance(field, string_types):
            raise ArgumentError('invalid group field: %r' % (field,))
        if field in fields[:i]:
            raise ArgumentError('duplicate group field: %r' % (field,))
    return fields


def _stream_single_field(name, label):
    if not isinstance(name, string_types):
        raise ArgumentError('%s must be a single field name: %r'
                            % (label, name))
    return name


def _stream_header(hdr, side):
    try:
        fields = tuple(hdr)
    except TypeError:
        raise ArgumentError('%s header must be a sequence of strings' % side)
    if not all(isinstance(f, str) for f in fields):
        raise ArgumentError('%s header fields must be strings' % side)
    if not fields:
        raise ArgumentError('%s table has no header row' % side)
    for i, field in enumerate(fields):
        if field in fields[:i]:
            raise ArgumentError(
                'duplicate %s header field: %r' % (side, field))
    return fields


def _stream_field_index(fields, name, side, label):
    try:
        return fields.index(name)
    except ValueError:
        raise FieldSelectionError(
            '%s field %r not found in %s header %r'
            % (label, name, side, fields))


def _stream_check_width(row, rownum, side, width):
    if len(row) != width:
        raise ArgumentError(
            '%s data row %d has %d values but the header has %d fields'
            % (side, rownum, len(row), width))


def _stream_check_group(value, rownum, side):
    if value is not None and not isinstance(value, (str, int)):
        raise ArgumentError(
            '%s data row %d has an invalid group value: %r'
            % (side, rownum, value))
    if isinstance(value, bool):
        raise ArgumentError(
            '%s data row %d has an invalid group value: %r'
            % (side, rownum, value))


def _stream_check_coordinate(value, rownum, side, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ArgumentError(
            '%s data row %d has an invalid %s coordinate: %r'
            % (side, rownum, label, value))


def _stream_check_interval(row, rownum, side, startidx, stopidx, groupidx):
    start = row[startidx]
    stop = row[stopidx]
    _stream_check_coordinate(start, rownum, side, 'start')
    _stream_check_coordinate(stop, rownum, side, 'stop')
    if not start < stop:
        raise ArgumentError(
            '%s data row %d has start >= stop: %r >= %r'
            % (side, rownum, start, stop))
    for idx in groupidx:
        _stream_check_group(row[idx], rownum, side)


def _stream_output_header(lfields, rfields, groupindices, lprefix, rprefix):
    if lprefix is None:
        out = list(lfields)
    else:
        out = [lprefix + f for f in lfields]
    for idx, field in enumerate(rfields):
        if idx not in groupindices:
            out.append(field if rprefix is None else rprefix + field)
    for i, field in enumerate(out):
        if field in out[:i]:
            raise ArgumentError('duplicate output field: %r' % (field,))
    return tuple(out)


def _stream_order_key(group, start):
    # Allowed group values map naturally onto Comparable's existing
    # relaxed ordering semantics.
    if len(group) == 1:
        return (Comparable(group[0]), start)
    return (Comparable(group), start)


def _stream_validate_row(row, rownum, side, width, startidx, stopidx,
                         groupindices):
    row = tuple(row)
    _stream_check_width(row, rownum, side, width)
    _stream_check_interval(row, rownum, side, startidx, stopidx, groupindices)
    return row


def _stream_encode_unsigned(value):
    # One leading length byte followed by big-endian value bytes keeps both
    # small and multi-byte integers lexically ordered.
    body = []
    while value:
        body.append(value & 0xff)
        value >>= 8
    body.reverse()
    return bytes((len(body),)) + bytes(body)


def _stream_encode_signed(value):
    encoded = _stream_encode_unsigned(abs(value))
    if value < 0:
        return b'\x00' + encoded
    if value == 0:
        return b'\x01' + encoded
    return b'\x02' + encoded


def _stream_numeric_key(value):
    if value == 0:
        return b'\x01'
    with localcontext() as context:
        # Avoid Decimal's default 28-digit precision truncating very large
        # integer coordinates during normalization.
        context.prec = max(40, len(str(abs(int(value))))
                           if isinstance(value, int) else 40)
        decimal_value = Decimal(repr(float(value))
                                if isinstance(value, float) else value)
        sign, digits, exponent = decimal_value.normalize().as_tuple()
    digit_text = ''.join(str(d) for d in digits)
    magnitude_exponent = exponent + len(digits) - 1
    width = max(39, len(digit_text))
    padded_digits = digit_text.ljust(width, chr(0)).encode('ascii')
    body = (_stream_encode_signed(magnitude_exponent) + bytes((0xff,))
            + padded_digits)
    if sign:
        return b'\x00' + bytes(255 - b for b in body)
    return b'\x02' + body


def _stream_open_database(tempdir, buffersize):
    if tempdir is not None and (not isinstance(tempdir, string_types)
                                or not os.path.isdir(tempdir)):
        raise ArgumentError('tempdir must be an existing directory or None')
    # One database file and one connection mean the number of open data files
    # does not depend on the number of sort chunks or emitted matches.
    fd, filename = tempfile.mkstemp(prefix='petl-interval-join-',
                                    suffix='.sqlite', dir=tempdir)
    os.close(fd)
    connection = sqlite3.connect(filename, isolation_level=None)
    connection.execute('PRAGMA journal_mode = OFF')
    connection.execute('PRAGMA synchronous = OFF')
    # Keep SQLite's own page cache small and fixed; records above the caller's
    # buffer remain on disk rather than in Python lists.
    connection.execute('PRAGMA cache_size = %d' % min(buffersize, 16))
    connection.execute('PRAGMA temp_store = FILE')
    if tempdir is not None:
        connection.execute('PRAGMA temp_store_directory = %s'
                           % _stream_sqlite_quote(tempdir))
    return connection, filename


def _stream_sqlite_quote(value):
    return "'" + value.replace("'", "''") + "'"


def _stream_group_columns(prefix, ngroups):
    if not ngroups:
        return ''
    return ', ' + ', '.join('%sg%d TEXT' % (prefix, i)
                            for i in range(ngroups))


def _stream_prepare_database(connection, ngroups):
    connection.execute(
        'CREATE TABLE right_stage ('
        'id INTEGER PRIMARY KEY, seq INTEGER, start_key BLOB, stop_key BLOB'
        + _stream_group_columns('', ngroups) + ', row BLOB)')
    connection.execute(
        'CREATE TABLE left_stage ('
        'id INTEGER PRIMARY KEY, seq INTEGER, start_key BLOB, stop_key BLOB'
        + _stream_group_columns('', ngroups) + ', row BLOB)')
    index_columns = ['start_key', 'seq']
    if ngroups:
        index_columns = ['g%d' % i for i in range(ngroups)] + index_columns
    connection.execute(
        'CREATE INDEX right_stage_lookup ON right_stage(%s)'
        % ', '.join(index_columns))


def _stream_group_sql(ngroups):
    if not ngroups:
        return '', []
    cols = ', '.join('g%d' % i for i in range(ngroups))
    return ', ' + cols, [None] * ngroups


def _stream_insert(connection, table, record_id, seq, start, stop, groups,
                   row, ngroups):
    gcols, _ = _stream_group_sql(ngroups)
    placeholders = ', '.join('?' for _ in range(5 + ngroups))
    connection.execute(
        'INSERT INTO %s (id, seq, start_key, stop_key%s, row) VALUES (%s)'
        % (table, gcols, placeholders),
        [record_id, seq, _stream_numeric_key(start),
         _stream_numeric_key(stop)] + list(groups)
        + [pickle.dumps(row, protocol=-1)])


def _stream_load_sorted_table(connection, iterator, side, width, startidx,
                              stopidx, groupindices, ngroups, buffersize,
                              table, presorted):
    rownum = 0
    in_transaction = False
    try:
        for row in iterator:
            if not in_transaction:
                connection.execute('BEGIN')
                in_transaction = True
            rownum += 1
            row = _stream_validate_row(row, rownum, side, width, startidx,
                                       stopidx, groupindices)
            groups = tuple(row[i] for i in groupindices)
            _stream_insert(connection, table, rownum, rownum,
                           row[startidx], row[stopidx], groups, row, ngroups)
            if rownum % buffersize == 0:
                connection.execute('COMMIT')
                in_transaction = False
        if in_transaction:
            connection.execute('COMMIT')
    except Exception:
        if in_transaction:
            connection.rollback()
        raise
    if not presorted and table != 'right_stage':
        order = ', '.join(['g%d' % i for i in range(ngroups)] +
                          ['start_key', 'seq'])
        connection.execute(
            'CREATE INDEX %s_order ON %s(%s)'
            % (table, table, order))


def _stream_group_clause(ngroups, suffix):
    if not ngroups:
        return ''
    return ' AND '.join('g%d IS ?%s' % (i, suffix) for i in range(ngroups))


def _stream_active_matches(connection, groups, start, stop, ngroups):
    start_key = _stream_numeric_key(start)
    stop_key = _stream_numeric_key(stop)
    where = ['start_key < ?', 'stop_key > ?']
    values = [stop_key, start_key]
    clause = _stream_group_clause(ngroups, '')
    if clause:
        where.append(clause)
        values.extend(groups)
    sql = ('SELECT row FROM right_stage WHERE ' + ' AND '.join(where)
           + ' ORDER BY start_key, seq')
    cursor = connection.execute(sql, values)
    # The sqlite3 module pre-fetches this many rows; keep it fixed so the
    # number of rows held for a large match set does not grow with input size.
    cursor.arraysize = 32
    return cursor


def _stream_presorted_groups(row, groupindices):
    return tuple(row[i] for i in groupindices) if groupindices else ()


def _stream_check_order(previous, current, rownum, side):
    if previous is not None:
        prev_groups, prev_start, prev_seq = previous
        cur_key = _stream_order_key(current[0], current[1])
        prev_key = _stream_order_key(prev_groups, prev_start)
        if cur_key < prev_key:
            raise ArgumentError(
                '%s input is out of order at data row %d' % (side, rownum))


def _stream_join_presorted(connection, leftit, rightit, lwidth, rwidth,
                           lstartidx, lstopidx, rstartidx, rstopidx,
                           lgroupidx, rgroupidx, ngroups, how, missing,
                           rfields, routindices):
    lrownum = 0
    rrownum = 0
    prev_left = None
    prev_right = None
    right_exhausted = False

    def next_left():
        nonlocal lrownum, prev_left
        row = next(leftit)
        lrownum += 1
        row = _stream_validate_row(row, lrownum, 'left', lwidth, lstartidx,
                                   lstopidx, lgroupidx)
        groups = _stream_presorted_groups(row, lgroupidx)
        _stream_check_order(prev_left, (groups, row[lstartidx]), lrownum,
                            'left')
        prev_left = (groups, row[lstartidx], lrownum)
        return groups, row

    def next_right():
        nonlocal rrownum, prev_right
        row = next(rightit)
        rrownum += 1
        row = _stream_validate_row(row, rrownum, 'right', rwidth, rstartidx,
                                   rstopidx, rgroupidx)
        groups = _stream_presorted_groups(row, rgroupidx)
        _stream_check_order(prev_right, (groups, row[rstartidx]), rrownum,
                            'right')
        prev_right = (groups, row[rstartidx], rrownum)
        return groups, row

    pending = None
    while True:
        try:
            groups, lrow = next_left()
        except StopIteration:
            break

        # Re-consume a retained boundary right row first.
        if pending is not None:
            rid, lg, rrow = pending
            same_group = not ngroups or tuple(lg) == tuple(groups)
            if not same_group and ngroups and \
                    Comparable(lg) < Comparable(groups):
                # A boundary retained for the previous left group cannot
                # overlap any row in this or later groups.
                pending = None
            elif same_group and rrow[rstartidx] < lrow[lstopidx]:
                _stream_insert(connection, 'right_stage', rid, rid,
                               rrow[rstartidx], rrow[rstopidx], lg, rrow,
                               ngroups)
                pending = None

        while pending is None and not right_exhausted:
            try:
                rgroups, rrow = next_right()
            except StopIteration:
                right_exhausted = True
                break
            rid = rrownum
            if ngroups and Comparable(rgroups) < Comparable(groups):
                # A right group without a later left row can never match.
                continue
            if ngroups and Comparable(rgroups) > Comparable(groups):
                # Keep one future-group lookahead.  It will be reconsidered
                # when the left table reaches that group.
                pending = (rid, rgroups, rrow)
                break
            if rrow[rstartidx] >= lrow[lstopidx]:
                pending = (rid, rgroups, rrow)
                break
            _stream_insert(connection, 'right_stage', rid, rid,
                           rrow[rstartidx], rrow[rstopidx], rgroups, rrow,
                           ngroups)

        cursor = _stream_active_matches(
            connection, groups, lrow[lstartidx], lrow[lstopidx], ngroups)
        matched = False
        try:
            for (blob,) in cursor:
                rrow = pickle.loads(blob)
                matched = True
                yield _stream_output_row(lrow, rrow, routindices)
        finally:
            cursor.close()
        if how == 'left' and not matched:
            yield _stream_missing_row(lrow, rfields, routindices, missing)

    # Exhaustion must validate every remaining right row, including an
    # unmatched or invalid tail.  The retained pending row has already been
    # read and validated.
    while not right_exhausted:
        try:
            next_right()
        except StopIteration:
            right_exhausted = True


def _stream_output_row(lrow, rrow, routindices):
    return tuple(lrow) + tuple(rrow[i] for i in routindices)


def _stream_missing_row(lrow, rfields, routindices, missing):
    return tuple(lrow) + tuple(missing for _ in routindices)


def _stream_unsorted_order_sql(ngroups):
    return ', '.join(['g%d' % i for i in range(ngroups)] +
                     ['start_key', 'seq'])


def _stream_join_staged(connection, ngroups, how, missing, routindices,
                        lstartidx, lstopidx):
    order = _stream_unsorted_order_sql(ngroups)
    group_select = ', '.join('g%d' % i for i in range(ngroups)) \
        if ngroups else 'NULL'
    lcursor = connection.execute(
        'SELECT row, %s FROM left_stage ORDER BY %s'
        % (group_select, order))
    try:
        record = lcursor.fetchone()
        while record is not None:
            lblob = record[0]
            groups = tuple(record[1:])
            lrow = pickle.loads(lblob)
            lstart = lrow[lstartidx]
            lstop = lrow[lstopidx]
            cursor = _stream_active_matches(connection, groups, lstart,
                                            lstop, ngroups)
            matched = False
            try:
                for match in cursor:
                    rblob = match[0]
                    matched = True
                    yield _stream_output_row(lrow, pickle.loads(rblob),
                                             routindices)
            finally:
                cursor.close()
            if how == 'left' and not matched:
                yield _stream_missing_row(lrow, None, routindices, missing)
            record = lcursor.fetchone()
    finally:
        lcursor.close()


def _stream_safe_close(obj, label):
    close = getattr(obj, 'close', None)
    if close is None:
        return
    try:
        close()
    except Exception:
        # Cleanup must not mask an exception from the source or transform.
        pass


def iterintervaljoinstream(left, right, lstart, lstop, rstart, rstop, by,
                           how, presorted, buffersize, tempdir, lprefix,
                           rprefix, missing):
    if how not in ('inner', 'left'):
        raise ArgumentError("how must be 'inner' or 'left', got %r" % (how,))
    if not isinstance(presorted, bool):
        raise ArgumentError('presorted must be True or False')
    _stream_check_buffersize(buffersize)
    group_fields = _stream_group_fields(by)
    lstart = _stream_single_field(lstart, 'lstart')
    lstop = _stream_single_field(lstop, 'lstop')
    rstart = _stream_single_field(rstart, 'rstart')
    rstop = _stream_single_field(rstop, 'rstop')
    if lstart == lstop:
        raise ArgumentError('lstart and lstop must name different fields')
    if rstart == rstop:
        raise ArgumentError('rstart and rstop must name different fields')

    leftit = None
    rightit = None
    connection = None
    dbname = None

    def close_resources():
        _stream_safe_close(leftit, 'left source')
        _stream_safe_close(rightit, 'right source')
        if connection is not None:
            _stream_safe_close(connection, 'temporary database')
        if dbname is not None:
            try:
                os.unlink(dbname)
            except OSError:
                pass

    try:
        leftit = iter(left)
        rightit = iter(right)
        try:
            lhdr = next(leftit)
        except StopIteration:
            raise ArgumentError('left table has no header row')
        try:
            rhdr = next(rightit)
        except StopIteration:
            raise ArgumentError('right table has no header row')

        lfields = _stream_header(lhdr, 'left')
        rfields = _stream_header(rhdr, 'right')
        lstartidx = _stream_field_index(lfields, lstart, 'left', 'lstart')
        lstopidx = _stream_field_index(lfields, lstop, 'left', 'lstop')
        rstartidx = _stream_field_index(rfields, rstart, 'right', 'rstart')
        rstopidx = _stream_field_index(rfields, rstop, 'right', 'rstop')
        lgroupidx = [_stream_field_index(lfields, f, 'left', 'group')
                     for f in group_fields]
        rgroupidx = [_stream_field_index(rfields, f, 'right', 'group')
                     for f in group_fields]
        if lstartidx in lgroupidx or lstopidx in lgroupidx:
            raise ArgumentError(
                'left interval fields cannot overlap group fields')
        if rstartidx in rgroupidx or rstopidx in rgroupidx:
            raise ArgumentError(
                'right interval fields cannot overlap group fields')

        routindices = tuple(i for i in range(len(rfields))
                            if i not in rgroupidx)
        outhdr = _stream_output_header(lfields, rfields, set(rgroupidx),
                                       lprefix, rprefix)
        yield outhdr

        ngroups = len(group_fields)
        connection, dbname = _stream_open_database(tempdir, buffersize)
        _stream_prepare_database(connection, ngroups)

        if presorted:
            for outrow in _stream_join_presorted(
                    connection, leftit, rightit, len(lfields), len(rfields),
                    lstartidx, lstopidx, rstartidx, rstopidx, lgroupidx,
                    rgroupidx, ngroups, how, missing, rfields, routindices):
                yield outrow
        else:
            _stream_load_sorted_table(
                connection, leftit, 'left', len(lfields), lstartidx,
                lstopidx, lgroupidx, ngroups, buffersize, 'left_stage',
                presorted)
            _stream_load_sorted_table(
                connection, rightit, 'right', len(rfields), rstartidx,
                rstopidx, rgroupidx, ngroups, buffersize, 'right_stage',
                presorted)
            for outrow in _stream_join_staged(
                    connection, ngroups, how, missing, routindices,
                    lstartidx, lstopidx):
                yield outrow
    finally:
        close_resources()


def intervalantijoin(left, right, lstart='start', lstop='stop', rstart='start',
                     rstop='stop', lkey=None, rkey=None, include_stop=False,
                     missing=None):
    """
    Return rows from the `left` table with no overlapping rows from the `right`
    table.

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """

    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalAntiJoinView(left, right, lstart=lstart, lstop=lstop,
                                rstart=rstart, rstop=rstop, lkey=lkey,
                                rkey=rkey, include_stop=include_stop,
                                missing=missing)


Table.intervalantijoin = intervalantijoin


class IntervalAntiJoinView(Table):

    def __init__(self, left, right, lstart='start', lstop='stop',
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 missing=None, include_stop=False):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.missing = missing
        self.include_stop = include_stop

    def __iter__(self):
        return iterintervaljoin(
            left=self.left,
            right=self.right,
            lstart=self.lstart,
            lstop=self.lstop,
            rstart=self.rstart,
            rstop=self.rstop,
            lkey=self.lkey,
            rkey=self.rkey,
            include_stop=self.include_stop,
            missing=self.missing,
            lprefix=None,
            rprefix=None,
            leftouter=True,
            anti=True
        )


def iterintervaljoin(left, right, lstart, lstop, rstart, rstop, lkey,
                     rkey, include_stop, missing, lprefix, rprefix, leftouter,
                     anti=False):

    # create iterators and obtain fields
    lit = iter(left)
    lhdr = next(lit)
    lflds = list(map(text_type, lhdr))
    rit = iter(right)
    rhdr = next(rit)
    rflds = list(map(text_type, rhdr))

    # check fields via petl.util.asindices (raises FieldSelectionError if spec
    # is not valid)
    asindices(lhdr, lstart)
    asindices(lhdr, lstop)
    if lkey is not None:
        asindices(lhdr, lkey)
    asindices(rhdr, rstart)
    asindices(rhdr, rstop)
    if rkey is not None:
        asindices(rhdr, rkey)

    # determine output fields
    if lprefix is None:
        outhdr = list(lflds)
        if not anti:
            outhdr.extend(rflds)
    else:
        outhdr = list(lprefix + f for f in lflds)
        if not anti:
            outhdr.extend(rprefix + f for f in rflds)
    yield tuple(outhdr)
    
    # create getters for start and stop positions
    getlstart = itemgetter(lflds.index(lstart))
    getlstop = itemgetter(lflds.index(lstop))

    if rkey is None:
        # build interval lookup for right table
        lookup = intervallookup(right, rstart, rstop, include_stop=include_stop)
        search = lookup.search
        # main loop
        for lrow in lit:
            start = getlstart(lrow)
            stop = getlstop(lrow)
            rrows = search(start, stop)
            if rrows:
                if not anti:
                    for rrow in rrows:
                        outrow = list(lrow)
                        outrow.extend(rrow)
                        yield tuple(outrow)
            elif leftouter:
                outrow = list(lrow)
                if not anti:
                    outrow.extend([missing] * len(rflds))
                yield tuple(outrow)

    else:
        # build interval lookup for right table
        lookup = facetintervallookup(right, key=rkey, start=rstart,
                                     stop=rstop, include_stop=include_stop)
        search = dict()
        for f in lookup:
            search[f] = lookup[f].search
        # getter for facet key values in left table
        getlkey = itemgetter(*asindices(lflds, lkey))
        # main loop
        for lrow in lit:
            lkey = getlkey(lrow)
            start = getlstart(lrow)
            stop = getlstop(lrow)
            
            try:
                rrows = search[lkey](start, stop)
            except KeyError:
                rrows = None
            except AttributeError:
                rrows = None
                
            if rrows:
                if not anti:
                    for rrow in rrows:
                        outrow = list(lrow)
                        outrow.extend(rrow)
                        yield tuple(outrow)
            elif leftouter:
                outrow = list(lrow)
                if not anti:
                    outrow.extend([missing] * len(rflds))
                yield tuple(outrow)


def intervaljoinvalues(left, right, value, lstart='start', lstop='stop',
                       rstart='start', rstop='stop', lkey=None, rkey=None,
                       include_stop=False):
    """
    Convenience function to join the left table with values from a specific 
    field in the right hand table.
    
    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    
    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    if lkey is None:
        lkp = intervallookup(right, start=rstart, stop=rstop, value=value,
                             include_stop=include_stop)
        f = lambda row: lkp.search(row[lstart], row[lstop])
    else:
        lkp = facetintervallookup(right, rkey, start=rstart, stop=rstop,
                                  value=value, include_stop=include_stop)

        def f(row):
            # a key missing from the right table has no overlapping intervals
            facet = lkp.get(row[lkey])
            if facet is None:
                return []
            return facet.search(row[lstart], row[lstop])
    return addfield(left, value, f)
        
        
Table.intervaljoinvalues = intervaljoinvalues


def intervalsubtract(left, right, lstart='start', lstop='stop', rstart='start',
                     rstop='stop', lkey=None, rkey=None, include_stop=False):
    """
    Subtract intervals in the right hand table from intervals in the left hand 
    table.
    
    """

    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalSubtractView(left, right, lstart=lstart, lstop=lstop,
                                rstart=rstart, rstop=rstop, lkey=lkey,
                                rkey=rkey, include_stop=include_stop)


Table.intervalsubtract = intervalsubtract


class IntervalSubtractView(Table):
    
    def __init__(self, left, right, lstart='start', lstop='stop', 
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 include_stop=False):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.include_stop = include_stop

    def __iter__(self):
        return iterintervalsubtract(self.left, self.right, self.lstart,
                                    self.lstop, self.rstart, self.rstop,
                                    self.lkey, self.rkey, self.include_stop)
        

def iterintervalsubtract(left, right, lstart, lstop, rstart, rstop, lkey, rkey,
                         include_stop):

    # create iterators and obtain fields
    lit = iter(left)
    lhdr = next(lit)
    lflds = list(map(text_type, lhdr))
    rit = iter(right)
    rhdr = next(rit)

    # check fields via petl.util.asindices (raises FieldSelectionError if spec
    # is not valid)
    asindices(lhdr, lstart)
    asindices(lhdr, lstop)
    if lkey is not None:
        asindices(lhdr, lkey)
    asindices(rhdr, rstart)
    asindices(rhdr, rstop)
    if rkey is not None:
        asindices(rhdr, rkey)

    # determine output fields
    outhdr = list(lflds)
    yield tuple(outhdr)
    
    # create getters for start and stop positions
    lstartidx, lstopidx = asindices(lhdr, (lstart, lstop))
    getlcoords = itemgetter(lstartidx, lstopidx)
    getrcoords = itemgetter(*asindices(rhdr, (rstart, rstop)))

    if rkey is None:
        # build interval lookup for right table
        lookup = intervallookup(right, rstart, rstop, include_stop=include_stop)
        search = lookup.search
        # main loop
        for lrow in lit:
            start, stop = getlcoords(lrow)
            rrows = search(start, stop)
            if not rrows:
                yield tuple(lrow)
            else:
                rivs = sorted([getrcoords(rrow) for rrow in rrows],
                              key=itemgetter(0))  # sort by start
                for x, y in _subtract(start, stop, rivs):
                    out = list(lrow)
                    out[lstartidx] = x
                    out[lstopidx] = y
                    yield tuple(out)
                
    else:
        # build interval lookup for right table
        lookup = facetintervallookup(right, key=rkey, start=rstart, stop=rstop,
                                     include_stop=include_stop)
        # getter for facet key values in left table
        getlkey = itemgetter(*asindices(lhdr, lkey))
        # main loop
        for lrow in lit:
            lkey = getlkey(lrow)
            start, stop = getlcoords(lrow)
            try:
                rrows = lookup[lkey].search(start, stop)
            except KeyError:
                rrows = None
            except AttributeError:
                rrows = None
            if not rrows:
                yield tuple(lrow)
            else:
                rivs = sorted([getrcoords(rrow) for rrow in rrows],
                              key=itemgetter(0))  # sort by start
                for x, y in _subtract(start, stop, rivs):
                    out = list(lrow)
                    out[lstartidx] = x
                    out[lstopidx] = y
                    yield tuple(out)


from collections import namedtuple
_Interval = namedtuple('Interval', 'start stop')


def collapsedintervals(table, start='start', stop='stop', key=None):
    """
    Utility function to collapse intervals in a table. 
    
    If no facet `key` is given, returns an iterator over `(start, stop)` tuples.
    
    If facet `key` is given, returns an iterator over `(key, start, stop)`
    tuples.
    
    """
    
    if key is None:
        table = sort(table, key=start)
        for iv in _collapse(values(table, (start, stop))):
            yield iv
    else:
        table = sort(table, key=(key, start))
        for k, g in rowgroupby(table, key=key, value=(start, stop)):
            for iv in _collapse(g):
                yield (k,) + iv


Table.collapsedintervals = collapsedintervals


def _collapse(intervals):
    """
    Collapse an iterable of intervals sorted by start coord.
    
    """
    span = None
    for start, stop in intervals:
        if span is None:
            span = _Interval(start, stop)
        elif start <= span.stop < stop:
            span = _Interval(span.start, stop)
        elif start > span.stop:
            yield span
            span = _Interval(start, stop)
    if span is not None:
        yield span
    
    
def _subtract(start, stop, intervals):
    """
    Subtract intervals from a spanning interval.
    
    """
    remainder_start = start
    sub_stop = None
    for sub_start, sub_stop in _collapse(intervals):
        if remainder_start < sub_start:
            yield _Interval(remainder_start, sub_start)
        remainder_start = sub_stop
    if sub_stop is not None and sub_stop < stop:
        yield _Interval(sub_stop, stop)
