from __future__ import absolute_import, print_function, division


import gc
import operator
import os
import random
import shutil
import tempfile

import pytest


from collections import OrderedDict
from petl.test.helpers import ieq
from petl.errors import ArgumentError, FieldSelectionError
from petl.comparison import Comparable
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
    mergeduplicates, merge, Conflict, fold, rollingaggregate


def test_rowreduce():
    
    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))
    
    def sumbar(key, rows):
        return [key, sum(row[1] for row in rows)]
        
    table2 = rowreduce(table1, key='foo', reducer=sumbar, 
                       header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'),
               ('a', 10),
               ('b', 12),
               ('c', 4))
    ieq(expect2, table2)
    

def test_rowreduce_fieldnameaccess():
    
    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))
    
    def sumbar(key, records):
        return [key, sum([rec['bar'] for rec in records])]
        
    table2 = rowreduce(table1, key='foo', reducer=sumbar, 
                       header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'),
               ('a', 10),
               ('b', 12),
               ('c', 4))
    ieq(expect2, table2)
    

def test_rowreduce_more():
    
    table1 = (('foo', 'bar'),
              ('aa', 3),
              ('aa', 7),
              ('bb', 2),
              ('bb', 1),
              ('bb', 9),
              ('cc', 4))
    
    def sumbar(key, records):
        return [key, sum(rec['bar'] for rec in records)]
        
    table2 = rowreduce(table1, key='foo', reducer=sumbar, 
                       header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'),
               ('aa', 10),
               ('bb', 12),
               ('cc', 4))
    ieq(expect2, table2)
    

def test_rowreduce_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    reducer = lambda key, rows: (key, [r[0] for r in rows])
    actual = rowreduce(table, key='foo', reducer=reducer, 
                       header=('foo', 'bar'))
    ieq(expect, actual)


def test_aggregate_simple():
    
    table1 = (('foo', 'bar', 'baz'),
              ('a', 3, True),
              ('a', 7, False),
              ('b', 2, True),
              ('b', 2, False),
              ('b', 9, False),
              ('c', 4, True))

    # simplest signature - aggregate whole rows
    table2 = aggregate(table1, 'foo', len)
    expect2 = (('foo', 'value'),
               ('a', 2),
               ('b', 3),
               ('c', 1))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # next simplest signature - aggregate single field
    table3 = aggregate(table1, 'foo', sum, 'bar')
    expect3 = (('foo', 'value'),
               ('a', 10),
               ('b', 13),
               ('c', 4))
    ieq(expect3, table3)
    ieq(expect3, table3)
    
    # alternative signature for simple aggregation
    table4 = aggregate(table1, key=('foo', 'bar'), aggregation=list, 
                       value=('bar', 'baz'))
    expect4 = (('foo', 'bar', 'value'),
               ('a', 3, [(3, True)]),
               ('a', 7, [(7, False)]),
               ('b', 2, [(2, True), (2, False)]),
               ('b', 9, [(9, False)]),
               ('c', 4, [(4, True)]))
    ieq(expect4, table4)
    ieq(expect4, table4)
    
    table5 = aggregate(table1, 'foo', len, field='nrows')
    expect5 = (('foo', 'nrows'),
               ('a', 2),
               ('b', 3),
               ('c', 1))
    ieq(expect5, table5)
    ieq(expect5, table5)


def test_aggregate_simple_key_is_None():

    table1 = (('foo', 'bar', 'baz'),
              ('a', 3, True),
              ('a', 7, False),
              ('b', 2, True),
              ('b', 2, False),
              ('b', 9, False),
              ('c', 4, True))

    # simplest signature - aggregate whole rows
    table2 = aggregate(table1, None, len)
    expect2 = (('value',), (6,))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # next simplest signature - aggregate single field
    table3 = aggregate(table1, None, sum, 'bar')
    expect3 = (('value',), (27,))
    ieq(expect3, table3)
    ieq(expect3, table3)

    # alternative signature for simple aggregation
    table4 = aggregate(table1, key=None, aggregation=list,
                       value=('bar', 'baz'))
    expect4 = (('value',),
               ( [(3, True), (7, False),
                 (2, True), (2, False), (9, False),
                 (4, True)],),)

    ieq(expect4, table4)
    ieq(expect4, table4)

    table5 = aggregate(table1, None, len, field='nrows')
    expect5 = (('nrows',), (6,))
    ieq(expect5, table5)
    ieq(expect5, table5)

def test_aggregate_simple_whole_records():
    table = (('foo', 'bar'), ('a', 3), ('b', 7))

    def aggregation(rows):
        return sum(row['bar'] for row in rows)

    actual = aggregate(table, None, aggregation)
    expect = (('value',), (10,))
    ieq(expect, actual)


def test_aggregate_simple_whole_records_single_pass():
    table = iter((('foo', 'bar'), ('a', 3), ('b', 7)))
    actual = aggregate(table, None, list)
    expect = (('value',), ([('a', 3), ('b', 7)],))
    ieq(expect, actual)


def test_aggregate_multifield():
    
    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))
    
    # dict arg
    
    aggregators = OrderedDict()
    aggregators['count'] = len
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar', list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, 'foo', aggregators)
    expect2 = (('foo', 'count', 'minbar', 'maxbar', 'sumbar', 'listbar', 
                'bars'),
               ('a', 2, 3, 7, 10, [3, 7], '3, 7'),
               ('b', 3, 1, 9, 12, [2, 1, 9], '2, 1, 9'),
               ('c', 1, 4, 4, 4, [4], '4'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice
    
    # use suffix notation
    
    table3 = aggregate(table1, 'foo')
    table3['count'] = len
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)
    
    # list arg

    aggregators = [('count', len),
                   ('minbar', 'bar', min),
                   ('maxbar', 'bar', max),
                   ('sumbar', 'bar', sum),
                   ('listbar', 'bar', list),
                   ('bars', 'bar', strjoin(', '))]

    table4 = aggregate(table1, 'foo', aggregators)
    ieq(expect2, table4)
    ieq(expect2, table4)  # check can iterate twice

def _aggregate_multifield(table, key, form, presorted):
    if form == 'dict':
        aggregation = OrderedDict([('total', ('bar', sum)), ('count', len)])
    elif form == 'list':
        aggregation = [('total', 'bar', sum), ('count', len)]
    else:
        aggregation = None
    result = aggregate(table, key, aggregation, presorted=presorted)
    if form == 'suffix':
        result['total'] = 'bar', sum
        result['count'] = len
    return result


@pytest.mark.parametrize('key_type', [list, tuple])
@pytest.mark.parametrize('key_values', [
    ('alpha', 'beta'),
    (1, 2),
    (None, 1),
    ('', 'a'),
    ((1, 2), (3, 4)),
    (False, True),
])
@pytest.mark.parametrize('form', ['dict', 'list', 'suffix'])
@pytest.mark.parametrize('presorted', [False, True])
def test_aggregate_multifield_singleton_key(key_type, key_values, form,
                                          presorted):
    first, second = key_values
    rows = [(first, 3), (first, 7), (second, 5)]
    if not presorted:
        rows.reverse()
    table = [('foo', 'bar')] + rows
    original = list(table)
    key = key_type(['foo'])
    result = _aggregate_multifield(table, key, form, presorted)
    expected = [('foo', 'total', 'count'), (first, 10, 2), (second, 5, 1)]

    assert list(result) == expected
    assert list(result) == expected
    assert key == key_type(['foo'])
    assert table == original


@pytest.mark.parametrize('key_type', [list, tuple])
@pytest.mark.parametrize('form', ['dict', 'list', 'suffix'])
@pytest.mark.parametrize('presorted', [False, True])
def test_aggregate_multifield_singleton_key_empty(key_type, form, presorted):
    result = _aggregate_multifield([('foo', 'bar')], key_type(['foo']),
                                  form, presorted)
    expected = [('foo', 'total', 'count')]
    assert list(result) == expected
    assert list(result) == expected


@pytest.mark.parametrize('presorted', [False, True])
@pytest.mark.parametrize('key, expected', [
    ('foo', [('foo', 'total', 'count'), ('alpha', 10, 2), ('beta', 5, 1)]),
    (('foo', 'baz'), [('foo', 'baz', 'total', 'count'),
                      ('alpha', 1, 10, 2), ('beta', 2, 5, 1)]),
    (operator.itemgetter(0, 2), [('key', 'total', 'count'),
                               (('alpha', 1), 10, 2), (('beta', 2), 5, 1)]),
    (None, [('total', 'count'), (15, 3)]),
])
def test_aggregate_multifield_key_shapes(key, expected, presorted):
    rows = [('alpha', 3, 1), ('alpha', 7, 1), ('beta', 5, 2)]
    if not presorted:
        rows.reverse()
    result = _aggregate_multifield([('foo', 'bar', 'baz')] + rows, key,
                                  'dict', presorted)
    if callable(key) and not presorted:
        # Sorting does not accept callable field selectors.
        with pytest.raises(FieldSelectionError):
            list(result)
        return
    assert list(result) == expected
    assert list(result) == expected


def test_aggregate_multifield_key_is_None():

    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))

    # dict arg
    aggregators = OrderedDict()
    aggregators['count'] = len
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar', list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, None, aggregators)
    expect2 = (('count', 'minbar', 'maxbar', 'sumbar', 'listbar',
                'bars'),
               (6, 1, 9, 26, [3, 7, 2, 1, 9, 4], '3, 7, 2, 1, 9, 4'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice

    # use suffix notation
    table3 = aggregate(table1, None)
    table3['count'] = len
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)

    # list arg
    aggregators = [('count', len),
                   ('minbar', 'bar', min),
                   ('maxbar', 'bar', max),
                   ('sumbar', 'bar', sum),
                   ('listbar', 'bar', list),
                   ('bars', 'bar', strjoin(', '))]

    table4 = aggregate(table1, None, aggregators)
    ieq(expect2, table4)
    ieq(expect2, table4)  # check can iterate twice

def test_aggregate_more():

    table1 = (('foo', 'bar'),
              ('aa', 3),
              ('aa', 7),
              ('bb', 2),
              ('bb', 1),
              ('bb', 9),
              ('cc', 4),
              ('dd', 3))

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar'  # default aggregation is list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, 'foo', aggregators)
    expect2 = (('foo', 'minbar', 'maxbar', 'sumbar', 'listbar', 'bars'),
               ('aa', 3, 7, 10, [3, 7], '3, 7'),
               ('bb', 1, 9, 12, [2, 1, 9], '2, 1, 9'),
               ('cc', 4, 4, 4, [4], '4'),
               ('dd', 3, 3, 3, [3], '3'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice

    table3 = aggregate(table1, 'foo')
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)

def test_aggregate_more_key_is_None():

    table1 = (('foo', 'bar'),
              ('aa', 3),
              ('aa', 7),
              ('bb', 2),
              ('bb', 1),
              ('bb', 9),
              ('cc', 4),
              ('dd', 3))

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar'  # default aggregation is list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, None, aggregators)
    expect2 = (('minbar', 'maxbar', 'sumbar', 'listbar', 'bars'),
               (1, 9, 29,
                   [3, 7, 2, 1, 9, 4, 3],
                   '3, 7, 2, 1, 9, 4, 3'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice

    table3 = aggregate(table1, None)
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)

def test_aggregate_multiple_source_fields():

    table = (('foo', 'bar', 'baz'),
             ('a', 3, True),
             ('a', 7, False),
             ('b', 2, True),
             ('b', 2, False),
             ('b', 9, False),
             ('c', 4, True))

    expect = (('foo', 'bar', 'value'),
              ('a', 3, [(3, True)]),
              ('a', 7, [(7, False)]),
              ('b', 2, [(2, True), (2, False)]),
              ('b', 9, [(9, False)]),
              ('c', 4, [(4, True)]))

    actual = aggregate(table, ('foo', 'bar'), list, ('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)

    actual = aggregate(table, key=('foo', 'bar'), aggregation=list, 
                       value=('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)
    
    actual = aggregate(table, key=('foo', 'bar'))
    actual['value'] = ('bar', 'baz'), list
    ieq(expect, actual)
    ieq(expect, actual)    

def test_aggregate_multiple_source_fields_key_is_None():

    table = (('foo', 'bar', 'baz'),
             ('a', 3, True),
             ('a', 7, False),
             ('b', 2, True),
             ('b', 2, False),
             ('b', 9, False),
             ('c', 4, True))

    expect  = (('value',),
               ( [(3, True), (7, False),
                 (2, True), (2, False), (9, False),
                 (4, True)],),)

    actual = aggregate(table, None, list, ('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)

    actual = aggregate(table, key=None, aggregation=list,
                       value=('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)

    actual = aggregate(table, key=None)
    actual['value'] = ('bar', 'baz'), list
    ieq(expect, actual)
    ieq(expect, actual)


def test_aggregate_empty():

    table = (('foo', 'bar'),)

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum

    actual = aggregate(table, 'foo', aggregators)
    expect = (('foo', 'minbar', 'maxbar', 'sumbar'),)
    ieq(expect, actual)

def test_aggregate_empty_key_is_None():

    table = (('foo', 'bar'),)

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum

    actual = aggregate(table, None, aggregators)
    expect = (('minbar', 'maxbar', 'sumbar'),)
    ieq(expect, actual)

def test_mergeduplicates():

    table = (('foo', 'bar', 'baz'),
             ('A', 1, 2),
             ('B', '2', None),
             ('D', 'xyz', 9.4),
             ('B', None, u'7.8', True),
             ('E', None, 42.),
             ('D', 'xyz', 12.3),
             ('A', 2, None))

    # value overrides missing
    result = mergeduplicates(table, 'foo', missing=None)
    expectation = (('foo', 'bar', 'baz'),
                   ('A', Conflict([1, 2]), 2),
                   ('B', '2', u'7.8'),
                   ('D', 'xyz', Conflict([9.4, 12.3])),
                   ('E', None, 42.))
    ieq(expectation, result)
    
    
def test_mergeduplicates_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = mergeduplicates(table, key='foo')
    ieq(expect, actual)


def test_mergeduplicates_shortrows():
    table = [['foo', 'bar', 'baz'], 
             ['a', 1, True], 
             ['b', 2, True], 
             ['b', 3]]
    actual = mergeduplicates(table, 'foo')
    expect = [('foo', 'bar', 'baz'), 
              ('a', 1, True), 
              ('b', Conflict([2, 3]), True)]
    ieq(expect, actual)
        
    
def test_mergeduplicates_compoundkey():
    table = [['foo', 'bar', 'baz'], 
             ['a', 1, True], 
             ['a', 1, True], 
             ['a', 2, False],
             ['a', 2, None],
             ['c', 3, True],
             ['c', 3, False],
             ]
    actual = mergeduplicates(table, key=('foo', 'bar'))
    expect = [('foo', 'bar', 'baz'), 
              ('a', 1, True), 
              ('a', 2, False), 
              ('c', 3, Conflict([True, False]))]
    ieq(expect, actual)
        

def test_mergeduplicates_field_index():
    table = [['foo', 'bar', 'baz'],
             ['aa', 1, True],
             ['aa', None, False],
             ['bb', 2, True]]
    expect = [('foo', 'bar', 'baz'),
              ('aa', 1, Conflict([True, False])),
              ('bb', 2, True)]
    for key in 0, -3, [0], ['foo'], (-3,):
        ieq(expect, mergeduplicates(table, key))
        ieq(expect, mergeduplicates(table, key, presorted=True))
        ieq(expect, merge(table[:2], table[:1] + table[2:], key=key))

    # key fields come first in the output, as with field names
    expect = [('baz', 'foo', 'bar'),
              (False, 'aa', None),
              (True, Conflict(['aa', 'bb']), Conflict([1, 2]))]
    ieq(expect, mergeduplicates(table, -1))
    ieq(expect, mergeduplicates(table, 'baz'))


def test_mergeduplicates_compoundkey_field_index():
    table = [['foo', 'bar', 'baz'],
             ['a', 1, True],
             ['a', 1, None],
             ['a', 2, False],
             ['c', 3, True],
             ['c', 3, False]]
    expect = [('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('a', 2, False),
              ('c', 3, Conflict([True, False]))]
    for key in (0, 1), [-3, -2], ('foo', 1):
        ieq(expect, mergeduplicates(table, key))
        ieq(expect, mergeduplicates(table, key, presorted=True))


def test_mergeduplicates_field_index_out_of_range():
    table = [['foo', 'bar'],
             ['a', 1]]
    for key in 2, -3, [0, 2]:
        with pytest.raises(FieldSelectionError):
            mergeduplicates(table, key).nrows()
        with pytest.raises(FieldSelectionError):
            mergeduplicates(table, key, presorted=True).nrows()


def test_mergeduplicates_field_index_ragged_rows():
    # a negative index counts from the end of the header, not of the row
    table = [['foo', 'bar', 'baz'],
             ['s1', 'a'],
             ['f', 0, 'x'],
             ['s2', 'z']]
    expect = [('baz', 'foo', 'bar'),
              (None, Conflict(['s1', 's2']), Conflict(['a', 'z'])),
              ('x', 'f', 0)]
    for key in 'baz', 2, -1:
        ieq(expect, mergeduplicates(table, key))

    table = [['foo', 'bar'],
             ['a', 1, 'zz'],
             ['b', 2, 'mm'],
             ['c', 1, 'aa']]
    expect = [('bar', 'foo'),
              (1, Conflict(['a', 'c'])),
              (2, 'b')]
    for key in 'bar', 1, -1:
        ieq(expect, mergeduplicates(table, key))


def test_mergeduplicates_header_change_between_iterations():
    # the key is resolved against the header of each iteration, so a reused
    # view still groups equal keys after its source changes the header
    table = [['id', 'value'],
             ['a', 1],
             ['b', 2],
             ['a', 3]]
    views = [mergeduplicates(table, key, cache=False) for key in ('id', -2)]
    for view in views:
        ieq([('id', 'value'), ('a', Conflict([1, 3])), ('b', 2)], view)

    table[:] = [['x', 'id', 'value'],
                [1, 'a', 1],
                [2, 'b', 2],
                [3, 'a', 3]]
    expect = [('id', 'x', 'value'),
              ('a', Conflict([1, 3]), Conflict([1, 3])),
              ('b', 2, 2)]
    for view in views:
        ieq(expect, view)


def test_merge_field_index_reordered_fields():
    # an index refers to the combined header, whatever the order of the
    # fields in each table
    table1 = [['id', 'value'],
              [1, 'z'],
              [3, 'c']]
    table2 = [['value', 'id'],
              ['a', 2],
              ['b', 1]]
    expect = [('id', 'value'),
              (1, Conflict(['z', 'b'])),
              (2, 'a'),
              (3, 'c')]
    for key in 'id', 0, -2, [0]:
        ieq(expect, merge(table1, table2, key=key))
        ieq(expect[:1] + expect[:0:-1],
            merge(table1, table2, key=key, reverse=True))
    for key in 2, -3:
        with pytest.raises(FieldSelectionError):
            merge(table1, table2, key=key).nrows()


def test_merge_field_index_header_change_between_iterations():
    # an index key is looked up by name in each table, so a table that
    # changes the order of its fields is still sorted by the same field
    table1 = [['id', 'value'],
              [1, 'z'],
              [3, 'c']]
    table2 = [['id', 'value'],
              [2, 'a'],
              [1, 'b']]
    expect = [('id', 'value'),
              (1, Conflict(['z', 'b'])),
              (2, 'a'),
              (3, 'c')]
    view = merge(table1, table2, key=0, cache=False)
    ieq(expect, view)
    table2[:] = [['value', 'id'],
                 ['a', 2],
                 ['b', 1]]
    ieq(expect, view)


def test_fold():

    t1 = (('id', 'count'), (1, 3), (1, 5), (2, 4), (2, 8))
    t2 = fold(t1, 'id', operator.add, 'count', presorted=True)
    expect = (('key', 'value'), (1, 8), (2, 12))
    ieq(expect, t2)
    ieq(expect, t2)


# ---------------------------------------------------------------------------
# rollingaggregate
# ---------------------------------------------------------------------------


def _rolling_indices(hdr, time, by):
    tidx = hdr.index(time)
    if by is None:
        bidx = ()
    elif isinstance(by, str):
        bidx = (hdr.index(by),)
    else:
        bidx = tuple(hdr.index(b) for b in by)
    return tidx, bidx


def _rolling_sorted_rows(rows, hdr, time, by):
    """Stable sort of the data rows by (by..., time) using petl's table
    sort semantics, mirroring what the default path does internally."""
    tidx, bidx = _rolling_indices(hdr, time, by)

    def key(row):
        return Comparable(tuple(row[i] for i in bidx) + (row[tidx],))

    return sorted(rows, key=key)


def _naive_rollingaggregate(table, time, value, width, by=None,
                            prefix='window_'):
    """Independent naive reference implementation. Recomputes the window
    from scratch for every row, O(W) per row."""
    table = [tuple(row) for row in table]
    hdr, rows = table[0], table[1:]
    tidx, bidx = _rolling_indices(hdr, time, by)
    vidx = hdr.index(value)
    outhdr = hdr + (prefix + 'count', prefix + 'sum', prefix + 'min',
                    prefix + 'max')
    out = [outhdr]
    window = []
    current = None
    started = False
    for row in _rolling_sorted_rows(rows, hdr, time, by):
        if bidx:
            gkey = Comparable(tuple(row[i] for i in bidx))
        else:
            gkey = None
        if not started or not (gkey == current):
            window = []
            current = gkey
            started = True
        t = row[tidx]
        # window holds rows of this group with time strictly greater than
        # t - width, up to and including the current row
        window = [w for w in window if w[tidx] > t - width]
        window.append(row)
        vals = [w[vidx] for w in window if w[vidx] is not None]
        out.append(row + (len(window),
                          sum(vals),
                          min(vals) if vals else None,
                          max(vals) if vals else None))
    return out


def test_rollingaggregate_handchecked():
    # hand-checked example, independent of the naive reference
    table = [['id', 'time', 'value'],
             ['a', 1, 10],
             ['a', 2, 20],
             ['b', 3, 5],
             ['a', 4, None],
             ['a', 5, 7]]
    expect = (('id', 'time', 'value', 'window_count', 'window_sum',
               'window_min', 'window_max'),
              ('a', 1, 10, 1, 10, 10, 10),
              ('a', 2, 20, 2, 30, 10, 20),
              ('a', 4, None, 2, 20, 20, 20),
              ('a', 5, 7, 2, 7, 7, 7),
              ('b', 3, 5, 1, 5, 5, 5))
    ieq(expect, rollingaggregate(table, 'time', 'value', 3, by='id'))
    # same result via the presorted path
    sorted_table = [table[0]] + _rolling_sorted_rows(table[1:], table[0],
                                                     'time', 'id')
    ieq(expect, rollingaggregate(sorted_table, 'time', 'value', 3, by='id',
                                 presorted=True))
    # without grouping the whole table is one window, sorted by time only
    expect2 = (('id', 'time', 'value', 'window_count', 'window_sum',
                'window_min', 'window_max'),
               ('a', 1, 10, 1, 10, 10, 10),
               ('a', 2, 20, 2, 30, 10, 20),
               ('b', 3, 5, 2, 25, 5, 20),
               ('a', 4, None, 2, 5, 5, 5),
               ('a', 5, 7, 2, 7, 7, 7))
    ieq(expect2, rollingaggregate(table, 'time', 'value', 2))


def test_rollingaggregate_window_left_boundary():
    # a record exactly at time - width is *not* in the window
    table = [('t', 'v')] + [(i, i) for i in range(6)]
    expect = [('t', 'v', 'window_count', 'window_sum', 'window_min',
               'window_max'),
              (0, 0, 1, 0, 0, 0),
              (1, 1, 2, 1, 0, 1),
              (2, 2, 3, 3, 0, 2),
              (3, 3, 3, 6, 1, 3),   # time 0 is exactly at 3 - 3, excluded
              (4, 4, 3, 9, 2, 4),
              (5, 5, 3, 12, 3, 5)]
    ieq(expect, rollingaggregate(table, 't', 'v', 3))
    ieq(expect, rollingaggregate(table, 't', 'v', 3, presorted=True))


def test_rollingaggregate_negative_times():
    table = [('t', 'v'),
             (-4, 3),
             (-10, 1),
             (0, 4),
             (-5, 2)]
    expect = [('t', 'v', 'window_count', 'window_sum', 'window_min',
               'window_max'),
              (-10, 1, 1, 1, 1, 1),
              (-5, 2, 1, 2, 2, 2),    # -10 is exactly at -5 - 5, excluded
              (-4, 3, 2, 5, 2, 3),
              (0, 4, 2, 7, 3, 4)]     # -5 is exactly at 0 - 5, excluded
    ieq(expect, rollingaggregate(table, 't', 'v', 5))
    sorted_table = [table[0]] + _rolling_sorted_rows(table[1:], table[0],
                                                     't', None)
    ieq(expect, rollingaggregate(sorted_table, 't', 'v', 5, presorted=True))


def test_rollingaggregate_big_integers_exact():
    big = 2 ** 70 + 12345
    table = [('t', 'v'),
             (1, big),
             (2, big + 1),
             (3, -big),
             (4, None)]
    rows = list(rollingaggregate(table, 't', 'v', 10))
    # sums are exact integers, no float rounding of large integers
    assert rows[1][3] == big and isinstance(rows[1][3], int)
    assert rows[2][3] == 2 * big + 1 and isinstance(rows[2][3], int)
    assert rows[3][3] == big + 1 and isinstance(rows[3][3], int)
    assert rows[3][4] == -big and rows[3][5] == big + 1
    assert rows[4][2] == 4  # count includes the None-valued row
    assert rows[4][3] == big + 1  # None contributes nothing to the sum
    ieq(_naive_rollingaggregate(table, 't', 'v', 10),
        rollingaggregate(table, 't', 'v', 10))


def test_rollingaggregate_same_time_stable_prefix():
    # rows with equal times advance one at a time, in input order
    table = [('g', 't', 'v', 'tag'),
             ('a', 5, 1, 'first'),
             ('a', 5, 2, 'second'),
             ('a', 5, None, 'third'),
             ('a', 5, 4, 'fourth')]
    expect = [('g', 't', 'v', 'tag', 'window_count', 'window_sum',
               'window_min', 'window_max'),
              ('a', 5, 1, 'first', 1, 1, 1, 1),
              ('a', 5, 2, 'second', 2, 3, 1, 2),
              ('a', 5, None, 'third', 3, 3, 1, 2),
              ('a', 5, 4, 'fourth', 4, 7, 1, 4)]
    ieq(expect, rollingaggregate(table, 't', 'v', 3, by='g'))
    ieq(expect, rollingaggregate(table, 't', 'v', 3, by='g',
                                 presorted=True))


def test_rollingaggregate_duplicate_and_expiring_extremes():
    table = [('t', 'v'),
             (1, 5),
             (2, 5),
             (3, 1),
             (4, 2),
             (5, 3)]
    expect = [('t', 'v', 'window_count', 'window_sum', 'window_min',
               'window_max'),
              (1, 5, 1, 5, 5, 5),
              (2, 5, 2, 10, 5, 5),
              (3, 1, 2, 6, 1, 5),   # first 5 expired, duplicate remains
              (4, 2, 2, 3, 1, 2),   # both 5s expired, max drops
              (5, 3, 2, 5, 2, 3)]
    ieq(expect, rollingaggregate(table, 't', 'v', 2))
    ieq(expect, rollingaggregate(table, 't', 'v', 2, presorted=True))


def test_rollingaggregate_group_switch_no_leak():
    table = [('g', 't', 'v'),
             ('a', 1, 10),
             ('b', 1, 20),
             ('a', 2, 30),
             ('b', 2, 40)]
    expect = [('g', 't', 'v', 'window_count', 'window_sum', 'window_min',
               'window_max'),
              ('a', 1, 10, 1, 10, 10, 10),
              ('a', 2, 30, 2, 40, 10, 30),
              ('b', 1, 20, 1, 20, 20, 20),  # group a's window must not leak
              ('b', 2, 40, 2, 60, 20, 40)]
    ieq(expect, rollingaggregate(table, 't', 'v', 3, by='g'))
    sorted_table = [table[0]] + _rolling_sorted_rows(table[1:], table[0],
                                                     't', 'g')
    ieq(expect, rollingaggregate(sorted_table, 't', 'v', 3, by='g',
                                 presorted=True))


def test_rollingaggregate_none_group_and_values():
    table = [('g', 't', 'v'),
             (None, 1, None),
             (None, 2, None),
             ('a', 1, 5),
             (None, 3, 7)]
    expect = [('g', 't', 'v', 'window_count', 'window_sum', 'window_min',
               'window_max'),
              (None, 1, None, 1, 0, None, None),
              (None, 2, None, 2, 0, None, None),
              (None, 3, 7, 3, 7, 7, 7),
              ('a', 1, 5, 1, 5, 5, 5)]
    ieq(expect, rollingaggregate(table, 't', 'v', 5, by='g'))
    sorted_table = [table[0]] + _rolling_sorted_rows(table[1:], table[0],
                                                     't', 'g')
    ieq(expect, rollingaggregate(sorted_table, 't', 'v', 5, by='g',
                                 presorted=True))


def test_rollingaggregate_compound_by():
    table = [('g1', 'g2', 't', 'v'),
             ('a', 'x', 1, 1),
             ('a', 'y', 1, 10),
             ('a', 'x', 2, 2),
             ('a', 'y', 2, 20),
             ('b', 'x', 1, 100)]
    expect = _naive_rollingaggregate(table, 't', 'v', 5, by=('g1', 'g2'))
    # spot-check the grouping: each (g1, g2) pair gets its own window
    assert expect[1] == ('a', 'x', 1, 1, 1, 1, 1, 1)
    assert expect[2] == ('a', 'x', 2, 2, 2, 3, 1, 2)
    assert expect[3] == ('a', 'y', 1, 10, 1, 10, 10, 10)
    assert expect[4] == ('a', 'y', 2, 20, 2, 30, 10, 20)
    assert expect[5] == ('b', 'x', 1, 100, 1, 100, 100, 100)
    ieq(expect, rollingaggregate(table, 't', 'v', 5, by=('g1', 'g2')))
    ieq(expect, rollingaggregate(table, 't', 'v', 5, by=['g1', 'g2'],
                                 buffersize=2))
    sorted_table = [table[0]] + _rolling_sorted_rows(table[1:], table[0],
                                                     't', ('g1', 'g2'))
    ieq(expect, rollingaggregate(sorted_table, 't', 'v', 5, by=('g1', 'g2'),
                                 presorted=True))


def test_rollingaggregate_custom_prefix():
    table = [('t', 'v'), (1, 5), (2, 7)]
    expect = [('t', 'v', 'r_count', 'r_sum', 'r_min', 'r_max'),
              (1, 5, 1, 5, 5, 5),
              (2, 7, 2, 12, 5, 7)]
    ieq(expect, rollingaggregate(table, 't', 'v', 2, prefix='r_'))


def test_rollingaggregate_empty_and_header_only():
    # header only -> only the output header is produced
    view = rollingaggregate([('g', 't', 'v')], 't', 'v', 3, by='g')
    assert list(view) == [('g', 't', 'v', 'window_count', 'window_sum',
                           'window_min', 'window_max')]
    view = rollingaggregate([('g', 't', 'v')], 't', 'v', 3, by='g',
                            presorted=True)
    assert list(view) == [('g', 't', 'v', 'window_count', 'window_sum',
                           'window_min', 'window_max')]
    # no header row at all -> error
    pytest.raises(ArgumentError, list, rollingaggregate([], 't', 'v', 3))
    pytest.raises(ArgumentError, list,
                  rollingaggregate([], 't', 'v', 3, presorted=True))


def test_rollingaggregate_randomised_matches_naive():
    rng = random.Random(20261112)
    hdr = ('g1', 'g2', 't', 'v', 'tag')
    for _ in range(30):
        n = rng.randint(0, 60)
        rows = []
        for i in range(n):
            g1 = rng.choice([None, 'a', 'b', 'c'])
            g2 = rng.choice([None, 'x', 'y'])
            t = rng.randint(-15, 15)
            r = rng.random()
            if r < 0.25:
                v = None
            elif r < 0.35:
                v = rng.choice([2 ** 70, -(2 ** 70), 0])
            else:
                v = rng.randint(-100, 100)
            rows.append((g1, g2, t, v, 'row%d' % i))
        width = rng.randint(1, 12)
        by = rng.choice([None, 'g1', 'g2', ('g1', 'g2'), ['g1', 'g2']])
        table = [hdr] + rows
        expect = _naive_rollingaggregate(table, 't', 'v', width, by=by)
        # default path, with a small buffer to force chunked sorting
        ieq(expect, rollingaggregate(table, 't', 'v', width, by=by,
                                     buffersize=5))
        # presorted path over the stably sorted input
        sorted_table = [hdr] + _rolling_sorted_rows(rows, hdr, 't', by)
        ieq(expect, rollingaggregate(sorted_table, 't', 'v', width, by=by,
                                     presorted=True))


def test_rollingaggregate_argument_validation():
    table = [('g', 't', 'v')]
    # time and value must be single field names
    pytest.raises(ArgumentError, rollingaggregate, table, 0, 'v', 3)
    pytest.raises(ArgumentError, rollingaggregate, table, ['t'], 'v', 3)
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 1, 3)
    pytest.raises(ArgumentError, rollingaggregate, table, 't', ['v'], 3)
    # time and value must not overlap each other or by
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 't', 3)
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by='t')
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by='v')
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by=('g', 't'))
    # by must be None, a field name, or a non-empty sequence of field names
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by=[])
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by=())
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by=5)
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by=('g', 5))
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  by=('g', 'g'))
    # width must be a positive integer and not a bool
    for badwidth in (0, -1, -100, True, False, 2.5, '3', None):
        pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v',
                      badwidth)
    # prefix must be a string
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  prefix=5)
    pytest.raises(ArgumentError, rollingaggregate, table, 't', 'v', 3,
                  prefix=None)
    # valid forms construct without touching the source
    rollingaggregate(table, 't', 'v', 3)
    rollingaggregate(table, 't', 'v', 3, by='g')
    rollingaggregate(table, 't', 'v', 3, by=['g'])
    rollingaggregate(table, 't', 'v', 1, prefix='w_')


def test_rollingaggregate_validation_is_lazy():
    # missing field: construction and iter() are fine, error on consumption
    view = rollingaggregate([('t', 'v')], 'missing', 'v', 3)
    it = iter(view)
    pytest.raises(FieldSelectionError, next, it)


def test_rollingaggregate_field_and_header_errors():
    for presorted in (False, True):
        kwargs = {'presorted': presorted}
        # missing fields -> FieldSelectionError when the header is consumed
        pytest.raises(FieldSelectionError, list,
                      rollingaggregate([('t', 'v')], 'missing', 'v', 3,
                                       **kwargs))
        pytest.raises(FieldSelectionError, list,
                      rollingaggregate([('t', 'v')], 't', 'missing', 3,
                                       **kwargs))
        pytest.raises(FieldSelectionError, list,
                      rollingaggregate([('t', 'v')], 't', 'v', 3,
                                       by='missing', **kwargs))
        # non-string and duplicate header field names
        pytest.raises(ArgumentError, list,
                      rollingaggregate([(1, 't', 'v')], 't', 'v', 3,
                                       **kwargs))
        pytest.raises(ArgumentError, list,
                      rollingaggregate([('t', 'v', 'v')], 't', 'v', 3,
                                       **kwargs))
        # appended columns must not clash with the existing header
        pytest.raises(ArgumentError, list,
                      rollingaggregate([('t', 'v', 'window_sum')], 't', 'v',
                                       3, **kwargs))
        pytest.raises(ArgumentError, list,
                      rollingaggregate([('t', 'v', 'w_max')], 't', 'v', 3,
                                       prefix='w_', **kwargs))


@pytest.mark.parametrize('badtime', [True, False, 1.5, '10', None])
def test_rollingaggregate_invalid_time(badtime):
    table = [('t', 'v'), (1, 10), (badtime, 20), (3, 30)]
    view = rollingaggregate(table, 't', 'v', 3, presorted=True)
    it = iter(view)
    next(it)  # header
    assert next(it)[:2] == (1, 10)  # row 1 is fine
    with pytest.raises(ArgumentError) as excinfo:
        next(it)
    assert 'row 2' in str(excinfo.value)


@pytest.mark.parametrize('badvalue', [True, False, 1.5, '10'])
def test_rollingaggregate_invalid_value(badvalue):
    table = [('t', 'v'), (1, 10), (2, badvalue), (3, 30)]
    view = rollingaggregate(table, 't', 'v', 3, presorted=True)
    it = iter(view)
    next(it)  # header
    assert next(it)[:2] == (1, 10)  # row 1 is fine
    with pytest.raises(ArgumentError) as excinfo:
        next(it)
    assert 'row 2' in str(excinfo.value)


def test_rollingaggregate_row_width_errors():
    # short row
    table = [('t', 'v'), (1, 10), (2,), (3, 30)]
    view = rollingaggregate(table, 't', 'v', 3, presorted=True)
    it = iter(view)
    next(it)
    next(it)
    with pytest.raises(ArgumentError) as excinfo:
        next(it)
    assert 'row 2' in str(excinfo.value)
    # long row
    table = [('t', 'v'), (1, 10, 99)]
    with pytest.raises(ArgumentError) as excinfo:
        list(rollingaggregate(table, 't', 'v', 3, presorted=True))
    assert 'row 1' in str(excinfo.value)


def test_rollingaggregate_error_row_numbers_are_original():
    # unsorted input: the bad row is the 3rd row of the original input,
    # and the error must point there even though sorting moves it around
    table = [('t', 'v'),
             (5, 1),
             (1, 2),
             (3, 'bad'),
             (2, 4)]
    view = rollingaggregate(table, 't', 'v', 3, buffersize=2)
    with pytest.raises(ArgumentError) as excinfo:
        list(view)
    assert 'row 3' in str(excinfo.value)


def test_rollingaggregate_presorted_order_regression():
    table = [('g', 't', 'v'),
             ('a', 1, 10),
             ('a', 3, 30),
             ('a', 2, 20)]  # time goes backwards within the group
    view = rollingaggregate(table, 't', 'v', 3, by='g', presorted=True)
    it = iter(view)
    next(it)
    next(it)
    next(it)  # rows 1 and 2 are produced before the bad row is reached
    with pytest.raises(ArgumentError) as excinfo:
        next(it)
    assert 'row 3' in str(excinfo.value)
    # group regression
    table = [('g', 't', 'v'), ('b', 1, 1), ('a', 1, 2)]
    with pytest.raises(ArgumentError) as excinfo:
        list(rollingaggregate(table, 't', 'v', 3, by='g', presorted=True))
    assert 'row 2' in str(excinfo.value)
    # equal keys are fine
    table = [('g', 't', 'v'), ('a', 1, 1), ('a', 1, 2), ('a', 2, 3)]
    assert len(list(rollingaggregate(table, 't', 'v', 3, by='g',
                                     presorted=True))) == 4
    # and the default path sorts, so it accepts the same input
    table = [('g', 't', 'v'), ('b', 1, 1), ('a', 1, 2)]
    assert len(list(rollingaggregate(table, 't', 'v', 3, by='g'))) == 3


class _RollingProbeSource(object):
    """A source that records how many iterators were requested from it,
    how many data rows were actually pulled, and how many iterators were
    closed. Can be made to fail at a given (1-based) data row."""

    def __init__(self, table, failat=None):
        self.table = [tuple(row) for row in table]
        self.failat = failat
        self.iterators = 0
        self.rows_consumed = 0
        self.closed = 0

    def __iter__(self):
        self.iterators += 1
        return self._gen()

    def _gen(self):
        try:
            yield self.table[0]
            for i, row in enumerate(self.table[1:], start=1):
                if self.failat is not None and i == self.failat:
                    raise RuntimeError('probe failure at row %d' % i)
                self.rows_consumed += 1
                yield row
        finally:
            self.closed += 1


def test_rollingaggregate_presorted_reads_rows_on_demand():
    hdr = ('g', 't', 'v')
    rows = [('a', i, i * 10) for i in range(1, 11)]
    rows += [('b', i, i) for i in range(1, 6)]
    probe = _RollingProbeSource([hdr] + rows)
    view = rollingaggregate(probe, 't', 'v', 3, by='g', presorted=True)
    assert probe.iterators == 0  # creating the view reads nothing
    it = iter(view)
    assert probe.iterators == 0  # iter(view) reads nothing either
    next(it)  # the header
    assert probe.iterators == 1  # exactly one iterator requested
    assert probe.rows_consumed == 0
    for k in range(1, len(rows) + 1):
        next(it)
        # each output row is available after reading exactly its own input
        # row: no look-ahead into future rows, groups, or the whole input
        assert probe.rows_consumed == k
    pytest.raises(StopIteration, next, it)
    assert probe.closed == 1  # exhausted iterator was released


def test_rollingaggregate_default_path_consumption():
    hdr = ('t', 'v')
    rows = [(i, i) for i in range(10)]
    probe = _RollingProbeSource([hdr] + rows)
    view = rollingaggregate(probe, 't', 'v', 3)
    assert probe.iterators == 0
    it = iter(view)
    assert probe.iterators == 0
    next(it)  # the header alone does not consume any data row
    assert probe.iterators == 1
    assert probe.rows_consumed == 0
    next(it)  # sorting consumes the whole input before the first data row
    assert probe.rows_consumed == len(rows)
    list(it)
    assert probe.closed == 1


def test_rollingaggregate_close_releases_source():
    table = [('t', 'v')] + [(i, i) for i in range(10)]
    # presorted: rows are pulled on demand, so closing early must release
    # the source and stop any further reading
    probe = _RollingProbeSource(table)
    view = rollingaggregate(probe, 't', 'v', 3, presorted=True)
    it = iter(view)
    next(it)
    next(it)
    assert probe.closed == 0
    assert probe.rows_consumed == 1
    it.close()
    assert probe.closed == 1
    assert probe.rows_consumed == 1
    pytest.raises(StopIteration, next, it)
    assert probe.rows_consumed == 1
    # default path: the sort has already consumed (and so released) the
    # source by the time the first data row appears; closing must be
    # harmless and read nothing more
    probe = _RollingProbeSource(table)
    view = rollingaggregate(probe, 't', 'v', 3, buffersize=3)
    it = iter(view)
    next(it)
    next(it)
    assert probe.rows_consumed == 10
    assert probe.closed == 1
    it.close()
    assert probe.closed == 1
    assert probe.rows_consumed == 10


def test_rollingaggregate_source_exception_propagates_and_cleans_up():
    table = [('t', 'v')] + [(i, i) for i in range(10)]
    for presorted in (True, False):
        probe = _RollingProbeSource(table, failat=4)
        view = rollingaggregate(probe, 't', 'v', 3, presorted=presorted,
                                buffersize=3)
        with pytest.raises(RuntimeError) as excinfo:
            list(view)
        # the source exception propagates as-is
        assert 'probe failure at row 4' in str(excinfo.value)
        assert probe.closed == 1


def test_rollingaggregate_cleanup_does_not_mask_source_exception():
    class BadCloseIterator(object):
        def __init__(self, table, failat):
            self.rows = iter(table)
            self.failat = failat
            self.pos = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            self.pos += 1
            if self.pos == self.failat:
                raise ValueError('original failure')
            return next(self.rows)

        def close(self):
            self.closed = True
            raise RuntimeError('close failure')

    class BadCloseSource(object):
        def __init__(self, table, failat):
            self.table = table
            self.failat = failat
            self.iterator = None

        def __iter__(self):
            self.iterator = BadCloseIterator(self.table, self.failat)
            return self.iterator

    table = [('t', 'v')] + [(i, i) for i in range(10)]
    source = BadCloseSource(table, failat=3)
    view = rollingaggregate(source, 't', 'v', 3, presorted=True)
    with pytest.raises(ValueError) as excinfo:
        list(view)
    # the original exception wins over the cleanup failure
    assert 'original failure' in str(excinfo.value)
    assert source.iterator.closed


def test_rollingaggregate_tempfiles_cleaned_up():
    tmpdir = tempfile.mkdtemp()
    try:
        table = [('g', 't', 'v')] + [('a', i, i) for i in range(20)]
        # early close: chunk files must be removed
        view = rollingaggregate(table, 't', 'v', 3, buffersize=2,
                                tempdir=tmpdir)
        it = iter(view)
        next(it)
        next(it)
        assert os.listdir(tmpdir) != []  # sanity: chunks were spilled
        it.close()
        gc.collect()
        assert os.listdir(tmpdir) == []
        # full exhaustion: same
        view = rollingaggregate(table, 't', 'v', 3, buffersize=2,
                                tempdir=tmpdir)
        list(view)
        gc.collect()
        assert os.listdir(tmpdir) == []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rollingaggregate_repeated_iteration():
    table = [['g', 't', 'v'], ['a', 2, 20], ['a', 1, 10]]
    view = rollingaggregate(table, 't', 'v', 3, by='g')
    first = list(view)
    assert first == _naive_rollingaggregate(table, 't', 'v', 3, by='g')
    # a second pass over a repeatable source gives the same result and
    # does not reuse cached window state
    assert list(view) == first
    # and it reflects the current state of the source
    table.append(['a', 3, 30])
    expect = _naive_rollingaggregate(table, 't', 'v', 3, by='g')
    assert list(view) == expect
    assert list(view) != first
    # independent iterators have independent window state
    it1, it2 = iter(view), iter(view)
    for expected_row in expect:
        assert next(it1) == expected_row
        assert next(it2) == expected_row


def test_rollingaggregate_fluent_and_exported():
    import petl as etl
    table = [('t', 'v'), (1, 10), (2, 20)]
    expect = _naive_rollingaggregate(table, 't', 'v', 2)
    ieq(expect, etl.rollingaggregate(table, 't', 'v', 2))
    ieq(expect, etl.wrap(table).rollingaggregate('t', 'v', 2))
    # the result chains back into the fluent pipeline
    chained = etl.wrap(table).rollingaggregate('t', 'v', 2) \
        .cut('t', 'window_sum')
    ieq((('t', 'window_sum'), (1, 10), (2, 30)), chained)


class _CountingInt(int):
    """An int that counts how many times it is compared."""

    ncomparisons = 0

    def __lt__(self, other):
        _CountingInt.ncomparisons += 1
        return int.__lt__(self, other)

    def __le__(self, other):
        _CountingInt.ncomparisons += 1
        return int.__le__(self, other)

    def __gt__(self, other):
        _CountingInt.ncomparisons += 1
        return int.__gt__(self, other)

    def __ge__(self, other):
        _CountingInt.ncomparisons += 1
        return int.__ge__(self, other)


@pytest.mark.parametrize('width', [500, 10 ** 9])
@pytest.mark.parametrize('shape', ['random', 'increasing', 'decreasing',
                                   'constant'])
def test_rollingaggregate_workload_is_linear_in_rows(width, shape):
    # For N rows and peak window W, maintaining the statistics must cost
    # O(N) amortised (better than the allowed O(N log(W+1))): each value
    # enters and leaves each monotone deque at most once. An implementation
    # that rescans the window for every row would need ~N*W comparisons.
    n = 4000
    rng = random.Random(7)
    rows = []
    for i in range(n):
        if shape == 'random':
            v = rng.randint(0, 10 ** 6)
        elif shape == 'increasing':
            v = i
        elif shape == 'decreasing':
            v = n - i
        else:
            v = 42
        rows.append((i, _CountingInt(v)))
    table = [('t', 'v')] + rows
    _CountingInt.ncomparisons = 0
    result = list(rollingaggregate(table, 't', 'v', width, presorted=True))
    ncomps = _CountingInt.ncomparisons
    assert len(result) == n + 1
    # spot-check correctness of one row deep inside a large window;
    # row i has time i, so its window covers rows lo..k (times > k - width)
    k = 1234
    lo = max(0, k - width + 1)
    vals = [int(rows[i][1]) for i in range(lo, k + 1)]
    expect_row = (k, rows[k][1], len(vals), sum(vals), min(vals),
                  max(vals))
    assert result[k + 1] == expect_row
    # amortised O(1) per row: a generous bound of 10 comparisons per row;
    # a per-row window rescan would need ~(n - width) * (width - 1) * 2
    # comparisons for width=500, i.e. millions
    assert 0 < ncomps <= 10 * n


def test_rollingaggregate_drops_expired_history():
    # values are tracked for liveness; once a row has expired from every
    # window its value objects must be released, so the retained set is
    # bounded by the live window, not by the history seen so far
    n = 3000
    width = 10

    class TrackedInt(int):
        alive = 0

        def __new__(cls, x):
            TrackedInt.alive += 1
            return int.__new__(cls, x)

        def __del__(self):
            TrackedInt.alive -= 1

    def source():
        yield ('t', 'v')
        for i in range(n):
            yield (i, TrackedInt(i))

    view = rollingaggregate(source(), 't', 'v', width, presorted=True)
    it = iter(view)
    next(it)  # header
    halfway = n // 2
    for _ in range(halfway):
        next(it)
    gc.collect()
    # only the current window (plus the row in flight) may still be alive;
    # an implementation keeping expired history would still hold ~halfway
    assert 0 < TrackedInt.alive <= 4 * width
    for row in it:
        pass
    del row, it
    gc.collect()
    assert TrackedInt.alive <= 4 * width
