from __future__ import absolute_import, print_function, division

import gc
import os
import resource
import subprocess
import sys
import tempfile

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError


HEADER_LEFT = ('g', 'h', 'start', 'stop', 'left_value')
HEADER_RIGHT = ('g', 'h', 'start', 'stop', 'right_value')
GROUPS = ('a', 'b', None, 1)


def reference_join(left, right, how='inner', missing=None):
    """Deliberately simple O(n*m) reference implementation."""
    rows = [tuple(row) for row in left]
    lhdr, left_rows = rows[0], rows[1:]
    rows = [tuple(row) for row in right]
    rhdr, right_rows = rows[0], rows[1:]
    group_fields = ('g', 'h')
    rgroup_positions = [rhdr.index(f) for f in group_fields]
    rout_positions = [i for i, _ in enumerate(rhdr)
                      if i not in rgroup_positions]
    out = tuple(lhdr) + tuple('right_' + rhdr[i] for i in rout_positions)
    lkey = [lhdr.index(f) for f in group_fields]
    rkey = [rhdr.index(f) for f in group_fields]
    ls, le = lhdr.index('start'), lhdr.index('stop')
    rs, re_ = rhdr.index('start'), rhdr.index('stop')
    ordered_left = sorted(
        enumerate(left_rows),
        key=lambda item: (etl.comparison.Comparable(
            tuple(item[1][i] for i in lkey)), item[1][ls], item[0]))
    result = [out]
    for lpos, lrow in ordered_left:
        matches = []
        for rpos, rrow in enumerate(right_rows):
            if tuple(lrow[i] for i in lkey) != tuple(rrow[i] for i in rkey):
                continue
            if lrow[ls] < rrow[re_] and rrow[rs] < lrow[le]:
                matches.append((rrow[rs], rpos, rrow))
        matches.sort(key=lambda item: (item[0], item[1]))
        if matches:
            for _, _, rrow in matches:
                result.append(tuple(lrow) + tuple(rrow[i] for i in rout_positions))
        elif how == 'left':
            result.append(tuple(lrow) + tuple(missing for _ in rout_positions))
    return result


def make_rows(seed, side):
    rnd = _DeterministicRandom(seed)
    rows = []
    for group_index, group in enumerate(GROUPS):
        for i in range(18):
            start = rnd.nint(0, 24)
            stop = start + rnd.nint(1, 9)
            second = rnd.choice(('x', 'y', None, 2))
            rows.append((group, second, start, stop, '%s-%d-%d' % (side, group_index, i)))
    # Explicit edge cases: touching, nested, crossing and duplicate intervals.
    rows.extend([
        ('edge', 'x', 0, 10, 'long'),
        ('edge', 'x', 1, 9, 'nested'),
        ('edge', 'x', 9, 12, 'touch-and-overlap'),
        ('edge', 'x', 10, 11, 'touch-only'),
        ('edge', 'x', 3, 4, 'dup'),
        ('edge', 'x', 3, 4, 'dup'),
        ('edge', 'y', 0, 2, 'other-key'),
    ])
    return rows


class _DeterministicRandom(object):
    def __init__(self, seed):
        self.state = seed & 0xffffffff

    def nint(self, low, high):
        # xorshift keeps these tests independent of Python hashing or random
        # module version.
        x = self.state
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        self.state = x or 1
        return low + x % (high - low + 1)

    def choice(self, values):
        return values[self.nint(0, len(values) - 1)]


def sorted_table(rows):
    rows = sorted(
        enumerate(rows),
        key=lambda item: (etl.comparison.Comparable((item[1][0], item[1][1])),
                          item[1][2], item[0]))
    return [HEADER_LEFT] + [r for _, r in rows]


def test_intervaljoinstream_matches_reference_inner_and_left():
    left_rows = make_rows(17, 'L')
    right_rows = make_rows(29, 'R')
    left = [HEADER_LEFT] + left_rows
    right = [HEADER_RIGHT] + right_rows
    ordered_left = [HEADER_LEFT] + sorted_table(left_rows)[1:]
    ordered_right = [HEADER_RIGHT] + sorted_table(right_rows)[1:]

    for how in ('inner', 'left'):
        expected = reference_join([HEADER_LEFT] + left_rows,
                                  [HEADER_RIGHT] + right_rows, how=how,
                                  missing='<missing>')
        actual_unsorted = list(etl.intervaljoinstream(
            left, right, by=('g', 'h'), how=how, buffersize=7,
            missing='<missing>'))
        actual_sorted = list(etl.intervaljoinstream(
            ordered_left, ordered_right, by=('g', 'h'), how=how,
            presorted=True, buffersize=7, missing='<missing>'))
        assert actual_unsorted == expected
        assert actual_sorted == expected


def test_intervaljoinstream_duplicate_rows_and_touching_endpoints():
    left = [('g', 'start', 'stop'), ('a', 1, 5), ('a', 1, 5)]
    right = [('g', 'start', 'stop'), ('a', 1, 2), ('a', 1, 2), ('a', 1, 2),
             ('a', 5, 6), ('b', 1, 2)]
    view = etl.intervaljoinstream(left, right, by='g', buffersize=2)
    rows = list(view)[1:]
    assert len(rows) == 6
    assert all(row[-2:] == (1, 2) for row in rows)


def test_intervaljoinstream_empty_tables_and_headers():
    left = [HEADER_LEFT]
    right = [HEADER_RIGHT]
    expected = [tuple(HEADER_LEFT) +
                tuple('right_' + f for f in HEADER_RIGHT if f not in ('g', 'h'))]
    assert list(etl.intervaljoinstream(left, right, by=('g', 'h'))) == expected
    assert list(etl.intervaljoinstream(left, right, by=('g', 'h'),
                                       presorted=True)) == expected

    for side, args in (('left', ([], right)), ('right', (left, []))):
        try:
            list(etl.intervaljoinstream(*args, by=('g', 'h')))
        except ArgumentError as exc:
            assert side in str(exc)
        else:
            raise AssertionError('missing header should fail')


def test_intervaljoinstream_validation_and_field_conflicts():
    left = [('g', 'start', 'stop'), ('a', 1, 2)]
    right = [('g', 'start', 'stop'), ('a', 1, 2)]

    invalid_arguments = [
        (dict(by='g', how='outer'), False),
        (dict(by='g', buffersize=0), False),
        (dict(by='g', buffersize=True), False),
        (dict(by=()), True),
        (dict(by=('g', 'g')), True),
        (dict(by='g', lstart='stop', lstop='stop'), False),
        (dict(by='g', rstart='stop', rstop='stop'), False),
        (dict(by='g', presorted='yes'), False)]
    for kwargs, _ in invalid_arguments:
        try:
            list(etl.intervaljoinstream(left, right, **kwargs))
        except ArgumentError:
            pass
        else:
            raise AssertionError('expected ArgumentError for %r' % kwargs)

    try:
        list(etl.intervaljoinstream([('g', 'start', 'start')], right, by='g'))
    except ArgumentError:
        pass
    else:
        raise AssertionError('duplicate header should fail')

    try:
        list(etl.intervaljoinstream([('x', 'start', 'stop')], right, by='g'))
    except FieldSelectionError:
        pass
    else:
        raise AssertionError('missing field should fail')

    bad_rows = [
        ([('g', 'start', 'stop'), ('a', 1, 1)], right, 'start >= stop'),
        ([('g', 'start', 'stop'), ('a', True, 2)], right, 'invalid'),
        ([('g', 'start', 'stop'), ('a', 1, float('nan'))], right, 'invalid'),
        ([('g', 'start', 'stop'), (1.2, 1, 2)], right, 'invalid group'),
        (left, [('g', 'start', 'stop'), ('a', 1)], 'has 2 values'),
    ]
    for bad_left, bad_right, message in bad_rows:
        try:
            list(etl.intervaljoinstream(bad_left, bad_right, by='g'))
        except ArgumentError as exc:
            if message is not None:
                assert message in str(exc)
            assert 'data row 1' in str(exc)
            if message == 'has 2 values':
                assert 'right data row 1' in str(exc)
        else:
            raise AssertionError('expected validation failure: %s' % message)

    conflicting = etl.intervaljoinstream(
        left, [('g', 'start', 'stop', 'value'), ('a', 1, 2, 'x')],
        by='g', lprefix='', rprefix='')
    try:
        list(conflicting)
    except ArgumentError:
        pass
    else:
        raise AssertionError('output field conflict should fail')


def test_intervaljoinstream_preserves_large_integer_coordinate_order():
    big = 10 ** 30
    left = [('start', 'stop'), (big, big + 10), (big + 1, big + 2),
            (-big - 10, -big), (-big - 9, -big - 8)]
    right = [('start', 'stop'), (big + 1, big + 2), (-big - 9, -big - 8)]
    expected_values = {(big, big + 10), (big + 1, big + 2),
                       (-big - 10, -big), (-big - 9, -big - 8)}
    actual = {row[:2] for row in
              list(etl.intervaljoinstream(left, right))[1:]}
    assert actual == expected_values


def test_intervaljoinstream_presorted_order_regression_and_tail_errors():
    left = [('g', 'start', 'stop'), ('a', 1, 2), ('a', 3, 4)]
    backwards_right = [('g', 'start', 'stop'), ('a', 5, 6), ('a', 2, 3)]
    try:
        list(etl.intervaljoinstream(left, backwards_right, by='g',
                                    presorted=True))
    except ArgumentError as exc:
        assert 'right input is out of order at data row 2' in str(exc)
    else:
        raise AssertionError('expected backwards right row')

    backwards_left = [('g', 'start', 'stop'), ('b', 1, 2), ('a', 3, 4)]
    try:
        list(etl.intervaljoinstream(backwards_left,
                                    [('g', 'start', 'stop')], by='g',
                                    presorted=True))
    except ArgumentError as exc:
        assert 'left input is out of order at data row 2' in str(exc)
    else:
        raise AssertionError('expected backwards left row')

    # The invalid right tail must be reached even though no left row can
    # match it.
    unmatched_left = [('g', 'start', 'stop'), ('a', 1, 2)]
    invalid_tail = [('g', 'start', 'stop'), ('z', 1, 2), ('z', 2, 1)]
    for presorted in (False, True):
        try:
            list(etl.intervaljoinstream(unmatched_left, invalid_tail, by='g',
                                        how='left', presorted=presorted))
        except ArgumentError as exc:
            assert 'right data row 2' in str(exc)
        else:
            raise AssertionError('invalid right tail must be validated')


class CountingSource(object):
    def __init__(self, rows):
        self.rows = rows
        self.iterators = 0
        self.closed = 0

    def __iter__(self):
        self.iterators += 1
        return _CountingIterator(self.rows, self)


class _CountingIterator(object):
    def __init__(self, rows, source):
        self.iterator = iter(rows)
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    next = __next__

    def close(self):
        self.source.closed += 1


def test_intervaljoinstream_lazy_rereadable_and_independent_iterators():
    left = CountingSource([('g', 'start', 'stop'), ('a', 1, 3)])
    right = CountingSource([('g', 'start', 'stop'), ('a', 2, 4)])
    view = etl.intervaljoinstream(left, right, by='g')
    assert left.iterators == right.iterators == 0
    iterator = iter(view)
    assert left.iterators == right.iterators == 0
    assert next(iterator)
    assert left.iterators == right.iterators == 1
    first = list(iterator)
    second = list(view)
    assert first == second[1:]
    assert left.iterators == right.iterators == 2
    assert left.closed == right.closed == 2

    one = iter(view)
    two = iter(view)
    assert next(one) == next(two)
    assert left.iterators == right.iterators == 4
    list(one)
    list(two)
    assert left.iterators == right.iterators == 4


class Payload(object):
    live = 0
    peak = 0

    def __init__(self, marker):
        self.marker = marker
        Payload.live += 1
        Payload.peak = max(Payload.peak, Payload.live)

    def __del__(self):
        Payload.live -= 1


def _payload_rows(side, n, long_interval=False):
    yield ('g', 'start', 'stop', 'payload')
    if long_interval:
        yield ('g', 0, n + 10, Payload('%s-long' % side))
        return
    for i in range(n):
        # Yield directly: retaining the whole source list in this test would
        # make the payload probe measure the test fixture rather than the
        # transform.
        yield ('g', i, i + 2, Payload('%s-%d' % (side, i)))


def test_intervaljoinstream_bounded_retention_and_early_delivery():
    Payload.live = Payload.peak = 0
    n = 1200
    left = _payload_rows('L', n)
    right = _payload_rows('R', n)
    view = etl.intervaljoinstream(left, right, by='g', presorted=True,
                                  buffersize=5)
    iterator = iter(view)
    next(iterator)
    first = next(iterator)
    # The first pair is available while most right rows have not been read.
    assert first[-1].marker == 'R-0'
    peak_partial = Payload.peak
    for _ in iterator:
        pass
    # SQLite owns serialized row blobs; Python payload retention must not grow
    # with the number of active intervals or total rows.
    assert peak_partial < 50
    assert Payload.peak < 50

    # A short left interval following a long one still receives all of its
    # matches, even though the long left interval consumed many right rows.
    short_left = [
        ('g', 'start', 'stop', 'id'),
        ('g', 0, 1000, 'long'),
        ('g', 100, 102, 'short'),
    ]
    right_rows = [('g', 'start', 'stop', 'id')] + [
        ('g', i, i + 1, i) for i in range(1000)]
    result = list(etl.intervaljoinstream(short_left, right_rows, by='g',
                                         presorted=True))[1:]
    assert [r[-1] for r in result if r[3] == 'short'] == [100, 101]
    assert len([r for r in result if r[3] == 'long']) == 1000


def _many_match_right(n):
    yield ('g', 'start', 'stop', 'payload')
    for i in range(n):
        yield ('g', i, i + 1, Payload(i))


def test_intervaljoinstream_many_matches_for_one_left_row():
    Payload.live = Payload.peak = 0
    left = [('g', 'start', 'stop'), ('g', 0, 5000)]
    view = etl.intervaljoinstream(left, _many_match_right(5000), by='g',
                                  presorted=True, buffersize=3)
    iterator = iter(view)
    next(iterator)
    first = next(iterator)
    assert first[-1].marker == 0
    partial_peak = Payload.peak
    count = 1
    for _ in iterator:
        count += 1
    assert count == 5000
    assert partial_peak < 30
    assert Payload.peak < 30


def test_intervaljoinstream_tempdir_cleanup_and_staging_failure():
    with tempfile.TemporaryDirectory() as directory:
        before = set(os.listdir(directory))
        left = [('g', 'start', 'stop')] + [('a', i, i + 2) for i in range(100)]
        right = [('g', 'start', 'stop')] + [('a', i, i + 2) for i in range(100)]
        list(etl.intervaljoinstream(left, right, by='g', buffersize=3,
                                    tempdir=directory))
        gc.collect()
        assert set(os.listdir(directory)) == before

    class FailSource(object):
        def __iter__(self):
            return self._iter()

        def _iter(self):
            yield ('g', 'start', 'stop')
            for i in range(50):
                yield ('a', i, i + 1)
            raise RuntimeError('source failed during staging')

    with tempfile.TemporaryDirectory() as directory:
        try:
            list(etl.intervaljoinstream(FailSource(), right, by='g',
                                        buffersize=3, tempdir=directory))
        except RuntimeError as exc:
            assert str(exc) == 'source failed during staging'
        else:
            raise AssertionError('source exception should propagate')
        gc.collect()
        assert os.listdir(directory) == []


def test_intervaljoinstream_presorted_source_failure_cleanup():
    class FailingRight(object):
        def __iter__(self):
            yield ('g', 'start', 'stop')
            yield ('a', 1, 2)
            raise RuntimeError('presorted source failed')

    with tempfile.TemporaryDirectory() as directory:
        left = [('g', 'start', 'stop'), ('a', 1, 3)]
        view = etl.intervaljoinstream(left, FailingRight(), by='g',
                                      presorted=True, tempdir=directory)
        try:
            list(view)
        except RuntimeError as exc:
            assert str(exc) == 'presorted source failed'
        else:
            raise AssertionError('presorted source exception should propagate')
        gc.collect()
        assert os.listdir(directory) == []


def test_intervaljoinstream_explicit_close_stops_consumption_and_closes_source():
    left = CountingSource([('g', 'start', 'stop'), ('a', 1, 3)])
    right_rows = [('g', 'start', 'stop')] + [('a', i, i + 1)
                                             for i in range(100)]
    right = CountingSource(right_rows)
    iterator = iter(etl.intervaljoinstream(left, right, by='g',
                                           presorted=True))
    next(iterator)
    next(iterator)
    iterator.close()
    assert left.closed == right.closed == 1
    # Generators are closed before the remaining source rows are consumed.
    assert right.iterators == 1


def _fd_limited_child_script(directory):
    import petl as etl
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(24, soft), hard))
    left = [('g', 'h', 'start', 'stop')]
    right = [('g', 'h', 'start', 'stop')]
    for g in range(12):
        for i in range(160):
            left.append((g, i % 3, i, i + 5))
            right.append((g, i % 3, i + 1, i + 7))
    result = etl.intervaljoinstream(left, right, by=('g', 'h'),
                                    buffersize=5, tempdir=directory)
    count = len(list(result))
    assert count > 1000


def test_intervaljoinstream_runs_with_tight_file_descriptor_limit():
    with tempfile.TemporaryDirectory() as directory:
        subprocess.check_call([sys.executable, '-I', '-c',
                               'import sys; sys.path.insert(0, %r); '
                               'from petl.test.transform.test_intervaljoinstream '
                               'import _fd_limited_child_script; '
                               '_fd_limited_child_script(%r)'
                               % (os.getcwd(), directory)])
