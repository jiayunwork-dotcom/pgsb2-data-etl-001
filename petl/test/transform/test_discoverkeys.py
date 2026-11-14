from __future__ import absolute_import, print_function, division

import math
import os
import random

import pytest

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError


# ---------------------------------------------------------------------------
# Exhaustive reference implementation (deliberately simple, in-memory)
# ---------------------------------------------------------------------------


def _py_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def reference_discoverkeys(header, rows, columns=None, max_width=None):
    """Brute-force reference: enumerate subsets, test keys directly."""
    header = tuple(header)
    rows = [tuple(r) for r in rows]
    if columns is None:
        chosen = tuple(range(len(header)))
    else:
        chosen = tuple(header.index(c) for c in columns)

    # columns with any missing value can never belong to a key
    viable = tuple(i for i in chosen
                   if not any(_py_missing(r[i]) for r in rows))

    def is_key(combo):
        if not combo:
            return False
        seen = set()
        for r in rows:
            proj = tuple(r[i] for i in combo)
            if proj in seen:
                return False
            seen.add(proj)
        return True

    limit = len(viable) if max_width is None else min(max_width, len(viable))
    keys = []
    for width in range(1, limit + 1):
        for positions in _combinations(range(len(viable)), width):
            combo = tuple(viable[p] for p in positions)
            if is_key(combo):
                keys.append(combo)
    # minimality: drop every key that contains another key as a proper subset
    minimal = []
    for k in keys:
        kset = set(k)
        if not any(set(other) < kset for other in keys):
            minimal.append(k)
    minimal.sort(key=lambda combo: (len(combo), combo))
    return tuple(tuple(header[i] for i in combo) for combo in minimal)


def _combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# ---------------------------------------------------------------------------
# Source test doubles
# ---------------------------------------------------------------------------


class CountingSource(object):
    """Records how often its iterator is requested and consumed."""

    def __init__(self, rows):
        self.rows = [tuple(r) for r in rows]
        self.iterations = 0
        self.consumed = 0
        self.closed = 0

    def __iter__(self):
        self.iterations += 1
        rows = self.rows
        source = self

        def gen():
            try:
                for row in rows:
                    source.consumed += 1
                    yield list(row)
            finally:
                source.closed += 1

        return gen()


class FailingSource(object):
    """Raises a specific error from the iterator at a given data row."""

    def __init__(self, rows, error, fail_at=None):
        self.rows = [tuple(r) for r in rows]
        self.error = error
        self.fail_at = fail_at
        self.closed = 0

    def __iter__(self):
        rows = self.rows
        source = self

        def gen():
            try:
                for idx, row in enumerate(rows):
                    if source.fail_at is not None and idx == source.fail_at:
                        raise source.error
                    yield list(row)
                if source.fail_at is None:
                    raise source.error
            finally:
                source.closed += 1

        return gen()


class CloseTrackingIterator(object):
    def __init__(self, rows):
        self.rows = iter(rows)
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.rows)

    next = __next__

    def close(self):
        self.close_calls += 1


def table(header, *rows):
    return [list(header)] + [list(r) for r in rows]


# ---------------------------------------------------------------------------
# Basic behaviour and ordering
# ---------------------------------------------------------------------------


def test_spec_example():
    t = table(('a', 'b', 'c'),
              (0, 0, 'x'), (0, 1, 'y'), (1, 0, 'z'))
    assert etl.discoverkeys(t) == (('c',), ('a', 'b'))


def test_duplicate_full_rows_mean_no_key():
    t = table(('a', 'b', 'c'),
              (0, 0, 'x'), (0, 1, 'y'), (1, 0, 'z'), (0, 0, 'x'))
    assert etl.discoverkeys(t) == ()


def test_result_is_tuple_of_tuples():
    t = table(('a', 'b'), (1, 2), (3, 4))
    result = etl.discoverkeys(t)
    assert isinstance(result, tuple)
    assert all(isinstance(k, tuple) for k in result)


def test_ordering_by_width_then_header_positions():
    # single-column key plus wider minimal keys, out-of-order discovery check
    t = table(('a', 'b', 'c', 'd'),
              (0, 0, 0, 'u'),
              (0, 1, 1, 'v'),
              (1, 0, 1, 'w'),
              (1, 1, 0, 'x'))
    result = etl.discoverkeys(t)
    positions = [tuple(t[0].index(name) for name in k) for k in result]
    widths = [len(k) for k in result]
    assert widths == sorted(widths)
    # within each width, position tuples are lexicographically sorted
    by_width = {}
    for k, p in zip(result, positions):
        by_width.setdefault(len(k), []).append(p)
    for ps in by_width.values():
        assert ps == sorted(ps)
    assert result == reference_discoverkeys(t[0], t[1:])


def test_minimal_keys_of_several_widths_coexist():
    t = table(('a', 'b', 'c'),
              (0, 0, 'x'), (0, 1, 'y'), (1, 0, 'z'))
    assert etl.discoverkeys(t) == reference_discoverkeys(t[0], t[1:])


def test_superset_keys_are_not_listed():
    # 'id' is a key; ('id', 'name') contains it and must not be returned
    t = table(('id', 'name'), (1, 'a'), (2, 'b'), (3, 'c'))
    assert etl.discoverkeys(t) == (('id',), ('name',))
    t = table(('id', 'grp'), (1, 'x'), (2, 'x'), (3, 'y'))
    assert etl.discoverkeys(t) == (('id',),)


# ---------------------------------------------------------------------------
# Empty / single-row batches
# ---------------------------------------------------------------------------


def test_header_only_every_non_missing_column_is_minimal_key():
    t = table(('a', 'b', 'c'))
    assert etl.discoverkeys(t) == (('a',), ('b',), ('c',))


def test_single_row_non_missing_columns_are_minimal_keys():
    t = table(('a', 'b', 'c'), (1, None, 'x'))
    assert etl.discoverkeys(t) == (('a',), ('c',))


def test_single_row_nan_column_excluded():
    t = table(('a', 'b'), (1, float('nan')))
    assert etl.discoverkeys(t) == (('a',),)


# ---------------------------------------------------------------------------
# Missing values and value semantics
# ---------------------------------------------------------------------------


def test_none_and_nan_are_missing_and_exclude_columns():
    t = table(('a', 'b', 'c'),
              (1, 2.0, 'x'),
              (2, float('nan'), 'y'),
              (3, 4.0, 'z'))
    assert etl.discoverkeys(t) == (('a',), ('c',))
    t = table(('a', 'b'), (None, 1), (2, 2))
    assert etl.discoverkeys(t) == (('b',),)


def test_infinities_are_legal_values():
    t = table(('a', 'b'),
              (float('inf'), 1), (float('-inf'), 2), (0.0, 3))
    assert etl.discoverkeys(t) == (('a',), ('b',))


def test_python_equality_true_one_float_one_collapse():
    t = table(('a', 'b', 'c'),
              (True, 0.0, 1), (1, -0.0, 2), (1.0, 5.0, 3))
    # column a collapses all rows and column b collapses all rows; c is key
    assert etl.discoverkeys(t) == (('c',),)
    assert etl.discoverkeys(t) == reference_discoverkeys(t[0], t[1:])


def test_signed_zero_equal():
    t = table(('a', 'b'), (0.0, 1), (-0.0, 2))
    assert etl.discoverkeys(t) == (('b',),)


def test_string_and_bytes_are_distinct():
    t = table(('a'), ('x'), (b'x'), ('y'))
    assert etl.discoverkeys(t) == (('a',),)


def test_large_integers_keep_precision():
    big = 10 ** 60
    t = table(('a'), (big,), (big + 1,), (big + 2,))
    assert etl.discoverkeys(t) == (('a',),)


def test_hash_collisions_do_not_merge_values():
    # Allowed built-in types whose hashes coincide cannot be fabricated
    # directly, but force the spill machinery into tiny buffers and heavy
    # collisions: correctness must rely on equality, never hashing alone.
    vals = [i for i in range(200)] + [i for i in range(200)]
    t = table(('a',), *[(v,) for v in vals])
    assert etl.discoverkeys(t, buffersize=7) == ()
    t2 = table(('a',), *[(v,) for v in range(400)])
    assert etl.discoverkeys(t2, buffersize=7) == (('a',),)


# ---------------------------------------------------------------------------
# columns / max_width / buffersize arguments
# ---------------------------------------------------------------------------


def test_columns_restricts_search():
    t = table(('a', 'b', 'c'),
              (1, 1, 'x'), (1, 2, 'y'), (2, 2, 'z'))
    # restricted to (a, b): neither alone unique (both repeat), pair is key
    assert etl.discoverkeys(t, columns=['a', 'b']) == (('a', 'b'),)
    # passing order must not change the normalised result
    assert etl.discoverkeys(t, columns=['b', 'a']) == (('a', 'b'),)
    # restricting to a single unique column returns just that key
    assert etl.discoverkeys(t, columns=['c']) == (('c',),)


def test_columns_missing_field_raises_field_selection_error():
    t = table(('a', 'b'), (1, 2))
    with pytest.raises(FieldSelectionError):
        etl.discoverkeys(t, columns=['a', 'nope'])


def test_columns_rejects_bad_shapes():
    t = table(('a', 'b'), (1, 2))
    for bad in ('a', '', 1, 1.5, True, object(), ['a', 'a'], (), []):
        with pytest.raises(ArgumentError):
            etl.discoverkeys(t, columns=bad)


def test_columns_non_string_entries_rejected():
    t = table(('a', 'b'), (1, 2))
    with pytest.raises(ArgumentError):
        etl.discoverkeys(t, columns=['a', 1])


def test_max_width_limits_results():
    t = table(('a', 'b', 'c'),
              (0, 0, 'x'), (0, 1, 'y'), (1, 0, 'z'))
    assert etl.discoverkeys(t, max_width=1) == (('c',),)
    assert etl.discoverkeys(t, max_width=2) == (('c',), ('a', 'b'))
    assert etl.discoverkeys(t, max_width=100) == (('c',), ('a', 'b'))


def test_max_width_with_no_key_that_narrow():
    t = table(('a', 'b'), (1, 1), (1, 2), (2, 1))
    assert etl.discoverkeys(t, max_width=1) == ()
    assert etl.discoverkeys(t, max_width=2) == (('a', 'b'),)


def test_bad_max_width_and_buffersize_rejected():
    t = table(('a'), (1,), (2,))
    for bad in (0, -1, 1.0, True, False, '8'):
        with pytest.raises(ArgumentError):
            etl.discoverkeys(t, max_width=bad)
        with pytest.raises(ArgumentError):
            etl.discoverkeys(t, buffersize=bad)


def test_argument_errors_raised_before_source_touched():
    source = CountingSource(table(('a', 'b'), (1, 2)))
    with pytest.raises(ArgumentError):
        etl.discoverkeys(source, buffersize=0)
    with pytest.raises(ArgumentError):
        etl.discoverkeys(source, max_width=True)
    with pytest.raises(ArgumentError):
        etl.discoverkeys(source, columns='a')
    assert source.iterations == 0


def test_field_selection_error_also_before_data_consumed():
    source = CountingSource(table(('a', 'b'), (1, 2)))
    with pytest.raises(FieldSelectionError):
        etl.discoverkeys(source, columns=['nope'])
    # header must be read to resolve field names, but no data row consumed
    assert source.iterations == 1
    assert source.consumed == 1  # header only


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------


def test_bad_headers():
    for bad in ([], (), [''], (''), ['a', 'a'], ('a', ''), [1, 2]):
        with pytest.raises(ArgumentError):
            etl.discoverkeys([list(bad) if isinstance(bad, (list, tuple))
                              else bad, [1, 2]])


def test_missing_header_row():
    with pytest.raises(ArgumentError):
        etl.discoverkeys(iter([]))


# ---------------------------------------------------------------------------
# Row width and cell-type errors
# ---------------------------------------------------------------------------


def test_short_and_long_rows_rejected_with_row_number():
    t = table(('a', 'b'), (1, 2), (3,))
    with pytest.raises(ArgumentError) as exc:
        etl.discoverkeys(t)
    assert 'row 2' in str(exc.value)
    t = table(('a', 'b'), (1, 2), (3, 4, 5))
    with pytest.raises(ArgumentError) as exc:
        etl.discoverkeys(t)
    assert 'row 2' in str(exc.value)


def test_unsupported_cell_type_located_by_row_and_field():
    t = table(('a', 'b'), (1, 2), (3, ['nope']))
    with pytest.raises(ArgumentError) as exc:
        etl.discoverkeys(t)
    message = str(exc.value)
    assert 'row 2' in message
    assert "'b'" in message


def test_unselected_column_cell_type_is_unconstrained():
    t = table(('a', 'b'), (1, ['anything']), (2, {'also': 'ok'}))
    assert etl.discoverkeys(t, columns=['a']) == (('a',),)


def test_tail_is_validated_even_when_no_key_is_possible(tmp_path):
    # duplicate rows prove "no key" early, but the bad (short) tail row must
    # still raise rather than the call succeeding
    good = table(('a', 'b'), (1, 1), (1, 1))
    rows = good + [('only',)]
    source = CountingSource(rows)
    with pytest.raises(ArgumentError):
        etl.discoverkeys(source, buffersize=2, tempdir=str(tmp_path))
    assert source.consumed == len(rows)


def test_source_error_in_tail_propagates(tmp_path):
    err = RuntimeError('boom-tail')
    rows = table(('a', 'b'), (1, 1), (1, 1), (2, 2))
    source = FailingSource(rows, err)
    with pytest.raises(RuntimeError) as exc:
        etl.discoverkeys(source, buffersize=2, tempdir=str(tmp_path))
    assert exc.value is err


# ---------------------------------------------------------------------------
# Resource semantics
# ---------------------------------------------------------------------------


def test_single_source_iterator_and_close(tmp_path):
    rows = table(('a', 'b'), *[(i, i % 3) for i in range(100)])
    source = CountingSource(rows)
    result = etl.discoverkeys(source, buffersize=8, tempdir=str(tmp_path))
    assert result == (('a',),)
    assert source.iterations == 1
    assert source.consumed == len(rows)
    assert source.closed == 1
    assert os.listdir(str(tmp_path)) == []


def test_close_called_on_supported_iterator():
    rows = table(('a'), *[(i,) for i in range(50)])
    it = CloseTrackingIterator(rows)
    etl.discoverkeys(it, buffersize=4)
    assert it.close_calls == 1


def test_tempdir_isolation_and_cleanup(tmp_path):
    marker = tmp_path / 'belongs-to-caller'
    marker.write_text('keep me')
    rows = table(('a', 'b'), *[(i, j) for i in range(20) for j in range(5)])
    etl.discoverkeys(rows, buffersize=4, tempdir=str(tmp_path))
    assert marker.exists()
    # only the caller's marker remains; every call temp area is gone
    assert os.listdir(str(tmp_path)) == ['belongs-to-caller']


def test_temp_files_cleaned_after_source_error(tmp_path):
    err = IOError('disk exploded')
    rows = table(('a'), *[(i,) for i in range(200)])
    source = FailingSource(rows, err, fail_at=150)
    with pytest.raises(IOError) as exc:
        etl.discoverkeys(source, buffersize=5, tempdir=str(tmp_path))
    assert exc.value is err
    assert os.listdir(str(tmp_path)) == []


def test_input_rows_are_not_mutated():
    rows = table(('a', 'b'), (1, 2), (3, 4))
    snapshot = [list(r) for r in rows]
    etl.discoverkeys(rows, buffersize=1)
    assert rows == snapshot


def test_repeated_calls_recompute_independently(tmp_path):
    rows1 = table(('a', 'b'), (1, 1), (2, 2))
    source = CountingSource(rows1)
    assert etl.discoverkeys(source, tempdir=str(tmp_path)) == (('a',), ('b',))
    # new batch with a duplicate: must not reuse any earlier state
    source.rows = [tuple(r) for r in table(('a', 'b'), (1, 1), (1, 2))]
    assert etl.discoverkeys(source, tempdir=str(tmp_path)) == (('b',),)


def test_call_after_failure_still_works(tmp_path):
    bad = table(('a', 'b'), (1, 2), (3,))
    with pytest.raises(ArgumentError):
        etl.discoverkeys(CountingSource(bad), tempdir=str(tmp_path))
    good = table(('a', 'b'), (1, 2), (3, 4))
    assert etl.discoverkeys(good, tempdir=str(tmp_path)) == (('a',), ('b',))


def test_chained_table_entry():
    t = etl.wrap(table(('a', 'b'), (1, 1), (2, 2)))
    assert t.discoverkeys() == (('a',), ('b',))


# ---------------------------------------------------------------------------
# Exhaustive cross-check against the brute-force reference
# ---------------------------------------------------------------------------


ALLOWED = [0, 1, True, False, 1.0, 0.0, -0.0, float('inf'), float('-inf'),
           None, float('nan'), 'x', 'y', b'x', b'y', 10 ** 30, 10 ** 30 + 1]


def test_exhaustive_reference_small_tables():
    rng = random.Random(20240517)
    for trial in range(120):
        ncols = rng.randint(1, 5)
        nrows = rng.randint(0, 8)
        header = tuple('c%d' % i for i in range(ncols))
        rows = []
        for _ in range(nrows):
            rows.append(tuple(rng.choice(ALLOWED) for _ in range(ncols)))
        for buffersize in (1, 3, 1000):
            got = etl.discoverkeys([list(header)] + [list(r) for r in rows],
                                   buffersize=buffersize)
            expected = reference_discoverkeys(header, rows)
            assert got == expected, (trial, header, rows, buffersize)


def test_exhaustive_with_columns_and_max_width():
    rng = random.Random(7719)
    for trial in range(80):
        ncols = rng.randint(2, 6)
        nrows = rng.randint(1, 7)
        header = tuple('c%d' % i for i in range(ncols))
        rows = [tuple(rng.choice(ALLOWED) for _ in range(ncols))
                for _ in range(nrows)]
        chosen = sorted(rng.sample(range(ncols), rng.randint(1, ncols)))
        columns = [header[i] for i in chosen]
        max_width = rng.choice([None, 1, 2, ncols, ncols + 5])
        got = etl.discoverkeys([list(header)] + [list(r) for r in rows],
                               columns=columns, max_width=max_width,
                               buffersize=rng.choice([1, 2, 5, 1000]))
        expected = reference_discoverkeys(header, rows, columns=columns,
                                          max_width=max_width)
        assert got == expected, (trial, columns, max_width)


def test_row_and_column_reordering_renaming_equivalence():
    rng = random.Random(424242)
    header = tuple('c%d' % i for i in range(6))
    rows = [tuple(rng.choice(ALLOWED[:12]) for _ in range(6))
            for _ in range(12)]
    base = etl.discoverkeys([list(header)] + [list(r) for r in rows],
                            buffersize=4)

    # rename columns collision-free: same key structure under new names
    renamed = tuple('f%d' % i for i in range(6))
    got_renamed = etl.discoverkeys([list(renamed)] + [list(r) for r in rows],
                                   buffersize=4)
    map_names = dict(zip(header, renamed))
    assert got_renamed == tuple(tuple(map_names[c] for c in k) for k in base)

    # reorder columns: results renormalise to the new header order
    perm = [3, 0, 5, 1, 4, 2]
    new_header = tuple(header[i] for i in perm)
    new_rows = [tuple(r[i] for i in perm) for r in rows]
    got_reorder = etl.discoverkeys([list(new_header)] +
                                   [list(r) for r in new_rows], buffersize=4)
    assert got_reorder == reference_discoverkeys(new_header, new_rows)
    # the same field sets are keys in both layouts
    assert {frozenset(k) for k in got_reorder} == \
        {frozenset(k) for k in base}


def test_buffer_size_does_not_change_result():
    rng = random.Random(99)
    header = tuple('c%d' % i for i in range(5))
    rows = [tuple(rng.choice(ALLOWED) for _ in range(5)) for _ in range(60)]
    results = {etl.discoverkeys([list(header)] + [list(r) for r in rows],
                                buffersize=bs)
               for bs in (1, 2, 7, 32, 1000)}
    assert len(results) == 1
    assert results.pop() == reference_discoverkeys(header, rows)


def test_row_order_does_not_change_key_set():
    rng = random.Random(5150)
    header = tuple('c%d' % i for i in range(5))
    rows = [tuple(rng.choice(ALLOWED) for _ in range(5)) for _ in range(40)]
    shuffled = list(rows)
    rng.shuffle(shuffled)
    a = {frozenset(k) for k in etl.discoverkeys(
        [list(header)] + [list(r) for r in rows], buffersize=3)}
    b = {frozenset(k) for k in etl.discoverkeys(
        [list(header)] + [list(r) for r in shuffled], buffersize=3)}
    assert a == b


# ---------------------------------------------------------------------------
# Wide-table scale check (36 columns, 9**4 minimal keys)
# ---------------------------------------------------------------------------


def _binary_attribute_table(extra_constants=0, extra_missing=0,
                            duplicate=False):
    from itertools import product
    names = ['%s%d' % (attr, k)
             for attr in ('p', 'q', 'r', 's') for k in range(9)]
    names += ['k%d' % i for i in range(extra_constants + extra_missing)]
    rows = [names]
    for bits in product((0, 1), repeat=4):
        row = []
        for b in bits:
            row.extend([b] * 9)
        row.extend(['const'] * extra_constants)
        row.extend([None] * extra_missing)
        rows.append(row)
    if duplicate:
        rows.append(list(rows[1]))
    return rows


def test_wide_table_all_minimal_keys():
    from itertools import product
    rows = _binary_attribute_table()
    result = etl.discoverkeys(rows)
    header = rows[0]
    # The minimal keys are exactly the 9**4 combinations choosing one column
    # per binary attribute group; checking the combinatorial expectation is
    # far cheaper than brute-forcing all 2**36 subsets.
    groups = [[i for i in range(36) if i // 9 == g] for g in range(4)]
    expected = {tuple(header[i] for i in picks)
                for picks in product(*groups)}
    assert len(result) == 9 ** 4
    assert set(result) == expected
    assert all(len(k) == 4 for k in result)
    # result ordering: width then header-position lexicographic
    positions = [tuple(header.index(name) for name in k) for k in result]
    assert positions == sorted(positions)
    # minimality: no returned key contains another returned key
    keysets = [frozenset(k) for k in result]
    for k in keysets:
        assert not any(other < k for other in keysets)


def test_wide_table_variants():
    assert len(etl.discoverkeys(_binary_attribute_table(
        extra_constants=2))) == 9 ** 4
    assert len(etl.discoverkeys(_binary_attribute_table(
        extra_missing=2))) == 9 ** 4
    assert etl.discoverkeys(_binary_attribute_table(duplicate=True)) == ()
    rows = _binary_attribute_table()
    assert etl.discoverkeys(rows, max_width=3) == ()
    assert len(etl.discoverkeys(rows, max_width=4)) == 9 ** 4


def test_wide_table_completes_well_under_budget():
    # The 6561-key discovery must not fall back to enumerating all 2**36
    # subsets; a generous wall-clock guard keeps that regression visible.
    import time
    rows = _binary_attribute_table()
    start = time.perf_counter()
    result = etl.discoverkeys(rows)
    elapsed = time.perf_counter() - start
    assert len(result) == 9 ** 4
    assert elapsed < 30.0


# ---------------------------------------------------------------------------
# Large-row streaming: no whole-table materialisation
# ---------------------------------------------------------------------------


class _GeneratorRowTable(object):
    """Repeatable table whose rows are generated, never stored as a list."""

    def __init__(self, nrows, ncols=8):
        self.nrows = nrows
        self.ncols = ncols
        self.iterations = 0
        self.rows_yielded = 0

    def __iter__(self):
        nrows = self.nrows
        ncols = self.ncols
        source = self
        header = tuple('c%d' % i for i in range(ncols))

        def gen():
            yield header
            for i in range(nrows):
                source.rows_yielded += 1
                # first two columns form the only key; the rest are constant
                yield (i // 1000, i % 1000) + (0,) * (ncols - 2)

        source.iterations += 1
        return gen()


@pytest.mark.parametrize('nrows', [2000, 20000])
def test_large_row_count_streams_with_single_iterator(nrows, tmp_path):
    table = _GeneratorRowTable(nrows)
    result = etl.discoverkeys(table, buffersize=64, tempdir=str(tmp_path))
    assert result == (('c0', 'c1'),)
    # exactly one source iterator, every row read exactly once
    assert table.iterations == 1
    assert table.rows_yielded == nrows
    # all temporary files are released again
    assert os.listdir(str(tmp_path)) == []


def test_large_row_count_result_matches_in_memory_reference(tmp_path):
    nrows = 5000
    streamed = etl.discoverkeys(_GeneratorRowTable(nrows), buffersize=64,
                                tempdir=str(tmp_path))
    materialised = [tuple('c%d' % i for i in range(8))]
    materialised += [(i // 1000, i % 1000) + (0,) * 6
                     for i in range(nrows)]
    assert streamed == reference_discoverkeys(materialised[0],
                                              materialised[1:])
