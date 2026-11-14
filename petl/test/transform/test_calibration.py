from __future__ import absolute_import, print_function, division


import math
from collections import OrderedDict

import pytest

import petl as etl
from petl.errors import (ArgumentError, CalibrationError,
                        FieldSelectionError)


# ---------------------------------------------------------------------
# independent high-precision reference
# ---------------------------------------------------------------------

def _incidence(table, margins):
    """Build the sparse category incidence (members lists) and targets."""
    header = table[0]
    members = []
    targets = []
    for fields, tmap in margins:
        pos = [header.index(f) for f in fields]
        buckets = OrderedDict((cat, []) for cat in tmap)
        for i, row in enumerate(table[1:]):
            cat = tuple(row[p] for p in pos)
            buckets[cat].append(i)
        for cat, target in tmap.items():
            members.append(tuple(buckets[cat]))
            targets.append(float(target))
    return members, targets


def reference_weights(table, margins, weight, lower, upper):
    """Independent optimum via damped Newton on the smooth dual KKT
    system, with explicit box clipping. This is deliberately written
    independently of the production IPF solver."""
    np = pytest.importorskip('numpy')
    header = table[0]
    wpos = header.index(weight)
    d = np.array([float(r[wpos]) for r in table[1:]], dtype=float)
    n = len(d)
    members, targets = _incidence(table, margins)
    m = len(targets)
    t = np.array(targets, dtype=float)
    # dense incidence is acceptable for the small reference cases
    A = np.zeros((m, n))
    for k, mem in enumerate(members):
        for i in mem:
            A[k, i] = 1.0
    llo, lhi = math.log(lower), math.log(upper)
    lm = np.zeros(m)
    for _ in range(500):
        z = np.clip(A.T.dot(lm), llo, lhi)
        x = np.exp(z)
        g = A.dot(d * x) - t
        if np.abs(g).max() < 1e-13:
            break
        H = (A * (d * x)).dot(A.T)
        step = -np.linalg.solve(H + 1e-12 * np.eye(m), g)
        alpha = 1.0
        g0 = np.abs(g).max()
        for _ in range(60):
            z2 = np.clip(A.T.dot(lm + alpha * step), llo, lhi)
            g2 = A.dot(d * np.exp(z2)) - t
            if np.abs(g2).max() <= g0:
                break
            alpha *= 0.5
        lm = lm + alpha * step
    z = np.clip(A.T.dot(lm), llo, lhi)
    return (d * np.exp(z)).tolist()


def _weighted_totals(table, out_rows, margins):
    header = out_rows[0]
    wpos = header.index('calibrated_weight') \
        if 'calibrated_weight' in header else len(header) - 1
    totals = {}
    for mi, (fields, tmap) in enumerate(margins):
        pos = [header.index(f) for f in fields]
        for row in out_rows[1:]:
            key = (mi, tuple(row[p] for p in pos))
            totals[key] = totals.get(key, 0.0) + row[wpos]
    return totals


def _assert_margins(table, out_rows, margins, tol=1e-8):
    totals = _weighted_totals(table, out_rows, margins)
    for mi, (fields, tmap) in enumerate(margins):
        for cat, target in tmap.items():
            actual = totals[(mi, cat)]
            assert abs(actual - target) <= tol * max(1.0, abs(target)), \
                (fields, cat, actual, target)


def _assert_bounds(out_rows, weight, lower, upper):
    h = out_rows[0]
    dpos = h.index(weight)
    wpos = len(h) - 1
    for row in out_rows[1:]:
        ratio = row[wpos] / row[dpos]
        assert lower - 1e-12 <= ratio <= upper + 1e-12
        assert isinstance(row[wpos], float)
        assert math.isfinite(row[wpos]) and row[wpos] > 0


# ---------------------------------------------------------------------
# basic behaviour and optimality
# ---------------------------------------------------------------------

def test_basic_two_overlapping_margins():
    table = [['region', 'sex', 'd'],
             ['east', 'm', 1.0], ['east', 'f', 2.0],
             ['west', 'm', 3.0], ['west', 'f', 4.0]]
    margins = [
        (('region',), {('east',): 9.0, ('west',): 11.0}),
        (('sex',), {('m',): 8.0, ('f',): 12.0}),
    ]
    out = etl.calibrateweights(table, margins, 'd')
    rows = list(out)
    assert rows[0] == ('region', 'sex', 'd', 'calibrated_weight')
    assert [r[:3] for r in rows[1:]] == [tuple(r) for r in table[1:]]
    _assert_margins(table, rows, margins)
    _assert_bounds(rows, 'd', 0.2, 5.0)
    ref = reference_weights(table, margins, 'd', 0.2, 5.0)
    got = [r[-1] for r in rows[1:]]
    for a, b in zip(got, ref):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)


def test_matches_reference_on_random_overlapping_margins():
    rng = pytest.importorskip('numpy').random.default_rng(0)
    n = 60
    region = rng.integers(0, 4, n)
    sex = rng.integers(0, 2, n)
    age = rng.integers(0, 3, n)
    d = rng.uniform(0.5, 5.0, n)
    mult = rng.uniform(0.6, 1.6, n)
    ws = d * mult
    table = [['region', 'sex', 'age', 'd']]
    for i in range(n):
        table.append([int(region[i]), int(sex[i]), int(age[i]),
                      float(d[i])])
    margins = [
        (('region',),
         {(int(v),): float(ws[region == v].sum()) for v in range(4)}),
        (('sex',),
         {(int(v),): float(ws[sex == v].sum()) for v in range(2)}),
        (('region', 'age'),
         {(int(r), int(a)):
          float(ws[(region == r) & (age == a)].sum())
          for r in range(4) for a in range(3)}),
    ]
    out = etl.calibrateweights(table, margins, 'd')
    rows = list(out)
    _assert_margins(table, rows, margins)
    ref = reference_weights(table, margins, 'd', 0.2, 5.0)
    got = [r[-1] for r in rows[1:]]
    for a, b in zip(got, ref):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
    # objective must not exceed the reference objective
    def obj(w):
        return sum(wi * math.log(wi / di) - wi + di
                   for wi, di in zip(w, d.tolist()))
    assert obj(got) <= obj(ref) + 1e-9


def test_redundant_margins_accepted():
    # a margin duplicated exactly, and a finer margin whose categories
    # partition the coarse margin's categories (so it implies the coarse)
    table = [['r', 's', 'd'],
             ['e', 'm', 1.0], ['e', 'f', 2.0],
             ['w', 'm', 3.0], ['w', 'f', 4.0]]
    # feasible targets derived from multipliers (0.9, 1.1, 1.2, 0.8)
    # east total = 0.9 + 2.2 = 3.1, west total = 3.6 + 3.2 = 6.8
    coarse = (('r',), {('e',): 3.1, ('w',): 6.8})
    fine = (('r', 's'),
            {('e', 'm'): 0.9, ('e', 'f'): 2.2,
             ('w', 'm'): 3.6, ('w', 'f'): 3.2})
    margins = [coarse, fine, coarse]
    out = etl.calibrateweights(table, margins, 'd')
    rows = list(out)
    _assert_margins(table, rows, margins)
    ref = reference_weights(table, [coarse, fine], 'd', 0.2, 5.0)
    got = [r[-1] for r in rows[1:]]
    for a, b in zip(got, ref):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)


def test_none_and_mixed_categories():
    table = [['g', 'd'],
             [None, 1.0], [1, 2.0], ['1', 3.0], [10 ** 20, 4.0]]
    targets = {(None,): 1.5, (1,): 3.0, ('1',): 4.5, (10 ** 20,): 6.0}
    out = etl.calibrateweights(table, [(('g',), targets)], 'd')
    rows = list(out)
    _assert_margins(table, rows, [(('g',), targets)])
    # int 1 and str '1' kept distinct in the output cells
    assert rows[2][0] == 1 and rows[3][0] == '1'


def test_zero_target_absent_category_legal():
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0]]
    margins = [(('g',), {('a',): 1.0, ('b',): 2.0, ('c',): 0.0})]
    out = etl.calibrateweights(table, margins, 'd')
    assert [r[-1] for r in list(out)[1:]] == [1.0, 2.0]


def test_base_weights_meeting_targets_are_kept():
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0], ['a', 3.0]]
    margins = [(('g',), {('a',): 4.0, ('b',): 2.0})]
    out = etl.calibrateweights(table, margins, 'd')
    assert [r[-1] for r in list(out)[1:]] == [1.0, 2.0, 3.0]


def test_boundary_only_feasible_points():
    table = [['g', 'd'], ['a', 1.0], ['b', 1.0]]
    low = [(('g',), {('a',): 0.5, ('b',): 0.5})]
    out = etl.calibrateweights(table, low, 'd', bounds=(0.5, 2.0))
    assert [r[-1] for r in list(out)[1:]] == [0.5, 0.5]
    high = [(('g',), {('a',): 2.0, ('b',): 2.0})]
    out = etl.calibrateweights(table, high, 'd', bounds=(0.5, 2.0))
    assert [r[-1] for r in list(out)[1:]] == [2.0, 2.0]


def test_boundary_optimum_matches_reference():
    table = [['g', 'd'], ['a', 1.0], ['b', 10.0]]
    margins = [(('g',), {('a',): 2.0, ('b',): 10.0})]
    out = etl.calibrateweights(table, margins, 'd', bounds=(0.5, 2.0))
    rows = list(out)
    assert math.isclose(rows[1][-1], 2.0, abs_tol=1e-10)
    assert math.isclose(rows[2][-1], 10.0, abs_tol=1e-9)
    ref = reference_weights(table, margins, 'd', 0.5, 2.0)
    got = [r[-1] for r in rows[1:]]
    for a, b in zip(got, ref):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)


# ---------------------------------------------------------------------
# infeasibility and iteration limits
# ---------------------------------------------------------------------

def test_infeasible_absent_positive_category():
    table = [['g', 'd'], ['a', 1.0]]
    margins = [(('g',), {('a',): 1.0, ('b',): 1.0})]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd')
    assert exc.value.reason == 'infeasible'
    assert 'never occurs' in str(exc.value)


def test_infeasible_zero_target_observed_category():
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0]]
    margins = [(('g',), {('a',): 0.0, ('b',): 3.0})]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd')
    assert exc.value.reason == 'infeasible'


def test_infeasible_per_category_bound_conflict():
    table = [['g', 'd'], ['a', 1.0], ['b', 1.0]]
    margins = [(('g',), {('a',): 0.4, ('b',): 1.6})]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd', bounds=(0.5, 2.0))
    assert exc.value.reason == 'infeasible'


def test_infeasible_grand_total_disagreement():
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0]]
    margins = [
        (('g',), {('a',): 1.0, ('b',): 2.0}),
        (('g',), {('a',): 2.0, ('b',): 2.0}),
    ]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd')
    assert exc.value.reason == 'infeasible'
    assert 'grand total' in str(exc.value)


def test_infeasible_structural_cross_category_lp():
    # full 2x2 grid; each 1D category is individually feasible, but the
    # joint boxed system has no solution (requires an LP certificate)
    pytest.importorskip('scipy')
    table = [['r', 's', 'd'],
             ['e', 'm', 1.0], ['e', 'f', 1.0],
             ['w', 'm', 1.0], ['w', 'f', 1.0]]
    margins = [
        (('r',), {('e',): 3.5, ('w',): 1.5}),
        (('s',), {('m',): 1.5, ('f',): 3.5}),
    ]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd', bounds=(0.5, 2.0))
    assert exc.value.reason == 'infeasible'


def test_iteration_limit_reason():
    rng = pytest.importorskip('numpy').random.default_rng(7)
    n = 200
    a = rng.integers(0, 8, n)
    b = rng.integers(0, 8, n)
    d = rng.uniform(1.0, 3.0, n)
    ws = d * rng.uniform(0.6, 1.8, n)
    table = [['a', 'b', 'd']]
    for i in range(n):
        table.append([int(a[i]), int(b[i]), float(d[i])])
    margins = [
        (('a',), {(int(v),): float(ws[a == v].sum()) for v in range(8)}),
        (('b',), {(int(v),): float(ws[b == v].sum()) for v in range(8)}),
    ]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd', max_iter=1)
    assert exc.value.reason == 'iteration_limit'
    # a proper budget converges
    out = etl.calibrateweights(table, margins, 'd', max_iter=1000)
    _assert_margins(table, list(out), margins)


def test_loose_tolerance_met_within_small_budget_succeeds():
    # when the user tolerance is reachable within the budget, no failure
    # is reported even though a tighter internal optimum was not reached
    rng = pytest.importorskip('numpy').random.default_rng(7)
    n = 200
    a = rng.integers(0, 8, n)
    b = rng.integers(0, 8, n)
    d = rng.uniform(1.0, 3.0, n)
    ws = d * rng.uniform(0.6, 1.8, n)
    table = [['a', 'b', 'd']]
    for i in range(n):
        table.append([int(a[i]), int(b[i]), float(d[i])])
    margins = [
        (('a',), {(int(v),): float(ws[a == v].sum()) for v in range(8)}),
        (('b',), {(int(v),): float(ws[b == v].sum()) for v in range(8)}),
    ]
    out = etl.calibrateweights(table, margins, 'd',
                               max_iter=3, tol=5e-4)
    rows = list(out)
    # every margin meets the loose requested tolerance
    totals = _weighted_totals(table, rows, margins)
    for mi, (fields, tmap) in enumerate(margins):
        for cat, target in tmap.items():
            assert abs(totals[(mi, cat)] - target) \
                <= 5e-4 * max(1.0, abs(target))
    _assert_bounds(rows, 'd', 0.2, 5.0)


def test_calibration_error_is_argument_error():
    assert issubclass(CalibrationError, ArgumentError)
    err = CalibrationError('msg', 'infeasible')
    assert err.reason == 'infeasible'
    assert 'msg' in str(err)


# ---------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize('kwargs', [
    {'bounds': (1.0, 5.0)},
    {'bounds': (0.2, 0.9)},
    {'bounds': (0.0, 5.0)},
    {'bounds': (True, 5.0)},
    {'bounds': (0.2, float('inf'))},
])
def test_invalid_bounds(kwargs):
    table = [['g', 'd'], ['a', 1.0]]
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd', **kwargs)


@pytest.mark.parametrize('max_iter', [0, -1, True, 1.5, 2.0])
def test_invalid_max_iter(max_iter):
    table = [['g', 'd'], ['a', 1.0]]
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd', max_iter=max_iter)


@pytest.mark.parametrize('tol', [0.0, -1e-9, 2e-3, True, float('nan')])
def test_invalid_tol(tol):
    table = [['g', 'd'], ['a', 1.0]]
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd', tol=tol)


def test_invalid_margins_structure():
    table = [['g', 'd'], ['a', 1.0]]
    bad_margins = [
        'not-a-sequence', [],
        [(('g',),)],                       # not a pair
        [((), {('a',): 1.0})],             # empty fields
        [((1,), {(1,): 1.0})],             # non-string field
        [(('g', 'g'), {('a', 'b'): 1.0})],  # duplicate fields
        [(('g',), [('a', 1.0)])],          # targets not a mapping
        [(('g',), {})],                    # empty targets
        [(('g',), {('a', 'b'): 1.0})],     # wrong key length
        [(('g',), {(1.5,): 1.0})],         # bad component type
        [(('g',), {(True,): 1.0})],        # bool component
        [(('g',), {('a',): True})],        # bool target
        [(('g',), {('a',): -1.0})],        # negative target
        [(('g',), {('a',): float('nan')})],
        [(('g',), {('a',): float('inf')})],
    ]
    for margins in bad_margins:
        with pytest.raises(ArgumentError):
            etl.calibrateweights(table, margins, 'd')


def test_missing_fields_raise_field_selection_error():
    table = [['g', 'd'], ['a', 1.0]]
    with pytest.raises(FieldSelectionError):
        etl.calibrateweights(table, [(('nope',), {('a',): 1.0})], 'd')
    with pytest.raises(FieldSelectionError):
        etl.calibrateweights(table, [(('g',), {('a',): 1.0})], 'nope')


def test_invalid_output_name():
    table = [['g', 'd'], ['a', 1.0]]
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd', output='g')
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd', output='')
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd', output=5)


def test_header_and_row_validation():
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError):
        etl.calibrateweights([], margins, 'd')           # empty
    with pytest.raises(ArgumentError):
        etl.calibrateweights([[]], margins, 'd')         # empty header
    with pytest.raises(ArgumentError):
        etl.calibrateweights([['g', 'g'], ['a', 1]],
                             margins, 'g')               # dup header
    with pytest.raises(ArgumentError):
        etl.calibrateweights([['g', 'd']], margins, 'd')  # no data rows


def test_data_errors_carry_row_and_column():
    margins = [(('g',), {('a',): 1.0, ('b',): 2.0})]
    with pytest.raises(ArgumentError) as exc:
        etl.calibrateweights(
            [['g', 'd'], ['a', 0.0], ['b', 2.0]], margins, 'd')
    assert 'row 1' in str(exc.value)
    with pytest.raises(ArgumentError) as exc:
        etl.calibrateweights(
            [['g', 'd'], ['a', 1.0], ['b', True]], margins, 'd')
    assert 'row 2' in str(exc.value)
    with pytest.raises(ArgumentError) as exc:
        etl.calibrateweights(
            [['g', 'd'], [1.5, 1.0], ['b', 2.0]], margins, 'd')
    assert 'row 1' in str(exc.value) and "'g'" in str(exc.value)
    with pytest.raises(ArgumentError) as exc:
        etl.calibrateweights(
            [['g', 'd'], ['a', 1.0], ['c', 2.0]], margins, 'd')
    assert 'row 2' in str(exc.value) and 'undeclared' in str(exc.value)
    with pytest.raises(ArgumentError) as exc:
        etl.calibrateweights(
            [['g', 'd'], ['a', 1.0], ['b']], margins, 'd')
    assert 'row 2' in str(exc.value)


# ---------------------------------------------------------------------
# source iteration semantics
# ---------------------------------------------------------------------

class _TrackingTable(object):
    def __init__(self, rows):
        self._rows = rows
        self.iter_count = 0
        self.closed = 0

    def __iter__(self):
        self.iter_count += 1
        outer = self
        data = iter(list(self._rows))

        class _It(object):
            def __iter__(self):
                return self

            def __next__(self):
                return next(data)

            def close(self):
                outer.closed += 1
        return _It()


def test_single_iterator_read_once_and_closed_on_success():
    table = _TrackingTable([['g', 'd'], ['a', 1.0], ['b', 2.0]])
    margins = [(('g',), {('a',): 1.0, ('b',): 2.0})]
    out = etl.calibrateweights(table, margins, 'd')
    assert table.iter_count == 1
    assert table.closed == 1
    for _ in range(3):
        list(out)
    assert table.iter_count == 1 and table.closed == 1


def test_iterator_closed_on_data_error():
    table = _TrackingTable([['g', 'd'], ['a', 0.0]])
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError):
        etl.calibrateweights(table, margins, 'd')
    assert table.closed == 1 and table.iter_count == 1


class _Boom(Exception):
    pass


class _BoomTable(object):
    def __init__(self):
        self.closed = False

    def __iter__(self):
        outer = self

        def gen():
            yield ['g', 'd']
            yield ['a', 1.0]
            raise _Boom('source exploded')
        g = gen()

        class _It(object):
            def __iter__(self):
                return self

            def __next__(self):
                return next(g)

            def close(self):
                outer.closed = True
        return _It()


def test_source_exception_identity_preserved_and_closed():
    table = _BoomTable()
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(_Boom) as exc:
        etl.calibrateweights(table, margins, 'd')
    assert str(exc.value) == 'source exploded'
    assert table.closed is True


class _BadCloseTable(object):
    def __iter__(self):
        def gen():
            yield ['g', 'd']
            yield ['a', 0.0]
        g = gen()

        class _It(object):
            def __iter__(self):
                return self

            def __next__(self):
                return next(g)

            def close(self):
                raise RuntimeError('close broken')
        return _It()


def test_cleanup_error_does_not_mask_original():
    margins = [(('g',), {('a',): 1.0})]
    with pytest.raises(ArgumentError) as exc:
        etl.calibrateweights(_BadCloseTable(), margins, 'd')
    assert 'base weight' in str(exc.value)


def test_inputs_not_modified():
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0]]
    snapshot = [list(r) for r in table]
    targets = OrderedDict([(('a',), 1.0), (('b',), 2.0)])
    target_snapshot = list(targets.items())
    etl.calibrateweights(table, [(('g',), targets)], 'd')
    assert table == snapshot
    assert list(targets.items()) == target_snapshot


# ---------------------------------------------------------------------
# representation invariances
# ---------------------------------------------------------------------

def _invariance_fixture():
    rng = pytest.importorskip('numpy').random.default_rng(11)
    n = 80
    region = rng.integers(0, 4, n)
    sex = rng.integers(0, 2, n)
    age = rng.integers(0, 3, n)
    d = rng.uniform(1.0, 4.0, n)
    ws = d * rng.uniform(0.7, 1.6, n)
    table = [['id', 'region', 'sex', 'age', 'd']]
    for i in range(n):
        table.append([i, int(region[i]), int(sex[i]), int(age[i]),
                      float(d[i])])
    margins = [
        (('region',),
         {(int(v),): float(ws[region == v].sum())
          for v in set(region.tolist())}),
        (('sex',),
         {(int(v),): float(ws[sex == v].sum())
          for v in set(sex.tolist())}),
        (('region', 'age'),
         {(int(r), int(a)):
          float(ws[(region == r) & (age == a)].sum())
          for r in set(region.tolist()) for a in set(age.tolist())}),
    ]
    return table, margins, d


def test_shuffle_invariance():
    table, margins, _ = _invariance_fixture()
    base = {r[0]: r[-1] for r in
            list(etl.calibrateweights(table, margins, 'd'))[1:]}
    shuffled = [table[0]] + list(reversed(table[1:]))
    out = etl.calibrateweights(shuffled, margins, 'd')
    for row in list(out)[1:]:
        assert math.isclose(row[-1], base[row[0]],
                            rel_tol=1e-6, abs_tol=1e-8)


def test_margin_permutation_invariance():
    table, margins, _ = _invariance_fixture()
    base = {r[0]: r[-1] for r in
            list(etl.calibrateweights(table, margins, 'd'))[1:]}
    permuted = [margins[2], margins[0], margins[1]]
    permuted = [(f, dict(list(tg.items())[::-1])) for f, tg in permuted]
    out = etl.calibrateweights(table, permuted, 'd')
    for row in list(out)[1:]:
        assert math.isclose(row[-1], base[row[0]],
                            rel_tol=1e-6, abs_tol=1e-8)


def test_row_split_invariance():
    table, margins, d = _invariance_fixture()
    out = etl.calibrateweights(table, margins, 'd')
    base = {r[0]: r[-1] for r in list(out)[1:]}
    idx = 7
    row = table[idx + 1]
    d1 = d[idx] * 0.3
    d2 = d[idx] - d1
    split = [list(r) for r in table]
    split[idx + 1] = row[:-1] + [d1]
    split.insert(idx + 2, row[:-1] + [d2])
    sp = list(etl.calibrateweights(split, margins, 'd'))[1:]
    m1 = sp[idx][-1] / d1
    m2 = sp[idx + 1][-1] / d2
    expected_mult = base[row[0]] / d[idx]
    assert math.isclose(m1, expected_mult, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(m2, expected_mult, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(sp[idx][-1] + sp[idx + 1][-1],
                        base[row[0]], rel_tol=1e-6, abs_tol=1e-8)


def test_common_scaling_invariance():
    table, margins, d = _invariance_fixture()
    base = {r[0]: r[-1] for r in
            list(etl.calibrateweights(table, margins, 'd'))[1:]}
    c = 3.7
    scaled = [table[0]]
    for r in table[1:]:
        scaled.append(r[:-1] + [r[-1] * c])
    scaled_margins = [
        (f, {cat: tgt * c for cat, tgt in tg.items()})
        for f, tg in margins
    ]
    out = etl.calibrateweights(scaled, scaled_margins, 'd')
    for r in list(out)[1:]:
        assert math.isclose(r[-1], c * base[r[0]],
                            rel_tol=1e-6, abs_tol=1e-8)


# ---------------------------------------------------------------------
# repeatability, Table method, and groupstate integration
# ---------------------------------------------------------------------

def test_repeatable_and_interleaved_calls_independent():
    table = [['r', 's', 'd'],
             ['e', 'm', 1.0], ['e', 'f', 2.0],
             ['w', 'm', 3.0], ['w', 'f', 4.0]]
    margins = [
        (('r',), {('e',): 5.0, ('w',): 9.0}),
        (('s',), {('m',): 7.0, ('f',): 7.0}),
    ]
    a = etl.calibrateweights(table, margins, 'd', output='w1')
    b = etl.calibrateweights(table, margins, 'd', output='w2',
                             bounds=(0.3, 4.0))
    assert list(a) == list(a)
    la = list(a)
    assert la[0][-1] == 'w1'
    assert list(b)[0][-1] == 'w2'
    # interleaving does not change either result
    assert list(a) == la
    for r in list(b)[1:]:
        ratio = r[-1] / r[2]
        assert 0.3 - 1e-12 <= ratio <= 4.0 + 1e-12


def test_table_method_parity():
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0]]
    margins = [(('g',), {('a',): 1.5, ('b',): 2.5})]
    fn = etl.calibrateweights(table, margins, 'd')
    method = etl.wrap(table).calibrateweights(margins, 'd')
    assert list(fn) == list(method)


def test_runs_without_scientific_stack(monkeypatch):
    # the iterative solver only uses the standard library; blocking
    # numpy/scipy imports must not affect a successful calibration
    import builtins
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == 'numpy' or name.startswith('numpy.') \
                or name == 'scipy' or name.startswith('scipy.'):
            raise ImportError('scientific stack blocked for the test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    table = [['g', 'd'], ['a', 1.0], ['b', 2.0], ['a', 3.0]]
    margins = [(('g',), {('a',): 7.0, ('b',): 3.5})]
    out = etl.calibrateweights(table, margins, 'd')
    rows = list(out)
    _assert_margins(table, rows, margins)
    _assert_bounds(rows, 'd', 0.2, 5.0)


def test_structural_infeasibility_certified_without_scipy(monkeypatch):
    # with the LP backend unavailable, the solver must still recognize a
    # boxed joint infeasibility via its fixed-point certificate
    from petl.transform import calibration as calibration_mod
    monkeypatch.setattr(calibration_mod, '_lp_feasible',
                        lambda problem, lower, upper: None)
    table = [['r', 's', 'd'],
             ['e', 'm', 1.0], ['e', 'f', 1.0],
             ['w', 'm', 1.0], ['w', 'f', 1.0]]
    margins = [
        (('r',), {('e',): 3.5, ('w',): 1.5}),
        (('s',), {('m',): 1.5, ('f',): 3.5}),
    ]
    with pytest.raises(CalibrationError) as exc:
        etl.calibrateweights(table, margins, 'd', bounds=(0.5, 2.0))
    assert exc.value.reason == 'infeasible'


def test_groupstate_hits_margins_and_merges_serialize():
    table = [['region', 'sex', 'd'],
             ['east', 'm', 1.0], ['east', 'f', 2.0],
             ['west', 'm', 3.0], ['west', 'f', 4.0],
             ['east', 'm', 5.0]]
    margins = [
        (('region',), {('east',): 12.0, ('west',): 9.0}),
        (('sex',), {('m',): 11.0, ('f',): 10.0}),
    ]
    calibrated = etl.calibrateweights(table, margins, 'd')
    state = etl.groupstate(
        calibrated, key=('region', 'sex'),
        sets=[('region',), ('sex',), ()],
        measures=[('pop', 'sum', 'calibrated_weight'),
                  ('n', 'count', None)],
        batch='b1')
    rendered = list(state.totable())
    for row in rendered:
        if row[0] is not None and row[1] is None and row[2] == 1:
            target = margins[0][1][(row[0],)]
            assert abs(row[3] - target) <= 1e-8 * max(1, abs(target))
    # serialization round-trips
    assert list(etl.loadgroupstate(state.dumps()).totable()) == rendered


def test_sparse_scaling_and_no_dense_matrix():
    # a few thousand rows spread over hundreds of sparse categories; the
    # internal problem stores membership lists only (one entry per row
    # per margin), never a rows x categories dense matrix
    np = pytest.importorskip('numpy')
    from petl.transform.calibration import (_build_problem,
                                            _resolve_margins,
                                            _validate_margins)
    rng = np.random.default_rng(5)
    n = 4000
    n_region, n_age, n_prod = 20, 10, 300
    region = rng.integers(0, n_region, n)
    age = rng.integers(0, n_age, n)
    product = rng.integers(0, n_prod, n)
    d = rng.uniform(1.0, 5.0, n)
    ws = d * rng.uniform(0.7, 1.6, n)
    table = [['region', 'age', 'product', 'd']]
    for i in range(n):
        table.append([int(region[i]), int(age[i]), int(product[i]),
                      float(d[i])])
    m1 = OrderedDict(((int(v),), float(ws[region == v].sum()))
                     for v in range(n_region))
    m2 = OrderedDict(((int(v),), float(ws[age == v].sum()))
                     for v in range(n_age))
    m3 = OrderedDict()
    for r in range(n_region):
        idx = np.where(region == r)[0]
        for p in np.unique(product[idx]):
            sel = idx[product[idx] == p]
            m3[(int(r), int(p))] = float(ws[sel].sum())
    margins = [(('region',), m1), (('age',), m2),
               (('region', 'product'), m3)]
    out = etl.calibrateweights(table, margins, 'd')
    _assert_margins(table, list(out), margins)

    # reconstruct the internal problem to inspect its storage
    normed = _validate_margins(margins)
    spec_objs = _resolve_margins(normed, tuple(table[0]))
    problem = _build_problem(spec_objs,
                             [float(r[3]) for r in table[1:]],
                             [tuple(r) for r in table[1:]])
    stored = sum(len(mem) for mem in problem.members)
    n_categories = len(problem.members)
    # sparse membership storage is exactly one entry per row per margin,
    # which is far smaller than an n x categories dense matrix
    assert stored == n * len(margins)
    assert stored < n * n_categories
