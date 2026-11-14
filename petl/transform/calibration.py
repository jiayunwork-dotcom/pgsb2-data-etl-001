"""
Calibration of sample weights to known population margins.

The :func:`calibrateweights` function adjusts per-row base weights so that
the weighted totals of one or more *margins* (one- or multi-dimensional
population tables, possibly overlapping or redundant) match given targets,
while keeping every final weight within a multiplicative band around its
base weight.

The adjustment is the minimum-information one: among all weight vectors
that satisfy the margins and the multiplier bounds, it minimizes the
entropy distance

    sum_i ( w_i * log(w_i / d_i) - w_i + d_i )

from the base weights ``d_i``. The objective is strictly convex in the
weights (its second derivative is ``1 / w_i > 0``), so the minimizer is
unique whenever the feasible region is non-empty.

Numerical method
----------------

The core solver is a bounded, active-set iterative proportional fitting
(raking) procedure written only with the Python standard library: one
sweep updates every margin category in turn, moving all of its member
multipliers by a common factor and clipping them to the multiplier box;
rows that would cross a bound are pinned at that bound. Each update is
an exact coordinate step on the convex dual, so the dual decreases
monotonically and the iterates converge to the unique optimum.

If NumPy and SciPy are installed they are used additionally to certify
feasibility of the full boxed linear system with a linear program
(HiGHS). Both the iterative solver and the LP work on sparse incidence
structures (each row belongs to exactly one category per margin); no
dense ``rows x categories`` matrix is ever built. The optional packages
are imported lazily, so importing petl never requires them, every other
petl entry point works without them, and calibration itself runs on the
standard-library solver when they are absent.

"""

from __future__ import absolute_import, print_function, division


import math
from collections import OrderedDict

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 2 fallback
    from collections import Mapping


from petl.compat import string_types
from petl.errors import ArgumentError, CalibrationError, FieldSelectionError
from petl.util.base import Table


__all__ = ['calibrateweights', 'CalibratedTable']


_INSTALL_HINT = (
    'calibrateweights runs on a standard-library solver by default; the '
    'optional scientific stack speeds it up - install with e.g. '
    '`python -m pip install numpy scipy`')

# reasons carried by CalibrationError
INFEASIBLE = 'infeasible'
ITERATION_LIMIT = 'iteration_limit'
NUMERICAL_FAILURE = 'numerical_failure'


def calibrateweights(table, margins, weight, *, output='calibrated_weight',
                     bounds=(0.2, 5.0), max_iter=1000, tol=1e-8):
    """
    Calibrate per-row base weights of `table` to population `margins`.

    `margins` is a non-empty sequence of ``(fields, targets)`` pairs.
    `fields` is a non-empty sequence of unique field name strings; each
    row's category for that margin is the tuple of its values in those
    fields. `targets` is a non-empty mapping keyed by category tuples of
    the same length as `fields`, mapping to non-negative finite target
    totals (``bool`` values are rejected). Category components must be
    ``str``, non-``bool`` ``int`` or ``None``; ``1`` and ``'1'`` are
    different categories and ``None`` is an ordinary category. Every
    category actually observed in the data must be declared in the
    corresponding `targets`; missing declarations raise
    :class:`petl.errors.ArgumentError`. A declared category that never
    occurs is legal with target 0; a positive target for an absent
    category is infeasible.

    `weight` is the name of an existing column holding the strictly
    positive, finite base weights (``int`` or ``float``, not ``bool``).
    `output` is the name of the appended column (a non-empty string that
    does not collide with an existing field); its values are ``float``.

    `bounds` is ``(lower, upper)`` with finite positive reals satisfying
    ``0 < lower < 1 < upper``; every final weight ``w_i`` obeys
    ``lower * d_i <= w_i <= upper * d_i`` where ``d_i`` is the base
    weight. `max_iter` is a non-``bool`` positive integer and `tol` a
    finite real in ``(0, 1e-3]`` (``bool`` rejected); a margin with
    target ``t`` is satisfied when the absolute residual is at most
    ``tol * max(1, abs(t))``.

    Iteration counting and stopping. One iteration is one complete
    sweep over every declared margin category. In a sweep each category
    updates all of its member multipliers once, in declaration order,
    by an exact bounded common-factor step. The loop stops at the first
    sweep whose residuals all meet the `tol` gate, or after `max_iter`
    sweeps; a run that exhausts the budget without reaching the gate
    raises :class:`petl.errors.CalibrationError` with
    ``reason='iteration_limit'``. In practice feasible, well-conditioned
    batches converge in a few dozen sweeps regardless of the row count.

    The source table is read exactly once, through a single iterator, and
    the whole batch is calibrated in one go. The returned table retains
    the original header, row order and every original cell, with the
    `output` column appended; it can be iterated repeatedly without
    re-reading the source or re-solving. The input table is never
    modified.

    A minimal example::

        >>> import petl as etl
        >>> table = [['region', 'd'], ['east', 1.0], ['west', 2.0]]
        >>> margins = [(('region',), {('east',): 3.0, ('west',): 2.0})]
        >>> result = etl.calibrateweights(table, margins, 'd')
        >>> list(result)  # doctest: +SKIP
        [('region', 'd', 'calibrated_weight'),
         ('east', 1.0, 3.0),
         ('west', 2.0, 2.0)]


    Raises :class:`petl.errors.FieldSelectionError` for missing fields,
    :class:`petl.errors.ArgumentError` for other invalid arguments or
    data, and :class:`petl.errors.CalibrationError` (a subclass of
    :class:`petl.errors.ArgumentError`) when no acceptable weight vector
    exists (``reason='infeasible'``), the iteration budget is exhausted
    (``reason='iteration_limit'``) or the numerical procedure fails
    (``reason='numerical_failure'``).

    """

    # ---- configuration validation (no source access yet) -------------
    output = _validate_output(output)
    lower, upper = _validate_bounds(bounds)
    max_iter = _validate_max_iter(max_iter)
    tol = _validate_tol(tol)
    normed_margins = _validate_margins(margins)

    # ---- single-pass read of the source ------------------------------
    it = iter(table)
    try:
        try:
            header_t = next(it)
        except StopIteration:
            raise ArgumentError('expected a table with a header row, '
                                'found an empty table')
        header = tuple(header_t)
        _validate_header(header)

        fieldset = set(header)
        if weight not in fieldset:
            raise FieldSelectionError(weight)
        weight_pos = header.index(weight)
        if output in fieldset:
            raise ArgumentError("output field name %r collides with an "
                                "existing header field" % output)

        margin_specs = _resolve_margins(normed_margins, header)

        rows = []
        base_weights = []
        # observed categories per margin, in first-seen order
        observed = [OrderedDict() for _ in margin_specs]
        ncols = len(header)
        for rownum, row in enumerate(it, start=1):
            if len(row) != ncols:
                raise ArgumentError('row %d: expected %d fields, found %d'
                                    % (rownum, ncols, len(row)))
            row = tuple(row)
            d = _check_base_weight(row[weight_pos], rownum, weight)
            for mi, spec in enumerate(margin_specs):
                cat = tuple(_check_category_value(spec.fields[j],
                                                  row[spec.positions[j]],
                                                  rownum)
                            for j in range(len(spec.fields)))
                if cat not in spec.targets:
                    raise ArgumentError(
                        'row %d: margin %d on fields %r: undeclared '
                        'category %r'
                        % (rownum, mi + 1, spec.fields, cat))
                observed[mi].setdefault(cat, None)
            rows.append(row)
            base_weights.append(d)
    except BaseException:
        _close_iterator(it)
        raise
    _close_iterator(it)

    if not rows:
        raise ArgumentError('expected at least one data row, found a '
                            'header-only table')

    # ---- structural pre-checks that prove infeasibility --------------
    # the pre-checks also build the sparse problem once, reused below
    problem = _precheck_infeasibility(margin_specs, observed,
                                      base_weights, rows, lower, upper)

    # ---- solve --------------------------------------------------------
    calibrated = _solve(problem, lower, upper, max_iter, tol)

    return CalibratedTable(header, rows, calibrated, output)


def _close_iterator(it):
    close = getattr(it, 'close', None)
    if close is not None:
        try:
            close()
        except Exception:
            # cleanup errors must never mask the original failure
            pass


class CalibratedTable(Table):
    """Result table of :func:`calibrateweights`.

    A materialized, repeatably iterable view: it holds the original rows
    in their original order and the solved calibrated weights, and emits
    the original header plus one trailing float column. Iterating it
    again never re-reads the source table or re-runs the solver.

    """

    def __init__(self, header, rows, weights, output):
        self._header = tuple(header) + (output,)
        self._rows = rows
        self._weights = weights

    def __iter__(self):
        yield self._header
        for row, w in zip(self._rows, self._weights):
            yield row + (float(w),)


# ---------------------------------------------------------------------
# margin specification
# ---------------------------------------------------------------------

class MarginSpec(object):

    __slots__ = ('fields', 'positions', 'targets', 'categories')

    def __init__(self, fields, positions, targets, categories):
        self.fields = fields
        self.positions = positions
        self.targets = targets
        self.categories = categories


class Problem(object):
    """Sparse description of the calibration system.

    `n` rows, one constraint per *observed* declared category. Only the
    sparse membership lists are stored; the dense ``n x m`` incidence
    matrix is never materialized.

    """

    __slots__ = ('n', 'specs', 'base_weights', 'members', 'targets',
                 'constraint_margin', 'constraint_category',
                 'row_constraints')

    def __init__(self, n, specs, base_weights, members, targets,
                 constraint_margin, constraint_category,
                 row_constraints):
        self.n = n
        self.specs = specs
        self.base_weights = base_weights
        # members[k]: tuple of row indices in constraint k
        self.members = members
        self.targets = targets
        self.constraint_margin = constraint_margin
        self.constraint_category = constraint_category
        # row_constraints[i]: tuple of constraint indices touching row i
        self.row_constraints = row_constraints


def _is_real_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_output(output):
    if not isinstance(output, string_types) or not output:
        raise ArgumentError('output must be a non-empty field name '
                            'string, found %r' % (output,))
    return output


def _validate_bounds(bounds):
    if isinstance(bounds, string_types) \
            or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ArgumentError('bounds must be a (lower, upper) pair, '
                            'found %r' % (bounds,))
    lower, upper = bounds
    for name, value in (('lower', lower), ('upper', upper)):
        if isinstance(value, bool) \
                or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0:
            raise ArgumentError('bounds %s must be a finite positive '
                                'real number, found %r' % (name, value))
    lower = float(lower)
    upper = float(upper)
    if not (0.0 < lower < 1.0 < upper):
        raise ArgumentError('bounds must satisfy 0 < lower < 1 < upper, '
                            'found lower=%r upper=%r' % (lower, upper))
    return lower, upper


def _validate_max_iter(max_iter):
    if not _is_real_int(max_iter) or max_iter <= 0:
        raise ArgumentError('max_iter must be a positive integer, '
                            'found %r' % (max_iter,))
    return int(max_iter)


def _validate_tol(tol):
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) \
            or not math.isfinite(tol) or not (0.0 < tol <= 1e-3):
        raise ArgumentError('tol must be a finite real number in '
                            '(0, 1e-3], found %r' % (tol,))
    return float(tol)


def _validate_margins(margins):
    if isinstance(margins, string_types) \
            or not isinstance(margins, (list, tuple)) or len(margins) == 0:
        raise ArgumentError('margins must be a non-empty sequence of '
                            '(fields, targets) pairs, found %r'
                            % (margins,))
    normed = []
    for mi, margin in enumerate(margins):
        if not isinstance(margin, (list, tuple)) or len(margin) != 2:
            raise ArgumentError('margin %d must be a (fields, targets) '
                                'pair, found %r' % (mi + 1, margin))
        fields, targets = margin
        if isinstance(fields, string_types) \
                or not isinstance(fields, (list, tuple)) \
                or len(fields) == 0:
            raise ArgumentError('margin %d: fields must be a non-empty '
                                'sequence of field name strings, found %r'
                                % (mi + 1, fields))
        fields = tuple(fields)
        for f in fields:
            if not isinstance(f, string_types) or not f:
                raise ArgumentError('margin %d: field names must be '
                                    'non-empty strings, found %r'
                                    % (mi + 1, f))
        if len(set(fields)) != len(fields):
            raise ArgumentError('margin %d: field names must be unique, '
                                'found %r' % (mi + 1, fields))
        if not isinstance(targets, Mapping):
            raise ArgumentError('margin %d: targets must be a Mapping, '
                                'found %r'
                                % (mi + 1, type(targets).__name__))
        if len(targets) == 0:
            raise ArgumentError('margin %d: targets must be non-empty'
                                % (mi + 1,))
        target_map = OrderedDict()
        for cat, target in targets.items():
            if not isinstance(cat, tuple) or len(cat) != len(fields):
                raise ArgumentError(
                    'margin %d: target key %r must be a tuple of length '
                    '%d' % (mi + 1, cat, len(fields)))
            for component in cat:
                if not (component is None
                        or isinstance(component, string_types)
                        or _is_real_int(component)):
                    raise ArgumentError(
                        'margin %d: target key %r contains a component of '
                        'type %s; only str, non-bool int and None are '
                        'allowed'
                        % (mi + 1, cat, type(component).__name__))
            if isinstance(target, bool) \
                    or not isinstance(target, (int, float)) \
                    or not math.isfinite(target) or target < 0:
                raise ArgumentError(
                    'margin %d: target for category %r must be a '
                    'non-negative finite real number, found %r'
                    % (mi + 1, cat, target))
            if cat in target_map:
                raise ArgumentError('margin %d: duplicate target '
                                    'category %r' % (mi + 1, cat))
            target_map[cat] = float(target)
        normed.append((fields, target_map))
    return tuple(normed)


def _validate_header(header):
    if len(header) == 0:
        raise ArgumentError('header must contain at least one field')
    for h in header:
        if not isinstance(h, string_types) or not h:
            raise ArgumentError('header fields must be non-empty '
                                'strings, found %r' % (h,))
    if len(set(header)) != len(header):
        raise ArgumentError('header fields must be unique, found %r'
                            % (header,))


def _resolve_margins(normed_margins, header):
    specs = []
    for mi, (fields, targets) in enumerate(normed_margins):
        positions = []
        for f in fields:
            try:
                positions.append(header.index(f))
            except ValueError:
                raise FieldSelectionError(f)
        specs.append(MarginSpec(tuple(fields), tuple(positions),
                                targets, tuple(targets.keys())))
    return tuple(specs)


def _check_base_weight(value, rownum, weight):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ArgumentError(
            'row %d: base weight field %r must hold a strictly positive '
            'finite int or float, found %r' % (rownum, weight, value))
    return float(value)


def _check_category_value(field, value, rownum):
    if value is None or isinstance(value, string_types) \
            or _is_real_int(value):
        return value
    raise ArgumentError(
        'row %d: margin field %r has value %r of type %s; allowed '
        'category values are str, non-bool int or None'
        % (rownum, field, value, type(value).__name__))


# ---------------------------------------------------------------------
# structural infeasibility pre-checks (exact, solver-independent)
# ---------------------------------------------------------------------

def _precheck_infeasibility(specs, observed, base_weights, rows,
                            lower, upper):
    for mi, spec in enumerate(specs):
        lo_sum = {}
        hi_sum = {}
        for i, row in enumerate(rows):
            cat = tuple(row[p] for p in spec.positions)
            d = base_weights[i]
            lo_sum[cat] = lo_sum.get(cat, 0.0) + lower * d
            hi_sum[cat] = hi_sum.get(cat, 0.0) + upper * d
        for cat in spec.categories:
            target = spec.targets[cat]
            if cat not in observed[mi]:
                # declared but never observed in the sample
                if target > 0.0:
                    raise CalibrationError(
                        _margin_prefix(mi, spec)
                        + 'category %r is declared with positive target '
                        '%g but never occurs in the sample'
                        % (cat, target), INFEASIBLE)
                continue
            lo = lo_sum[cat]
            hi = hi_sum[cat]
            if target < lo or target > hi:
                raise CalibrationError(
                    _margin_prefix(mi, spec)
                    + 'category %r target %g cannot be reached within '
                      'the multiplier bounds: feasible weighted total is '
                      'in [%g, %g]'
                    % (cat, target, lo, hi), INFEASIBLE)

    # grand-total consistency across margins
    grand_targets = [sum(spec.targets.values()) for spec in specs]
    t0 = grand_targets[0]
    for mi in range(1, len(grand_targets)):
        ti = grand_targets[mi]
        if abs(ti - t0) > 1e-10 * max(1.0, abs(t0), abs(ti)):
            raise CalibrationError(
                'margin target grand totals disagree: margin 1 on fields '
                '%r sums to %g but margin %d on fields %r sums to %g'
                % (specs[0].fields, t0, mi + 1, specs[mi].fields, ti),
                INFEASIBLE)

    # The full boxed linear system is jointly feasible only if a linear
    # program over the multiplier box has a solution. That certificate
    # (see _lp_feasible) can be expensive on very large systems, so it is
    # not run here; the cheap exact structural checks above cover all
    # margin-local contradictions. The sparse problem is built once and
    # the LP is run lazily, only if the iterative solver stalls.
    return _build_problem(specs, base_weights, rows)


def _margin_prefix(mi, spec):
    return 'margin %d on fields %r: ' % (mi + 1, spec.fields)


# ---------------------------------------------------------------------
# problem construction
# ---------------------------------------------------------------------

def _build_problem(specs, base_weights, rows):
    n = len(rows)
    members = []
    targets = []
    constraint_margin = []
    constraint_category = []
    # map (margin, category) -> constraint index
    cat_index = {}
    for mi, spec in enumerate(specs):
        for cat in spec.categories:
            k = len(members)
            cat_index[(mi, cat)] = k
            members.append([])
            targets.append(spec.targets[cat])
            constraint_margin.append(mi)
            constraint_category.append(cat)

    row_constraints = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        for mi, spec in enumerate(specs):
            cat = tuple(row[p] for p in spec.positions)
            k = cat_index[(mi, cat)]
            members[k].append(i)
            row_constraints[i].append(k)

    # drop constraints for declared categories that never occur (target
    # 0 was validated); keep index alignment by rebuilding compactly
    kept = [k for k in range(len(members)) if members[k]]
    old_to_new = {}
    new_members = []
    new_targets = []
    new_margin = []
    new_category = []
    for new_k, old_k in enumerate(kept):
        old_to_new[old_k] = new_k
        new_members.append(tuple(members[old_k]))
        new_targets.append(targets[old_k])
        new_margin.append(constraint_margin[old_k])
        new_category.append(constraint_category[old_k])
    new_row_constraints = []
    for i in range(n):
        new_row_constraints.append(
            tuple(sorted(old_to_new[old_k] for old_k in row_constraints[i]
                         if old_k in old_to_new)))
    return Problem(n, specs, base_weights, tuple(new_members),
                   tuple(new_targets), tuple(new_margin),
                   tuple(new_category), tuple(new_row_constraints))


# ---------------------------------------------------------------------
# LP feasibility certificate (optional SciPy)
# ---------------------------------------------------------------------

def _lp_feasible(problem, lower, upper):
    """Certify feasibility of the boxed margin system with an LP.

    Variables are the multipliers ``x_i = w_i / d_i`` in
    ``[lower, upper]``. Constraints are ``sum_i a_ki d_i x_i = t_k``.

    Returns ``True`` when a feasible point is found (HiGHS status 0),
    ``False`` when the LP is certified infeasible (status 2), and
    ``None`` when no certificate is available - either because NumPy or
    SciPy is not installed or the solver reported an inconclusive
    status - so callers never mistake "unknown" for "infeasible".

    """

    try:
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError:
        return None

    try:
        n = problem.n
        m = len(problem.members)
        row_idx = []
        col_idx = []
        vals = []
        b = []
        for k, member_rows in enumerate(problem.members):
            t = problem.targets[k]
            # scale each row by its target (or 1 for zero targets) to
            # keep the matrix numerically balanced
            scale = t if t > 0.0 else 1.0
            b.append(t / scale)
            for i in member_rows:
                row_idx.append(k)
                col_idx.append(i)
                vals.append(problem.base_weights[i] / scale)
        a_eq = coo_matrix((np.asarray(vals, dtype=float),
                           (np.asarray(row_idx), np.asarray(col_idx))),
                          shape=(m, n))
        result = linprog(c=np.zeros(n), A_eq=a_eq,
                         b_eq=np.asarray(b, dtype=float),
                         bounds=[(lower, upper)] * n, method='highs')
    except Exception:
        return None

    if result.status == 0:
        return True
    if result.status == 2:
        return False
    return None


# ---------------------------------------------------------------------
# residual diagnostics
# ---------------------------------------------------------------------

def _residual_text(problem):
    totals = [0.0] * len(problem.members)
    for k, member_rows in enumerate(problem.members):
        totals[k] = sum(problem.base_weights[i] for i in member_rows)
    parts = []
    order = sorted(range(len(totals)),
                   key=lambda k: abs(totals[k] - problem.targets[k]),
                   reverse=True)
    for k in order[:5]:
        mi = problem.constraint_margin[k]
        spec = problem.specs[mi]
        parts.append('margin %d fields %r category %r: total %.12g vs '
                     'target %.12g'
                     % (mi + 1, spec.fields,
                        problem.constraint_category[k], totals[k],
                        problem.targets[k]))
    return 'largest residual constraints: ' + '; '.join(parts)


def _check_solution(weights, problem, tol, lower, upper):
    for w in weights:
        if not math.isfinite(w) or w <= 0.0:
            return False
    for i, w in enumerate(weights):
        ratio = w / problem.base_weights[i]
        if ratio < lower * (1.0 - 1e-12) - 1e-15 \
                or ratio > upper * (1.0 + 1e-12) + 1e-15:
            return False
    for k, member_rows in enumerate(problem.members):
        total = 0.0
        for i in member_rows:
            total += weights[i]
        target = problem.targets[k]
        if abs(total - target) > tol * max(1.0, abs(target)):
            return False
    return True


# ---------------------------------------------------------------------
# solver dispatch
# ---------------------------------------------------------------------

def _solve(problem, lower, upper, max_iter, tol):
    # The active-set coordinate ascent (IPF) solver runs on the Python
    # standard library only: it is linear in the data per sweep, handles
    # the multiplier box exactly, and converges on these sparse margin
    # systems in a small number of sweeps. The optional NumPy/SciPy
    # stack is used solely to certify infeasibility of the full boxed
    # linear system via a linear program (see _lp_feasible); the
    # iterative solve itself never needs it.
    #
    # `tol` is the user's acceptance gate. The solve is driven toward a
    # tighter internal target when the iteration budget allows, so the
    # returned weights approximate the true optimum rather than merely
    # being any tol-feasible point; but reaching the user tolerance is
    # always sufficient, even with a small `max_iter`.
    work_tol = min(tol, 1e-12)
    return _solve_pure(problem, lower, upper, max_iter, tol,
                       work_tol=work_tol)


# ---------------------------------------------------------------------
# pure-Python bounded coordinate ascent (standard library only)
# ---------------------------------------------------------------------

def _solve_pure(problem, lower, upper, max_iter, tol, warm_x=None,
                work_tol=None):
    # Bounded iterative proportional fitting (raking) on the multipliers.
    #
    # Let x_i = w_i / d_i be the multiplier of row i, kept in
    # [lower, upper]. One iteration is one full sweep over every margin
    # constraint. For constraint k with target t_k > 0, all multipliers
    # of its member rows are multiplied by a common factor q and clipped
    # to the box; q is chosen so the constraint's weighted total equals
    # its target. Because the total is linear in q once the set of
    # clipped rows is fixed, q is found exactly with a short active-set
    # procedure (typically a single pass): rows that would cross a bound
    # are pinned at that bound and the common factor is recomputed over
    # the remaining free rows. Each update is an exact bounded
    # coordinate step of the dual problem, so the dual objective
    # decreases monotonically and the iterates converge to the unique
    # optimum. Active bounds are handled exactly by the clipping.
    #
    # Constraint totals are recomputed from scratch after every sweep
    # rather than maintained incrementally, which keeps the residuals
    # free of accumulated round-off at a cost linear in the data.
    #
    # `tol` is the user acceptance gate: the first sweep meeting it is
    # returned. `work_tol`, when tighter, is the precision the sweeps
    # keep pursuing while iterations remain; it improves the accuracy of
    # the returned optimum without ever requiring more than `max_iter`.
    #
    # `warm_x`, when given, is a starting multiplier vector (e.g. the
    # output of an external solve); otherwise all multipliers start at 1,
    # corresponding to the base weights.
    if work_tol is None:
        work_tol = tol
    n = problem.n
    d = problem.base_weights
    m = len(problem.members)
    members = problem.members
    targets = problem.targets

    if warm_x is None:
        x = [1.0] * n
    else:
        x = [min(max(float(v), lower), upper) for v in warm_x]

    def recompute_totals():
        totals = [0.0] * m
        for k, member_rows in enumerate(members):
            total = 0.0
            for i in member_rows:
                total += d[i] * x[i]
            totals[k] = total
        return totals

    def worst_residual(totals):
        worst = -1.0
        worst_k = 0
        for k in range(m):
            r = abs(totals[k] - targets[k]) \
                / max(1.0, abs(targets[k]))
            if r > worst:
                worst = r
                worst_k = k
        return worst, worst_k

    # residual history used to recognize a stalled (non-contracting)
    # iteration, which on a boxed linear system indicates an infeasible
    # constraint set rather than a slow convergence
    stall_window = 10
    history = []

    for sweep in range(max_iter):
        # snapshot the multipliers to detect an exact fixed point
        before = x[:]
        for k, member_rows in enumerate(members):
            target = targets[k]
            if target <= 0.0:
                # only reachable when every member sits at the lower
                # bound; pre-checks reject this for observed rows, so
                # pin defensively and move on
                for i in member_rows:
                    x[i] = lower
                continue

            # active-set common-factor update
            free = list(member_rows)
            pinned_lo_sum = 0.0
            pinned_hi_sum = 0.0
            while free:
                free_sum = 0.0
                for i in free:
                    free_sum += d[i] * x[i]
                free_target = target - pinned_lo_sum - pinned_hi_sum
                if free_sum <= 0.0:
                    break
                q = free_target / free_sum
                crossed = []
                for j, i in enumerate(free):
                    new_value = q * x[i]
                    if new_value <= lower:
                        crossed.append((j, i, lower))
                    elif new_value >= upper:
                        crossed.append((j, i, upper))
                if not crossed:
                    for i in free:
                        x[i] = q * x[i]
                    break
                # pin every row that crossed a bound and re-solve over
                # the remaining free rows
                for j, i, bound in crossed:
                    x[i] = bound
                    if bound == lower:
                        pinned_lo_sum += d[i] * lower
                    else:
                        pinned_hi_sum += d[i] * upper
                crossed_set = set(i for _, i, _ in crossed)
                free = [i for i in free if i not in crossed_set]

        totals = recompute_totals()
        worst, worst_k = worst_residual(totals)
        # keep refining until the tight internal target is reached; the
        # user's looser tol only gates acceptance at the end, so the
        # returned weights are the optimum rather than an arbitrary
        # tol-feasible point. A fixed point at the user tolerance with no
        # further movement is handled by the stationarity check below.
        if worst <= work_tol:
            weights = [d[i] * x[i] for i in range(n)]
            if _check_solution(weights, problem, tol, lower, upper):
                return weights
        history.append(worst)

        # A full sweep that leaves every multiplier unchanged while
        # residuals remain is a stationary point of the boxed dual that
        # fails the constraints. A feasible system has exactly one
        # stationary point (the KKT optimum, which satisfies every
        # constraint), so this is a solver-independent infeasibility
        # certificate. The comparison tolerance allows for accumulated
        # round-off once the iterates have settled at a bound.
        stationary = all(
            abs(x[i] - before[i])
            <= 1e-12 * max(1.0, abs(x[i]), abs(before[i]))
            for i in range(n))
        if stationary and worst > work_tol:
            certificate = _lp_feasible(problem, lower, upper)
            if certificate is False:
                raise CalibrationError(
                    'the margin constraints have no common feasible '
                    'point within the multiplier bounds; %s'
                    % _residual_text(problem), INFEASIBLE)
            if certificate is None:
                # no LP backend available: the fixed-point argument
                # still certifies infeasibility on its own
                raise CalibrationError(
                    'the margin constraints have no common feasible '
                    'point within the multiplier bounds (the iterative '
                    'solver reached a fixed point with scaled residual '
                    '%.6g); %s'
                    % (worst, _residual_text(problem)), INFEASIBLE)

        # residual-based stall detection as a second, earlier signal;
        # it is only conclusive when an LP certificate is available
        if len(history) > stall_window \
                and sweep % stall_window == stall_window - 1:
            past = history[-stall_window - 1]
            now = history[-1]
            if past > 0.0 and now >= 0.5 * past and now > work_tol:
                if _lp_feasible(problem, lower, upper) is False:
                    raise CalibrationError(
                        'the margin constraints have no common feasible '
                        'point within the multiplier bounds (the iterative '
                        'solver stalled at scaled residual %.6g); %s'
                        % (now, _residual_text(problem)), INFEASIBLE)

    weights = [d[i] * x[i] for i in range(n)]
    if _check_solution(weights, problem, tol, lower, upper):
        return weights
    # a non-finite or non-positive result is a genuine numerical
    # breakdown, distinct from merely failing to converge in budget
    if any(not math.isfinite(w) or w <= 0.0 for w in weights):
        raise CalibrationError(
            'the numerical procedure produced a non-finite or '
            'non-positive weight; check the scale of the base weights '
            'and targets. %s' % _INSTALL_HINT, NUMERICAL_FAILURE)
    if _lp_feasible(problem, lower, upper) is False:
        raise CalibrationError(
            'the margin constraints have no common feasible point within '
            'the multiplier bounds; ' + _residual_text(problem),
            INFEASIBLE)
    totals = recompute_totals()
    worst, worst_k = worst_residual(totals)
    raise CalibrationError(
        'iteration limit %d reached without satisfying all margins; '
        'worst scaled residual %.6g at margin %d category %r; %s'
        % (max_iter, worst,
           problem.constraint_margin[worst_k] + 1,
           problem.constraint_category[worst_k],
           _residual_text(problem)), ITERATION_LIMIT)



Table.calibrateweights = calibrateweights
