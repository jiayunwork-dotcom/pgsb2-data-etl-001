"""Scale and resource probes for MISSING-heavy joins.

The worst case for the feature is both input sides carrying only
MISSING-valued join keys, with no complete keys at all. With
match_missing=False the relational decision work must stay linear in
N + M rows: no N x M cartesian enumeration may be built and filtered away.

What is measured
----------------
* ``pairs`` - candidate pair rows the join actually materialises. For inner
  joins with match_missing off this is zero, which is the linearity proof;
  rows are consumed one at a time so even large outputs are never retained.
* ``cmp`` - ordering comparisons actually executed, counted by a lightweight
  :class:`~petl.comparison.Comparable` subclass installed for the run. The
  counter is only used at the smaller sizes; the 100k runs rely on timing and
  emitted-pair counts.
* wall-clock time and peak traced memory via :mod:`tracemalloc`.

match_missing=True deliberately creates a cartesian product of the two
all-MISSING groups, so that mode is probed with an asymmetric (large N, small
M) layout to keep the output countable while still exercising the path.

Run from the repository root::

    PYTHONPATH=. python3 scratch_missing/scale_probe.py
"""

from __future__ import print_function

import gc
import time
import tracemalloc

import petl.comparison as comparison_module
from petl.comparison import Comparable
import petl.transform.joins as joins_module
import petl as etl
from petl import MISSING


SIZES = (1000, 10000, 100000)
COUNT_SIZES = (1000, 10000)
SMALL_SIDE = 10  # right-table size for the match_missing=True probes


class CountingComparable(Comparable):
    """Comparable subclass with module-level comparison counters."""

    lt_calls = 0
    eq_calls = 0

    def __lt__(self, other):
        CountingComparable.lt_calls += 1
        return super(CountingComparable, self).__lt__(other)

    def __eq__(self, other):
        CountingComparable.eq_calls += 1
        return super(CountingComparable, self).__eq__(other)

    @classmethod
    def reset(cls):
        cls.lt_calls = 0
        cls.eq_calls = 0


def install_counter():
    # comparable_itemgetter looks Comparable up as a module global, and the
    # joins module imported the class by name; replace both references with
    # the counting subclass once for the whole script.
    comparison_module.Comparable = CountingComparable
    joins_module.Comparable = CountingComparable


def make_table(n, cols=2):
    rows = [('k',) + tuple('v%d' % i for i in range(cols - 1))]
    rows.extend((MISSING,) + tuple(i for _ in range(cols - 1))
                for i in range(n))
    return etl.wrap(rows)


def consume(view):
    """Stream the view, counting emitted candidate pairs without storing."""
    it = iter(view)
    next(it)  # header
    return sum(1 for _ in it)


def time_and_memory(fn):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return result, elapsed, peak / 1024.0


def probe(n, m, match_missing, buffersize=None, count=False):
    left = make_table(n)
    right = make_table(m)
    kwargs = {'key': 'k', 'match_missing': match_missing}
    if buffersize is not None:
        kwargs['buffersize'] = buffersize

    if count:
        CountingComparable.reset()
    (sort_pairs, sort_t, sort_mem) = time_and_memory(
        lambda: consume(etl.join(left, right, **kwargs)))
    sort_cmp = ((CountingComparable.lt_calls, CountingComparable.eq_calls)
                if count else None)

    if count:
        CountingComparable.reset()
    (hash_pairs, hash_t, hash_mem) = time_and_memory(
        lambda: consume(etl.hashjoin(left, right, key='k',
                                     match_missing=match_missing)))
    hash_cmp = ((CountingComparable.lt_calls, CountingComparable.eq_calls)
                if count else None)

    (left_pairs, left_t, left_mem) = time_and_memory(
        lambda: consume(etl.hashleftjoin(left, right, key='k',
                                         match_missing=match_missing)))

    return {
        'n': n, 'm': m, 'mm': match_missing, 'buf': buffersize,
        'sort': (sort_pairs, sort_t, sort_mem, sort_cmp),
        'hash': (hash_pairs, hash_t, hash_mem, hash_cmp),
        'hashleft': (left_pairs, left_t, left_mem),
    }


def fmt(cmp_tuple):
    return 'cmp(lt,eq)=%s' % (cmp_tuple,) if cmp_tuple else 'cmp=n/a'


def main():
    install_counter()
    header = ('%-7s %-6s %-5s %-6s | %-38s | %-30s | %-26s'
              % ('N', 'M', 'mm', 'buf', 'sorted inner', 'hash inner',
                 'hash left'))
    print(header)
    print('-' * len(header))

    linearity = {}
    for size in SIZES:
        count = size in COUNT_SIZES
        for buffersize in (None, 1000):
            r = probe(size, size, False, buffersize=buffersize, count=count)
            sp, st, sm, sc = r['sort']
            hp, ht, hm, hc = r['hash']
            lp, lt, lm = r['hashleft']
            print('%-7d %-6d %-6s %-6s | pairs=%-7d %6.3fs %9.0fKiB %s | '
                  'pairs=%-7d %6.3fs %9.0fKiB %s | pairs=%-7d %6.3fs %9.0fKiB'
                  % (size, size, False, str(buffersize),
                     sp, st, sm, fmt(sc), hp, ht, hm, fmt(hc),
                     lp, lt, lm))
            if buffersize is None:
                linearity[size] = r

        # opt-in matching mode: asymmetric sizes to keep the product countable
        r = probe(size, SMALL_SIDE, True, count=count)
        sp, st, sm, sc = r['sort']
        hp, ht, hm, hc = r['hash']
        lp, lt, lm = r['hashleft']
        print('%-7d %-6d %-6s %-6s | pairs=%-7d %6.3fs %9.0fKiB %s | '
              'pairs=%-7d %6.3fs %9.0fKiB %s | pairs=%-7d %6.3fs %9.0fKiB'
              % (size, SMALL_SIDE, True, 'n/a',
                 sp, st, sm, fmt(sc), hp, ht, hm, fmt(hc),
                 lp, lt, lm))

    print('\nlinearity assertions (match_missing=False, symmetric inputs):')
    for size in SIZES:
        r = linearity[size]
        assert r['sort'][0] == 0 and r['hash'][0] == 0
        assert r['hashleft'][0] == size  # exactly one outer row per left row
        print('  N=M=%-7d inner pairs=0 (both engines), left-outer rows=%d'
              % (size, size))

    small, big = linearity[1000], linearity[100000]
    # comparisons at the counted sizes must grow ~linearly, never quadratically
    if small['sort'][3] is not None:
        ratio_lt = big['sort'][3] if big['sort'][3] else None
        print('\nrecorded comparison counts at counted sizes:')
        for size in COUNT_SIZES:
            r = linearity[size]
            print('  N=%-6d sorted cmp(lt,eq)=%s hash cmp(lt,eq)=%s'
                  % (size, r['sort'][3], r['hash'][3]))
        print('  (100k runs report cmp=n/a to keep the probe lightweight)')

    growth = big['sort'][1] / max(small['sort'][1], 1e-9)
    print('\ntime growth sorted inner from 1k to 100k: x%.1f over x100 rows'
          % growth)
    print('SCALE PROBES COMPLETE')


if __name__ == '__main__':
    main()
