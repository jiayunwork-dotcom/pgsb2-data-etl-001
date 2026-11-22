"""Scale probe for NaN-key joins: left table has K NaN-keyed rows, right
table has 2K NaN-keyed rows, for K in (256, 4096, 65536).

Verifies and evidences:

* candidate pairing / relational decision work is O(N + M) - no N*M
  pair enumeration - for both the sorted merge joins and the hash joins;
* costs are reported separately for sorting, hash-index building and
  the pairing/probe phase;
* results are consumed row by row; the probe itself stores nothing
  proportional to the output (counters only);
* peak memory (tracemalloc) and wall time per phase.

Run:  PYTHONPATH=/workspace python3 scratch_nan/scale_probe_nan.py
"""

import gc
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import petl as etl
import petl.comparison as comparison
import petl.transform.joins as joins_mod
import petl.transform.hashjoins as hashjoins_mod


class Counters(object):
    lt_calls = 0          # Comparable.__lt__ invocations (sort + merge)
    merge_kcn_calls = 0   # key_contains_nan decisions in merge joins
    hash_kcn_calls = 0    # key_contains_nan guards in hash joins
    dict_contains = 0     # actual `k in rlookup` probes
    dict_getitem = 0      # actual rlookup[k] fetches

    @classmethod
    def reset(cls):
        cls.lt_calls = 0
        cls.merge_kcn_calls = 0
        cls.hash_kcn_calls = 0
        cls.dict_contains = 0
        cls.dict_getitem = 0


_orig_lt = comparison.Comparable.__lt__
_orig_kcn = comparison.key_contains_nan
_orig_hash_lookup = hashjoins_mod._hash_lookup


def _counting_lt(self, other):
    Counters.lt_calls += 1
    return _orig_lt(self, other)


def _counting_merge_kcn(key):
    Counters.merge_kcn_calls += 1
    return _orig_kcn(key)


def _counting_hash_kcn(key):
    Counters.hash_kcn_calls += 1
    return _orig_kcn(key)


class CountingDict(dict):
    def __contains__(self, k):
        Counters.dict_contains += 1
        return dict.__contains__(self, k)

    def __getitem__(self, k):
        Counters.dict_getitem += 1
        return dict.__getitem__(self, k)


def _counting_hash_lookup(table, key):
    return CountingDict(_orig_hash_lookup(table, key))


def instrument():
    comparison.Comparable.__lt__ = _counting_lt
    joins_mod.key_contains_nan = _counting_merge_kcn
    hashjoins_mod.key_contains_nan = _counting_hash_kcn
    hashjoins_mod._hash_lookup = _counting_hash_lookup


def uninstrument():
    comparison.Comparable.__lt__ = _orig_lt
    joins_mod.key_contains_nan = _orig_kcn
    hashjoins_mod.key_contains_nan = _orig_kcn
    hashjoins_mod._hash_lookup = _orig_hash_lookup


def nan_table(nrows, tag):
    # every row has a distinct float NaN key object
    return [('k', 'v')] + [(float('nan'), '%s%d' % (tag, i))
                           for i in range(nrows)]


def consume(view):
    """Consume rows one by one, counting only - nothing is stored."""
    n = 0
    for _ in view:
        n += 1
    return n


class Phase(object):
    """Time + peak-memory measurement for one phase. Wall time is the
    best of `repeat` runs *without* tracemalloc (which itself distorts
    timings); peak memory comes from a single separate traced run."""

    def __init__(self, fn, repeat=3):
        self.fn = fn
        self.repeat = repeat

    def measure(self):
        self.dt = min(self._timed() for _ in range(self.repeat))
        self.peak = self._traced()
        return self

    def _timed(self):
        gc.collect()
        t0 = time.perf_counter()
        self.fn()
        return time.perf_counter() - t0

    def _traced(self):
        gc.collect()
        tracemalloc.start()
        try:
            self.fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak


def fmt_mb(nbytes):
    return '%.1fMB' % (nbytes / 1e6)


def main():
    print('%-8s %-14s %-10s %-10s %-10s %-10s %-10s %-12s' %
          ('K', 'phase', 'rows_out', 'lt_calls', 'kcn_merge',
           'kcn_hash', 'dictprobe', 'time/peak'))
    prev = {}
    for k in (256, 4096, 65536):
        left = nan_table(k, 'L')
        right = nan_table(2 * k, 'R')

        # ----------------------------------------------------------
        # instrumented pass: count relational decisions (no timing)
        instrument()
        Counters.reset()
        rows_inner = consume(etl.join(left, right, key='k',
                                      buffersize=None, cache=False))
        m_inner = (Counters.lt_calls, Counters.merge_kcn_calls)
        Counters.reset()
        rows_outer = consume(etl.outerjoin(left, right, key='k',
                                           buffersize=None, cache=False))
        m_outer = (Counters.lt_calls, Counters.merge_kcn_calls)
        Counters.reset()
        rows_hash = consume(etl.hashjoin(left, right, key='k',
                                         cache=False))
        h_inner = (Counters.hash_kcn_calls, Counters.dict_contains,
                   Counters.dict_getitem)
        Counters.reset()
        rows_hleft = consume(etl.hashleftjoin(left, right, key='k',
                                              cache=False))
        h_left = (Counters.hash_kcn_calls, Counters.dict_contains,
                  Counters.dict_getitem)
        uninstrument()

        # ----------------------------------------------------------
        # clean pass: time and peak memory per cost component

        # 1. sorting alone (both sides, memory path)
        def do_sort():
            consume(etl.sort(left, 'k', buffersize=None, cache=False))
            consume(etl.sort(right, 'k', buffersize=None, cache=False))
        sort_ph = Phase(do_sort).measure()

        # 2. full merge join (sort + merge/pairing), memory path
        def do_join_mem():
            consume(etl.join(left, right, key='k', buffersize=None,
                             cache=False))
        join_mem_ph = Phase(do_join_mem).measure()

        # 3. pairing phase alone: presorted=True skips both sorts, so
        # this is the pure merge/pairing cost (all-NaN input is already
        # in the documented order)
        def do_join_presorted():
            consume(etl.join(left, right, key='k', presorted=True))
        join_pre_ph = Phase(do_join_presorted).measure()

        # 4. full merge join, spill path (buffersize=1024)
        def do_join_spill():
            consume(etl.join(left, right, key='k', buffersize=1024,
                             cache=False))
        join_spill_ph = Phase(do_join_spill).measure()

        # 5. hash index build alone (right table lookup)
        holder = []

        def do_index():
            holder.append(_orig_hash_lookup(right, 'k'))
        index_ph = Phase(do_index).measure()
        del holder[:]

        # 6. full hash join (index + probes)
        def do_hashjoin():
            consume(etl.hashjoin(left, right, key='k', cache=False))
        hash_ph = Phase(do_hashjoin).measure()

        # ----------------------------------------------------------
        # report
        print('K=%-6d N+2N rows=%-7d hypothetical N*M pairs=%d' %
              (k, 3 * k, 2 * k * k))
        print('  merge inner : rows=%-8d lt=%-9d kcn=%-5d '
              '(pairs emitted = 0 expected)' % (rows_inner, m_inner[0],
                                                m_inner[1]))
        print('  merge outer : rows=%-8d lt=%-9d kcn=%-5d (rows = 3K '
              'expected)' % (rows_outer, m_outer[0], m_outer[1]))
        print('  hash  inner : rows=%-8d kcn=%-9d contains=%d getitem=%d'
              % (rows_hash, h_inner[0], h_inner[1], h_inner[2]))
        print('  hash  left  : rows=%-8d kcn=%-9d contains=%d getitem=%d'
              % (rows_hleft, h_left[0], h_left[1], h_left[2]))
        print('  costs: sort(both) %.3fs/%s | join mem %.3fs/%s | '
              'join presorted(pairing only) %.3fs/%s | join spill '
              '%.3fs/%s'
              % (sort_ph.dt, fmt_mb(sort_ph.peak),
                 join_mem_ph.dt, fmt_mb(join_mem_ph.peak),
                 join_pre_ph.dt, fmt_mb(join_pre_ph.peak),
                 join_spill_ph.dt, fmt_mb(join_spill_ph.peak)))
        print('         hash index %.3fs/%s | hashjoin %.3fs/%s | '
              'hash probe share %.3fs'
              % (index_ph.dt, fmt_mb(index_ph.peak),
                 hash_ph.dt, fmt_mb(hash_ph.peak),
                 hash_ph.dt - index_ph.dt))

        # linearity assertions: decisions and outputs grow like K, and
        # nothing grows like K*K
        assert rows_inner == 1, rows_inner          # header only
        assert rows_outer == 1 + 3 * k, rows_outer  # header + 3K rows
        assert rows_hash == 1, rows_hash
        assert rows_hleft == 1 + k, rows_hleft
        assert h_inner[1] == 0 and h_inner[2] == 0  # zero dict probes
        assert h_inner[0] == k                      # one guard per row
        if prev:
            ratio = (m_outer[1] + h_inner[0]) / float(prev['decisions'])
            assert ratio <= 20, ratio  # 16x K growth => ~16x decisions
        prev['decisions'] = m_outer[1] + h_inner[0]
        print()


def control_duplicates_still_multiply():
    # control: duplicate *finite* keys keep their cartesian product
    k = 256
    left = [('k', 'v')] + [(1, 'L%d' % i) for i in range(k)]
    right = [('k', 'v')] + [(1, 'R%d' % i) for i in range(2 * k)]
    n = consume(etl.join(left, right, key='k', buffersize=None,
                         cache=False))
    print('control: finite duplicate keys 1 x K x 2K -> %d pairs '
          '(cartesian product preserved)' % (n - 1))
    assert n - 1 == 2 * k * k


if __name__ == '__main__':
    main()
    control_duplicates_still_multiply()
    print('OK')
