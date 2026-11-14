"""Large-row resource evidence for discoverkeys (kept out of the unit suite).

The input is a generator: the full table is never materialised. Run with
``python3 -I scripts_internal/bench_discoverkeys_rows.py`` from a checkout
that has the petl source on sys.path (the script inserts the repo root).
"""
import os
import resource
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import petl as etl


class GeneratorTable(object):
    """One-shot-ish repeatable table backed by a generator factory.

    Each iteration creates a fresh generator, so the 200k-row table is never
    held in memory as a list.
    """

    def __init__(self, ncols, nrows):
        self.header = tuple('c%d' % i for i in range(ncols))
        self.nrows = nrows
        self.iterations = 0
        self.rows_yielded = 0

    def __iter__(self):
        self.iterations += 1
        header = self.header
        nrows = self.nrows
        counter = self

        def gen():
            yield header
            for i in range(nrows):
                counter.rows_yielded += 1
                yield (i // 1000, i % 1000, 0, 0, 0, 0, 0, 0)

        return gen()


def run(nrows):
    table = GeneratorTable(8, nrows)
    tmp = tempfile.mkdtemp(prefix='petl-bench-%d-' % nrows)
    before_files = set(os.listdir(tmp))
    t0 = time.perf_counter()
    result = etl.discoverkeys(table, buffersize=64, tempdir=tmp)
    dt = time.perf_counter() - t0
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    leftovers = set(os.listdir(tmp)) - before_files
    expected = (('c0', 'c1'),)
    assert result == expected, result
    assert leftovers == set(), leftovers
    print('rows=%d time=%.2fs source_iters=%d rows_yielded=%d '
          'peak_rss=%.1fMiB temp_leftovers=%d'
          % (nrows, dt, table.iterations, table.rows_yielded,
             rss_kib / 1024.0, len(leftovers)))


if __name__ == '__main__':
    run(20000)
    run(200000)
