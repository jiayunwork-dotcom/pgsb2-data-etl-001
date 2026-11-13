import os
import sys
import time

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import petl as etl

root, mode = sys.argv[1], sys.argv[2]
N = 5000


def gen():
    yield ('p', 'v')
    for i in range(N):
        # slow enough that the parent can kill us mid-stream
        yield (i % 50, 'row-%d' % i)
        if i % 100 == 0:
            time.sleep(0.002)


etl.topartitioned(gen(), root, partition_by=['p'], mode=mode,
                  max_open_files=4)
print('COMPLETED')  # should never print when killed
