import os
import sys
import time

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import petl as etl
from petl.errors import PartitionCommitConflict

root, tag, hold = sys.argv[1], sys.argv[2], float(sys.argv[3])


def gen():
    yield ('p', 'v')
    for i in range(200):
        yield (tag, '%s-%d' % (tag, i))
        if i == 100:
            time.sleep(hold)  # widen the overlap window


try:
    etl.topartitioned(gen(), root, partition_by=['p'], mode='append')
    print('COMMITTED ' + tag)
except PartitionCommitConflict:
    print('CONFLICT ' + tag)
