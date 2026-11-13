import os
import random
import resource
import sys

# make the petl checkout importable regardless of the working directory
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import petl as etl

# Hard cap on file descriptors: far fewer than the number of partitions.
# std streams (3) plus a handful of metadata fds must still fit.
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

root = sys.argv[1]
N = 300
rows = [['p', 'v']]
order = list(range(N)) * 4
# deterministic shuffle so partitions arrive out of order
random.seed(7)
random.shuffle(order)
for i in order:
    rows.append([i, 'r%d' % i])

etl.topartitioned(rows, root, partition_by=['p'], max_open_files=8)
print('WRITE_OK')
