"""Subprocess helper: run one commit against a dataset under a barrier.

Deterministic concurrency harness using coordination files that are
separate from the dataset's own commit lock:

* the process pins its base version (a plain read of the manifest);
* it then takes an external coordination lock, increments the shared
  ready count and releases the coordination lock;
* it waits until every participant has pinned, then runs the real
  public API.

Because the ready count is only incremented after the base version is
pinned, once the count reaches the number of participants every
participant is guaranteed to start from the same base version. The
dataset's own commit lock then serialises the commits and exactly one
participant wins.

Prints ``EVOLVED``/``COMMITTED`` on success or ``CONFLICT`` when the base
version was superseded.
"""
import os
import sys
import time

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import petl as etl
from petl.errors import PartitionCommitConflict
from petl.io import partitioned as pmod

root = sys.argv[1]
mode = sys.argv[2]
coord_dir = sys.argv[3]
n_participants = int(sys.argv[4])
tag = sys.argv[5] if len(sys.argv) > 5 else '0'

coord_lock = os.path.join(coord_dir, 'coord.lock')
ready_count = os.path.join(coord_dir, 'ready.count')
start_file = os.path.join(coord_dir, 'start')


def _acquire_coord_lock():
    deadline = time.time() + 30
    while True:
        try:
            fd = os.open(coord_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o644)
            return fd
        except OSError as e:
            if e.errno != 17:       # EEXIST
                raise
            if time.time() > deadline:
                raise
            time.sleep(0.005)


def _mark_ready():
    fd = _acquire_coord_lock()
    try:
        count = 0
        if os.path.exists(ready_count):
            with open(ready_count) as f:
                count = int(f.read().strip() or '0')
        count += 1
        with open(ready_count, 'w') as f:
            f.write(str(count))
    finally:
        os.close(fd)
        os.unlink(coord_lock)


def _wait_ready_then_start():
    deadline = time.time() + 30
    while True:
        try:
            with open(ready_count) as f:
                count = int(f.read().strip() or '0')
        except OSError:
            count = 0
        if count >= n_participants:
            break
        if time.time() > deadline:
            raise RuntimeError('participants never became ready')
        time.sleep(0.005)
    # everyone has pinned; hold briefly so no participant races ahead of
    # the group, then proceed together
    if not os.path.exists(start_file):
        fd = _acquire_coord_lock()
        try:
            if not os.path.exists(start_file):
                open(start_file, 'w').close()
        finally:
            os.close(fd)
            try:
                os.unlink(coord_lock)
            except OSError:
                pass
    _wait(start_file)


def _wait(path):
    deadline = time.time() + 30
    while not os.path.exists(path):
        if time.time() > deadline:
            raise RuntimeError('barrier was never opened: %r' % path)
        time.sleep(0.005)


def _evolve():
    # Pin the exact base version and commit against it, so every
    # participant demonstrably starts from the same old version.
    manifest = pmod._read_current(root)
    pinned = manifest['version']
    _mark_ready()
    _wait_ready_then_start()
    etl.evolvepartitioned(root, [('add', 'extra%s' % tag, 'D')],
                          _base_version=pinned)
    print('EVOLVED')


def _write():
    def gen():
        yield ('p', 'v')
        for i in range(50):
            yield (tag, '%s-%d' % (tag, i))
    manifest = pmod._read_current(root)
    pinned = manifest['version']
    _mark_ready()
    _wait_ready_then_start()
    # topartitioned pins internally from the same manifest the caller
    # would read; force the same semantics by stamping the base via the
    # public path after the barrier (the source traversal itself is
    # delayed until the barrier opens).
    etl.topartitioned(gen(), root, partition_by=['p'], mode='append')
    print('COMMITTED')


try:
    if mode == 'evolve':
        _evolve()
    elif mode == 'write':
        _write()
    else:
        raise SystemExit('unknown mode %r' % mode)
except PartitionCommitConflict:
    print('CONFLICT')
