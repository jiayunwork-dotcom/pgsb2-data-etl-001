# -*- coding: utf-8 -*-
"""Tests for Hive-style partitioned dataset reading and writing."""
from __future__ import absolute_import, print_function, division

import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

import petl as etl
from petl.errors import ArgumentError, PartitionCommitConflict
from petl.io.partitioned import (
    _decode_partition_value,
    _encode_partition_value,
    _is_dataset,
)
from petl.util.base import Table


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_partitioned_scripts')


@pytest.fixture
def tmp_root():
    d = tempfile.mkdtemp()
    root = os.path.join(d, 'dataset')
    yield root
    shutil.rmtree(d, ignore_errors=True)


def _data_files(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if _META(root) in dirpath:
            continue
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return out


def _META(root):
    return os.path.join(root, '_petl_partitioned')


# ---------------------------------------------------------------------------
# Encoding round trips
# ---------------------------------------------------------------------------

def test_encode_basic_types():
    assert _encode_partition_value(None) == '__HIVE_DEFAULT_PARTITION__'
    assert _encode_partition_value(0) == '0'
    assert _encode_partition_value(-17) == '-17'
    assert _encode_partition_value('') == ''
    assert _encode_partition_value('plain text') == 'plain text'


def test_encode_special_characters():
    specials = '"#%\'*/:=?\\[]^{}'
    encoded = _encode_partition_value(specials)
    assert '%' in encoded
    # every unsafe byte is percent encoded; nothing raw remains
    for ch in specials:
        assert ch not in encoded.replace('%', '') or ch == '%'
    assert _decode_partition_value(encoded) == specials


def test_encode_controls_and_non_ascii():
    value = 'a\x00b\x1fc\x7fd caf\xc3\xa9 日本'
    encoded = _encode_partition_value(value)
    # controls and DEL are encoded, spaces and non-ascii stay literal
    assert '%00' in encoded and '%1F' in encoded and '%7F' in encoded
    assert ' ' in encoded and '日本' in encoded
    assert _decode_partition_value(encoded) == value


def test_encode_rejected_values():
    with pytest.raises(ArgumentError):
        _encode_partition_value('__HIVE_DEFAULT_PARTITION__')
    for bad in (True, False, 1.5, b'bytes', object()):
        with pytest.raises(TypeError):
            _encode_partition_value(bad)


def test_roundtrip_all_value_kinds(tmp_root):
    values = [None, '', 'a b', 'caf\xe9', '日本', 'a/b', '100%',
              'x:y=z?', 0, 42, -7, '"q"', '#h', '*s', '[b]', '{c}',
              'ctrl\x01', 'del\x7f']
    table = [['p', 'v']]
    for i, value in enumerate(values):
        table.append([value, 'r%d' % i])
    etl.topartitioned(table, tmp_root, partition_by=['p'])
    rows = list(etl.frompartitioned(tmp_root))
    assert rows[0] == ('v', 'p')
    got = {r[1]: r[0] for r in rows[1:]}
    expected = {None if v is None else str(v): 'r%d' % i
                for i, v in enumerate(values)}
    assert got == expected


def test_default_partition_decodes_to_none(tmp_root):
    etl.topartitioned([['p', 'v'], [None, 'x']], tmp_root,
                      partition_by=['p'])
    rows = list(etl.frompartitioned(tmp_root))
    assert rows == [('v', 'p'), ('x', None)]
    marker_dir = os.path.join(tmp_root, 'p=__HIVE_DEFAULT_PARTITION__')
    assert os.path.isdir(marker_dir)


# ---------------------------------------------------------------------------
# Layout and CSV format
# ---------------------------------------------------------------------------

def test_directory_layout_and_csv(tmp_root):
    table = [['country', 'city', 'people'],
             ['FR', 'Paris', 10],
             ['FR', 'Lyon', 5],
             ['UK', None, 8]]
    etl.topartitioned(table, tmp_root, partition_by=['country', 'city'])
    expected_dirs = [
        os.path.join('country=FR', 'city=Paris'),
        os.path.join('country=FR', 'city=Lyon'),
        os.path.join('country=UK',
                     'city=__HIVE_DEFAULT_PARTITION__'),
    ]
    for rel in expected_dirs:
        assert os.path.isdir(os.path.join(tmp_root, rel))
    for path in _data_files(tmp_root):
        assert not os.path.basename(path).startswith(('_', '.'))
        with open(path, encoding='utf-8', newline='') as f:
            content = f.read()
        # only the non-partition field is written, header first
        assert content.splitlines()[0] == 'people'
        assert 'country' not in content.splitlines()[0]


def test_header_is_non_partition_then_partition_fields(tmp_root):
    etl.topartitioned([['a', 'b', 'c'], [1, 2, 3]], tmp_root,
                      partition_by=['b'])
    assert etl.header(etl.frompartitioned(tmp_root)) == ('a', 'c', 'b')


def test_chained_method(tmp_root):
    tbl = etl.wrap([['p', 'v'], ['a', 1]])
    assert isinstance(tbl, Table)
    tbl.topartitioned(tmp_root, partition_by=['p'])
    assert list(etl.frompartitioned(tmp_root)) == [('v', 'p'), ('1', 'a')]


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_invalid_headers(tmp_root):
    for header in ((), ('a', 'a'), ('', 'b'), (1, 'b')):
        table = [tuple(header)] if header else []
        if not table:
            table = []
        with pytest.raises(ArgumentError):
            etl.topartitioned(table, tmp_root, partition_by=['a'])


def test_invalid_partition_by(tmp_root):
    table = [['a', 'b'], ['x', 'y']]
    for pb in ([], ['a', 'a'], ['missing'], ['a', 'b'], [1]):
        with pytest.raises(ArgumentError):
            etl.topartitioned(table, tmp_root, partition_by=pb)
    with pytest.raises(ArgumentError):
        etl.topartitioned([['a/b', 'b'], ['x', 'y']], tmp_root,
                          partition_by=['a/b'])


def test_row_width_error_carries_row_number(tmp_root):
    table = [['p', 'v'], ['a', '1'], ['b', '2', 'extra']]
    with pytest.raises(ArgumentError) as exc:
        etl.topartitioned(table, tmp_root, partition_by=['p'])
    assert 'row 2' in str(exc.value)
    assert not _is_dataset(tmp_root)


def test_bad_partition_value_carries_row_number(tmp_root):
    table = [['p', 'v'], ['a', '1'], [1.5, '2']]
    with pytest.raises(ArgumentError) as exc:
        etl.topartitioned(table, tmp_root, partition_by=['p'])
    assert 'row 2' in str(exc.value)
    table = [['p', 'v'], ['a', '1'], [True, '2']]
    with pytest.raises(ArgumentError) as exc:
        etl.topartitioned(table, tmp_root, partition_by=['p'])
    assert 'row 2' in str(exc.value)


def test_max_open_files_validation(tmp_root):
    table = [['a', 'b'], ['x', 'y']]
    for bad in (0, -3, True, False, 1.0, '4'):
        with pytest.raises(ArgumentError):
            etl.topartitioned(table, tmp_root, partition_by=['a'],
                              max_open_files=bad)


def test_unknown_mode(tmp_root):
    with pytest.raises(ArgumentError):
        etl.topartitioned([['a', 'b'], ['x', 'y']], tmp_root,
                          partition_by=['a'], mode='nope')


def test_non_dataset_root_is_never_modified(tmp_root):
    os.makedirs(tmp_root)
    sentinel = os.path.join(tmp_root, 'keep.txt')
    with open(sentinel, 'w') as f:
        f.write('important')
    for mode in ('error', 'overwrite', 'append', 'overwrite_partitions'):
        with pytest.raises(ArgumentError):
            etl.topartitioned([['p', 'v'], ['a', 1]], tmp_root,
                              partition_by=['p'], mode=mode)
    assert open(sentinel).read() == 'important'


def test_reader_rejects_non_dataset(tmp_root):
    with pytest.raises(ArgumentError):
        list(etl.frompartitioned(tmp_root))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_error_mode(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    with pytest.raises(ArgumentError):
        etl.topartitioned([['p', 'v'], ['a', '2']], tmp_root,
                          partition_by=['p'])
    # original data untouched
    assert list(etl.frompartitioned(tmp_root)) == [('v', 'p'), ('1', 'a')]


def test_overwrite_mode(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1'], ['b', '2']], tmp_root,
                      partition_by=['p'])
    etl.topartitioned([['p', 'v'], ['z', '9']], tmp_root,
                      partition_by=['p'], mode='overwrite')
    assert list(etl.frompartitioned(tmp_root)) == [('v', 'p'), ('9', 'z')]


def test_append_mode(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    etl.topartitioned([['p', 'v'], ['a', '2'], ['b', '3']], tmp_root,
                      partition_by=['p'], mode='append')
    rows = list(etl.frompartitioned(tmp_root))
    assert rows == [('v', 'p'), ('1', 'a'), ('2', 'a'), ('3', 'b')]


def test_overwrite_partitions_mode(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1'], ['b', '2']], tmp_root,
                      partition_by=['p'])
    etl.topartitioned([['p', 'v'], ['b', '20']], tmp_root,
                      partition_by=['p'], mode='overwrite_partitions')
    assert list(etl.frompartitioned(tmp_root)) == \
        [('v', 'p'), ('1', 'a'), ('20', 'b')]


def test_overwrite_partitions_empty_input_is_noop(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1'], ['b', '2']], tmp_root,
                      partition_by=['p'])
    before = list(etl.frompartitioned(tmp_root))
    etl.topartitioned([['p', 'v']], tmp_root, partition_by=['p'],
                      mode='overwrite_partitions')
    assert list(etl.frompartitioned(tmp_root)) == before


def test_modes_create_when_root_missing(tmp_root):
    for mode in ('overwrite', 'append', 'overwrite_partitions'):
        root = tmp_root + '-' + mode
        etl.topartitioned([['p', 'v'], ['a', '1']], root,
                          partition_by=['p'], mode=mode)
        assert _is_dataset(root)


def test_schema_mismatch_rejected_before_data_rows(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])

    class ExplodingSource:
        def __iter__(self):
            yield ('p', 'other')
            raise RuntimeError('data rows must never be consumed')

    for mode in ('append', 'overwrite_partitions'):
        with pytest.raises(ArgumentError):
            etl.topartitioned(ExplodingSource(), tmp_root,
                              partition_by=['p'], mode=mode)
    with pytest.raises(ArgumentError):
        etl.topartitioned([['q', 'v'], ['x', '1']], tmp_root,
                          partition_by=['q'], mode='append')


def test_empty_dataset_header_only(tmp_root):
    etl.topartitioned([['p', 'v']], tmp_root, partition_by=['p'])
    assert list(etl.frompartitioned(tmp_root)) == [('v', 'p')]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_partition_and_within_partition_ordering(tmp_root):
    table = [['p', 'q', 'v'],
             ['b', 1, 'b1'],
             ['a', 1, 'a1'],
             [None, 1, 'n1'],
             ['b', 1, 'b2'],
             ['a', 1, 'a2']]
    etl.topartitioned(table, tmp_root, partition_by=['p'])
    rows = list(etl.frompartitioned(tmp_root))
    # None sorts first, then a, b; input order kept inside a partition
    assert [r[2] for r in rows[1:]] == [None, 'a', 'a', 'b', 'b']
    assert [r[1] for r in rows[1:]] == ['n1', 'a1', 'a2', 'b1', 'b2']


def test_append_ordering_across_commits(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', 'first']], tmp_root,
                      partition_by=['p'])
    etl.topartitioned([['p', 'v'], ['a', 'second']], tmp_root,
                      partition_by=['p'], mode='append')
    rows = list(etl.frompartitioned(tmp_root))
    assert [r[0] for r in rows[1:]] == ['first', 'second']


# ---------------------------------------------------------------------------
# Resource limits and single traversal
# ---------------------------------------------------------------------------

def test_shuffled_partitions_respect_open_file_cap(tmp_root):
    import builtins
    import petl.io.partitioned as mod
    n_partitions = 60
    order = list(range(n_partitions)) * 4
    import random
    random.seed(11)
    random.shuffle(order)
    table = [['p', 'v']] + [[i, 'r%d' % i] for i in order]

    original_open = builtins.open
    live = [0]
    peak = [0]

    def counting_open(file, *args, **kwargs):
        f = original_open(file, *args, **kwargs)
        # only count data files opened inside the partitioned module
        if 'wb' in args or kwargs.get('mode') == 'wb':
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            original_close = f.close

            def close():
                if not f.closed:
                    live[0] -= 1
                return original_close()
            f.close = close
        return f

    mod.builtins_open = counting_open
    try:
        etl.topartitioned(table, tmp_root, partition_by=['p'],
                          max_open_files=2)
    finally:
        mod.builtins_open = original_open
    assert peak[0] <= 2, peak

    rows = list(etl.frompartitioned(tmp_root))
    assert len(rows) == 1 + n_partitions * 4
    counts = {}
    for r in rows[1:]:
        counts.setdefault(r[1], []).append(r[0])
    for i in range(n_partitions):
        assert counts[str(i)] == ['r%d' % i] * 4


def test_input_traversed_once(tmp_root):
    class Once:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            yield ('p', 'v')
            for i in range(20):
                yield (i % 5, i)

    source = Once()
    etl.topartitioned(source, tmp_root, partition_by=['p'],
                      max_open_files=1)
    assert source.iterations == 1


def test_fd_limited_subprocess(tmp_root):
    # 300 shuffled partitions, hard FD cap of 32, max_open_files=8.
    script = os.path.join(SCRIPT_DIR, 'fd_limit_writer.py')
    proc = subprocess.run(
        [sys.executable, '-I', script, tmp_root],
        capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=''))
    assert proc.returncode == 0, proc.stderr
    assert 'WRITE_OK' in proc.stdout
    rows = list(etl.frompartitioned(tmp_root))
    assert len(rows) == 1 + 300 * 4
    counts = {}
    for r in rows[1:]:
        counts.setdefault(r[1], []).append(r[0])
    for i in range(300):
        assert counts[str(i)] == ['r%d' % i] * 4


# ---------------------------------------------------------------------------
# Failure atomicity
# ---------------------------------------------------------------------------

def test_source_exception_leaves_previous_version(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    before = list(etl.frompartitioned(tmp_root))

    class Boom(Exception):
        pass

    class Source:
        def __iter__(self):
            yield ('p', 'v')
            yield ('b', '9')
            raise Boom('broken')

    with pytest.raises(Boom):
        etl.topartitioned(Source(), tmp_root, partition_by=['p'],
                          mode='append')
    assert list(etl.frompartitioned(tmp_root)) == before
    assert os.listdir(os.path.join(tmp_root, '_petl_partitioned',
                                   'tmp')) == []


def test_keyboard_interrupt_leaves_previous_version(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    before = list(etl.frompartitioned(tmp_root))

    def interrupted():
        yield ('p', 'v')
        yield ('b', '9')
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        etl.topartitioned(interrupted(), tmp_root,
                          partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == before


def test_failed_write_can_be_retried(tmp_root):
    with pytest.raises(ArgumentError):
        etl.topartitioned([['p', 'v'], ['a', '1'], ['bad']], tmp_root,
                          partition_by=['p'])
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    assert list(etl.frompartitioned(tmp_root)) == [('v', 'p'), ('1', 'a')]


def test_sigkill_mid_write(tmp_root):
    etl.topartitioned([['p', 'v'], ['k1', 'old1'], ['k2', 'old2']],
                      tmp_root, partition_by=['p'])
    v1 = list(etl.frompartitioned(tmp_root))

    script = os.path.join(SCRIPT_DIR, 'killable_writer.py')
    proc = subprocess.Popen(
        [sys.executable, '-I', script, tmp_root, 'append'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.5)
    proc.send_signal(9)
    proc.wait()
    assert proc.returncode == -9

    # previous version fully readable; staging leftovers invisible
    assert _is_dataset(tmp_root)
    assert list(etl.frompartitioned(tmp_root)) == v1

    # a later successful write ignores the leftovers
    etl.topartitioned([['p', 'v'], ['k9', 'new']], tmp_root,
                      partition_by=['p'], mode='append')
    rows = list(etl.frompartitioned(tmp_root))
    assert sorted(r[0] for r in rows[1:]) == \
        ['new', 'old1', 'old2']


# ---------------------------------------------------------------------------
# Concurrency and read consistency
# ---------------------------------------------------------------------------

def test_concurrent_commit_conflict(tmp_root):
    etl.topartitioned([['p', 'v'], ['seed', 's']], tmp_root,
                      partition_by=['p'])
    script = os.path.join(SCRIPT_DIR, 'concurrent_writer.py')
    outcomes = []
    procs = []
    for i in range(4):
        procs.append(subprocess.Popen(
            [sys.executable, '-I', script, tmp_root, 'w%d' % i, '0.3'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for proc in procs:
        out, err = proc.communicate()
        assert proc.returncode == 0, err.decode()
        outcomes.append(out.decode().strip())
    committed = [o for o in outcomes if o.startswith('COMMITTED')]
    conflicts = [o for o in outcomes if o.startswith('CONFLICT')]
    assert len(committed) >= 1
    assert len(committed) + len(conflicts) == 4
    # the winner's full payload is present; losers left nothing behind
    rows = list(etl.frompartitioned(tmp_root))
    counts = {}
    for r in rows[1:]:
        counts[r[1]] = counts.get(r[1], 0) + 1
    for line in committed:
        tag = line.split()[1]
        assert counts[tag] == 200
    for line in conflicts:
        tag = line.split()[1]
        assert tag not in counts


def test_reader_pinned_or_conflicts_on_new_commit(tmp_root):
    etl.topartitioned(
        [['p', 'v']] + [[i, 'v1-%d' % i] for i in range(300)],
        tmp_root, partition_by=['p'], max_open_files=8)
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', '..'))
    reader = (
        "import sys, time\n"
        "sys.path.insert(0, {root!r})\n"
        "import petl as etl\n"
        "from petl.errors import PartitionCommitConflict\n"
        "it = iter(etl.frompartitioned(sys.argv[1]))\n"
        "next(it)\n"
        "n = 0\n"
        "try:\n"
        "    for i, _ in enumerate(it):\n"
        "        n += 1\n"
        "        if i == 50:\n"
        "            time.sleep(1.5)\n"
        "    print('READ_COMPLETE', n)\n"
        "except PartitionCommitConflict:\n"
        "    print('READ_CONFLICT', n)\n"
    ).format(root=repo_root)
    writer = (
        "import sys, time\n"
        "sys.path.insert(0, {root!r})\n"
        "import petl as etl\n"
        "time.sleep(0.5)\n"
        "etl.topartitioned(\n"
        "    [['p', 'v']] + [[i, 'v2-%d' % i] for i in range(300)],\n"
        "    sys.argv[1], partition_by=['p'], mode='overwrite')\n"
        "print('V2_COMMITTED')\n"
    ).format(root=repo_root)
    rp = subprocess.Popen([sys.executable, '-I', '-c', reader, tmp_root],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wp = subprocess.Popen([sys.executable, '-I', '-c', writer, tmp_root],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wout, werr = wp.communicate()
    rout, rerr = rp.communicate()
    assert wp.returncode == 0, werr.decode()
    assert rp.returncode == 0, rerr.decode()
    line = rout.decode().strip()
    # either the pinned version is read in full or an explicit conflict;
    # never a mix or a silently truncated iteration
    assert line == 'READ_COMPLETE 300' or line.startswith('READ_CONFLICT')


# ---------------------------------------------------------------------------
# No file growth under repeated overwrites
# ---------------------------------------------------------------------------

def _disk_usage(root):
    n = 0
    size = 0
    for path in _data_files(root):
        n += 1
        size += os.path.getsize(path)
    return n, size


def test_repeated_overwrite_does_not_grow(tmp_root):
    table = [['p', 'v']] + [[i % 20, 'x' * 100] for i in range(500)]
    etl.topartitioned(table, tmp_root, partition_by=['p'],
                      mode='overwrite', max_open_files=4)
    snapshot = _disk_usage(tmp_root)
    for _ in range(10):
        etl.topartitioned(table, tmp_root, partition_by=['p'],
                          mode='overwrite', max_open_files=4)
    assert _disk_usage(tmp_root) == snapshot


def test_repeated_overwrite_partitions_does_not_grow(tmp_root):
    etl.topartitioned(
        [['p', 'v']] + [[i % 20, 'x' * 100] for i in range(500)],
        tmp_root, partition_by=['p'], max_open_files=4)
    etl.topartitioned([['p', 'v'], [0, 'y'], [1, 'z']], tmp_root,
                      partition_by=['p'], mode='overwrite_partitions')
    snapshot = _disk_usage(tmp_root)
    for _ in range(10):
        etl.topartitioned([['p', 'v'], [0, 'y'], [1, 'z']], tmp_root,
                          partition_by=['p'],
                          mode='overwrite_partitions')
        assert _disk_usage(tmp_root) == snapshot


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

def test_partition_commit_conflict_is_exception():
    assert issubclass(PartitionCommitConflict, Exception)


def test_multi_level_encoding_roundtrip(tmp_root):
    values = [('a/b', None), ('a/b', 'c d'), ('x', 42), ('', '')]
    table = [['g', 'p', 'v']]
    for g, p in values:
        table.append([g, p, 'v'])
    etl.topartitioned(table, tmp_root, partition_by=['g', 'p'])
    rows = list(etl.frompartitioned(tmp_root))
    assert rows[0] == ('v', 'g', 'p')
    got = {(r[1], r[2]) for r in rows[1:]}
    expected = {(g, None if p is None else str(p)) for g, p in values}
    assert got == expected
    # nested directories exist with encoded components
    assert os.path.isdir(os.path.join(
        tmp_root, 'g=a%2Fb', 'p=__HIVE_DEFAULT_PARTITION__'))


def test_cleanup_error_does_not_mask_original(tmp_root, monkeypatch):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    before = list(etl.frompartitioned(tmp_root))

    class Boom(Exception):
        pass

    # force staging cleanup itself to fail; the source error must survive
    import petl.io.partitioned as mod
    original_remove = mod._remove_tree

    def failing_remove(path):
        original_remove(path)
        raise OSError('cleanup failed')

    monkeypatch.setattr(mod, '_remove_tree', failing_remove)

    class Source:
        def __iter__(self):
            yield ('p', 'v')
            raise Boom('source error')

    try:
        etl.topartitioned(Source(), tmp_root, partition_by=['p'],
                          mode='append')
    except Boom:
        pass
    else:
        raise AssertionError('original exception was masked')
    # restore and confirm the dataset is still usable
    monkeypatch.undo()
    assert list(etl.frompartitioned(tmp_root)) == before


def test_write_then_append_then_overwrite_full_cycle(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    etl.topartitioned([['p', 'v'], ['a', '2'], ['b', '3']], tmp_root,
                      partition_by=['p'], mode='append')
    etl.topartitioned([['p', 'v'], ['c', '4']], tmp_root,
                      partition_by=['p'], mode='overwrite')
    rows = list(etl.frompartitioned(tmp_root))
    assert rows == [('v', 'p'), ('4', 'c')]
    # only the current version's files are referenced
    import json
    with open(os.path.join(tmp_root, '_petl_partitioned', 'current'),
              encoding='utf-8') as f:
        manifest = json.load(f)
    referenced = set()
    for entry in manifest['partitions']:
        for fdesc in entry['files']:
            referenced.add(fdesc['path'])
    on_disk = {os.path.relpath(p, tmp_root) for p in _data_files(tmp_root)}
    assert on_disk == referenced


def test_reader_is_reiterable_independently(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1'], ['b', '2']], tmp_root,
                      partition_by=['p'])
    view = etl.frompartitioned(tmp_root)
    first = list(view)
    etl.topartitioned([['p', 'v'], ['z', '9']], tmp_root,
                      partition_by=['p'], mode='overwrite')
    # a fresh iteration sees the newly committed version
    second = list(view)
    assert first == [('v', 'p'), ('1', 'a'), ('2', 'b')]
    assert second == [('v', 'p'), ('9', 'z')]
