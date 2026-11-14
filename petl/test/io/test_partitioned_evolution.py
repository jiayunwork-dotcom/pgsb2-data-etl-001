# -*- coding: utf-8 -*-
"""Tests for schema evolution of Hive-style partitioned datasets."""
from __future__ import absolute_import, print_function, division

import builtins
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import pytest

import petl as etl
from petl.errors import ArgumentError, PartitionCommitConflict
import petl.io.partitioned as mod


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_partitioned_scripts')
META = '_petl_partitioned'


@pytest.fixture
def tmp_root():
    d = tempfile.mkdtemp()
    root = os.path.join(d, 'dataset')
    yield root
    shutil.rmtree(d, ignore_errors=True)


def _data_files(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if META in dirpath.split(os.sep):
            continue
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return sorted(out)


def _snapshot(root):
    snap = {}
    for path in _data_files(root):
        with open(path, 'rb') as f:
            snap[os.path.relpath(path, root)] = (
                os.stat(path).st_ino, f.read())
    return snap


def _current(root):
    with open(os.path.join(root, META, 'current'),
              encoding='utf-8') as f:
        return json.load(f)


def _strip_to_legacy(root):
    """Rewrite the manifest in the pre-evolution on-disk format."""
    manifest = _current(root)
    manifest.pop('schema', None)
    for entry in manifest['partitions']:
        for f in entry['files']:
            f.pop('columns', None)
    payload = json.dumps(manifest, sort_keys=True,
                         ensure_ascii=False).encode('utf-8')
    mod._atomic_write_bytes(mod._current_path(root), payload)


# ---------------------------------------------------------------------------
# Basic add / rename / drop / reorder semantics
# ---------------------------------------------------------------------------

def test_add_column_fills_historical_defaults_only(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2'], ['y', '3', '4']],
        tmp_root, partition_by=['p'])
    assert etl.evolvepartitioned(tmp_root, [('add', 'c', 'D')]) is None
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'c', 'p'),
        ('1', '2', 'D', 'x'),
        ('3', '4', 'D', 'y'),
    ]
    # rows written afterwards must supply the column explicitly
    etl.topartitioned(
        [['a', 'b', 'c', 'p'], ['5', '6', '7', 'x']],
        tmp_root, partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'c', 'p'),
        ('1', '2', 'D', 'x'),
        ('5', '6', '7', 'x'),
        ('3', '4', 'D', 'y'),
    ]


def test_add_empty_string_default_and_real_empty_values(tmp_root):
    etl.topartitioned(
        [['p', 'v'], ['x', 'real-empty'], ['y', 'keep']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('add', 'c', '')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('v', 'c', 'p'),
        ('real-empty', '', 'x'),
        ('keep', '', 'y'),
    ]
    # an explicit empty string written later is indistinguishable in value
    # but comes from the new file, not from defaults
    etl.topartitioned(
        [['v', 'c', 'p'], ['new', '', 'z']],
        tmp_root, partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('v', 'c', 'p'),
        ('real-empty', '', 'x'),
        ('keep', '', 'y'),
        ('new', '', 'z'),
    ]


def test_rename_keeps_position_and_values(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2']], tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('rename', 'a', 'alpha')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('alpha', 'b', 'p'), ('1', '2', 'x')]
    manifest = _current(tmp_root)
    assert manifest['header'] == ['alpha', 'b']


def test_rename_then_use_old_name_is_unknown(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('rename', 'a', 'b')])
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, [('rename', 'a', 'c')])
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, [('drop', 'a')])
    # dataset unchanged after rejected batches
    assert etl.header(etl.frompartitioned(tmp_root)) == ('b', 'p')


def test_drop_removes_values(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2'], ['y', '3', '4']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('drop', 'a')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('b', 'p'), ('2', 'x'), ('4', 'y')]


def test_drop_then_readd_same_name_is_fresh_column(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', 'old-a', '2']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(
        tmp_root, [('drop', 'a'), ('add', 'a', 'DEF')])
    # the old values of 'a' never come back; historical rows get the
    # default of the brand-new column
    assert list(etl.frompartitioned(tmp_root)) == [
        ('b', 'a', 'p'), ('2', 'DEF', 'x')]
    etl.topartitioned(
        [['b', 'a', 'p'], ['6', 'new-a', 'x']],
        tmp_root, partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('b', 'a', 'p'),
        ('2', 'DEF', 'x'),
        ('6', 'new-a', 'x'),
    ]
    manifest = _current(tmp_root)
    ids = {c['name']: c['id'] for c in manifest['schema']['columns']}
    # re-added column has a new id; the old id is never reused
    old_file = manifest['partitions'][0]['files'][0]
    assert ids['a'] not in old_file['columns']


def test_reorder_changes_logical_order_only(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b', 'c'], ['x', '1', '2', '3']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('reorder', ['c', 'a', 'b'])])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('c', 'a', 'b', 'p'), ('3', '1', '2', 'x')]


def test_chained_operations_use_previous_results(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2'], ['y', '3', '4']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('add', 'c', 'D'),
        ('rename', 'a', 'A'),
        ('reorder', ['b', 'A', 'c']),
        ('drop', 'b'),
        ('add', 'b', 'E'),
        ('reorder', ['A', 'b', 'c']),
    ])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('A', 'b', 'c', 'p'),
        ('1', 'E', 'D', 'x'),
        ('3', 'E', 'D', 'y'),
    ]
    # later rename/reorder cannot change defaults already assigned
    etl.evolvepartitioned(tmp_root, [
        ('rename', 'c', 'cc'), ('reorder', ['cc', 'A', 'b'])])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('cc', 'A', 'b', 'p'),
        ('D', '1', 'E', 'x'),
        ('D', '3', 'E', 'y'),
    ]


def test_quoted_and_multiline_csv_values_survive_evolution(tmp_root):
    table = [['p', 'v'],
             ['x', 'has,comma'],
             ['y', 'line1\nline2'],
             ['z', 'say "hi"']]
    etl.topartitioned(table, tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('rename', 'v', 'w'), ('add', 'n', '0')])
    rows = list(etl.frompartitioned(tmp_root))
    assert rows == [
        ('w', 'n', 'p'),
        ('has,comma', '0', 'x'),
        ('line1\nline2', '0', 'y'),
        ('say "hi"', '0', 'z'),
    ]


def test_none_partition_values_survive_evolution(tmp_root):
    etl.topartitioned(
        [['p', 'v'], [None, 'n1'], ['x', 'x1'], [None, 'n2']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('add', 'd', 'D'), ('rename', 'v', 'vv'), ('drop', 'd')])
    rows = list(etl.frompartitioned(tmp_root))
    assert rows == [
        ('vv', 'p'),
        ('n1', None),
        ('n2', None),
        ('x1', 'x'),
    ]


def test_multiline_and_quoted_values_in_mixed_schema_files(tmp_root):
    etl.topartitioned(
        [['p', 'v'], ['x', 'a,b'], ['x', 'q"q']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('add', 'd', 'D0')])
    etl.topartitioned(
        [['v', 'd', 'p'], ['line1\nline2', 'D1', 'x']],
        tmp_root, partition_by=['p'], mode='append')
    etl.evolvepartitioned(tmp_root, [('reorder', ['d', 'v'])])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('d', 'v', 'p'),
        ('D0', 'a,b', 'x'),
        ('D0', 'q"q', 'x'),
        ('D1', 'line1\nline2', 'x'),
    ]


def test_repeated_evolutions_stable_across_reopens(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', '1'], ['y', '2']], tmp_root,
        partition_by=['p'])
    batches = [
        [('add', 'b', 'B')],
        [('rename', 'a', 'aa')],
        [('add', 'c', 'C'), ('reorder', ['c', 'b', 'aa'])],
        [('drop', 'b')],
        [('add', 'b', 'B2')],
    ]
    for batch in batches:
        etl.evolvepartitioned(tmp_root, batch)
        # reopen through a fresh view every time, as a new process would
        seen = list(etl.frompartitioned(tmp_root))
    assert seen == [
        ('c', 'aa', 'b', 'p'),
        ('C', '1', 'B2', 'x'),
        ('C', '2', 'B2', 'y'),
    ]


def test_legacy_dataset_evolve_then_drop_and_readd(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', 'secret', '2']],
        tmp_root, partition_by=['p'])
    _strip_to_legacy(tmp_root)
    etl.evolvepartitioned(tmp_root, [
        ('drop', 'a'), ('add', 'a', 'FRESH')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('b', 'a', 'p'), ('2', 'FRESH', 'x')]


def test_mixed_schema_files_in_one_partition(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', 'v1-a'], ['y', 'v1-y']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    etl.topartitioned(
        [['a', 'b', 'p'], ['v2-a', 'v2-b', 'x']],
        tmp_root, partition_by=['p'], mode='append')
    etl.evolvepartitioned(tmp_root, [('rename', 'a', 'aa')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('aa', 'b', 'p'),
        ('v1-a', 'B', 'x'),
        ('v2-a', 'v2-b', 'x'),
        ('v1-y', 'B', 'y'),
    ]


# ---------------------------------------------------------------------------
# Validation and whole-batch rollback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [
    None, [], (), 'nope', 42, {}, [('add',)],
    [['add', 'c', 'D'], 'not-an-op'],
    [['add', 'c', 'D'], []],
])
def test_bad_changes_shape(tmp_root, bad):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    before = _current(tmp_root)
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, bad)
    assert _current(tmp_root) == before


def test_unknown_operation(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, [('frobnicate', 'a')])


def test_add_validation(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    bad_batches = [
        [('add', '', 'D')],
        [('add', 1, 'D')],
        [('add', 'a', 'D')],                  # duplicates data field
        [('add', 'p', 'D')],                  # collides with partition
        [('add', 'c', 1)],                    # default not a string
        [('add', 'c', None)],
        [('add', 'c', True)],
        [('add', 'c')],                       # wrong arity
        [('add', 'c', 'D', 'x')],
    ]
    before = _current(tmp_root)
    for batch in bad_batches:
        with pytest.raises(ArgumentError):
            etl.evolvepartitioned(tmp_root, batch)
        assert _current(tmp_root) == before


def test_rename_validation(tmp_root):
    etl.topartitioned([['p', 'a', 'b'], ['x', '1', '2']],
                      tmp_root, partition_by=['p'])
    bad_batches = [
        [('rename', 'missing', 'c')],
        [('rename', 'a', 'b')],               # name in use
        [('rename', 'a', 'p')],               # partition name
        [('rename', 'a', '')],
        [('rename', '', 'c')],
        [('rename', 'a', 1)],
        [('rename', 'a')],
        [('rename', 'a', 'c', 'x')],
        [('rename', 'p', 'q')],               # partition fields immutable
    ]
    before = _current(tmp_root)
    for batch in bad_batches:
        with pytest.raises(ArgumentError):
            etl.evolvepartitioned(tmp_root, batch)
        assert _current(tmp_root) == before


def test_drop_validation(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    bad_batches = [
        [('drop', 'missing')],
        [('drop', 'p')],                      # cannot drop partition field
        [('drop', '')],
        [('drop', 1)],
        [('drop',)],
        [('drop', 'a', 'x')],
        [('drop', 'a')],                      # would leave zero fields
    ]
    before = _current(tmp_root)
    for batch in bad_batches:
        with pytest.raises(ArgumentError):
            etl.evolvepartitioned(tmp_root, batch)
        assert _current(tmp_root) == before


def test_reorder_validation(tmp_root):
    etl.topartitioned([['p', 'a', 'b'], ['x', '1', '2']],
                      tmp_root, partition_by=['p'])
    bad_batches = [
        [('reorder', ['a'])],                       # missing fields
        [('reorder', ['a', 'b', 'c'])],             # unknown field
        [('reorder', ['a', 'a'])],                  # duplicated
        [('reorder', ['a', 'p'])],                  # partition field
        [('reorder', ['a', ''])],
        [('reorder', 'ab')],                        # not a sequence
        [('reorder', [])],
        [('reorder', ['a', 'b'], 'extra')],
    ]
    before = _current(tmp_root)
    for batch in bad_batches:
        with pytest.raises(ArgumentError):
            etl.evolvepartitioned(tmp_root, batch)
        assert _current(tmp_root) == before


def test_invalid_step_rolls_back_whole_batch(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2']], tmp_root, partition_by=['p'])
    before = _current(tmp_root)
    before_snapshot = _snapshot(tmp_root)
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, [
            ('add', 'c', 'D'),
            ('rename', 'a', 'A'),
            ('drop', 'missing'),                # fails at step 3
        ])
    assert _current(tmp_root) == before
    assert _snapshot(tmp_root) == before_snapshot
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('1', '2', 'x')]


def test_evolution_rejects_non_dataset_without_creating_one(tmp_root):
    # missing root
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, [('add', 'c', 'D')])
    assert not os.path.exists(tmp_root)
    # non-empty non-dataset directory
    os.makedirs(tmp_root)
    with open(os.path.join(tmp_root, 'keep.txt'), 'w') as f:
        f.write('important')
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(tmp_root, [('add', 'c', 'D')])
    assert open(os.path.join(tmp_root, 'keep.txt')).read() == 'important'
    # a meta-only directory (failed first write) is also not a dataset
    root2 = tmp_root + '2'
    os.makedirs(os.path.join(root2, META))
    with pytest.raises(ArgumentError):
        etl.evolvepartitioned(root2, [('add', 'c', 'D')])
    assert os.listdir(os.path.join(root2, META)) == []


# ---------------------------------------------------------------------------
# Metadata-only behaviour: no CSV file is ever touched
# ---------------------------------------------------------------------------

def test_evolution_never_opens_csv_files(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2']], tmp_root, partition_by=['p'])
    data_paths = set(_data_files(tmp_root))
    before = _snapshot(tmp_root)
    opened = []
    original_open = builtins.open

    def probe_open(file, *args, **kwargs):
        if os.path.abspath(str(file)) in {
                os.path.abspath(p) for p in data_paths}:
            opened.append(str(file))
        return original_open(file, *args, **kwargs)

    mod.builtins_open = probe_open
    try:
        etl.evolvepartitioned(
            tmp_root, [('add', 'c', 'D'), ('rename', 'a', 'A'),
                       ('drop', 'b'), ('reorder', ['A', 'c'])])
        etl.evolvepartitioned(tmp_root, [('add', 'b', '')])
    finally:
        mod.builtins_open = original_open
    assert opened == []
    # every CSV keeps the exact path, inode and bytes recorded beforehand
    assert _snapshot(tmp_root) == before
    assert {os.path.relpath(p, tmp_root) for p in _data_files(tmp_root)} \
        == set(before)


def test_csv_bytes_and_inodes_unchanged_across_many_evolutions(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', '1'], ['y', '2']], tmp_root,
        partition_by=['p'])
    before = _snapshot(tmp_root)
    for i in range(5):
        etl.evolvepartitioned(tmp_root, [('add', 'c%d' % i, 'd%d' % i)])
    etl.evolvepartitioned(tmp_root, [('rename', 'a', 'A')])
    etl.evolvepartitioned(tmp_root, [('drop', 'c0')])
    etl.evolvepartitioned(tmp_root, [('reorder', ['A', 'c4', 'c3',
                                                  'c2', 'c1'])])
    assert _snapshot(tmp_root) == before
    rows = list(etl.frompartitioned(tmp_root))
    assert rows[0] == ('A', 'c4', 'c3', 'c2', 'c1', 'p')
    assert rows[1] == ('1', 'd4', 'd3', 'd2', 'd1', 'x')
    assert rows[2] == ('2', 'd4', 'd3', 'd2', 'd1', 'y')


# ---------------------------------------------------------------------------
# Legacy datasets, empty datasets and all write modes
# ---------------------------------------------------------------------------

def test_legacy_dataset_reads_directly(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2']], tmp_root, partition_by=['p'])
    _strip_to_legacy(tmp_root)
    assert 'schema' not in _current(tmp_root)
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('1', '2', 'x')]


def test_legacy_dataset_can_evolve_and_keep_writing(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', 'old']], tmp_root, partition_by=['p'])
    legacy_snapshot = _snapshot(tmp_root)
    _strip_to_legacy(tmp_root)
    etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    assert _snapshot(tmp_root) == legacy_snapshot
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('old', 'B', 'x')]
    etl.topartitioned(
        [['a', 'b', 'p'], ['new', 'n', 'x']],
        tmp_root, partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('old', 'B', 'x'), ('new', 'n', 'x')]


def test_legacy_dataset_supports_all_write_modes(tmp_root):
    etl.topartitioned(
        [['p', 'v'], ['a', '1'], ['b', '2']], tmp_root,
        partition_by=['p'])
    _strip_to_legacy(tmp_root)
    etl.topartitioned([['p', 'v'], ['a', '11']], tmp_root,
                      partition_by=['p'], mode='append')
    assert [r[0] for r in list(etl.frompartitioned(tmp_root))[1:]] == \
        ['1', '11', '2']
    etl.topartitioned([['p', 'v'], ['b', '22']], tmp_root,
                      partition_by=['p'], mode='overwrite_partitions')
    rows = list(etl.frompartitioned(tmp_root))
    assert rows == [('v', 'p'), ('1', 'a'), ('11', 'a'), ('22', 'b')]
    etl.topartitioned([['p', 'v'], ['c', '9']], tmp_root,
                      partition_by=['p'], mode='overwrite')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('v', 'p'), ('9', 'c')]


def test_evolve_empty_dataset(tmp_root):
    etl.topartitioned([['p', 'a']], tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('add', 'b', 'B'), ('rename', 'a', 'aa')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('aa', 'b', 'p')]
    etl.topartitioned([['aa', 'b', 'p'], ['1', '2', 'x']],
                      tmp_root, partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('aa', 'b', 'p'), ('1', '2', 'x')]


def test_write_modes_after_evolution(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', '1'], ['y', '2']], tmp_root,
        partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    # append keeps old-format files and adds current-schema files
    snap = _snapshot(tmp_root)
    etl.topartitioned([['a', 'b', 'p'], ['3', '4', 'x']],
                      tmp_root, partition_by=['p'], mode='append')
    assert all(rel in {os.path.relpath(p, tmp_root)
                       for p in _data_files(tmp_root)}
               for rel in snap)
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'),
        ('1', 'B', 'x'),
        ('3', '4', 'x'),
        ('2', 'B', 'y'),
    ]
    # overwrite_partitions preserves other partitions' old files
    etl.topartitioned([['a', 'b', 'p'], ['9', '8', 'x']],
                      tmp_root, partition_by=['p'],
                      mode='overwrite_partitions')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'),
        ('9', '8', 'x'),
        ('2', 'B', 'y'),
    ]
    # whole-table overwrite uses the current logical header
    etl.topartitioned([['a', 'b', 'p'], ['z', 'z', 'q']],
                      tmp_root, partition_by=['p'], mode='overwrite')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('z', 'z', 'q')]
    manifest = _current(tmp_root)
    assert manifest['header'] == ['a', 'b']


def test_write_header_mismatch_rejected_before_data_rows(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('rename', 'a', 'b'), ('add', 'c', 'C')])

    class ExplodingSource:
        def __iter__(self):
            yield ('a', 'c', 'p')
            raise RuntimeError('data rows must never be consumed')

    before = _current(tmp_root)
    for mode in ('append', 'overwrite_partitions', 'overwrite'):
        with pytest.raises(ArgumentError):
            etl.topartitioned(ExplodingSource(), tmp_root,
                              partition_by=['p'], mode=mode)
    # wrong order with the right names is also a mismatch
    with pytest.raises(ArgumentError):
        etl.topartitioned([['c', 'b', 'p'], ['1', '2', 'x']],
                          tmp_root, partition_by=['p'], mode='append')
    # wrong partition_by
    with pytest.raises(ArgumentError):
        etl.topartitioned([['b', 'c', 'q'], ['1', '2', 'x']],
                          tmp_root, partition_by=['q'], mode='append')
    assert _current(tmp_root) == before


def test_short_rows_are_not_filled_with_defaults(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    with pytest.raises(ArgumentError):
        etl.topartitioned([['a', 'b', 'p'], ['only-one', 'x']],
                          tmp_root, partition_by=['p'], mode='append')
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('1', 'B', 'x')]


# ---------------------------------------------------------------------------
# Persistence: separate process and relocated directory
# ---------------------------------------------------------------------------

def test_readback_from_separate_process(tmp_root):
    etl.topartitioned(
        [['p', 'a', 'b'], ['x', '1', '2'], ['y', '3', '4']],
        tmp_root, partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('add', 'c', 'D'), ('rename', 'a', 'A'), ('drop', 'b')])
    etl.topartitioned([['A', 'c', 'p'], ['9', '8', 'z']],
                      tmp_root, partition_by=['p'], mode='append')
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', '..'))
    reader = (
        "import sys, json\n"
        "sys.path.insert(0, {root!r})\n"
        "import petl as etl\n"
        "print(json.dumps(list(etl.frompartitioned(sys.argv[1]))))\n"
    ).format(root=repo_root)
    proc = subprocess.run(
        [sys.executable, '-I', '-c', reader, tmp_root],
        capture_output=True, text=True, env=dict(os.environ,
                                                 PYTHONPATH=''))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [
        ['A', 'c', 'p'],
        ['1', 'D', 'x'],
        ['3', 'D', 'y'],
        ['9', '8', 'z'],
    ]


def test_readback_after_moving_dataset_directory(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', '1'], [None, '2']], tmp_root,
        partition_by=['p'])
    etl.evolvepartitioned(tmp_root, [
        ('add', 'b', 'B'), ('rename', 'a', 'aa')])
    moved = tmp_root + '-moved'
    shutil.move(tmp_root, moved)
    try:
        assert list(etl.frompartitioned(moved)) == [
            ('aa', 'b', 'p'),
            ('2', 'B', None),
            ('1', 'B', 'x'),
        ]
        # evolution and writes keep working at the new location
        etl.evolvepartitioned(moved, [('add', 'c', '')])
        assert list(etl.frompartitioned(moved))[0] == (
            'aa', 'b', 'c', 'p')
    finally:
        shutil.rmtree(moved, ignore_errors=True)


def test_view_reiteration_picks_up_new_schema(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    view = etl.frompartitioned(tmp_root)
    first = list(view)
    etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    second = list(view)
    assert first == [('a', 'p'), ('1', 'x')]
    assert second == [('a', 'b', 'p'), ('1', 'B', 'x')]


# ---------------------------------------------------------------------------
# Concurrency and conflicts
# ---------------------------------------------------------------------------

def test_concurrent_evolutions_only_one_wins(tmp_root):
    etl.topartitioned([['p', 'v'], ['seed', 's']], tmp_root,
                      partition_by=['p'])
    script = os.path.join(SCRIPT_DIR, 'concurrent_evolve.py')
    coord_dir = tempfile.mkdtemp()
    n = 4
    procs = []
    for i in range(n):
        procs.append(subprocess.Popen(
            [sys.executable, '-I', script, tmp_root, 'evolve',
             coord_dir, str(n), str(i)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    outcomes = []
    for proc in procs:
        out, err = proc.communicate()
        assert proc.returncode == 0, err.decode()
        outcomes.append(out.decode().strip())
    assert outcomes.count('EVOLVED') == 1
    assert outcomes.count('CONFLICT') == n - 1
    manifest = _current(tmp_root)
    # exactly one new column exists with one new version
    names = [c['name'] for c in manifest['schema']['columns']]
    assert names == ['v'] + [name for name in names
                             if name.startswith('extra')]
    assert len([name for name in names if name.startswith('extra')]) == 1
    assert list(etl.frompartitioned(tmp_root)) == [
        tuple(names + ['p']),
        ('s', 'D', 'seed'),
    ]


def test_evolution_vs_write_conflict_in_process(tmp_root):
    etl.topartitioned([['p', 'v'], ['seed', 's']], tmp_root,
                      partition_by=['p'])
    base = _current(tmp_root)
    barrier = threading.Barrier(2)
    errors = []

    def do_evolve():
        try:
            barrier.wait(10)
            etl.evolvepartitioned(tmp_root, [('add', 'e', 'E')])
        except Exception as exc:                       # noqa: BLE001
            errors.append(('evolve', exc))

    def do_write():
        try:
            barrier.wait(10)
            etl.topartitioned(
                [['p', 'v'], ['w', 'x']], tmp_root,
                partition_by=['p'], mode='append')
        except Exception as exc:                       # noqa: BLE001
            errors.append(('write', exc))

    t1 = threading.Thread(target=do_evolve)
    t2 = threading.Thread(target=do_write)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert len(errors) == 1
    kind, exc = errors[0]
    assert isinstance(exc, PartitionCommitConflict)
    manifest = _current(tmp_root)
    if kind == 'evolve':
        # write won: schema unchanged, new rows present
        assert manifest['version'] != base['version']
        assert manifest['header'] == ['v']
        rows = list(etl.frompartitioned(tmp_root))
        assert ('s', 'seed') in rows and ('x', 'w') in rows
    else:
        # evolve won: new schema, writer's rows absent
        assert manifest['header'] == ['v', 'e']
        rows = list(etl.frompartitioned(tmp_root))
        assert rows == [('v', 'e', 'p'), ('s', 'E', 'seed')]


def test_write_while_evolving_conflicts_even_with_matching_header(tmp_root):
    etl.topartitioned([['p', 'v'], ['seed', 's']], tmp_root,
                      partition_by=['p'])
    # rename away and back so the final header equals the original one
    etl.evolvepartitioned(tmp_root, [
        ('rename', 'v', 'tmp'), ('rename', 'tmp', 'v')])
    assert _current(tmp_root)['header'] == ['v']

    # A writer whose source is being consumed while an evolution commits
    # must detect at commit time that the base version is gone, even
    # though the final header happens to equal the original one.
    barrier = threading.Barrier(2)
    conflict = []

    def writer():
        try:
            barrier.wait(10)
            etl.topartitioned(
                [['p', 'v'], ['w', 'x']], tmp_root,
                partition_by=['p'], mode='append')
        except Exception as exc:                       # noqa: BLE001
            conflict.append(exc)

    def evolver():
        try:
            barrier.wait(10)
            etl.evolvepartitioned(tmp_root, [('add', 'g', 'G')])
        except Exception as exc:                       # noqa: BLE001
            conflict.append(exc)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=evolver)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert len(conflict) == 1
    assert isinstance(conflict[0], PartitionCommitConflict)
    rows = list(etl.frompartitioned(tmp_root))
    # evolver won (writer lost): no 'w' rows; schema includes g
    assert all(r[-1] != 'w' for r in rows[1:])
    assert etl.header(etl.frompartitioned(tmp_root)) == ('v', 'g', 'p')


def test_reader_pinned_through_evolution_or_conflicts(tmp_root):
    etl.topartitioned(
        [['p', 'v']] + [[i, 'v1-%d' % i] for i in range(200)],
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
        "        if i == 40:\n"
        "            time.sleep(1.2)\n"
        "    print('READ_COMPLETE', n)\n"
        "except PartitionCommitConflict:\n"
        "    print('READ_CONFLICT', n)\n"
    ).format(root=repo_root)
    evolver = (
        "import sys, time\n"
        "sys.path.insert(0, {root!r})\n"
        "import petl as etl\n"
        "time.sleep(0.4)\n"
        "etl.evolvepartitioned(sys.argv[1], [('add', 'n', 'D')])\n"
        "print('EVOLVED')\n"
    ).format(root=repo_root)
    rp = subprocess.Popen([sys.executable, '-I', '-c', reader, tmp_root],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ep = subprocess.Popen([sys.executable, '-I', '-c', evolver, tmp_root],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eout, eerr = ep.communicate()
    rout, rerr = rp.communicate()
    assert ep.returncode == 0, eerr.decode()
    assert rp.returncode == 0, rerr.decode()
    assert eout.decode().strip() == 'EVOLVED'
    line = rout.decode().strip()
    assert line == 'READ_COMPLETE 200' or line.startswith('READ_CONFLICT')
    # afterwards the new schema is readable in full
    rows = list(etl.frompartitioned(tmp_root))
    assert rows[0] == ('v', 'n', 'p')
    assert len(rows) == 201


def test_legal_write_after_conflict(tmp_root):
    etl.topartitioned([['p', 'v'], ['a', '1']], tmp_root,
                      partition_by=['p'])
    barrier = threading.Barrier(2)
    results = []

    def evolve():
        barrier.wait(10)
        try:
            etl.evolvepartitioned(tmp_root, [('add', 'n', 'D')])
            results.append('evolved')
        except PartitionCommitConflict:
            results.append('conflict')

    def write():
        barrier.wait(10)
        try:
            etl.topartitioned([['p', 'v'], ['b', '2']], tmp_root,
                              partition_by=['p'], mode='append')
            results.append('written')
        except PartitionCommitConflict:
            results.append('conflict')

    t1 = threading.Thread(target=evolve)
    t2 = threading.Thread(target=write)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert results.count('conflict') == 1
    # whichever lost, a subsequent operation against the current version
    # must succeed
    header = etl.header(etl.frompartitioned(tmp_root))
    if 'n' in header:
        etl.topartitioned([['v', 'n', 'p'], ['3', '4', 'c']], tmp_root,
                          partition_by=['p'], mode='append')
    else:
        etl.evolvepartitioned(tmp_root, [('add', 'n', 'D')])
        etl.topartitioned([['v', 'n', 'p'], ['3', '4', 'c']], tmp_root,
                          partition_by=['p'], mode='append')
    rows = list(etl.frompartitioned(tmp_root))
    assert ('3', '4', 'c') in rows


# ---------------------------------------------------------------------------
# I/O failure boundary
# ---------------------------------------------------------------------------

def test_io_error_before_publish_keeps_old_metadata_and_bytes(tmp_root):
    etl.topartitioned(
        [['p', 'a'], ['x', '1'], ['y', '2']], tmp_root,
        partition_by=['p'])
    before_manifest = _current(tmp_root)
    before_snapshot = _snapshot(tmp_root)
    original = mod._atomic_write_bytes

    def failing_write(path, data):
        raise OSError('disk full')

    mod._atomic_write_bytes = failing_write
    try:
        with pytest.raises(OSError):
            etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    finally:
        mod._atomic_write_bytes = original
    assert _current(tmp_root) == before_manifest
    assert _snapshot(tmp_root) == before_snapshot
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'p'), ('1', 'x'), ('2', 'y')]
    # the dataset remains fully usable afterwards
    etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    assert list(etl.frompartitioned(tmp_root)) == [
        ('a', 'b', 'p'), ('1', 'B', 'x'), ('2', 'B', 'y')]


def test_lock_io_error_passes_through(tmp_root):
    etl.topartitioned([['p', 'a'], ['x', '1']], tmp_root,
                      partition_by=['p'])
    before = _current(tmp_root)

    def failing_lock(dummy_root):
        raise OSError('cannot lock')

    original = mod._Lock
    mod._Lock = failing_lock
    try:
        with pytest.raises(OSError):
            etl.evolvepartitioned(tmp_root, [('add', 'b', 'B')])
    finally:
        mod._Lock = original
    assert _current(tmp_root) == before
