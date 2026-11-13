# -*- coding: utf-8 -*-
"""Read and write Hive-style partitioned datasets.

A partitioned dataset is a directory tree whose layout mirrors the values
of one or more partition fields. Each partition directory contains one or
more UTF-8 CSV data files holding the non-partition fields. All changes to
a dataset are published through an atomic, versioned commit so that readers
either see the complete result of a write or the complete previous version,
never a mixture of the two.
"""
from __future__ import absolute_import, print_function, division

import builtins
import csv
import errno
import io
import json
import os
import uuid

from petl.comparison import Comparable
from petl.errors import ArgumentError, PartitionCommitConflict
from petl.util.base import Table


# Name of the directory holding dataset metadata and staged writes. The
# leading underscore guarantees it can never collide with a partition
# directory name (which are always of the form ``field=value``) and that it
# is never mistaken for a data file (data file names may not start with
# ``_`` or ``.``).
_META_DIR = '_petl_partitioned'
_CURRENT_NAME = 'current'
_LOCK_NAME = 'lock'
_TMP_NAME = 'tmp'
_MARKER = '__HIVE_DEFAULT_PARTITION__'
_CSV_DIALECT = 'excel'

# Indirection so callers (and tests) can observe/limit file openings without
# monkeypatching the built-in ``open`` globally.
builtins_open = builtins.open

# Characters that must be percent-encoded when they appear in a partition
# value: C0 controls (U+0000-U+001F), DEL (U+007F) and a fixed set of
# punctuation characters that are unsafe in Hive-style paths.
_ENCODE_CHARS = frozenset(
    [chr(c) for c in range(0x00, 0x20)]
    + [chr(0x7F)]
    + list('"#%\'*/:=?\\[]^{}')
)


# ---------------------------------------------------------------------------
# Partition value encoding / decoding
# ---------------------------------------------------------------------------

def _encode_partition_value(value):
    """Encode one partition value into its single directory-name component."""
    if value is None:
        return _MARKER
    if isinstance(value, bool):
        # bool is a subclass of int; reject it explicitly.
        raise TypeError('partition values must be str, int or None; '
                        'bool is not accepted: %r' % (value,))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if value == _MARKER:
            raise ArgumentError(
                'partition value %r is reserved for the default (None) '
                'partition' % _MARKER)
        out = []
        for ch in value:
            if ch in _ENCODE_CHARS:
                for byte in ch.encode('utf-8'):
                    out.append('%%%02X' % byte)
            else:
                out.append(ch)
        return ''.join(out)
    raise TypeError(
        'partition values must be str, int or None, got %r' % (value,))


def _decode_partition_value(text):
    """Reverse :func:`_encode_partition_value`, always returning str or
    None."""
    if text == _MARKER:
        return None
    if '%' not in text:
        return text
    out = bytearray()
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '%':
            if i + 2 >= n:
                raise ArgumentError(
                    'invalid encoded partition value: %r' % text)
            try:
                out.append(int(text[i + 1:i + 3], 16))
            except ValueError:
                raise ArgumentError(
                    'invalid encoded partition value: %r' % text)
            i += 3
        else:
            out.extend(ch.encode('utf-8'))
            i += 1
    return out.decode('utf-8')


def _encode_partition_tuple(values):
    return tuple(_encode_partition_value(v) for v in values)


# ---------------------------------------------------------------------------
# On-disk layout helpers
# ---------------------------------------------------------------------------

def _meta_dir(root):
    return os.path.join(root, _META_DIR)


def _current_path(root):
    return os.path.join(_meta_dir(root), _CURRENT_NAME)


def _lock_path(root):
    return os.path.join(_meta_dir(root), _LOCK_NAME)


def _tmp_root(root):
    return os.path.join(_meta_dir(root), _TMP_NAME)


def _partition_dir(root, field_names, encoded_values):
    parts = [f + '=' + v
             for f, v in zip(field_names, encoded_values)]
    return os.path.join(root, *parts)


def _is_dataset(root):
    return os.path.isdir(root) and os.path.isfile(_current_path(root))


def _read_current(root):
    """Read and parse the committed manifest. Returns a dict."""
    try:
        with builtins_open(_current_path(root), 'r',
                           encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, OSError) as e:
        raise ArgumentError(
            'root %r does not contain a readable partitioned dataset: %s'
            % (root, e))


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ArgumentError('partitioned dataset manifest is invalid')
    for key in ('version', 'partition_by', 'header', 'partitions'):
        if key not in manifest:
            raise ArgumentError(
                'partitioned dataset manifest is missing %r' % key)
    if not isinstance(manifest['partition_by'], list) \
            or not isinstance(manifest['header'], list) \
            or not isinstance(manifest['partitions'], list):
        raise ArgumentError('partitioned dataset manifest is invalid')


def _ensure_empty_or_dataset(root):
    """Classify an existing root.

    Returns the parsed manifest if a dataset exists there, None if the
    location is empty (or does not yet exist). Raises ArgumentError if the
    location is occupied by something that is not one of our datasets.
    Nothing under an unrecognised root is ever removed.
    """
    if not os.path.exists(root):
        return None
    if not os.path.isdir(root):
        raise ArgumentError(
            'root %r exists and is not a directory' % root)
    if _is_dataset(root):
        manifest = _read_current(root)
        _validate_manifest(manifest)
        return manifest
    # Existing directory. It counts as empty when it contains nothing at
    # all, or only this module's own metadata directory (e.g. left behind
    # by a write that failed before its first commit).
    entries = os.listdir(root)
    if entries == [] or entries == [_META_DIR]:
        return None
    raise ArgumentError(
        'root %r exists, is not empty and is not a partitioned dataset'
        % root)


def _atomic_write_bytes(path, data):
    """Write bytes to path via a sibling temporary file and atomic rename."""
    directory = os.path.dirname(path)
    tmp = os.path.join(directory, '.tmp-%s' % uuid.uuid4().hex)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp, path)


def _fsync_dir(path):
    # Best effort; some platforms do not support fsync on directories.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class _Lock(object):
    """Exclusive inter-process lock, used to serialise commits."""

    def __init__(self, root):
        self._path = _lock_path(root)
        self._fd = None

    def __enter__(self):
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        except ImportError:
            # Minimal fallback for platforms without fcntl: spin on an
            # exclusively-created lock file. Stale lock files are possible
            # after a hard kill but are reclaimable once their owner is gone.
            os.close(fd)
            import time
            while True:
                try:
                    fd = os.open(self._path,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                    break
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    time.sleep(0.05)
        self._fd = fd
        return self

    def __exit__(self, exc_type, exc, tb):
        fd = self._fd
        self._fd = None
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            except ImportError:
                try:
                    os.unlink(self._path)
                except OSError:
                    pass
        finally:
            os.close(fd)


def _remove_tree(path):
    """Remove a directory tree, tolerating files vanishing underneath us."""
    if not os.path.lexists(path):
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            try:
                os.unlink(os.path.join(dirpath, name))
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
        for name in dirnames:
            full = os.path.join(dirpath, name)
            try:
                os.rmdir(full)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
    try:
        os.rmdir(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


# ---------------------------------------------------------------------------
# Validation of arguments
# ---------------------------------------------------------------------------

def _validate_header(header):
    if not isinstance(header, (list, tuple)):
        raise ArgumentError('table header must be a sequence of field names')
    names = list(header)
    if not names:
        raise ArgumentError('table header must not be empty')
    seen = set()
    for name in names:
        if not isinstance(name, str) or name == '':
            raise ArgumentError(
                'header field names must be non-empty strings: %r' % (name,))
        if name in seen:
            raise ArgumentError('duplicate header field name: %r' % name)
        seen.add(name)
    return names


def _validate_partition_by(partition_by, header):
    if partition_by is None or not isinstance(partition_by,
                                               (list, tuple)):
        raise ArgumentError(
            'partition_by must be a non-empty sequence of field names')
    names = list(partition_by)
    if not names:
        raise ArgumentError(
            'partition_by must be a non-empty sequence of field names')
    seen = set()
    for name in names:
        if not isinstance(name, str) or name == '':
            raise ArgumentError(
                'partition field names must be non-empty strings: %r'
                % (name,))
        if name in seen:
            raise ArgumentError(
                'duplicate partition field name: %r' % name)
        if name not in header:
            raise ArgumentError(
                'partition field %r is not present in the table header'
                % name)
        if name == _MARKER or any(ch in _ENCODE_CHARS for ch in name):
            raise ArgumentError(
                'partition field name %r contains characters that require '
                'encoding' % name)
        seen.add(name)
    if len(seen) == len(header):
        raise ArgumentError(
            'at least one non-partition field must remain after partitioning')
    return names


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class _PartitionWriterPool(object):
    """LRU pool of open per-partition CSV data files."""

    def __init__(self, staging_dir, data_header, max_open_files):
        self._staging_dir = staging_dir
        self._data_header = data_header
        self._max_open_files = max_open_files
        # key: encoded partition tuple -> open state dict
        self._states = {}
        # LRU order: least recently used first
        self._order = []
        # key: encoded partition tuple -> list of (path, size), retained
        # after the underlying file has been closed/evicted
        self._files = {}

    def _open(self, key):
        # Enforce the hard cap before opening: close the least recently
        # used partition file first so the number of data files open never
        # exceeds max_open_files.
        while len(self._order) >= self._max_open_files:
            old_key = self._order.pop(0)
            state = self._states.pop(old_key)
            self._close_state(old_key, state)
        path = os.path.join(self._staging_dir,
                            'data-%s.csv' % uuid.uuid4().hex)
        buf = builtins_open(path, 'wb')
        text = io.TextIOWrapper(buf, encoding='utf-8', newline='')
        writer = csv.writer(text, dialect=_CSV_DIALECT)
        writer.writerow(self._data_header)
        state = {'path': path, 'buf': buf, 'text': text,
                 'writer': writer, 'bytes': 0}
        self._states[key] = state
        self._order.append(key)
        self._files.setdefault(key, [])
        return state

    def _close_state(self, key, state):
        text = state.pop('text')
        buf = state.pop('buf')
        state.pop('writer')
        text.flush()
        text.detach()
        buf.flush()
        os.fsync(buf.fileno())
        size = os.path.getsize(state['path'])
        state['bytes'] = size
        buf.close()
        self._files[key].append((state['path'], size))

    def write_row(self, key, row):
        state = self._states.get(key)
        if state is None:
            state = self._open(key)
        else:
            # mark most recently used
            self._order.remove(key)
            self._order.append(key)
        state['writer'].writerow(row)

    def close_all(self):
        while self._order:
            key = self._order.pop(0)
            state = self._states.pop(key)
            self._close_state(key, state)

    def close_all_suppress(self):
        # Used during failure cleanup: never let close errors mask the
        # original exception.
        try:
            self.close_all()
        except Exception:
            pass

    def files(self):
        """Return mapping key -> list of (path, size)."""
        return self._files


def topartitioned(table, root, partition_by, mode='error',
                  max_open_files=32):
    """Write a table to a Hive-style partitioned dataset at `root`.

    Partition directories are nested under `root`, one level per field in
    `partition_by`, each named ``field=encoded_value``. Data files are
    UTF-8 CSV files using the standard library ``excel`` dialect and carry
    only the non-partition fields (partition fields are represented by the
    directory structure).

    Each call publishes a new, complete version of the dataset atomically.
    Readers using :func:`frompartitioned` only ever see fully committed
    versions. Concurrent writers serialise through a lock and a version
    check; a writer whose base version was changed by another writer fails
    with :class:`petl.errors.PartitionCommitConflict` without altering the
    dataset.

    Modes:

    * ``error`` (default): create a new dataset at `root`. If a dataset
      already exists there, an :class:`petl.errors.ArgumentError` is raised
      before any data rows are read.
    * ``overwrite``: replace the whole dataset with the contents of `table`.
    * ``overwrite_partitions``: replace only the partitions that occur in
      `table`; untouched partitions are preserved. An empty input leaves
      the dataset unchanged.
    * ``append``: add the rows of `table` to existing partitions, creating
      new partitions as needed.

    For the last three modes, a missing `root` means a new dataset is
    created. When writing into an existing dataset, its non-partition fields
    and `partition_by` must match exactly.

    At most `max_open_files` data files are open at any time; partitions may
    arrive in any order and may be much more numerous than this limit. The
    input table is traversed exactly once and memory use is proportional to
    the number of partitions, never to the number of rows.
    """

    if isinstance(max_open_files, bool) or not isinstance(max_open_files,
                                                          int):
        raise ArgumentError(
            'max_open_files must be a positive integer (bool is not '
            'accepted)')
    if max_open_files < 1:
        raise ArgumentError('max_open_files must be a positive integer')
    if mode not in ('error', 'overwrite', 'overwrite_partitions', 'append'):
        raise ArgumentError(
            'mode must be one of error, overwrite, overwrite_partitions, '
            'append; got %r' % mode)

    it = iter(table)
    try:
        header = tuple(next(it))
    except StopIteration:
        raise ArgumentError('input table has no header row')
    header = _validate_header(header)
    partition_by = _validate_partition_by(partition_by, header)
    p_indices = [header.index(f) for f in partition_by]
    data_header = [f for f in header if f not in partition_by]
    d_indices = [i for i in range(len(header)) if i not in p_indices]

    # Root/mode validation happens before any data rows are consumed.
    existing = _ensure_empty_or_dataset(root)
    if existing is not None:
        if mode == 'error':
            raise ArgumentError(
                'a partitioned dataset already exists at %r' % root)
        if list(existing['partition_by']) != list(partition_by) or \
                list(existing['header']) != list(data_header):
            raise ArgumentError(
                'existing partitioned dataset at %r has a different schema: '
                'header=%r partition_by=%r'
                % (root, existing['header'], existing['partition_by']))
    base_version = None if existing is None else existing['version']

    os.makedirs(_tmp_root(root), exist_ok=True)
    txn_id = uuid.uuid4().hex
    staging_dir = os.path.join(_tmp_root(root), txn_id)
    os.makedirs(staging_dir)

    pool = _PartitionWriterPool(staging_dir, data_header, max_open_files)
    # Partition bookkeeping, one entry per distinct encoded partition key.
    # Memory is proportional to the number of distinct partitions only.
    partition_keys = set()

    try:
        row_number = 0
        for row in it:
            row_number += 1
            if len(row) != len(header):
                raise ArgumentError(
                    'row %d has %d values but the header has %d fields'
                    % (row_number, len(row), len(header)))
            raw_values = tuple(row[i] for i in p_indices)
            try:
                key = _encode_partition_tuple(raw_values)
            except ArgumentError as e:
                # keep the inner message without the repeated prefix
                raise ArgumentError(
                    'invalid partition value on row %d: %s'
                    % (row_number, e.message))
            except TypeError as e:
                raise ArgumentError(
                    'invalid partition value on row %d: %s'
                    % (row_number, e))
            pool.write_row(key, [row[i] for i in d_indices])
            partition_keys.add(key)
        pool.close_all()
    except BaseException:
        # Source failure, bad row, disk error, KeyboardInterrupt, ...:
        # the write must never become visible. Close and cleanup errors
        # must never mask the original exception.
        pool.close_all_suppress()
        try:
            _remove_tree(staging_dir)
        except Exception:
            pass
        raise

    committed = False
    try:
        with _Lock(root):
            # Re-read the committed version while holding the commit lock.
            if _is_dataset(root):
                current = _read_current(root)
                _validate_manifest(current)
                current_version = current['version']
            elif not os.path.exists(root) or not os.listdir(root) \
                    or os.listdir(root) == [_META_DIR]:
                # New dataset: root is empty apart from our own metadata.
                current = None
                current_version = None
            else:
                # Someone created unrelated content at the root.
                raise PartitionCommitConflict(
                    'root %r is no longer a partitioned dataset' % root)
            if current_version != base_version:
                raise PartitionCommitConflict(
                    'the dataset at %r was modified by another writer '
                    'since this write started' % root)

            new_manifest = _build_manifest(
                mode, current, partition_by, data_header,
                partition_keys, pool.files(), root, staging_dir)
            _publish(root, current, new_manifest, staging_dir)
            committed = True
    except BaseException:
        # Anything failing before the manifest swap (including failures
        # while moving staged files into partition directories) must leave
        # the previous version intact. Files already moved are not
        # referenced by any manifest, so removing them cannot disturb
        # committed data. _publish() never raises after a successful swap.
        if not committed:
            _cleanup_failed_commit(root, partition_by, staging_dir,
                                   pool.files())
        raise


def _build_manifest(mode, current, partition_by, data_header,
                    partition_keys, staged_files, root, staging_dir):
    """Assemble the post-commit manifest and move staged data files into
    their final partition directories."""
    if current is None or mode == 'overwrite':
        kept_partitions = {}
    else:
        # append and overwrite_partitions: carry forward existing entries.
        kept_partitions = {}
        for entry in current['partitions']:
            key = tuple(entry['key'])
            kept_partitions[key] = {
                'key': list(key),
                'values': list(entry['values']),
                'files': [dict(f) for f in entry['files']],
            }

    for key in sorted(partition_keys):
        if mode == 'overwrite_partitions' and key in kept_partitions:
            # Replaced: drop the old files of this partition.
            kept_partitions.pop(key)
        values = [_decode_partition_value(component) for component in key]
        entry = kept_partitions.setdefault(key, {
            'key': list(key),
            'values': values,
            'files': [],
        })
        for staged_path, size in staged_files.get(key, []):
            target_dir = _partition_dir(root, partition_by, key)
            os.makedirs(target_dir, exist_ok=True)
            name = os.path.basename(staged_path)
            target_path = os.path.join(target_dir, name)
            os.rename(staged_path, target_path)
            rel_dir = os.path.relpath(target_dir, root)
            entry['files'].append(
                {'path': os.path.join(rel_dir, name), 'size': size})

    partitions = [kept_partitions[k] for k in sorted(kept_partitions)]
    return {
        'version': uuid.uuid4().hex,
        'partition_by': list(partition_by),
        'header': list(data_header),
        'partitions': partitions,
    }


def _cleanup_failed_commit(root, partition_by, staging_dir, staged_files):
    """Remove staged files that had already been moved into partition
    directories, then the staging directory itself. Best effort: this only
    runs after a failure, so never raise (the original exception must be
    preserved)."""
    try:
        for key, files in staged_files.items():
            target_dir = _partition_dir(root, partition_by, key)
            for staged_path, _size in files:
                target = os.path.join(target_dir,
                                      os.path.basename(staged_path))
                try:
                    os.unlink(target)
                except OSError:
                    pass
    finally:
        _remove_tree(staging_dir)
        _prune_empty_dirs(root)


def _publish(root, current, new_manifest, staging_dir):
    """Swap in the new manifest and remove files it no longer references.

    No exception is raised once the manifest swap has succeeded: failures
    while deleting obsolete files leave files on disk that no manifest
    references, which is safe (they are never read and never counted as
    dataset data)."""
    obsolete = []
    if current is not None:
        live = set()
        for entry in new_manifest['partitions']:
            for f in entry['files']:
                live.add(f['path'])
        for entry in current['partitions']:
            for f in entry['files']:
                if f['path'] not in live:
                    obsolete.append(os.path.join(root, f['path']))

    payload = json.dumps(new_manifest, sort_keys=True,
                         ensure_ascii=False).encode('utf-8')
    _atomic_write_bytes(_current_path(root), payload)
    _fsync_dir(_meta_dir(root))

    # Files removed after the manifest swap are either still held open by
    # readers (POSIX keeps their data alive) or reported as a conflict by
    # those readers; they can never be read as part of the new version.
    for path in obsolete:
        try:
            os.unlink(path)
        except OSError:
            pass

    _remove_tree(staging_dir)
    _prune_empty_dirs(root)


def _prune_empty_dirs(root):
    meta_abs = os.path.abspath(_meta_dir(root))
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        abs_dir = os.path.abspath(dirpath)
        try:
            inside_meta = os.path.commonpath([abs_dir, meta_abs]) == meta_abs
        except ValueError:
            # different drives on some platforms
            inside_meta = False
        if inside_meta or dirpath == root:
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


Table.topartitioned = topartitioned


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def frompartitioned(root):
    """Read a partitioned dataset previously written with
    :func:`topartitioned`.

    Returns a :class:`petl.util.base.Table` whose header is the
    non-partition fields followed by the partition fields. Partition values
    are restored as strings, with the default partition marker restored to
    ``None``; data fields are strings as produced by CSV reading.

    Rows are ordered by partition value tuple (ascending, using petl's
    comparison semantics with ``None`` sorting first) and, within a
    partition, by commit order then by the original input order.

    Each iteration is pinned to the version that was committed when it
    started. If another writer commits while an iteration is in progress,
    the iteration either completes against its pinned version or raises
    :class:`petl.errors.PartitionCommitConflict`; rows from different
    versions are never mixed and rows are never silently dropped.
    """
    return PartitionedView(root)


class PartitionedView(Table):

    def __init__(self, root):
        self.root = root

    def __iter__(self):
        return iterpartitioned(self.root)


def iterpartitioned(root):
    if not _is_dataset(root):
        raise ArgumentError(
            '%r is not a partitioned dataset' % root)
    manifest = _read_current(root)
    _validate_manifest(manifest)
    version = manifest['version']
    partition_by = list(manifest['partition_by'])
    header = tuple(list(manifest['header']) + partition_by)
    yield header

    def conflict(msg):
        return PartitionCommitConflict(
            'dataset %r changed while reading: %s' % (root, msg))

    def check_version():
        if not _is_dataset(root):
            raise conflict('dataset removed')
        latest = _read_current(root)
        _validate_manifest(latest)
        if latest.get('version') != version:
            raise conflict('a new version was committed')

    entries = sorted(
        manifest['partitions'],
        key=lambda e: Comparable(tuple(e['values'])))
    for entry in entries:
        decoded = tuple(entry['values'])
        for f in entry['files']:
            rel_path = f['path']
            if os.path.isabs(rel_path) or rel_path.startswith('..'):
                raise ArgumentError(
                    'invalid data file path in manifest: %r' % rel_path)
            path = os.path.join(root, rel_path)
            expected_size = f.get('size')
            check_version()
            try:
                st = os.stat(path)
            except OSError:
                raise conflict('data file %r disappeared' % rel_path)
            if expected_size is not None and st.st_size != expected_size:
                raise conflict(
                    'data file %r changed size' % rel_path)
            try:
                with builtins_open(path, 'rb') as buf:
                    text = io.TextIOWrapper(buf, encoding='utf-8',
                                            newline='')
                    try:
                        reader = csv.reader(text, dialect=_CSV_DIALECT)
                        first = True
                        for row in reader:
                            if first:
                                # discard the per-file header row
                                first = False
                                continue
                            yield tuple(row) + decoded
                    finally:
                        text.detach()
            except FileNotFoundError:
                raise conflict('data file %r disappeared' % rel_path)
            check_version()
