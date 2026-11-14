# -*- coding: utf-8 -*-
"""Read and write Hive-style partitioned datasets.

A partitioned dataset is a directory tree whose layout mirrors the values
of one or more partition fields. Each partition directory contains one or
more UTF-8 CSV data files holding the non-partition fields. All changes to
a dataset are published through an atomic, versioned commit so that readers
either see the complete result of a write or the complete previous version,
never a mixture of the two.

The logical non-partition schema can evolve independently of the data
files: :func:`evolvepartitioned` adds, renames, drops and reorders columns
by metadata only. Columns are tracked by stable ids, so renames follow the
old values, dropped values stay gone (a re-added column of the same name is
a genuinely new column) and historical rows are filled with per-column
defaults recorded when the column was created.
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
    if 'schema' in manifest:
        _validate_schema(manifest['schema'])


def _validate_schema(schema):
    """Validate the schema-evolution section of a manifest."""
    if not isinstance(schema, dict):
        raise ArgumentError('partitioned dataset schema is invalid')
    columns = schema.get('columns')
    next_id = schema.get('next_id')
    if not isinstance(columns, list) or isinstance(next_id, bool) \
            or not isinstance(next_id, int) or next_id < len(columns):
        raise ArgumentError('partitioned dataset schema is invalid')
    ids = set()
    names = set()
    for col in columns:
        if not isinstance(col, dict):
            raise ArgumentError('partitioned dataset schema is invalid')
        cid = col.get('id')
        name = col.get('name')
        default = col.get('default')
        if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0:
            raise ArgumentError('partitioned dataset schema is invalid')
        if not isinstance(name, str) or name == '':
            raise ArgumentError('partitioned dataset schema is invalid')
        if not isinstance(default, str):
            raise ArgumentError('partitioned dataset schema is invalid')
        if cid in ids or name in names:
            raise ArgumentError('partitioned dataset schema is invalid')
        ids.add(cid)
        names.add(name)
    if not columns:
        raise ArgumentError(
            'partitioned dataset schema must keep at least one '
            'non-partition field')


def _normalize_manifest(manifest):
    """Return a manifest carrying explicit schema-evolution metadata.

    Datasets written before schema evolution existed have no ``schema``
    section and no per-file column lists. Their non-partition fields are
    assigned stable ids by header position and every existing file is
    treated as written under that schema with the empty default. This only
    annotates the in-memory manifest; nothing on disk is changed until a
    new commit is published.
    """
    if 'schema' in manifest:
        _validate_schema(manifest['schema'])
        schema = manifest['schema']
        columns = [dict(col) for col in schema['columns']]
        next_id = schema['next_id']
    else:
        columns = [{'id': i, 'name': name, 'default': ''}
                   for i, name in enumerate(manifest['header'])]
        next_id = len(columns)
    partitions = []
    for entry in manifest['partitions']:
        new_entry = {
            'key': list(entry['key']),
            'values': list(entry['values']),
            'files': [],
        }
        for f in entry['files']:
            new_f = dict(f)
            if 'columns' not in new_f:
                # A file committed before schema evolution carried no
                # column list: its columns were the schema of its time,
                # which for a manifest without a schema section is the
                # committed header in order.
                if len(manifest['header']) != len(columns):
                    raise ArgumentError(
                        'partitioned dataset manifest is invalid: data '
                        'file %r has no column list' % new_f.get('path'))
                new_f['columns'] = [col['id'] for col in columns]
            else:
                cols = new_f['columns']
                # File column lists may reference ids that have since been
                # dropped from the logical schema; the reader simply does
                # not project them. They must still be unique, non-negative
                # ints so that the positional mapping stays well defined.
                if not isinstance(cols, list) or not cols \
                        or any(isinstance(c, bool) or not isinstance(c, int)
                               or c < 0 for c in cols) \
                        or len(set(cols)) != len(cols):
                    raise ArgumentError(
                        'partitioned dataset manifest is invalid: bad column '
                        'list for data file %r' % new_f.get('path'))
            new_entry['files'].append(new_f)
        partitions.append(new_entry)
    normalized = dict(manifest)
    normalized['header'] = [col['name'] for col in columns]
    normalized['partitions'] = partitions
    normalized['schema'] = {'columns': columns, 'next_id': next_id}
    return normalized


def _is_nonempty_name(name):
    return isinstance(name, str) and name != ''


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
# Schema evolution
# ---------------------------------------------------------------------------

def _validate_changes_shape(changes):
    if not isinstance(changes, (list, tuple)) or not changes:
        raise ArgumentError(
            'changes must be a non-empty list or tuple of operations')
    for op in changes:
        if not isinstance(op, (list, tuple)) or not op:
            raise ArgumentError(
                'each change must be a non-empty list or tuple whose first '
                'element is the operation name')
        if not _is_nonempty_name(op[0]):
            raise ArgumentError(
                'unknown schema evolution operation: %r'
                % (op[0] if op else None,))


def _apply_changes(columns, next_id, partition_by, changes):
    """Apply the ordered change operations to an in-memory column list.

    ``columns`` is a list of ``{'id', 'name', 'default'}`` dicts in logical
    order. Either returns the new (columns, next_id) pair or raises
    ArgumentError; the inputs are never mutated on failure.
    """
    columns = [dict(col) for col in columns]

    def names():
        return [col['name'] for col in columns]

    for op in changes:
        action = op[0]
        current_names = names()
        name_set = set(current_names)
        partition_set = set(partition_by)
        if action == 'add':
            if len(op) != 3:
                raise ArgumentError(
                    "change %r is invalid: 'add' requires "
                    "('add', name, default)" % (op,))
            name = op[1]
            default = op[2]
            if not _is_nonempty_name(name):
                raise ArgumentError(
                    'the added field name must be a non-empty string: %r'
                    % (name,))
            if not isinstance(default, str):
                raise ArgumentError(
                    "the default for added field %r must be a string "
                    "(the empty string is allowed)" % name)
            if name in name_set:
                raise ArgumentError(
                    'a non-partition field named %r already exists' % name)
            if name in partition_set:
                raise ArgumentError(
                    'field name %r is already used by a partition field'
                    % name)
            columns.append({'id': next_id, 'name': name,
                            'default': default})
            next_id += 1
        elif action == 'rename':
            if len(op) != 3:
                raise ArgumentError(
                    "change %r is invalid: 'rename' requires "
                    "('rename', old, new)" % (op,))
            old = op[1]
            new = op[2]
            if not _is_nonempty_name(old) or not _is_nonempty_name(new):
                raise ArgumentError(
                    'renamed field names must be non-empty strings: %r'
                    % (op,))
            if old not in name_set:
                raise ArgumentError(
                    'cannot rename %r: no such non-partition field' % old)
            if new in name_set:
                raise ArgumentError(
                    'a non-partition field named %r already exists' % new)
            if new in partition_set:
                raise ArgumentError(
                    'field name %r is already used by a partition field'
                    % new)
            for col in columns:
                if col['name'] == old:
                    col['name'] = new
                    break
        elif action == 'drop':
            if len(op) != 2:
                raise ArgumentError(
                    "change %r is invalid: 'drop' requires ('drop', name)"
                    % (op,))
            name = op[1]
            if not _is_nonempty_name(name):
                raise ArgumentError(
                    'the dropped field name must be a non-empty string: %r'
                    % (name,))
            if name not in name_set:
                raise ArgumentError(
                    'cannot drop %r: no such non-partition field' % name)
            columns = [col for col in columns if col['name'] != name]
            if not columns:
                raise ArgumentError(
                    'dropping %r would leave no non-partition fields' % name)
        elif action == 'reorder':
            if len(op) != 2 or not isinstance(op[1], (list, tuple)):
                raise ArgumentError(
                    "change %r is invalid: 'reorder' requires "
                    "('reorder', names)" % (op,))
            ordered = list(op[1])
            if not ordered or any(not _is_nonempty_name(n)
                                  for n in ordered):
                raise ArgumentError(
                    'the reorder list must contain only non-empty field '
                    'name strings')
            if len(ordered) != len(current_names) \
                    or set(ordered) != name_set \
                    or len(set(ordered)) != len(ordered):
                raise ArgumentError(
                    'the reorder list must contain every current '
                    'non-partition field exactly once: %r' % (ordered,))
            by_name = {col['name']: col for col in columns}
            columns = [by_name[name] for name in ordered]
        else:
            raise ArgumentError(
                'unknown schema evolution operation: %r' % (action,))
    return columns, next_id


def evolvepartitioned(root, changes, _base_version=None):
    """Evolve the logical schema of a partitioned dataset without touching
    its CSV data files.

    `changes` is a non-empty list or tuple of operations; each operation is
    itself a list or tuple and the operations are applied in the order
    given:

    * ``('add', name, default)``: append a non-partition field named `name`
      at the end of the non-partition fields. `default` must be a string
      (the empty string is allowed) and is the value filled in for rows of
      data files committed before this operation. Rows written afterwards
      must supply the column explicitly; the default is never used to
      complete a short input row.
    * ``('rename', old, new)``: rename a non-partition field, keeping its
      position. Values in existing files follow the new name because
      columns are tracked by stable ids rather than names.
    * ``('drop', name)``: remove a non-partition field. Its values no
      longer appear when the dataset is read. Adding a field with the same
      name later creates a brand new column with a new id; values of the
      dropped column are never recovered from the old files.
    * ``('reorder', names)``: reorder the non-partition fields. `names`
      must be a list or tuple containing every current non-partition field
      exactly once.

    Partition fields never participate in an evolution and at least one
    non-partition field must remain after every operation. Field names are
    non-empty strings, may not be duplicated among the non-partition
    fields and may not collide with a partition field name. The result of
    one operation is available to the next; if any operation is invalid
    the whole batch is rejected and the dataset is left completely
    unchanged.

    Only metadata is written: no CSV data file is opened, scanned,
    rewritten or renamed, so the cost does not depend on the number of
    data rows. Existing files keep their paths and bytes through any number
    of evolutions; readers map their columns by id and fill historical
    defaults on the fly. The commit follows the same lock and version
    rules as :func:`topartitioned`: an evolution that started from a
    version since superseded by another evolution or write fails with
    :class:`petl.errors.PartitionCommitConflict` without changing the
    winner's schema or data. A dataset written by a petl version without
    schema evolution support is accepted directly and gains schema
    metadata on its first evolution; no separate migration is needed.

    Returns ``None`` on success.
    """
    _validate_changes_shape(changes)
    if not _is_dataset(root):
        # Distinguish a missing/unreadable dataset from a non-dataset
        # directory; never create an empty dataset here.
        _ensure_empty_or_dataset(root)
        raise ArgumentError(
            '%r is not a partitioned dataset' % root)
    base = _read_current(root)
    _validate_manifest(base)
    # The private pin lets a caller that read the manifest earlier commit
    # against that exact base version, mirroring topartitioned's optimistic
    # concurrency control. Public callers always use the current version.
    if _base_version is not None and not isinstance(_base_version, str):
        raise ArgumentError('_base_version must be a string or None')
    base_version = base['version'] if _base_version is None \
        else _base_version

    with _Lock(root):
        current = _read_current(root)
        _validate_manifest(current)
        if current['version'] != base_version:
            raise PartitionCommitConflict(
                'the dataset at %r was modified by another writer or '
                'evolution since this evolution started' % root)
        current = _normalize_manifest(current)
        # All change validation happens under the lock against the
        # committed state, so an invalid batch raises ArgumentError and a
        # concurrently superseded base raises PartitionCommitConflict;
        # neither mutates the dataset.
        locked_columns, locked_next_id = _apply_changes(
            current['schema']['columns'], current['schema']['next_id'],
            current['partition_by'], changes)
        # Carried-forward files keep their paths and bytes; only their
        # metadata is (re)annotated. No CSV file is opened or moved.
        new_partitions = []
        for entry in current['partitions']:
            new_partitions.append({
                'key': list(entry['key']),
                'values': list(entry['values']),
                'files': [dict(f) for f in entry['files']],
            })
        new_manifest = {
            'version': uuid.uuid4().hex,
            'partition_by': list(current['partition_by']),
            'header': [col['name'] for col in locked_columns],
            'partitions': new_partitions,
            'schema': {'columns': locked_columns,
                       'next_id': locked_next_id},
        }
        payload = json.dumps(new_manifest, sort_keys=True,
                             ensure_ascii=False).encode('utf-8')
        _atomic_write_bytes(_current_path(root), payload)
        _fsync_dir(_meta_dir(root))
    return None


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
    created. When writing into an existing dataset, its non-partition
    fields (in the current logical order, after any schema evolution) and
    `partition_by` must match exactly; a mismatch is rejected before any
    data rows are read. New files are written with the current logical
    header and are tagged with the current column ids, so files produced
    under older schemas can coexist in the same partitions and are mapped
    to the logical schema when read. Input rows must supply every current
    non-partition field; historical column defaults from
    :func:`evolvepartitioned` are never used to fill in missing values of
    newly written rows, and an explicit empty string is written as-is.

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
        existing_normalized = _normalize_manifest(existing)
        if list(existing_normalized['partition_by']) != list(partition_by) \
                or list(existing_normalized['header']) != list(data_header):
            raise ArgumentError(
                'existing partitioned dataset at %r has a different schema: '
                'header=%r partition_by=%r'
                % (root, existing_normalized['header'],
                   existing_normalized['partition_by']))
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
                current = _normalize_manifest(current)
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
    if current is None:
        # Brand new dataset: assign stable ids by header position.
        write_columns = list(range(len(data_header)))
    else:
        # The version check guarantees the logical header matches
        # data_header; resolve the current ids of the written columns.
        id_by_name = {col['name']: col['id']
                      for col in current['schema']['columns']}
        write_columns = [id_by_name[name] for name in data_header]
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
                {'path': os.path.join(rel_dir, name), 'size': size,
                 'columns': list(write_columns)})
    partitions = [kept_partitions[k] for k in sorted(kept_partitions)]
    if current is None:
        schema = {
            'columns': [{'id': i, 'name': name, 'default': ''}
                        for i, name in enumerate(data_header)],
            'next_id': len(data_header),
        }
    elif mode == 'overwrite':
        # A whole-table overwrite replaces every file but keeps the
        # logical schema, so stable column ids survive for future
        # evolutions of this dataset.
        schema = current['schema']
    else:
        # append / overwrite_partitions: the version check above
        # guarantees current matches the manifest validated earlier.
        schema = current['schema']
    return {
        'version': uuid.uuid4().hex,
        'partition_by': list(partition_by),
        'header': list(data_header),
        'partitions': partitions,
        'schema': {'columns': [dict(col) for col in schema['columns']],
                   'next_id': schema['next_id']},
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
    non-partition fields (in the current logical order, after any schema
    evolution) followed by the partition fields. Partition values are
    restored as strings, with the default partition marker restored to
    ``None``; data fields are strings as produced by CSV reading.

    Data files written under older schemas are mapped onto the current
    logical columns by stable column ids: renamed columns keep their old
    values, dropped columns disappear and columns added later are filled
    for historical rows with the default recorded when the column was
    created. Files from different schemas may coexist in one partition and
    are read in commit order without rewriting any bytes.

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
    manifest = _normalize_manifest(manifest)
    version = manifest['version']
    partition_by = list(manifest['partition_by'])
    columns = manifest['schema']['columns']
    logical_ids = [col['id'] for col in columns]
    default_by_id = {col['id']: col['default'] for col in columns}
    header = tuple(col['name'] for col in columns) + tuple(partition_by)
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
            file_columns = f['columns']
            # Per-file projection onto the current logical column order.
            # Columns absent from this file are historical columns added
            # (or re-added) after the file was committed; they are filled
            # with the default recorded when the column was created. This
            # workspace is bounded by the number of fields, not rows, and
            # retains nothing between files.
            positions = []
            for cid in logical_ids:
                positions.append(
                    file_columns.index(cid) if cid in file_columns else -1)
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
                            projected = tuple(
                                row[pos] if pos >= 0
                                else default_by_id[cid]
                                for pos, cid in zip(positions, logical_ids))
                            yield projected + decoded
                    finally:
                        text.detach()
            except FileNotFoundError:
                raise conflict('data file %r disappeared' % rel_path)
            check_version()
