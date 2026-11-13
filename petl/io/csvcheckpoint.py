# -*- coding: utf-8 -*-
"""Restartable, checkpointed reading of a local UTF-8 CSV file.

This module implements :func:`petl.io.csvcheckpoint.fromcsvcheckpoint`.  The
public surface is deliberately small: a table view, per-iteration cursors and
a JSON-serialisable checkpoint snapshot.  See the function docstring for the
cost model and the exact guarantees.
"""

from __future__ import absolute_import, print_function, division


import copy
import csv
import hashlib
import json
import os
import re
import stat
from collections import deque

from petl.errors import ArgumentError
from petl.util.base import Table


#: Indirection over the builtin so tests can probe read sizes and closing.
_open_binary = open

#: Indirection over ``csv.reader`` so tests can count the records the CSV
#: parser actually handles (restore validation must parse only the header).
_csv_reader = csv.reader


_FORMAT = 'petl.csv-checkpoint'
_FORMAT_VERSION = 1

_CONFIG_KEYS = (
    'delimiter', 'quotechar', 'escapechar', 'doublequote',
    'skipinitialspace', 'quoting', 'strict',
)

#: csv.reader's "excel" dialect, spelled out so the values are also stored in
#: the checkpoint and can be compared exactly on restore.
_CONFIG_DEFAULTS = {
    'delimiter': ',',
    'quotechar': '"',
    'escapechar': None,
    'doublequote': True,
    'skipinitialspace': False,
    'quoting': csv.QUOTE_MINIMAL,
    'strict': False,
}

_BOOLEAN_KEYS = ('doublequote', 'skipinitialspace', 'strict')

_HEX_SHA256 = re.compile(r'^[0-9a-f]{64}$')

_UTF8_BOM = b'\xef\xbb\xbf'


def fromcsvcheckpoint(path, checkpoint=None, chunksize=65536, **csvargs):
    """Extract a table from a local UTF-8 CSV file, with resumable cursors.

    ``path`` must be a string naming a local, regular, uncompressed file
    containing strict UTF-8 text, optionally with a UTF-8 BOM at the very
    start.  ``fromcsvcheckpoint`` does not accept URLs, compressed files,
    file-like sources or other petl ``Source`` objects.  The first CSV record
    is always used as the header; there is no explicit ``header`` argument.
    Data values keep the exact interpretation given by :mod:`csv`, including
    ``QUOTE_NONNUMERIC`` conversions.

    Each iteration of the returned table starts an independent cursor::

        view = etl.fromcsvcheckpoint('data.csv')
        it = iter(view)
        header = next(it)              # ('foo', 'bar')
        row = next(it)                 # ('a', '1')
        saved = it.checkpoint()        # JSON-serialisable dict
        it.close()

        view2 = etl.fromcsvcheckpoint('data.csv', checkpoint=saved)
        it2 = iter(view2)
        next(it2)                      # same header, from the checkpoint
        next(it2)                      # first row after the saved position

    Creating the view and a cursor does not touch the file.  The first
    ``next()`` opens it.  A cursor supports ``next``, ``close`` and
    ``checkpoint()``.  Every iteration yields the header first, even when
    resuming.  A checkpoint may be taken after the header and after every
    fully delivered data row.  Calling ``checkpoint()`` before the header has
    been obtained raises :class:`~petl.errors.ArgumentError`; it never
    consumes the following record.

    Checkpoint objects are plain JSON-serialisable dictionaries (paths are
    deliberately not part of them, so an identical file copied to another
    location can be resumed).  They bind the complete byte content of the
    file (SHA-256), the parse configuration, the header, and the committed
    data position.  A checkpoint can be saved with ``json.dumps`` and passed
    back either as the decoded dictionary or as the JSON text.  They are
    integrity checks against accidental corruption, not authentication: a
    caller able to rebuild every validation field can forge one.

    **Cost model.**  The first ``next()`` of a fresh cursor performs one
    sequential, bounded-memory pass over the whole file to compute its SHA-256
    digest (O(file size)); it then decodes and parses the file incrementally
    as rows are requested, so a complete fresh read costs two sequential
    passes in total.  A resumed cursor performs one such full-file digest
    pass to validate content (O(file size)), re-parses only the header
    record to verify it against the checkpoint, and then decodes from the
    saved record boundary; it never feeds the prefix's CSV records to the
    parser.  ``checkpoint()`` itself does no file I/O and is O(header size +
    checkpoint size), because the digest was already computed at ``next()``.
    Restore is therefore *not* constant time: validating that the input still
    has the same complete byte content is O(file size).  Memory use is
    bounded by ``chunksize`` plus the largest single CSV record (including
    quoted multi-line fields), never by the file size or by how much history
    has been processed.

    ``chunksize`` is a positive integer (``bool`` is rejected) and is the
    maximum number of bytes requested from the binary file by any single
    read.  A record may consume several chunks.  Using ``chunksize=1`` gives
    the same rows and the same checkpoint positions, only more read calls.

    Supported ``csv.reader`` format options are ``delimiter``,
    ``quotechar``, ``escapechar``, ``doublequote``, ``skipinitialspace``,
    ``quoting`` and ``strict``; their defaults are the standard library's
    ``excel`` dialect.  ``dialect`` and every other option (for example
    ``lineterminator``) are rejected with
    :class:`~petl.errors.ArgumentError`.

    Errors: malformed CSV remains a :class:`csv.Error`; invalid UTF-8 remains
    a :class:`UnicodeDecodeError`; invalid configuration, invalid checkpoints,
    content changes and data rows whose width differs from the header are
    :class:`~petl.errors.ArgumentError`.  Width and CSV errors mention the
    data-record number (the first record after the header is data record 1)
    and the 1-based physical line where that record starts.  On resume the
    numbering continues from the checkpoint rather than restarting at 1.

    A cursor is terminated by the first parsing, decoding or read error, but
    the last checkpoint returned before that failure remains usable.  A
    failed position is never published.  ``close()`` is idempotent; normal
    exhaustion, an error and an explicit close all release the file.  After
    close (or exhaustion) ``checkpoint()`` still returns the last valid
    checkpoint, while further ``next()`` calls do not reopen the file.

    The library never advances or rewrites a checkpoint stored by the caller;
    it only returns snapshots from ``checkpoint()``.  Rows delivered after the
    caller's last saved checkpoint may therefore be re-read after a restart,
    while rows covered by a saved checkpoint are neither repeated nor skipped.
    This is progress tracking for a trusted local file, not exactly-once
    delivery for an arbitrary external database.
    """

    _validate_path(path)
    _validate_chunksize(chunksize)
    config = _normalize_config(csvargs, where='fromcsvcheckpoint')
    saved = None
    if checkpoint is not None:
        saved = _validated_checkpoint(checkpoint, config)
    return CSVCheckpointView(path, config, chunksize, saved)


# ---------------------------------------------------------------------------
# Argument validation


def _validate_path(path):
    if not isinstance(path, str):
        raise ArgumentError(
            'path must be a string naming a local regular uncompressed file, '
            'not %r' % (type(path).__name__,))
    lowered = path.lower()
    if '://' in path or lowered.startswith(('http:', 'https:', 'ftp:')):
        raise ArgumentError('path must be a local file, not a URL: %r' % path)
    if lowered.endswith(('.gz', '.bgz', '.bz2', '.zip')):
        raise ArgumentError('path must name an uncompressed file: %r' % path)


def _check_local_regular_file(path):
    """Check the path at first iteration, not when the view is created."""
    try:
        info = os.stat(path)
    except OSError as err:
        raise ArgumentError('path cannot be opened: %s' % err)
    if not stat.S_ISREG(info.st_mode):
        raise ArgumentError('path must name a regular file: %r' % path)


def _validate_chunksize(chunksize):
    if isinstance(chunksize, bool) or not isinstance(chunksize, int):
        raise ArgumentError('chunksize must be a positive integer, not %r'
                            % (chunksize,))
    if chunksize <= 0:
        raise ArgumentError('chunksize must be a positive integer, got %r'
                            % (chunksize,))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_config(csvargs, where):
    unknown = sorted(set(csvargs) - set(_CONFIG_KEYS))
    if unknown:
        if 'dialect' in unknown:
            raise ArgumentError(
                '%s does not support the dialect argument; pass the format '
                'options explicitly' % where)
        raise ArgumentError('%s does not support these csv.reader options: %s'
                            % (where, ', '.join(unknown)))
    config = dict(_CONFIG_DEFAULTS)
    config.update(csvargs)
    _validate_config_values(config, where=where)
    return config


def _validate_config_values(config, where='checkpoint',
                            accept_int_bools=True):
    delimiter = config['delimiter']
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ArgumentError('%s: delimiter must be a 1-character string, not '
                            '%r' % (where, delimiter))
    for key in ('quotechar', 'escapechar'):
        value = config[key]
        if value is None:
            continue
        if not isinstance(value, str) or len(value) != 1:
            raise ArgumentError('%s: %s must be None or a 1-character string, '
                                'not %r' % (where, key, value))
    for key in _BOOLEAN_KEYS:
        value = config[key]
        if isinstance(value, bool):
            continue
        # csvargs may pass integers as truth values (the standard library
        # accepts them); normalise so the checkpoint stores a JSON boolean.
        # Checkpoints themselves must carry real booleans.
        if accept_int_bools and _is_int(value):
            config[key] = bool(value)
            continue
        raise ArgumentError('%s: %s must be a bool, not %r'
                            % (where, key, value))
    quoting = config['quoting']
    if not _is_int(quoting):
        raise ArgumentError('%s: quoting must be an integer, not %r'
                            % (where, quoting))
    # Let the standard library validate the option combination and any
    # version-specific quoting constant (e.g. QUOTE_STRINGS on 3.12+),
    # reporting the failure as an argument error rather than a bare TypeError
    # at first iteration.
    try:
        csv.reader((), **config)
    except (TypeError, ValueError) as err:
        raise ArgumentError('%s: invalid csv.reader format options: %s'
                            % (where, err))


class CSVCheckpointView(Table):
    """The table view returned by :func:`fromcsvcheckpoint`."""

    def __init__(self, path, config, chunksize, checkpoint):
        self.path = path
        self.csvargs = config
        self.chunksize = chunksize
        self._checkpoint = checkpoint

    def __iter__(self):
        return CSVCheckpointCursor(self.path, self.csvargs, self.chunksize,
                                   self._checkpoint)


class CSVCheckpointCursor(object):
    """One independent iteration cursor over a :class:`CSVCheckpointView`."""

    def __init__(self, path, config, chunksize, checkpoint):
        self.path = path
        # Take a private snapshot so later mutation of the view's public
        # attributes cannot change a cursor mid-read or corrupt checkpoints.
        self.config = dict(config)
        self.chunksize = chunksize
        self._initial_checkpoint = checkpoint

        self._handle = None
        self._line_reader = None
        self._reader = None
        self._closed = False
        self._terminated = False
        self._started = False
        self._has_header = False
        self._header = None
        self._size = None
        self._sha256 = None

        # Committed (delivered) position.  Only these values are ever
        # published by checkpoint(); a failed record cannot advance them.
        self._offset = 0
        self._data_records = 0
        self._physical_lines = 0

    # -- iterator protocol -------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._terminated:
            raise StopIteration
        if not self._started:
            return self._start()
        return self._next_data()

    next = __next__

    def close(self):
        """Release the file.  Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def checkpoint(self):
        """Return a detached, JSON-serialisable snapshot of the progress.

        The snapshot covers the header and the last fully delivered data
        record.  It does not read or consume anything from the file.
        """
        if not self._has_header:
            raise ArgumentError('checkpoint is not available before the CSV '
                                'header has been obtained')
        payload = {
            'format': _FORMAT,
            'version': _FORMAT_VERSION,
            'file': {'size': self._size, 'sha256': self._sha256},
            'config': dict(self.config),
            'header': list(self._header),
            'position': {
                'byte_offset': self._offset,
                'data_records': self._data_records,
                'physical_lines': self._physical_lines,
            },
        }
        payload['checksum'] = _checkpoint_checksum(payload)
        return copy.deepcopy(payload)

    # -- start-up ----------------------------------------------------------

    def _start(self):
        try:
            _check_local_regular_file(self.path)
            if self._initial_checkpoint is not None:
                header = self._start_restore()
            else:
                header = self._start_fresh()
            self._started = True
            self._has_header = True
            self._header = tuple(header)
            return self._header
        except Exception:
            self._terminate()
            raise

    def _start_fresh(self):
        size, sha256, _, _ = _scan_content(self.path, self.chunksize)
        handle, line_reader = _open_line_reader(
            self.path, self.chunksize, start=0, initial_lines=0,
            initial_offset=0)
        self._handle = handle
        self._line_reader = line_reader
        self._reader = _csv_reader(line_reader, **self.config)
        header = _read_header_with_reader(self._reader, self._line_reader,
                                          'header')
        self._size = size
        self._sha256 = sha256
        self._offset = line_reader.byte_pos
        self._physical_lines = line_reader.lines_emitted
        self._data_records = 0
        return header

    def _start_restore(self):
        saved = self._initial_checkpoint
        pos = saved['position']
        offset = pos['byte_offset']

        size, sha256, breaks, last_byte = _scan_content(
            self.path, self.chunksize, up_to=offset)
        if size != saved['file']['size'] or sha256 != saved['file']['sha256']:
            raise ArgumentError(
                'CSV file content does not match the checkpoint: the file was '
                'modified, truncated or appended to; refusing to resume')

        # Verify the stored header against the file.  Only the header record
        # is parsed here; the prefix's data records are never parsed.
        header, header_end, header_lines = _read_header_only(
            self.path, self.config, self.chunksize)
        if tuple(saved['header']) != header:
            raise ArgumentError(
                'checkpoint header %r does not match the CSV file header %r'
                % (saved['header'], list(header)))

        data_records = pos['data_records']
        physical_lines = pos['physical_lines']
        if offset < header_end or offset > size:
            raise ArgumentError(
                'checkpoint byte_offset %r is outside the resumable range '
                '[%r, %r]' % (offset, header_end, size))
        if data_records == 0:
            if offset != header_end or physical_lines != header_lines:
                raise ArgumentError(
                    'checkpoint claims no data records but its position is '
                    'not the end of the header')
        else:
            if offset <= header_end or physical_lines <= header_lines:
                raise ArgumentError(
                    'checkpoint position does not lie after the header')
            if physical_lines < header_lines + data_records:
                raise ArgumentError(
                    'checkpoint physical line count is inconsistent with its '
                    'data record count')

        expected = _expected_physical_lines(breaks, size, last_byte, offset)
        if expected != physical_lines:
            raise ArgumentError(
                'checkpoint physical line count %r is inconsistent with the '
                'file content at byte_offset %r (expected %r)'
                % (physical_lines, offset, expected))
        if offset < size and not _is_line_boundary(self.path, offset,
                                                   self.chunksize):
            raise ArgumentError(
                'checkpoint byte_offset %r is not at a record boundary'
                % (offset,))

        handle, line_reader = _open_line_reader(
            self.path, self.chunksize, start=offset,
            initial_lines=physical_lines, initial_offset=offset)
        self._handle = handle
        self._line_reader = line_reader
        self._reader = _csv_reader(line_reader, **self.config)
        self._size = size
        self._sha256 = sha256
        self._offset = offset
        self._data_records = data_records
        self._physical_lines = physical_lines
        return header

    # -- data --------------------------------------------------------------

    def _next_data(self):
        start_line = self._physical_lines + 1
        record_no = self._data_records + 1
        try:
            row = next(self._reader)
        except StopIteration:
            self._terminate()
            raise StopIteration
        except csv.Error as err:
            message = ('CSV parse error at data record %d (physical line %d): '
                       '%s' % (record_no, start_line, err))
            self._terminate()
            raise csv.Error(message)
        except UnicodeDecodeError as err:
            message = ('UTF-8 decode error at data record %d (physical line '
                       '%d): %s' % (record_no, start_line, err))
            self._terminate()
            raise UnicodeDecodeError(err.encoding, err.object, err.start,
                                     err.end, message)
        except ValueError as err:
            message = ('data record %d (physical line %d) could not be '
                       'interpreted under the configured csv quoting: %s'
                       % (record_no, start_line, err))
            self._terminate()
            raise ValueError(message)
        except Exception:
            self._terminate()
            raise

        if len(row) != len(self._header):
            message = ('data record %d (physical line %d) has %d fields, but '
                       'the header has %d'
                       % (record_no, start_line, len(row),
                          len(self._header)))
            self._terminate()
            raise ArgumentError(message)

        # Commit only after a complete, validated row has been produced.
        self._offset = self._line_reader.byte_pos
        self._physical_lines = self._line_reader.lines_emitted
        self._data_records = record_no
        return tuple(row)

    def _terminate(self):
        self._terminated = True
        self._release()

    def _release(self):
        handle, self._handle = self._handle, None
        self._line_reader = None
        self._reader = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Header parsing helpers


def _read_header_with_reader(reader, line_reader, context):
    try:
        row = next(reader)
    except StopIteration:
        raise ArgumentError('empty CSV file: a header record is required')
    except csv.Error as err:
        raise csv.Error('CSV parse error in the %s (physical line 1): %s'
                        % (context, err))
    except UnicodeDecodeError:
        raise
    except ValueError as err:
        raise ArgumentError('invalid %s: %s' % (context, err))
    return _validate_header(row, line_reader)


def _read_header_only(path, config, chunksize):
    handle, line_reader = _open_line_reader(
        path, chunksize, start=0, initial_lines=0, initial_offset=0)
    try:
        reader = _csv_reader(line_reader, **config)
        header = _read_header_with_reader(reader, line_reader, 'header')
        return header, line_reader.byte_pos, line_reader.lines_emitted
    finally:
        handle.close()


def _validate_header(row, line_reader):
    fields = list(row)
    if not fields:
        raise ArgumentError('CSV header (physical line 1) is empty; the '
                            'header must contain at least one field name')
    for value in fields:
        if not isinstance(value, str):
            raise ArgumentError(
                'CSV header fields must be strings; got %r (under '
                'QUOTE_NONNUMERIC an unquoted header is converted by csv)'
                % (value,))
        if value == '':
            raise ArgumentError('CSV header contains an empty field name')
    if len(set(fields)) != len(fields):
        raise ArgumentError('CSV header contains duplicate field names: %r'
                            % (fields,))
    return tuple(fields)


# ---------------------------------------------------------------------------
# Binary line reader


class _LineReader(object):
    """Yield physical CSV lines as decoded text, tracking exact byte positions.

    Lines are split on the raw bytes, like Python's universal-newline mode
    (``newline=''``): at CRLF, lone CR or lone LF, keeping the terminator in
    the yielded line.  The final line may have no terminator.  Because CR and
    LF bytes can never occur inside a UTF-8 multi-byte sequence, splitting on
    bytes is exact and each physical line decodes independently (strict
    UTF-8), so a decoding error surfaces only when the affected line is
    requested, never for a line that has not yet been consumed.

    ``byte_pos`` always points at the end of the most recently yielded
    physical line, no matter how far the binary handle has read ahead, and
    ``lines_emitted`` counts yielded lines.  Memory is bounded by
    ``chunksize`` plus the longest physical line.
    """

    def __init__(self, handle, chunksize, initial_bytes=b'', start_offset=0,
                 initial_lines=0):
        self.handle = handle
        self.chunksize = chunksize
        self._parts = []           # bytes of the line currently being built
        self._ready = deque()      # complete lines (bytes) not yet yielded
        self._pending_cr = False   # last byte seen was CR, awaits LF lookahead
        self.eof = False
        self.byte_pos = start_offset
        self.lines_emitted = initial_lines
        if initial_bytes:
            self._split(initial_bytes)

    def __iter__(self):
        return self

    def __next__(self):
        while not self._ready:
            if self.eof:
                self._finish_eof()
                if not self._ready:
                    raise StopIteration
                break
            self._read_more()
        raw = self._ready.popleft()
        text = raw.decode('utf-8')   # strict; error is raised for this line
        self.byte_pos += len(raw)
        self.lines_emitted += 1
        return text

    next = __next__

    def _split(self, data):
        if self._pending_cr:
            self._pending_cr = False
            if data.startswith(b'\n'):
                self._parts.append(b'\r\n')
                self._emit_parts()
                data = data[1:]
            else:
                self._parts.append(b'\r')
                self._emit_parts()
        start = 0
        n = len(data)
        while start < n:
            r = data.find(b'\r', start)
            lf = data.find(b'\n', start)
            if r < 0 and lf < 0:
                self._parts.append(data[start:])
                return
            if r < 0:
                idx = lf
                self._parts.append(data[start:idx])
                self._parts.append(b'\n')
                self._emit_parts()
                start = idx + 1
                continue
            if lf < 0 or r < lf:
                idx = r
                self._parts.append(data[start:idx])
                if idx + 1 >= n:
                    # A CR at the end of the chunk: wait to see whether the
                    # next chunk begins with LF.
                    self._pending_cr = True
                    return
                if data[idx + 1:idx + 2] == b'\n':
                    self._parts.append(b'\r\n')
                    start = idx + 2
                else:
                    self._parts.append(b'\r')
                    start = idx + 1
                self._emit_parts()
            else:
                idx = lf
                self._parts.append(data[start:idx])
                self._parts.append(b'\n')
                self._emit_parts()
                start = idx + 1

    def _emit_parts(self):
        self._ready.append(b''.join(self._parts))
        self._parts = []

    def _finish_eof(self):
        if self._pending_cr:
            self._pending_cr = False
            self._parts.append(b'\r')
            self._emit_parts()
        if self._parts:
            self._emit_parts()

    def _read_more(self):
        data = self.handle.read(self.chunksize)
        if not data:
            self.eof = True
            self._finish_eof()
        else:
            self._split(data)


def _open_line_reader(path, chunksize, start, initial_lines,
                      initial_offset):
    handle = _open_binary(path, 'rb')
    try:
        if start == 0:
            prefix = b''
            while len(prefix) < 3:
                want = min(chunksize, 3 - len(prefix))
                chunk = handle.read(want)
                if not chunk:
                    break
                prefix += chunk
            if prefix.startswith(_UTF8_BOM):
                initial = b''
                offset = 3
            else:
                initial = prefix
                offset = 0
        else:
            handle.seek(start)
            initial = b''
            offset = start
        reader = _LineReader(handle, chunksize, initial_bytes=initial,
                             start_offset=offset, initial_lines=initial_lines)
        return handle, reader
    except Exception:
        handle.close()
        raise


# ---------------------------------------------------------------------------
# Content validation


def _scan_content(path, chunksize, up_to=None):
    """Hash the file and count physical line breaks before ``up_to``.

    Returns ``(size, sha256, breaks, last_byte)``.  When ``up_to`` is ``None``
    the whole file is counted.  The scan is sequential and bounded-memory.
    """
    digest = hashlib.sha256()
    total = 0
    breaks = 0
    carry_cr = False
    last_byte = b''
    with _open_binary(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunksize)
            if not chunk:
                break
            digest.update(chunk)
            chunk_start = total
            total += len(chunk)
            if up_to is None:
                piece = chunk
            else:
                take = min(len(chunk), max(0, up_to - chunk_start))
                if take <= 0:
                    continue
                piece = chunk[:take]
            last_byte = piece[-1:]
            count, carry_cr = _count_breaks(piece, carry_cr)
            breaks += count
    return total, digest.hexdigest(), breaks, last_byte


def _count_breaks(piece, carry_cr):
    cr = piece.count(b'\r')
    lf = piece.count(b'\n')
    crlf = piece.count(b'\r\n')
    if carry_cr and piece.startswith(b'\n'):
        crlf += 1
    return cr + lf - crlf, piece.endswith(b'\r')


def _expected_physical_lines(breaks, size, last_byte, offset):
    if offset < size:
        return breaks
    if size == 0:
        return 0
    if last_byte in (b'\r', b'\n'):
        return breaks
    # The final record is not terminated, so its physical line is still
    # yielded (at end of file) but no break was counted for it.
    return breaks + 1


def _is_line_boundary(path, offset, chunksize):
    """True if ``offset`` falls at the end of a physical CSV line."""
    with _open_binary(path, 'rb') as handle:
        handle.seek(offset - 1)
        previous = handle.read(min(1, chunksize))
        next_byte = handle.read(min(1, chunksize))
    if previous not in (b'\r', b'\n'):
        return False
    # A boundary immediately after CR is only valid if the next byte is not
    # LF (otherwise the position splits a CRLF pair).
    if previous == b'\r' and next_byte == b'\n':
        return False
    return True


# ---------------------------------------------------------------------------
# Checkpoint validation


def _checkpoint_checksum(payload):
    body = {key: value for key, value in payload.items()
            if key != 'checksum'}
    try:
        canonical = json.dumps(body, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False)
    except (TypeError, ValueError) as err:
        raise ArgumentError('checkpoint is not JSON-serialisable: %s' % err)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _validated_checkpoint(checkpoint, config):
    if isinstance(checkpoint, str):
        try:
            obj = json.loads(checkpoint)
        except ValueError as err:
            raise ArgumentError('checkpoint is not valid JSON: %s' % err)
    else:
        obj = checkpoint
    if not isinstance(obj, dict):
        raise ArgumentError('checkpoint must be a dict or a JSON object '
                            'string, not %r' % (type(obj).__name__,))
    # Work on a detached copy so validation can normalise option values
    # without mutating the caller's object.
    try:
        obj = copy.deepcopy(obj)
    except Exception as err:
        raise ArgumentError('checkpoint could not be copied for validation: '
                            '%s' % err)

    required = ('format', 'version', 'file', 'config', 'header', 'position',
                'checksum')
    missing = [key for key in required if key not in obj]
    if missing:
        raise ArgumentError('checkpoint is missing required fields: %s'
                            % ', '.join(missing))
    unexpected = sorted(set(obj) - set(required))
    if unexpected:
        raise ArgumentError('checkpoint has unexpected fields: %s'
                            % ', '.join(unexpected))

    # Identity and version are checked before the integrity checksum so that
    # a checkpoint from another format or a future version is reported as
    # such rather than as generic corruption.
    if obj['format'] != _FORMAT:
        raise ArgumentError('unrecognised checkpoint format %r (expected %r)'
                            % (obj['format'], _FORMAT))
    if not _is_int(obj['version']):
        raise ArgumentError('checkpoint version must be an integer, not %r'
                            % (obj['version'],))
    if obj['version'] != _FORMAT_VERSION:
        raise ArgumentError('unsupported checkpoint version %r (this version '
                            'supports %r)' % (obj['version'], _FORMAT_VERSION))

    checksum = obj['checksum']
    if not isinstance(checksum, str) or not _HEX_SHA256.match(checksum):
        raise ArgumentError('checkpoint checksum must be a 64-character '
                            'lowercase hexadecimal string')
    # Verify the checksum before normalising any values below, so a tampered
    # or damaged checkpoint is rejected rather than silently reinterpreted.
    if _checkpoint_checksum(obj) != checksum:
        raise ArgumentError('checkpoint checksum does not match its contents; '
                            'the checkpoint is corrupted')

    saved_file = obj['file']
    if not isinstance(saved_file, dict):
        raise ArgumentError('checkpoint file field must be an object')
    extra_file = sorted(set(saved_file) - {'size', 'sha256'})
    if extra_file:
        raise ArgumentError('checkpoint file field has unexpected fields: %s'
                            % ', '.join(extra_file))
    if 'size' not in saved_file or 'sha256' not in saved_file:
        raise ArgumentError('checkpoint file field must contain size and '
                            'sha256')
    if not _is_int(saved_file['size']) or saved_file['size'] < 0:
        raise ArgumentError('checkpoint file size must be a non-negative '
                            'integer, not %r' % (saved_file['size'],))
    if not isinstance(saved_file['sha256'], str) or \
            not _HEX_SHA256.match(saved_file['sha256']):
        raise ArgumentError('checkpoint file sha256 must be a 64-character '
                            'lowercase hexadecimal string')

    saved_config = obj['config']
    if not isinstance(saved_config, dict):
        raise ArgumentError('checkpoint config must be an object')
    if set(saved_config) != set(_CONFIG_KEYS):
        raise ArgumentError('checkpoint config has missing or unexpected '
                            'fields: %r' % (sorted(saved_config),))
    _validate_config_values(saved_config, where='checkpoint config',
                            accept_int_bools=False)
    if saved_config != config:
        raise ArgumentError(
            'CSV parse configuration does not match the checkpoint '
            '(checkpoint has %r, this call has %r); refusing to resume'
            % (saved_config, config))

    saved_header = obj['header']
    if not isinstance(saved_header, list) or not saved_header:
        raise ArgumentError('checkpoint header must be a non-empty list of '
                            'field names')
    if not all(isinstance(value, str) and value != ''
               for value in saved_header):
        raise ArgumentError('checkpoint header fields must be non-empty '
                            'strings')
    if len(set(saved_header)) != len(saved_header):
        raise ArgumentError('checkpoint header contains duplicate field '
                            'names')

    position = obj['position']
    if not isinstance(position, dict):
        raise ArgumentError('checkpoint position must be an object')
    if set(position) - {'byte_offset', 'data_records', 'physical_lines'}:
        raise ArgumentError('checkpoint position has unexpected fields: %s'
                            % ', '.join(sorted(
                                set(position) -
                                {'byte_offset', 'data_records',
                                 'physical_lines'})))
    for key in ('byte_offset', 'data_records', 'physical_lines'):
        if key not in position:
            raise ArgumentError('checkpoint position is missing %r' % key)
        if not _is_int(position[key]) or position[key] < 0:
            raise ArgumentError('checkpoint position %s must be a '
                                'non-negative integer, not %r'
                                % (key, position[key]))

    checksum = obj['checksum']
    if not isinstance(checksum, str) or not _HEX_SHA256.match(checksum):
        raise ArgumentError('checkpoint checksum must be a 64-character '
                            'lowercase hexadecimal string')
    # Verify the checksum before normalising any values below, so a tampered
    # or damaged checkpoint is rejected rather than silently reinterpreted.
    if _checkpoint_checksum(obj) != checksum:
        raise ArgumentError('checkpoint checksum does not match its contents; '
                            'the checkpoint is corrupted')

    return obj
