from __future__ import absolute_import, print_function, division

import heapq
import math
import os
import pickle
import shutil
import tempfile
from itertools import combinations, product

from petl.compat import string_types, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, rowgetter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def discoverkeys(table, columns=None, max_width=None, buffersize=1000,
                 tempdir=None):
    """
    Discover the minimal candidate keys in the current batch of data.

    A *key* is a non-empty set of columns such that every row has a non-missing
    value in each of those columns and no two rows have the same projection
    onto them. A key is *minimal* if none of its non-empty proper subsets is
    itself a key; minimal keys of different widths may coexist. The result is
    a tuple of field-name tuples, sorted first by key width and then by the
    lexicographic order of the column positions in the table header, e.g.::

        >>> import petl as etl
        >>> table = [['a', 'b', 'c'],
        ...          [0, 0, 'x'],
        ...          [0, 1, 'y'],
        ...          [1, 0, 'z']]
        >>> etl.discoverkeys(table)
        (('c',), ('a', 'b'))

    The result describes **this batch only**; a later batch must be
    re-checked before the same columns can be relied upon as a key.

    The `columns` argument restricts the search. It must be ``None`` (search
    every column) or a non-empty, duplicate-free list or tuple of field
    names; a single string is not accepted as a sequence. Columns that ever
    contain a missing value never appear in a returned key.

    ``None`` and floating-point NaN are treated as missing. Values are
    compared using Python equality, so ``True``, ``1`` and ``1.0`` are equal
    and positive and negative zero are equal; strings and bytes are never
    coerced into one another and large integers keep their precision. Cells
    of columns excluded from `columns` are not type-checked, but every row
    must still be as wide as the header. Supported cell types in searched
    columns are ``str``, ``bytes``, ``int`` (including ``bool``), ``float``
    and ``None``; anything else raises :class:`petl.errors.ArgumentError`
    naming the original data row number (counting from one) and the field.

    `max_width` limits the width of the keys returned; ``None`` means no
    limit. Data rows are buffered at most `buffersize` records at a time and
    the overflow is staged in temporary files created under `tempdir`; all
    temporary files are removed again before the call returns. Even when the
    data clearly admit no key, the whole source is read so that trailing row
    widths, cell types and source errors are still validated.

    .. versionadded:: 1.7.32

    """

    # Everything that can be rejected without touching the source is checked
    # before the source iterator is requested.
    buffersize = _check_positive_int(buffersize, 'buffersize')
    if max_width is not None:
        max_width = _check_positive_int(max_width, 'max_width')
    selected = _check_columns_arg(columns)

    staged = None
    source_it = None
    try:
        # Exactly one source iterator per call: the header is read from the
        # same iterator that supplies the data, the source is never reopened.
        source_it = iter(table)
        try:
            header_row = next(source_it)
        except StopIteration:
            raise ArgumentError('table has no header row')
        header_row = _check_header(header_row)

        fieldnames = [text_type(f) for f in header_row]
        if selected is None:
            indices = tuple(range(len(header_row)))
        else:
            indices = _resolve_indices(fieldnames, selected)
        if max_width is not None and max_width > len(indices):
            # documented as legal; the search simply cannot grow that wide
            effective_max = len(indices)
        else:
            effective_max = max_width

        staged = _StagedRows(tempdir, buffersize)
        # The consume pass reads the source to its end and performs every
        # row-width, cell-type and missing-value check. Searching afterwards
        # only touches the staged data, so an early "not unique" result can
        # never skip validation of the source's tail.
        viable_local = staged.consume(source_it, indices, fieldnames,
                                      len(header_row))
        return _search(staged, indices, viable_local, effective_max,
                       header_row)
    finally:
        if staged is not None:
            staged.close()
        _close_iterator(source_it)


Table.discoverkeys = discoverkeys


# ---------------------------------------------------------------------------
# Argument and header validation
# ---------------------------------------------------------------------------


def _is_positive_int(value):
    # bool is a subclass of int but is explicitly not accepted here
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_positive_int(value, name):
    if not _is_positive_int(value):
        raise ArgumentError(
            '%s must be a positive integer (not a bool) or None, got %r'
            % (name, value))
    return value


def _check_columns_arg(columns):
    if columns is None:
        return None
    if isinstance(columns, string_types) or not isinstance(columns,
                                                           (list, tuple)):
        raise ArgumentError(
            'columns must be None or a non-empty list or tuple of field '
            'names; a single string is not accepted, got %r' % (columns,))
    if len(columns) == 0:
        raise ArgumentError('columns must not be empty')
    names = tuple(columns)
    for name in names:
        if not isinstance(name, string_types):
            raise ArgumentError(
                'columns entries must be field name strings, got %r' % (name,))
    if len(set(names)) != len(names):
        raise ArgumentError('columns contains duplicate field names: %r'
                            % (names,))
    return names


def _check_header(header_row):
    if not isinstance(header_row, (list, tuple)) or len(header_row) == 0:
        raise ArgumentError(
            'table header must be a non-empty sequence of field names')
    header_row = tuple(header_row)
    seen = set()
    for field in header_row:
        if not isinstance(field, string_types) or field == '':
            raise ArgumentError(
                'table header must contain only non-empty strings, got %r'
                % (field,))
        name = text_type(field)
        if name in seen:
            raise ArgumentError(
                'table header contains duplicate field name: %r' % (field,))
        seen.add(name)
    return header_row


def _resolve_indices(fieldnames, selected):
    # Resolve then sort by header position: the order in which columns are
    # passed must never affect the result, and every key tuple follows the
    # original header order.
    indices = []
    for name in selected:
        try:
            indices.append(fieldnames.index(name))
        except ValueError:
            raise FieldSelectionError(name)
    return tuple(sorted(indices))


# ---------------------------------------------------------------------------
# Source staging
# ---------------------------------------------------------------------------


_ALLOWED_TYPES = (str, bytes, int, float, type(None))


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


class _StagedRows(object):
    """Stages data projections in bounded chunks plus pickle spill files.

    At most ``buffersize`` rows are held in memory at any time. When the
    buffer fills it is written to a temporary file and a fresh buffer is
    started, so memory use is bounded by the buffer rather than the table
    length. Per-column missing-value flags are the only per-row state kept
    outside the staged chunks; no row numbers or values accumulate in the
    search structures themselves.
    """

    def __init__(self, tempdir, buffersize):
        self._tempdir = tempdir
        self._buffersize = buffersize
        self._temparea = None
        self._chunkfiles = []
        self._memory = []
        self._nrows = 0

    @property
    def temparea(self):
        if self._temparea is None:
            self._temparea = tempfile.mkdtemp(prefix='petl-discoverkeys-',
                                              dir=self._tempdir)
        return self._temparea

    @property
    def buffersize(self):
        return self._buffersize

    def consume(self, source_it, indices, fieldnames, width):
        """Read and validate every source row.

        Returns the *local* positions (into the searched projection tuple) of
        the candidate columns, i.e. the searched columns that never saw a
        missing value. The source is always read to its end, even once no key
        can possibly be returned, so tail validation is never skipped.
        """
        missing = [False] * len(indices)
        buffer_ = self._memory
        buffersize = self._buffersize

        rownumber = 0
        for raw_row in source_it:
            rownumber += 1
            row = tuple(raw_row)
            if len(row) != width:
                raise ArgumentError(
                    'row %d has %d fields, expected %d'
                    % (rownumber, len(row), width))

            # type checks apply only to searched columns
            for local, i in enumerate(indices):
                value = row[i]
                if not isinstance(value, _ALLOWED_TYPES):
                    raise ArgumentError(
                        'unsupported value type %s in field %r at row %d'
                        % (type(value).__name__, fieldnames[i],
                           rownumber))
                if _is_missing(value):
                    missing[local] = True

            # only the searched projection needs to be staged; unselected
            # cells have already been counted towards the row-width check
            projection = tuple(row[i] for i in indices)
            buffer_.append(projection)
            if len(buffer_) >= buffersize:
                self._spill(buffer_)
                buffer_ = []
                self._memory = buffer_

        self._nrows = rownumber
        return tuple(local for local, had_missing in enumerate(missing)
                     if not had_missing)

    def viable_projection_duplicate(self, viable_local):
        """Whether rows repeat after dropping every missing-bearing column.

        Computed from the staged data in a bounded pass; viable columns are
        only known once the whole source has been read and validated.
        """
        if self._nrows <= 1:
            return False
        return not _is_unique(self, tuple(viable_local))

    def _spill(self, rows):
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.temparea, delete=False,
                                             mode='wb') as f:
                name = f.name
                for row in rows:
                    pickle.dump(row, f, protocol=-1)
                f.flush()
        except Exception:
            if name is not None:
                _unlink_quiet(name)
            raise
        self._chunkfiles.append(name)

    def nrows(self):
        return self._nrows

    def fits_in_memory(self):
        """True when every row is held in the single in-memory chunk."""
        return not self._chunkfiles

    def memory_rows(self):
        return self._memory

    def iterchunks(self):
        """Yield the staged rows one bounded chunk at a time."""
        for name in self._chunkfiles:
            yield _read_chunk(name)
        if self._memory or not self._chunkfiles:
            yield self._memory

    def close(self):
        # Remove chunk files first, then the private temp area. Cleanup
        # failures must not mask the exception in flight, so they are
        # swallowed here; the caller's exception propagates untouched.
        for name in self._chunkfiles:
            _unlink_quiet(name)
        self._chunkfiles = []
        self._memory = []
        if self._temparea is not None:
            shutil.rmtree(self._temparea, ignore_errors=True)
            self._temparea = None


def _read_chunk(name):
    rows = []
    with open(name, 'rb') as f:
        while True:
            try:
                rows.append(pickle.load(f))
            except EOFError:
                break
    return rows


def _unlink_quiet(name):
    try:
        os.unlink(name)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Uniqueness testing with a bounded in-memory footprint
# ---------------------------------------------------------------------------


_MASK64 = (1 << 64) - 1
_MIX_A = 0x9e3779b97f4a7c15
_MIX_B = 0xbf58476d1ce4e5b9
_MIX_C = 0x94d049bb133111eb
# keep the number of concurrent in-memory sets in one pass bounded; memory
# beyond the row buffer then grows at most _BATCH_FACTOR * buffersize records
_BATCH_FACTOR = 8
# hash re-partitioning must give up eventually; external sort always finishes
_MAX_BUCKET_DEPTH = 32


def _bucket_index(value, salt, nbuckets):
    # Equality-preserving: Python guarantees hash(a) == hash(b) whenever
    # a == b, so True/1/1.0 and +/-0 stay together. The multiplicative mix
    # only re-spreads those hashes per salt for re-partitioning.
    h = hash(value) & _MASK64
    h = ((h ^ (_MIX_A * (salt + 1))) * _MIX_B + _MIX_C) & _MASK64
    return h % nbuckets


class _SpillDeduper(object):
    """Uniqueness test with a bounded in-memory footprint.

    While values fit in memory, membership is checked directly against a set
    and a duplicate is conclusive before any file is created. On overflow the
    values are hash-partitioned across a fixed number of bucket files; equal
    values always share a bucket, where they are checked with ordinary set
    equality, so a hash collision can never merge distinct values. Buckets
    that stay too large are re-partitioned with a salted mix; if a bucket
    still cannot shrink (e.g. it is dominated by repeated values in a
    non-unique projection) the check falls back to an external merge sort on
    that bucket, which always terminates.
    """

    _NBUCKETS = 16

    def __init__(self, buffersize, temparea):
        self._buffersize = buffersize
        self._temparea = temparea
        self._mem = set()
        self._bucketfiles = None
        self._partitioned = False

    def add_all(self, values):
        """Stream one chunk of values in.

        Returns False only when a duplicate is proven while everything still
        fits in memory. Once partitioning has started this only routes values
        and :meth:`finish` decides the verdict after every chunk is routed.
        """
        for value in values:
            if not self._partitioned:
                if value in self._mem:
                    return False
                self._mem.add(value)
                if len(self._mem) >= self._buffersize:
                    self._start_partition()
            else:
                self._route(value)
        return True

    def finish(self):
        """Return the uniqueness verdict once all chunks were streamed in."""
        if not self._partitioned:
            return True
        files = self._bucketfiles
        names = [f.name for f in files]
        for f in files:
            f.close()
        # ownership of the files moves here; close() must not touch them
        self._bucketfiles = None
        try:
            for name in names:
                if not self._check_bucket(name, 1):
                    return False
            return True
        finally:
            for name in names:
                _unlink_quiet(name)

    def close(self):
        """Release every bucket file without interpreting the contents."""
        if self._bucketfiles is not None:
            for f in self._bucketfiles:
                try:
                    f.close()
                except OSError:
                    pass
                _unlink_quiet(f.name)
            self._bucketfiles = None
        self._mem = set()
        self._partitioned = False

    def _start_partition(self):
        files = [self._new_bucket() for _ in range(self._NBUCKETS)]
        self._bucketfiles = files
        try:
            for value in self._mem:
                files[_bucket_index(value, 0, self._NBUCKETS)].write(
                    pickle.dumps(value, protocol=-1))
        except Exception:
            self.close()
            raise
        self._mem = set()
        self._partitioned = True

    def _route(self, value):
        try:
            self._bucketfiles[
                _bucket_index(value, 0, self._NBUCKETS)].write(
                pickle.dumps(value, protocol=-1))
        except Exception:
            self.close()
            raise

    def _check_bucket(self, name, depth):
        seen = set()
        limit = self._buffersize
        subnames = None
        try:
            with open(name, 'rb') as f:
                while True:
                    try:
                        value = pickle.load(f)
                    except EOFError:
                        break
                    if value in seen:
                        return False
                    seen.add(value)
                    if len(seen) >= limit:
                        subnames = self._repartition(f, seen, depth)
                        break
            if subnames is None:
                return True
            try:
                if depth >= _MAX_BUCKET_DEPTH or self._no_progress(
                        name, subnames):
                    # guaranteed-terminating path for unshrinkable buckets
                    for subname in subnames:
                        if not self._sort_dedup_file(subname):
                            return False
                else:
                    for subname in subnames:
                        if not self._check_bucket(subname, depth + 1):
                            return False
            finally:
                for subname in subnames:
                    _unlink_quiet(subname)
            return True
        finally:
            # belt and braces: the inner try normally removes these already
            if subnames is not None:
                for subname in subnames:
                    _unlink_quiet(subname)

    def _no_progress(self, parent, subnames):
        parent_size = os.path.getsize(parent)
        return any(os.path.getsize(n) >= parent_size for n in subnames)

    def _repartition(self, f, seed_values, depth):
        salt = depth
        files = [self._new_bucket() for _ in range(self._NBUCKETS)]
        names = [f.name for f in files]
        try:
            for value in seed_values:
                files[_bucket_index(value, salt, self._NBUCKETS)].write(
                    pickle.dumps(value, protocol=-1))
            while True:
                try:
                    value = pickle.load(f)
                except EOFError:
                    break
                files[_bucket_index(value, salt, self._NBUCKETS)].write(
                    pickle.dumps(value, protocol=-1))
        except Exception:
            for fh in files:
                try:
                    fh.close()
                except OSError:
                    pass
            for subname in names:
                _unlink_quiet(subname)
            raise
        for fh in files:
            fh.close()
        return names

    def _sort_dedup_file(self, name):
        """External merge-sort dedup for a single unshrinkable bucket."""
        runfiles = []
        try:
            with open(name, 'rb') as f:
                while True:
                    chunk = []
                    while len(chunk) < self._buffersize:
                        try:
                            chunk.append(pickle.load(f))
                        except EOFError:
                            break
                    if not chunk:
                        break
                    chunk.sort(key=Comparable)
                    with tempfile.NamedTemporaryFile(
                            dir=self._temparea, delete=False,
                            mode='wb') as rf:
                        runname = rf.name
                        for value in chunk:
                            rf.write(pickle.dumps(value, protocol=-1))
                    runfiles.append(runname)
            previous = _MISSING
            for value in heapq.merge(*[_iter_run(n) for n in runfiles],
                                     key=Comparable):
                if previous is not _MISSING and value == previous:
                    return False
                previous = value
            return True
        finally:
            for runname in runfiles:
                _unlink_quiet(runname)

    def _new_bucket(self):
        return tempfile.NamedTemporaryFile(dir=self._temparea, delete=False,
                                           mode='wb')


_MISSING = object()


def _iter_run(name):
    with open(name, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def _check_projections(staged, projections):
    """Test several projections in a single pass over the staged chunks.

    Returns a list of booleans aligned with ``projections``. Each projection
    owns an independent deduper, so every per-projection set stays bounded by
    ``buffersize`` records regardless of the table length.
    """
    if staged.nrows() <= 1:
        return [True] * len(projections)
    getters = [rowgetter(*local) for local in projections]
    if staged.fits_in_memory():
        seen_sets = [set() for _ in projections]
        results = [True] * len(projections)
        for row in staged.memory_rows():
            for k, getkey in enumerate(getters):
                if not results[k]:
                    continue
                value = getkey(row)
                seen = seen_sets[k]
                if value in seen:
                    results[k] = False
                else:
                    seen.add(value)
        return results

    dedupers = [_SpillDeduper(staged.buffersize, staged.temparea)
                for _ in projections]
    try:
        results = [True] * len(projections)
        for rows in staged.iterchunks():
            for k, getkey in enumerate(getters):
                if not results[k]:
                    continue
                if not dedupers[k].add_all(getkey(row) for row in rows):
                    # conclusive only while that deduper has not partitioned
                    results[k] = False
        for k, deduper in enumerate(dedupers):
            if results[k] and not deduper.finish():
                results[k] = False
        return results
    finally:
        for deduper in dedupers:
            deduper.close()


def _is_unique(staged, local_indices):
    """Test uniqueness of one projection over all staged chunks."""
    return _check_projections(staged, [tuple(local_indices)])[0]

# ---------------------------------------------------------------------------
# Level-wise minimal-key search
# ---------------------------------------------------------------------------


def _search(staged, searched_indices, viable_local, max_width, header_row):
    n = len(viable_local)
    if max_width is None or max_width > n:
        max_width = n
    if max_width <= 0:
        return ()

    # Global gate: if the rows repeat even after dropping every column that
    # held a missing value, no projection onto any subset of columns can be
    # unique either. Without it the level-wise search would otherwise walk
    # the entire subset lattice to discover that no key exists (e.g. a
    # duplicated full row). The source itself has already been fully read and
    # validated during staging, so this second pass over the staged data can
    # never skip a tail width, type or source error.
    gate_staged = tuple(viable_local)
    if staged.viable_projection_duplicate(gate_staged):
        return ()

    # Collapse value-identical columns into equivalence classes. Two columns
    # with equal values in every row are interchangeable in every key, so the
    # search only needs one representative per class; the final result is
    # expanded to every choice of member. This keeps a wide table whose
    # columns are repeated under different names (the 36-column/16-row case)
    # down to the number of distinct column patterns instead of the raw
    # column count, without changing the keys of any other relation.
    classes = _equivalence_classes(staged, viable_local)
    representatives = tuple(viable_local[members[0]] for members in classes)
    reduced_keys = _search_representatives(staged, representatives,
                                           min(max_width, len(classes)))

    return _expand_equivalent_keys(reduced_keys, classes, searched_indices,
                                   viable_local, header_row, max_width)


def _equivalence_classes(staged, viable_local):
    """Group viable columns whose value columns are identical in every row.

    Equivalence is decided by streaming partition refinement. Each column
    keeps a small dictionary mapping the values seen so far to a per-column
    integer code; two columns remain in the same class only while their code
    sequence agrees row for row. The memory held is proportional to the
    distinct values per column (which cannot exceed the number of rows but is
    bounded independently of the row buffer), never to the whole table.
    """
    width = len(viable_local)
    # one shared value->code encoder: the same value must receive the same
    # code in every column, otherwise distinct columns would look identical.
    # NaN and None never reach here because only missing-free columns are
    # viable, so every encoded value is hashable and equality-comparable.
    encoder = {}
    next_code = 0
    # cumulative class id of each column; initially all columns share one
    labels = [0] * width
    next_label = 1

    for rows in staged.iterchunks():
        for row in rows:
            # refine the current partition: columns in the same existing
            # class split apart when their encoded value at this row differs
            split = {}
            for p, staged_index in enumerate(viable_local):
                value = row[staged_index]
                code = encoder.get(value)
                if code is None:
                    code = next_code
                    encoder[value] = code
                    next_code += 1
                key = (labels[p], code)
                label = split.get(key)
                if label is None:
                    label = next_label
                    next_label += 1
                    split[key] = label
                labels[p] = label

    grouped = {}
    for p, label in enumerate(labels):
        grouped.setdefault(label, []).append(p)
    classes = list(grouped.values())
    # representatives stay in header order, matching the required ordering
    for members in classes:
        members.sort()
    classes.sort(key=lambda members: viable_local[members[0]])
    return classes


def _search_representatives(staged, representative_staged, max_width):
    """Level-wise Apriori over one representative column per equivalence
    class. Returns minimal keys as tuples of staged projection indices."""
    n = len(representative_staged)
    found_keys = []
    found_key_sets = set()
    prev_nonkey_sets = None
    width = 1

    while width <= max_width:
        local_combos = _next_candidates(n, width, prev_nonkey_sets,
                                        found_key_sets)
        if not local_combos:
            break

        staged_combos = [tuple(representative_staged[p] for p in combo)
                         for combo in local_combos]
        verdict = []
        for start in range(0, len(staged_combos), _BATCH_FACTOR):
            verdict.extend(
                _check_projections(
                    staged, staged_combos[start:start + _BATCH_FACTOR]))

        level_nonkey_sets = set()
        for combo, is_key in zip(local_combos, verdict):
            if is_key:
                found_keys.append(combo)
                found_key_sets.add(frozenset(combo))
            else:
                level_nonkey_sets.add(frozenset(combo))

        if width == max_width:
            break
        prev_nonkey_sets = level_nonkey_sets
        width += 1

    # translate representative-local positions to staged projection indices
    return [tuple(representative_staged[p] for p in combo)
            for combo in found_keys]


def _expand_equivalent_keys(reduced_keys, classes, searched_indices,
                            viable_local, header_row, max_width):
    """Expand each reduced key over the equivalent columns it represents.

    A reduced key choosing a representative from a class expands to every
    member of that class independently; the cross product over the key's
    classes yields all minimal keys of the original table. Keys wider than
    ``max_width`` are omitted, and the result follows the documented order.
    """
    # staged-index -> the local positions in the same equivalence class
    staged_to_members = {}
    for members in classes:
        for local_p in members:
            staged_to_members[viable_local[local_p]] = members

    expanded_keys = set()
    for reduced in reduced_keys:
        member_options = []
        for staged_index in reduced:
            members = staged_to_members[staged_index]
            member_options.append([viable_local[p] for p in members])
        for choice in product(*member_options):
            if len(choice) <= max_width:
                expanded_keys.add(tuple(sorted(choice)))

    # Order by width first, then by the lexicographic order of the column
    # positions in the original header. Sorting on the staged indices (which
    # follow header order) gives exactly that ordering; names are attached
    # only afterwards.
    ordered = sorted(expanded_keys, key=lambda key: (len(key), key))
    return tuple(
        tuple(header_row[searched_indices[staged_index]]
              for staged_index in key)
        for key in ordered)


def _next_candidates(n, width, prev_nonkey_sets, found_key_sets):
    """Build the minimal-key candidates of the given width.

    At width 1 every viable column is a candidate. At greater widths an
    Apriori prefix-join over the previous width's non-keys keeps only
    combinations whose every width-1 smaller subset was a verified non-key
    (any subset already known to be a key would make the combination
    non-minimal, and a never-seen subset cannot belong to a minimal key of
    this width). The join is linear in the number of previous non-keys times
    the combination width rather than quadratic in their count.
    """
    if width == 1 or not prev_nonkey_sets:
        return [tuple([p]) for p in range(n)]

    # Group previous (width-1)-sets by their sorted (width-2)-prefix; two
    # non-keys sharing a prefix join into a width-width combination exactly
    # when they differ in their final element.
    prefixes = {}
    for prev in prev_nonkey_sets:
        seq = tuple(sorted(prev))
        prefixes.setdefault(seq[:-1], []).append(seq[-1])

    candidates = []
    for prefix, tails in prefixes.items():
        if len(tails) < 2:
            continue
        tails = sorted(tails)
        for a in range(len(tails)):
            for b in range(a + 1, len(tails)):
                joined = prefix + (tails[a], tails[b])
                # every width-1 subset must itself be a known non-key
                if all(frozenset(s) in prev_nonkey_sets
                       for s in combinations(joined, width - 1)):
                    candidates.append(joined)

    # drop anything containing an already-found smaller minimal key
    return [combo for combo in candidates
            if not _contains_known_key(combo, found_key_sets)]


def _contains_known_key(combo, found_key_sets):
    """True when a previously found minimal key is a proper subset of combo.

    Lookups go through a set of frozensets, so the cost depends only on the
    combination width (at most ``2**width - 2`` bounded subset checks), never
    on how many keys have already been discovered.
    """
    width = len(combo)
    if width <= 1 or not found_key_sets:
        return False
    for sub_width in range(1, width):
        for subset in combinations(combo, sub_width):
            if frozenset(subset) in found_key_sets:
                return True
    return False


# ---------------------------------------------------------------------------
# Iterator housekeeping
# ---------------------------------------------------------------------------


def _close_iterator(it):
    if it is None:
        return
    close = getattr(it, 'close', None)
    if close is None:
        return
    try:
        close()
    except Exception:
        # cleanup must never mask an in-flight exception
        pass
