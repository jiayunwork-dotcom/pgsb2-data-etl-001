# -*- coding: utf-8 -*-
"""Streaming reader for JSON documents whose top level value is an array.

The reader never seeks and never issues an unbounded ``read()``: it pulls at
most ``chunksize`` bytes at a time from a binary stream and scans the raw
bytes one JSON token at a time. Record *text* is captured via byte slices and
decoded once, so the standard library remains the authority on numbers,
strings, escapes and grammar inside an object.

Error ordering independent of chunk boundaries
-----------------------------------------------

The structural scan works on bytes, but every decision it makes only looks at
ASCII bytes (braces, commas, quotes, whitespace), which never appear inside a
UTF-8 continuation sequence. Non-ASCII runs are handled as opaque data units
whose validity as UTF-8 is decided from the bytes themselves, never from how
they happened to be split across reads:

* an incomplete multi-byte sequence at the end of the buffered bytes waits
  for more input; only at real EOF is a dangling sequence reported as
  :class:`UnicodeDecodeError`;
* once enough bytes are assembled, a sequence is either valid UTF-8 or a
  definite :class:`UnicodeError` -- later bytes cannot repair it;
* when a bad byte is reached inside an unfinished object, the valid prefix
  of that object is parsed to check whether a JSON syntax error had already
  become definite *before* the bad byte. If so the earlier
  :class:`json.JSONDecodeError` wins; otherwise the
  :class:`UnicodeDecodeError` is raised.

Consequently the same byte string yields the same complete record prefix and
the same exception category no matter how the reads are split. A prefix that
is merely incomplete (truncated JSON or a half multi-byte character) is not
an error until EOF: more bytes may complete it.

The module is deliberately split into three responsibilities:

* :class:`_ChunkedByteStream` turns short binary reads into a stream of
  scan units (ASCII bytes and whole UTF-8 sequences) while tracking the
  character offset and line/column position in the original input;
* :func:`_iter_array_records` enforces the top level array grammar and
  yields one complete object's bytes at a time;
* :func:`petl.io.json.JsonView._iter_streaming` owns the petl concerns
  (header discovery, sampling, missing values, source lifecycle).

Design trade-offs
-----------------

* **Performance vs. compatibility.** Each object is extracted as bytes,
  decoded as UTF-8 and handed to the standard decoder rather than
  reimplementing number, string and escape semantics. That keeps decoded
  values (e.g. ``parse_float`` choices) byte-for-byte compatible with
  :func:`json.loads`. The structural scan is linear and record bytes are
  joined from buffer slices, so the cost does not grow quadratically as the
  chunk size shrinks.
* **Bounded memory vs. whole-document errors.** The reader only retains the
  current object (capped by ``max_record_chars``) plus a chunk's worth of
  surrounding bytes. A :class:`json.JSONDecodeError` therefore carries the
  correct line, column and offset in its attributes and message, but its
  ``doc`` field is not the full document; storing that would defeat the
  streaming memory bound. Positions are derived from counters maintained
  during the scan, never by re-reading or copying earlier input, so a very
  long first line or many preceding newlines cost nothing extra.
* **Eager error detection vs. early rows.** Header discovery samples up to
  ``sample`` objects before the header row is yielded, so any error inside
  that prefix surfaces on the first ``next()`` and neither header nor rows
  are delivered. With an explicit header (or ``sample=0``) nothing is
  buffered and valid records are yielded up to the point of a malformed
  suffix, matching the JSON lines behaviour.
* **Short/empty reads vs. EOF.** A non-seekable ``read()`` cannot by itself
  distinguish a momentarily empty stream from a closed one, so isolated
  empty reads are retried while an unbroken run of
  :data:`_EMPTY_READ_EOF` empty reads is accepted as end of stream.
"""
from __future__ import absolute_import, print_function, division

import json
from json import JSONDecodeError

from petl.errors import ArgumentError


# the four JSON whitespace characters, see RFC 8259
_WHITESPACE = frozenset(b' \t\n\r')

# Length of an unbroken run of empty reads accepted as end of stream.
# Ordinary files and compressed streams return b'' persistently once
# closed, while a live source that momentarily has no data either blocks
# in read() or returns an occasional isolated empty read before more
# bytes arrive. Requiring a long unbroken run keeps the two apart without
# busy-looping forever on a genuinely closed stream.
_EMPTY_READ_EOF = 16


def _utf8_lead_length(byte):
    """Number of bytes in the UTF-8 sequence started by ``byte``.

    Returns 1 for ASCII, a lead length of 2-4 for a multi-byte lead, and 0
    for a byte that can never start a sequence (a continuation byte,
    0xFF, 0xC0, 0xC1 or a 0xF8+ lead).
    """
    if byte < 0x80:
        return 1
    if 0xC2 <= byte <= 0xDF:
        return 2
    if 0xE0 <= byte <= 0xEF:
        return 3
    if 0xF0 <= byte <= 0xF4:
        return 4
    return 0


def _decode_utf8_unit(raw):
    """Validate exactly one complete UTF-8 sequence.

    The helper only runs once a structurally complete candidate sequence is
    buffered, so the resulting :class:`UnicodeDecodeError` describes the
    genuine illegal bytes (continuation problems, overlong encodings,
    surrogates or out-of-range code points) with a relative
    ``start``/``end``. Absolute document offsets are intentionally not
    synthesised: the type and the offending bytes are what callers need.
    """
    raw.decode('utf-8')


class _ChunkedByteStream(object):
    """Byte view over a short-read, non-seekable binary stream.

    ``read(n)`` on the wrapped binary stream may return fewer than ``n``
    bytes, including zero bytes while data is still coming. The buffer
    holds at most one record plus a chunk's worth of bytes; between
    records the consumed prefix is compacted away, so memory does not grow
    with the size of the file or with the length of whitespace runs.

    Scan units are either an ASCII byte (an ``int``) or a whole multi-byte
    UTF-8 sequence (``bytes``). A multi-byte sequence may span several
    reads; an incomplete lead at the buffer end waits for more input.
    Document positions are counted in *decoded characters* (0-based offset,
    1-based line/column), matching the convention used by
    :class:`json.JSONDecodeError`.
    """

    def __init__(self, binary, chunksize):
        self._binary = binary
        self._chunksize = chunksize
        self._buf = bytearray()
        # index of the first unconsumed byte in _buf
        self._head = 0
        # _head at the time mark() was called; -1 when unmarked
        self._mark = -1
        # cached result of peek_unit(): an int, bytes, or None at EOF
        self._peek = None
        self._have_peek = False
        self._eof = False
        # document level position, 0-based character offset and 1-based
        # line/column, all counted in decoded characters
        self._pos = 0
        self._line = 1
        self._col = 1
        # count of empty reads in the current unbroken run; any non-empty
        # read resets it, so genuine EOF (a long run of empties) is
        # distinguishable from isolated transient empty reads
        self._empty_run = 0

    # -- raw input -------------------------------------------------------

    def _refill(self):
        """Append at least one new byte, or set the EOF flag.

        A short read that yields no bytes is retried: non-seekable streams
        may return ``b''`` before more data arrives. Empty reads are
        tolerated as long as they stay isolated: any non-empty read resets
        the counter. A closed stream returns ``b''`` on every subsequent
        read, so an unbroken run reaching :data:`_EMPTY_READ_EOF` is
        accepted as EOF.
        """
        if self._mark < 0 and self._head >= len(self._buf):
            # everything buffered has been consumed: drop it so long
            # whitespace gaps cannot grow the buffer with the file
            self._buf = bytearray()
            self._head = 0
            self._have_peek = False
        while True:
            raw = self._binary.read(self._chunksize)
            if raw is None:
                raw = b''
            if raw == b'':
                self._empty_run += 1
                if self._empty_run >= _EMPTY_READ_EOF:
                    self._eof = True
                    return
                # isolated transient empty read: wait for more bytes
                continue
            self._empty_run = 0
            # data arriving after a run of empty reads proves the earlier
            # EOF conclusion was transient, so clear it before appending
            self._eof = False
            self._buf.extend(raw)
            return

    def _compact(self):
        """Drop consumed bytes unless a mark still needs them."""
        if self._mark >= 0 or self._head == 0:
            return
        del self._buf[:self._head]
        self._head = 0

    # -- scan units ------------------------------------------------------

    def peek_unit(self):
        """Return the next scan unit without consuming it.

        Returns an ASCII byte (``int``), the ``bytes`` of one complete
        multi-byte UTF-8 sequence, or ``None`` at EOF. A structurally bad
        sequence raises :class:`UnicodeDecodeError`; an incomplete lead at
        the buffer end waits for more input (or EOF, where the dangling
        bytes are reported as the illegal unit).
        """
        if self._have_peek:
            return self._peek
        # make sure at least one unconsumed byte is buffered
        while self._head >= len(self._buf):
            if self._eof:
                self._peek = None
                self._have_peek = True
                return None
            self._refill()
        lead = self._buf[self._head]
        length = _utf8_lead_length(lead)
        if length == 1:
            unit = lead
        elif length == 0:
            # a continuation or illegal lead byte is unrecoverable
            self._raise_bad_unit(1)
        else:
            # wait until the whole candidate sequence is buffered or the
            # stream ends with a dangling tail
            while len(self._buf) - self._head < length and not self._eof:
                self._refill()
            available = len(self._buf) - self._head
            if available < length:
                self._raise_bad_unit(available)
            raw = bytes(self._buf[self._head:self._head + length])
            # validate continuation bytes, overlong forms, surrogates and
            # out-of-range code points in one place
            _decode_utf8_unit(raw)
            unit = raw
        self._peek = unit
        self._have_peek = True
        return unit

    def _raise_bad_unit(self, size):
        """Raise a UnicodeDecodeError for the buffered illegal bytes.

        Decoding the faithful slice keeps the real reason and the real
        illegal bytes; ``start``/``end`` stay relative to that slice.
        """
        raw = bytes(self._buf[self._head:self._head + size])
        raw.decode('utf-8')
        raise AssertionError('expected UnicodeDecodeError')  # pragma: no cover

    def consume(self):
        """Advance over the peeked unit, updating character positions."""
        unit = self.peek_unit()
        if unit is None:
            return
        if isinstance(unit, int):
            size = 1
            if unit == 0x0A:  # '\n'
                self._line += 1
                self._col = 1
            else:
                self._col += 1
        else:
            size = len(unit)
            # one decoded character per validated multi-byte sequence
            self._col += 1
        self._head += size
        self._pos += 1
        self._have_peek = False
        self._peek = None

    def skip_ws(self):
        """Consume JSON whitespace; return the first unit after it/EOF."""
        unit = self.peek_unit()
        while isinstance(unit, int) and unit in _WHITESPACE:
            self.consume()
            unit = self.peek_unit()
        return unit

    # -- record capture --------------------------------------------------

    def mark(self):
        """Pin the buffer at the start of a record."""
        self._compact()
        self._mark = self._head

    def capture(self):
        """Return the bytes consumed since :meth:`mark`."""
        if self._mark < 0:
            return b''
        return bytes(self._buf[self._mark:self._head])

    def release(self):
        """Release a mark and compact bytes no longer needed."""
        self._mark = -1
        self._compact()

    # -- positions in the original document ------------------------------

    @property
    def pos(self):
        return self._pos

    @property
    def lineno(self):
        return self._line

    @property
    def colno(self):
        return self._col

    def close(self):
        """Drop buffered bytes. The binary stream is owned by the Source
        context manager and is closed by that layer."""
        self._buf = bytearray()
        self._have_peek = False
        self._peek = None


class _UnexpectedEndOfRecord(Exception):
    """Internal signal: EOF was reached before an object closed.

    Carries the record start position in the original document so the
    error reported by the standard decoder can be repositioned.
    """

    def __init__(self, start_pos_info):
        super(_UnexpectedEndOfRecord, self).__init__()
        self.start_pos_info = start_pos_info


class _BadUtf8InRecord(Exception):
    """Internal signal: a definite bad UTF-8 unit was hit inside an object.

    The object never closed, so the valid prefix captured so far is kept
    for the driver to decide whether an earlier JSON syntax error takes
    precedence over the encoding error.
    """

    def __init__(self, start_pos_info, decode_error):
        super(_BadUtf8InRecord, self).__init__()
        self.start_pos_info = start_pos_info
        self.decode_error = decode_error


def _make_json_error(msg, lineno, colno, pos):
    """Build a JSONDecodeError whose attributes and formatted message all
    refer to the same document position (the base class bakes its message
    from the constructor arguments, so set them together).

    The ``doc`` attribute is left empty rather than copying the whole
    input: keeping it would buffer the entire document and defeat the
    bounded-memory design. Position attributes and the formatted message
    still carry the line, column and offset of the original input text.
    """
    error = JSONDecodeError(msg, '', pos)
    error.lineno = lineno
    error.colno = colno
    error.args = (
        '%s: line %d column %d (char %d)' % (msg, lineno, colno, pos),
    )
    return error


def _json_error_at(stream, pos_info, msg):
    """Build a JSONDecodeError at a previously captured document position."""
    return _make_json_error(msg, pos_info[0], pos_info[1], pos_info[2])


def _eof_error(stream, msg):
    """Build a JSONDecodeError positioned at end-of-input."""
    return _make_json_error(msg, stream.lineno, stream.colno, stream.pos)


def _adjust_decode_error(error, start_pos_info, record_text):
    """Translate a record-relative JSONDecodeError to document positions.

    The record was decoded on its own, so ``error`` line/column numbers are
    relative to the record text. Absolute coordinates are derived from the
    record start plus counters over the record's own text -- no earlier
    input is re-read or copied -- and the message is rebuilt from them.
    """
    start_line, start_col, start_pos = start_pos_info
    rel_pos = error.pos
    abs_pos = start_pos + rel_pos
    prefix = record_text[:rel_pos]
    newlines = prefix.count('\n')
    if newlines:
        lineno = start_line + newlines
        colno = rel_pos - prefix.rfind('\n')
    else:
        lineno = start_line
        colno = start_col + rel_pos
    return _make_json_error(error.msg, lineno, colno, abs_pos)


def _scan_object(stream, start_pos_info, max_record_chars):
    """Consume one JSON object including its opening ``{``.

    The stream must be positioned on the opening brace. Returns the
    complete object bytes (brackets and internal whitespace included,
    inter-object whitespace excluded). String contents, including
    backslash and ``\\u`` escapes that happen to be split across chunks,
    are treated as data and left for :func:`json.loads` to interpret.

    Raises :class:`_UnexpectedEndOfRecord` if EOF falls inside the object,
    :class:`_BadUtf8InRecord` for a definite bad UTF-8 unit inside it,
    :class:`petl.errors.ArgumentError` when the decoded character count
    exceeds ``max_record_chars``, and :class:`json.JSONDecodeError` for
    structural problems detected while scanning.
    """
    stream.mark()
    unit = stream.peek_unit()
    assert unit == 0x7B  # '{'
    stream.consume()
    char_count = 1  # count decoded characters including the opening brace
    depth = 1
    in_string = False
    escaped = False
    while True:
        try:
            unit = stream.peek_unit()
        except UnicodeDecodeError as error:
            # definite bad bytes inside an unfinished object: keep the
            # valid prefix captured so far and let the driver order this
            # encoding error against any earlier syntax error
            raise _BadUtf8InRecord(start_pos_info, error)
        if unit is None:
            # keep the mark so the caller can capture the partial record
            raise _UnexpectedEndOfRecord(start_pos_info)
        char_count += 1
        if char_count > max_record_chars:
            stream.release()
            raise ArgumentError(
                'JSON object exceeds max_record_chars=%d (decoded text '
                'length %d at line %d column %d)'
                % (max_record_chars, char_count, stream.lineno,
                   stream.colno))
        if not isinstance(unit, int):
            # validated multi-byte UTF-8 sequence: opaque token data
            stream.consume()
            continue
        stream.consume()
        if in_string:
            if escaped:
                escaped = False
            elif unit == 0x5C:  # backslash
                escaped = True
            elif unit == 0x22:  # double quote
                in_string = False
            continue
        if unit == 0x22:  # '"'
            in_string = True
        elif unit == 0x7B:  # '{'
            depth += 1
        elif unit == 0x7D:  # '}'
            depth -= 1
            if depth == 0:
                raw = stream.capture()
                stream.release()
                return raw
        # any other token (including commas, colons, brackets and
        # whitespace) is simply data inside the object


def _iter_array_records(stream, max_record_chars):
    """Yield ``(record_bytes, start_position)`` for each top level object.

    ``start_position`` is ``(lineno, colno, pos)`` of the opening brace in
    the original document. The grammar around objects is enforced here; the
    grammar inside objects is delegated to :func:`json.loads` by the
    caller, which also translates any error back to document coordinates.
    A definite bad UTF-8 unit in the structural whitespace or in a member
    slot propagates as :class:`UnicodeDecodeError` from ``peek_unit``.
    """
    first = stream.skip_ws()
    if first is None:
        raise _eof_error(stream, 'Expecting value')
    if first != 0x5B:  # '['
        here = (stream.lineno, stream.colno, stream.pos)
        shown = chr(first) if isinstance(first, int) else repr(bytes(first))
        raise ArgumentError(
            'streaming mode requires a JSON array as the top level value; '
            'found %s at line %d column %d'
            % (shown, here[0], here[1]))
    stream.consume()  # '['

    expect_member = True
    while True:
        unit = stream.skip_ws()
        # positions are reported at the peeked (unconsumed) token, matching
        # the standard library caret convention
        here = (stream.lineno, stream.colno, stream.pos)
        if unit is None:
            # stdlib reports "Expecting value" while awaiting a member and
            # "Expecting ',' delimiter" once at least one member was read
            raise _eof_error(
                stream,
                'Expecting value' if expect_member
                else "Expecting ',' delimiter")
        if unit == 0x5D:  # ']'
            stream.consume()
            break
        if not expect_member:
            if unit != 0x2C:  # ','
                raise _json_error_at(stream, here,
                                     "Expecting ',' delimiter")
            stream.consume()
            unit = stream.skip_ws()
            here = (stream.lineno, stream.colno, stream.pos)
            if unit is None:
                raise _eof_error(stream, 'Expecting value')
            if unit in (0x5D, 0x7D):  # ']' or '}'
                # trailing comma (or a stray closing brace): the caret
                # lands on the offending bracket itself, matching json
                raise _json_error_at(stream, here, 'Expecting value')
        # member slot: an object is the only legal member type
        if not isinstance(unit, int):
            # valid multi-byte text cannot begin a JSON token
            raise _json_error_at(stream, here, 'Expecting value')
        if unit != 0x7B:  # '{'
            if (unit in (0x22, 0x2D, 0x74, 0x66, 0x6E)
                    or 0x30 <= unit <= 0x39):
                raise ArgumentError(
                    'streaming mode requires each array member to be a JSON '
                    'object; found %r at line %d column %d'
                    % (chr(unit), here[0], here[1]))
            raise _json_error_at(stream, here, 'Expecting value')
        start_pos_info = here
        record_bytes = _scan_object(stream, start_pos_info,
                                    max_record_chars)
        yield record_bytes, start_pos_info
        expect_member = False

    trailing = stream.skip_ws()
    if trailing is not None:
        here = (stream.lineno, stream.colno, stream.pos)
        raise _json_error_at(stream, here, 'Extra data')


# The single JSONDecodeError message whose caret points *inside* a prefix
# while the failure is really the missing continuation: an unterminated
# string reports where the string began, not where the input stopped.
# Every other truncation-style failure ("Expecting ...") places its caret
# exactly at the end of the bytes seen so far.
_UNTERMINATED_STRING = 'Unterminated string starting at'


def _is_truncation_error(error, prefix_length):
    """True when ``error`` merely reports that the prefix ends mid-token.

    A caret at or past the end of the prefix means no syntax violation had
    become definite before the interruption -- the reader was simply
    waiting for more input. An "unterminated string" caret points at the
    opening quote but describes the same situation. Any other error with a
    caret strictly inside the prefix is a real fault located before the
    bad byte and takes precedence over the encoding failure.
    """
    if error.pos >= prefix_length:
        return True
    return error.msg == _UNTERMINATED_STRING


def _diagnose_bad_utf8(stream, start_pos_info, record_bytes, decode_error,
                       decoder_kwargs):
    """Order a bad UTF-8 unit against earlier JSON syntax errors.

    ``record_bytes`` is the valid UTF-8 prefix of an object that never
    closed (EOF or a definite bad unit interrupted it). If parsing that
    prefix already fails with an error located *before* the interruption,
    the syntax error was determinable before the bad byte and wins.
    Errors that only describe the truncation itself do not pre-date the
    encoding failure, so the original :class:`UnicodeDecodeError` is
    re-raised. A hook raising inside the prefix propagates unchanged.
    """
    prefix_text = record_bytes.decode('utf-8')
    try:
        json.loads(prefix_text, **decoder_kwargs)
    except JSONDecodeError as error:
        if not _is_truncation_error(error, len(prefix_text)):
            return _adjust_decode_error(error, start_pos_info, prefix_text)
    except UnicodeDecodeError:
        # defensive: the prefix was assembled from validated units
        raise decode_error
    raise decode_error


def decode_streaming_records(binary, chunksize, max_record_chars,
                             decoder_kwargs):
    """Drive the streaming machinery over an open binary stream.

    Yields ``(record, record_text, start_position)`` for every top level
    object. Each object is decoded exactly once with the standard library
    decoder using ``decoder_kwargs``. Decoder errors raised inside an
    object are re-raised with positions translated to the original input.
    Records past the error point are never decoded, so their hooks never
    run; exceptions raised by user hooks propagate unchanged.
    """
    stream = _ChunkedByteStream(binary, chunksize)
    try:
        for record_bytes, start_pos_info in _iter_array_records(
                stream, max_record_chars):
            # every byte of a complete record has passed UTF-8 validation
            # during the scan, so this decode cannot fail
            record_text = record_bytes.decode('utf-8')
            try:
                record = json.loads(record_text, **decoder_kwargs)
            except JSONDecodeError as error:
                raise _adjust_decode_error(
                    error, start_pos_info, record_text)
            if not isinstance(record, dict):
                raise ArgumentError(
                    'streaming mode requires each decoded top level record '
                    'to be a dict; object hooks must return a dict '
                    '(got %r at line %d column %d)'
                    % (type(record).__name__, start_pos_info[0],
                       start_pos_info[1]))
            yield record, record_text, start_pos_info
    except _UnexpectedEndOfRecord as exc:
        # EOF inside an object: a dangling multi-byte tail is an encoding
        # error that must be ordered against any earlier syntax error;
        # otherwise let the standard decoder produce the canonical message
        # for the partial text, then reposition it.
        record_bytes = stream.capture()
        stream.release()
        try:
            record_text = record_bytes.decode('utf-8')
        except UnicodeDecodeError as error:
            diagnosed = _diagnose_bad_utf8(
                stream, exc.start_pos_info, record_bytes, error,
                decoder_kwargs)
            raise diagnosed
        try:
            json.loads(record_text, **decoder_kwargs)
        except JSONDecodeError as error:
            raise _adjust_decode_error(
                error, exc.start_pos_info, record_text)
        raise _eof_error(stream, "Expecting ',' delimiter")
    except _BadUtf8InRecord as exc:
        record_bytes = stream.capture()
        stream.release()
        raise _diagnose_bad_utf8(
            stream, exc.start_pos_info, record_bytes, exc.decode_error,
            decoder_kwargs)
    finally:
        stream.close()
