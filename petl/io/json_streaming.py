# -*- coding: utf-8 -*-
"""Streaming reader for JSON documents whose top level value is an array.

The reader never seeks and never issues an unbounded ``read()``: it pulls at
most ``chunksize`` bytes at a time from a binary stream, decodes the bytes
incrementally as UTF-8 (so multi-byte characters may straddle a chunk
boundary) and scans the resulting text one character at a time. Record text
is captured via chunk slices, keeping per-record handling linear in the size
of the record so that splitting a record into many small chunks does not
turn parsing into a quadratic operation.

The module is deliberately split into three responsibilities:

* :class:`_ChunkedTextStream` turns short binary reads into a character
  stream while tracking the character offset and line/column position in
  the original input;
* :func:`_iter_array_records` enforces the top level array grammar and
  yields one complete object text at a time;
* :func:`petl.io.json.JsonView._iter_streaming` owns the petl concerns
  (header discovery, sampling, missing values, source lifecycle).

Record *text* is decoded with :func:`json.loads`, so the standard library
remains the authority on numbers, strings, escapes and grammar inside an
object. Only the grammar *around* the objects (the array brackets, commas
and whitespace) is handled here, and errors from either layer are reported
as :class:`json.JSONDecodeError` positions relative to the original input.

Design trade-offs
-----------------

* **Performance vs. compatibility.** Each object is extracted as text and
  handed to the standard decoder rather than reimplementing number, string
  and escape semantics. That keeps decoded values (e.g. ``parse_float``
  choices) byte-for-byte compatible with :func:`json.loads` at the cost of
  one extra pass over each object's text. The structural scan is linear and
  record text is joined from chunk slices, so the cost does not grow
  quadratically as the chunk size shrinks.
* **Bounded memory vs. whole-document errors.** The reader only retains the
  current object (capped by ``max_record_chars``) plus a chunk's worth of
  surrounding text. A :class:`json.JSONDecodeError` therefore carries the
  correct line, column and offset in its attributes and message, but its
  ``doc`` field is not the full document; storing that would defeat the
  streaming memory bound.
* **Eager error detection vs. early rows.** Header discovery samples up to
  ``sample`` objects before the header row is yielded, so a syntax error
  inside that prefix surfaces on the first ``next()``. With an explicit
  header (or ``sample=0``) nothing is buffered and valid records are yielded
  up to the point of a malformed suffix, matching the JSON lines behaviour.
* **Short/empty reads vs. EOF.** A non-seekable ``read()`` cannot by itself
  distinguish a momentarily empty stream from a closed one, so isolated
  empty reads are retried while an unbroken run of
  :data:`_EMPTY_READ_EOF` empty reads is accepted as end of stream.
"""
from __future__ import absolute_import, print_function, division

import codecs
import json
from json import JSONDecodeError

from petl.errors import ArgumentError


# the four JSON whitespace characters, see RFC 8259
_WHITESPACE = frozenset(' \t\n\r')

# Length of an unbroken run of empty reads accepted as end of stream.
# Ordinary files and compressed streams return b'' persistently once
# closed, while a live source that momentarily has no data either blocks
# in read() or returns an occasional isolated empty read before more
# bytes arrive. Requiring a long unbroken run keeps the two apart without
# busy-looping forever on a genuinely closed stream.
_EMPTY_READ_EOF = 16


class _ChunkedTextStream(object):
    """Character view over a short-read, non-seekable binary stream.

    ``read(n)`` on the wrapped binary stream may return fewer than ``n``
    bytes, including zero bytes while data is still coming. Bytes are
    decoded incrementally as UTF-8 so a multi-byte character may span
    several reads.

    Between records the buffered text is compacted to the chunks that have
    not yet been consumed, so memory does not grow with the size of the
    file or with the length of whitespace runs. While a record is being
    scanned, :meth:`mark` pins the first chunk of the record and
    :meth:`capture` returns the accumulated record text by joining chunk
    slices, which is linear work per record rather than per character.
    """

    def __init__(self, binary, chunksize):
        self._binary = binary
        self._chunksize = chunksize
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')
        # decoded chunks not yet fully consumed
        self._chunks = []
        # index into _chunks of the chunk currently being read
        self._head = 0
        # character offset inside the current chunk
        self._offset = 0
        # _head/_offset at the time mark() was called
        self._mark_head = -1
        self._mark_offset = 0
        self._eof = False
        # document level position, 0-based offset and 1-based line/column
        self._pos = 0
        self._line = 1
        self._col = 1
        # count of empty reads in the current unbroken run; any non-empty
        # read resets it, so genuine EOF (a long run of empties) is
        # distinguishable from isolated transient empty reads
        self._empty_run = 0

    # -- raw input -------------------------------------------------------

    def _refill(self):
        """Make at least one decoded character available, or set EOF.

        A short read that yields no decoded characters (half of a
        multi-byte UTF-8 sequence) or even an empty read is retried:
        non-seekable streams may return ``b''`` before more data arrives.
        Empty reads are tolerated as long as they stay isolated: any
        non-empty read resets the counter. A closed stream returns
        ``b''`` on every subsequent read, so an unbroken run reaching
        :data:`_EMPTY_READ_EOF` is accepted as EOF. The incremental
        decoder is flushed at that point, which turns a dangling
        multibyte sequence into :class:`UnicodeDecodeError`.
        """
        while self._head >= len(self._chunks):
            raw = self._binary.read(self._chunksize)
            if raw is None:
                raw = b''
            if raw == b'':
                self._empty_run += 1
                if self._empty_run >= _EMPTY_READ_EOF:
                    tail = self._decoder.decode(b'', final=True)
                    if tail:
                        self._chunks.append(tail)
                    else:
                        self._eof = True
                    return
                # isolated transient empty read: wait for more bytes
                continue
            self._empty_run = 0
            text = self._decoder.decode(raw, final=False)
            if text:
                # more data arrived after a run of empty reads: any EOF
                # conclusion was transient, so clear it
                self._eof = False
                # a mark may pin older chunks; new text always appends
                self._chunks.append(text)
                return
            # bytes were read but decoded to no complete characters
            # (e.g. half of a multi-byte sequence): keep pulling

    def _ensure_char(self):
        """Advance to the next readable character, compacting spent chunks."""
        if self._eof:
            return False
        if self._head < len(self._chunks):
            chunk = self._chunks[self._head]
            if self._offset >= len(chunk):
                self._head += 1
                self._offset = 0
                self._compact()
        if self._head >= len(self._chunks):
            self._refill()
            if self._eof:
                return False
        return True

    def _compact(self):
        """Drop chunks before the head unless a mark still needs them."""
        if self._head <= 0 or self._mark_head >= 0:
            return
        self._chunks = self._chunks[self._head:]
        self._head = 0

    # -- public character API -------------------------------------------

    def read_char(self):
        """Consume and return the next character, or ``''`` at EOF."""
        if not self._ensure_char():
            return ''
        char = self._chunks[self._head][self._offset]
        self._offset += 1
        self._pos += 1
        if char == '\n':
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return char

    def peek_char(self):
        """Return the next character without consuming it, or ``''``."""
        if not self._ensure_char():
            return ''
        return self._chunks[self._head][self._offset]

    def skip_ws(self):
        """Consume JSON whitespace; returns the first non-ws char or EOF."""
        char = self.peek_char()
        while char in _WHITESPACE:
            self.read_char()
            char = self.peek_char()
        return char

    # -- record capture --------------------------------------------------

    def mark(self):
        """Pin the stream position at the start of a record."""
        self._mark_head = self._head
        self._mark_offset = self._offset

    def capture(self):
        """Return the text consumed since :meth:`mark`."""
        if self._mark_head < 0:
            return ''
        if self._mark_head == self._head:
            return self._chunks[self._head][
                self._mark_offset:self._offset]
        pieces = [self._chunks[self._mark_head][self._mark_offset:]]
        pieces.extend(self._chunks[self._mark_head + 1:self._head])
        if self._head < len(self._chunks) and self._offset:
            pieces.append(self._chunks[self._head][:self._offset])
        return ''.join(pieces)

    def release(self):
        """Release a mark and compact chunks no longer needed."""
        self._mark_head = -1
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
        """Drop buffered decoder text. The binary stream is owned by the
        Source context manager and is closed by that layer."""
        self._chunks = []


class _UnexpectedEndOfRecord(Exception):
    """Internal signal: EOF was reached before an object closed.

    Carries the record start position in the original document so the
    error reported by the standard decoder can be repositioned.
    """

    def __init__(self, start_pos_info):
        super(_UnexpectedEndOfRecord, self).__init__()
        self.start_pos_info = start_pos_info


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


def _adjust_decode_error(error, start_pos_info):
    """Translate a record-relative JSONDecodeError to document positions.

    The record was decoded on its own, so ``error`` line/column numbers are
    relative to the record text. A synthetic document prefix places the
    record back at its original line and column, which lets
    :class:`JSONDecodeError` derive the correct line and column; the
    absolute character offset is then substituted.
    """
    start_line, start_col, start_pos = start_pos_info
    abs_pos = start_pos + error.pos
    prefix = ('\n' * (start_line - 1)) + ('x' * (start_col - 1)) \
        + error.doc[:error.pos]
    positioned = JSONDecodeError(error.msg, prefix, len(prefix))
    return _make_json_error(error.msg, positioned.lineno,
                            positioned.colno, abs_pos)


def _scan_object(stream, start_pos_info, max_record_chars):
    """Consume one JSON object including its opening ``{``.

    The stream must be positioned on the opening brace. Returns the
    complete object text (brackets and internal whitespace included,
    inter-object whitespace excluded). String contents, including
    backslash and ``\\u`` escapes that happen to be split across chunks,
    are treated as data and left for :func:`json.loads` to interpret.

    Raises :class:`_UnexpectedEndOfRecord` if EOF falls inside the object,
    :class:`petl.errors.ArgumentError` when the decoded character count
    exceeds ``max_record_chars``, and :class:`json.JSONDecodeError` for
    structural problems detected while scanning.
    """
    stream.mark()
    char = stream.read_char()  # the opening '{'
    assert char == '{'
    char_count = 1  # count decoded characters including the opening brace
    depth = 1
    in_string = False
    escaped = False
    while True:
        char = stream.read_char()
        if char == '':
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
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                text = stream.capture()
                stream.release()
                return text
        # any other token (including commas, colons, brackets and
        # whitespace) is simply data inside the object


def _iter_array_records(stream, max_record_chars):
    """Yield ``(record_text, start_position)`` for each top level object.

    ``start_position`` is ``(lineno, colno, pos)`` of the opening brace in
    the original document. The grammar around objects is enforced here;
    the grammar inside objects is delegated to :func:`json.loads` by the
    caller, which also translates any error back to document coordinates.
    """
    first = stream.skip_ws()
    if first == '':
        raise _eof_error(stream, 'Expecting value')
    if first != '[':
        here = (stream.lineno, stream.colno, stream.pos)
        raise ArgumentError(
            'streaming mode requires a JSON array as the top level value; '
            'found %r at line %d column %d'
            % (first, here[0], here[1]))
    stream.read_char()  # consume '['

    expect_member = True
    while True:
        char = stream.skip_ws()
        # positions are reported at the peeked (unconsumed) token, matching
        # the standard library caret convention
        here = (stream.lineno, stream.colno, stream.pos)
        if char == '':
            # stdlib reports "Expecting value" while awaiting a member and
            # "Expecting ',' delimiter" once at least one member was read
            raise _eof_error(
                stream,
                'Expecting value' if expect_member
                else "Expecting ',' delimiter")
        if char == ']':
            stream.read_char()
            break
        if not expect_member:
            if char != ',':
                raise _json_error_at(stream, here,
                                     "Expecting ',' delimiter")
            stream.read_char()
            char = stream.skip_ws()
            here = (stream.lineno, stream.colno, stream.pos)
            if char == '':
                raise _eof_error(stream, 'Expecting value')
            if char in (']', '}'):
                # trailing comma (or a stray closing brace): the caret
                # lands on the offending bracket itself, matching json
                raise _json_error_at(stream, here, 'Expecting value')
        # member slot: an object is the only legal member type
        if char != '{':
            if char in ('"', '-', 't', 'f', 'n') or char.isdigit():
                raise ArgumentError(
                    'streaming mode requires each array member to be a JSON '
                    'object; found %r at line %d column %d'
                    % (char, here[0], here[1]))
            raise _json_error_at(stream, here, 'Expecting value')
        start_pos_info = here
        record_text = _scan_object(stream, start_pos_info,
                                   max_record_chars)
        yield record_text, start_pos_info
        expect_member = False

    trailing = stream.skip_ws()
    if trailing != '':
        here = (stream.lineno, stream.colno, stream.pos)
        raise _json_error_at(stream, here, 'Extra data')


def decode_streaming_records(binary, chunksize, max_record_chars,
                             decoder_kwargs):
    """Drive the streaming machinery over an open binary stream.

    Yields ``(record, record_text, start_position)`` for every top level
    object. Each object is decoded exactly once with the standard library
    decoder using ``decoder_kwargs``. Decoder errors raised inside an
    object are re-raised with positions translated to the original input.
    """
    stream = _ChunkedTextStream(binary, chunksize)
    try:
        for record_text, start_pos_info in _iter_array_records(
                stream, max_record_chars):
            try:
                record = json.loads(record_text, **decoder_kwargs)
            except JSONDecodeError as error:
                raise _adjust_decode_error(error, start_pos_info)
            if not isinstance(record, dict):
                raise ArgumentError(
                    'streaming mode requires each decoded top level record '
                    'to be a dict; object hooks must return a dict '
                    '(got %r at line %d column %d)'
                    % (type(record).__name__, start_pos_info[0],
                       start_pos_info[1]))
            yield record, record_text, start_pos_info
    except _UnexpectedEndOfRecord as exc:
        # EOF inside an object: let the standard decoder produce the
        # canonical message for the partial text, then reposition it
        record_text = stream.capture()
        stream.release()
        try:
            json.loads(record_text, **decoder_kwargs)
        except JSONDecodeError as error:
            raise _adjust_decode_error(error, exc.start_pos_info)
        raise _eof_error(stream, "Expecting ',' delimiter")
    finally:
        stream.close()
