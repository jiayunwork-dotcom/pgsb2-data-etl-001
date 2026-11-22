.. _nan_keys:

====================================
NaN keys: sort order versus matching
====================================

.. currentmodule:: petl

petl gives NaN key values a well-defined **sort order** and a separate,
stricter **matching rule**. The two notions are deliberately not the same:
sorting needs a total order (every value must sit somewhere), while
relational matching follows the IEEE 754 rule that NaN is not equal to
anything, not even itself.

The values concerned are Python ``float`` NaN and :class:`decimal.Decimal`
quiet NaN, in either sign and with any Decimal payload. Signaling NaN is
not covered. A *key containing NaN* means a scalar key that is NaN, or a
compound key with NaN in any component.

Sort order
==========

Under :func:`petl.transform.sorts.sort` (and everything built on it, such as
:func:`petl.transform.sorts.mergesort` and the sorted joins) the ascending
order is:

``MISSING`` < ``None`` < NaN < all other values

* All NaN values are mutually **co-equal** for ordering, regardless of
  category (float vs Decimal), sign or payload. Co-equal keys keep their
  original input order - the sort is stable, including through the
  external (spilled-to-disk) merge and with ``reverse=True``.
* Compound keys are compared component by component; a NaN component is
  co-equal with any other NaN component and sorts after ``None`` and
  before any other value at that position.
* ``reverse=True`` inverts the order of the groups and stays stable:
  co-equal rows keep their input order, exactly as
  ``sorted(..., reverse=True)`` does.
* All other values keep their existing relaxed comparison rules: ordinary
  strings, arbitrary-precision integers, finite floats and Decimals,
  positive and negative infinity, booleans and signed zero. Values that
  are already equal stay equal, so ``1``, ``1.0``, ``Decimal('1')`` and
  ``True`` still sort (and match) together, and big integers are
  compared exactly, never via float conversion.

Callers passing ``presorted=True`` to the joins or to
:func:`petl.transform.sorts.mergesort` should prepare their input in the
order above.

Matching rule
=============

For the sorted joins (:func:`petl.transform.joins.join`,
:func:`~petl.transform.joins.leftjoin`,
:func:`~petl.transform.joins.rightjoin`,
:func:`~petl.transform.joins.outerjoin`,
:func:`~petl.transform.joins.antijoin`,
:func:`~petl.transform.joins.lookupjoin`) and the hash joins
(:func:`petl.transform.hashjoins.hashjoin` and its variants), a key
containing a NaN component **never matches** another row - not even a row
carrying the identical NaN object, and regardless of the ``match_missing``
option (which governs only :data:`petl.MISSING`):

* inner joins drop such rows;
* outer joins emit them as unmatched rows on their own side, with the
  original NaN cell values carried through unchanged;
* anti joins always keep left rows with NaN keys.

NaN values in non-key fields do not affect matching and pass through to
the output unchanged. Output rows always contain the original cell values
and types; NaN cells keep their category, sign and Decimal payload (also
across the external sort's pickle round-trip).

Why co-equal but unmatchable?
=============================

Sorting must place every row somewhere deterministic, so NaN gets a single
co-equal bucket. But a join that paired NaN keys would silently multiply
rows (every NaN row on the left would pair with every NaN row on the
right) and would contradict IEEE semantics, so matching treats NaN keys
as unmatchable. The two facts travel through the shared machinery
separately:

* :class:`petl.comparison.Comparable` carries the sort order. It caches a
  per-value NaN flag when a key is wrapped, so the O(n log n) comparisons
  of a sort - and repeated compound-key comparisons - just read a slot.
* :func:`petl.comparison.key_contains_nan` carries the matching rule. The
  merge joins evaluate it once per key group and the hash joins once per
  row probe, before any comparison or dictionary lookup. NaN groups are
  consumed or emitted without ever being compared against the other side,
  and NaN keys never reach the hash lookup (where the interpreter's
  identity fast path would otherwise pair up a shared NaN object).

With N NaN-keyed rows on the left and M on the right, the pairing-decision
work is O(N + M): no N*M candidate pairs are ever enumerated. Sorting
costs the usual O(R log R) comparisons (plus chunk-file I/O when
spilling), and the hash joins keep their existing O(M) right-table index
build; NaN keys in the indexed table are simply never probed.

Decimal context safety
======================

All NaN handling avoids arithmetic and ordering comparisons on Decimal
NaN values, so it works whether or not the caller's decimal context traps
:class:`decimal.InvalidOperation`, and it never modifies the caller's
context, trap settings or status flags. (A native ``Decimal('NaN') < x``
comparison would otherwise raise ``InvalidOperation`` under the default
context.)
