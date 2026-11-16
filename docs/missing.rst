.. _missing:

============================================
Preserving field presence: the MISSING marker
============================================

.. currentmodule:: petl

.. autodata:: MISSING
   :annotation:

Why a separate marker?
======================

An upstream object that omits a key is not the same as one that carries an
explicit ``null``. By default petl collapses both cases into ``None`` when
loading dictionaries or JSON, which loses information that cannot be
recovered downstream.

:data:`petl.MISSING` is an opt-in singleton marker meaning *this field is
absent from the source record*. It is a cell value like any other: rows stay
ordinary ``tuple``/``list`` objects and every existing table access method
keeps working. The marker is:

* distinct from ``None``, from ordinary strings (including a string whose
  text is ``"MISSING"``) and from user objects, using identity-based
  equality;
* hashable, so it works as a dictionary key and inside compound tuple keys;
* stable under :func:`copy.copy`, :func:`copy.deepcopy` and
  :mod:`pickle`, including pickle round-trips in separate Python processes
  and the temporary chunk files used by external sorts.

Reading sparse data
===================

:func:`petl.io.json.fromdicts` and :func:`petl.io.json.fromjson` accept
``preserve_missing=False`` by default for full backwards compatibility. With
``preserve_missing=True``, within the selected header:

* an absent key produces ``MISSING``;
* an explicit ``null``/``None`` stays ``None``;
* every other value is passed through unchanged.

Header discovery via ``sample``, JSON array and JSON lines handling, and the
treatment of unknown fields are unchanged; the header is not expanded after
sampling. Combining ``preserve_missing=True`` with a non-``None`` ``missing``
argument raises :class:`petl.errors.ArgumentError`, because the two mechanisms
would both try to define what an absent field becomes.

Writing sparse data
===================

:func:`petl.io.json.tojson` accepts ``omit_missing=False`` by default. With
``omit_missing=True``, cells holding the marker are left out of the output
object, ``None`` cells are still written as ``null`` and all other values use
the normal JSON encoding. Both array and JSON lines output are supported. If
``omit_missing`` is not enabled and a ``MISSING`` cell is encountered, the
writer raises :class:`TypeError` rather than silently emitting ``null`` or the
string ``"MISSING"``.

Sorting
=======

:func:`petl.sort` needs no new flag: the marker participates in the library's
relaxed :class:`petl.comparison.Comparable` ordering. In ascending order
``MISSING`` sorts before ``None`` and every other value; ``reverse=True``
inverts the order; compound keys are compared component by component and the
sort stays stable. External merge sorts spill rows with :mod:`pickle`, so the
singleton identity is restored from every chunk file automatically.

Joining
=======

The sorted joins (:func:`petl.transform.joins.join`,
:func:`petl.transform.joins.leftjoin`, :func:`petl.transform.joins.rightjoin`,
:func:`petl.transform.joins.outerjoin`, :func:`petl.transform.joins.antijoin`,
:func:`petl.transform.joins.lookupjoin`) and the hash joins
(:func:`petl.transform.hashjoin`, :func:`petl.transform.hashleftjoin`,
:func:`petl.transform.hashrightjoin`, :func:`petl.transform.hashantijoin`,
:func:`petl.transform.hashlookupjoin`) accept ``match_missing=False`` by
default.

* With the default, any row whose (possibly compound) key contains a
  ``MISSING`` component can never match another row - not even a row whose
  key is missing in the same positions. The row is retained or excluded
  purely by the inner/left/full-outer semantics of the join. ``None`` keeps
  its historical equi-join behaviour and is completely unaffected.
* With ``match_missing=True`` the marker equals only the marker in the same
  key position, while all other components follow the existing rules.

Complete-key duplicates keep their existing cartesian product and
multiplicity in every mode.

Cells filled by an outer join for the absent side continue to come from the
join's ``missing`` argument, which may itself be ``petl.MISSING``; genuinely
absent input cells are never overwritten by the default ``missing=None``.

Design: how presence travels through the pipeline
=================================================

Two organisations were considered against the constraints above.

**Option A - in-cell sentinel (implemented).** The marker is a normal cell
value carried inside the existing plain-tuple rows. A single small helper
(:func:`petl.missing.key_contains_missing`) recognises it inside a raw or
compound key, :class:`~petl.comparison.Comparable` defines its sort position,
and the join/hash layers branch on the flag once per key group or lookup
probe. Pickle already preserves the singleton through the marker's
``__reduce__``, so external sort chunk files and pickle tables need no
separate side channel.

**Option B - parallel presence bitmap.** Every table would additionally carry
a per-cell boolean structure describing presence, and every transform would
have to thread, intersect and re-materialise that structure.

Option B was rejected. It doubles the state every view must maintain, breaks
the "rows are plain tuples" contract, requires bespoke external-sort and
pickle representations, and forces every existing transform to reason about a
second data plane even when no sparse source is involved. Option A keeps the
change confined to the marker itself, the comparison layer, the JSON
entry/exit points and the join matching plans.

Costs of the chosen design
--------------------------

* *Concentrated incomplete keys.* Because groups and probes with
  ``MISSING`` keys are handled by a dedicated branch instead of ordinary
  comparison or dictionary probing, two sides of N and M all-missing rows
  cost O(N + M) relational decisions with match_missing off; no
  N x M enumeration is ever built and discarded.
* *External intermediate representation.* No new format is needed: the
  marker pickles to a singleton reconstruction call, reusing the existing
  sort-cache resource model and lifecycle unchanged.
* *Cache reuse.* Hash joins keep their existing right-table lookup cache;
  the cache is keyed by raw row keys, under which the marker is already a
  stable hashable value. ``cache=False`` rebuilds from the current input on
  every pass and distinct views/iterators keep independent matching state.
* *Plain-row compatibility.* Only cell values are added; header handling,
  tuple/list rows, slicing and all ordinary table accessors are untouched.

.. autofunction:: petl.missing.key_contains_missing
