# MISSING field-presence feature: implementation notes and verification

## What was added

### Public API

| API | Where | Behaviour |
| --- | --- | --- |
| `petl.MISSING` | `petl/missing.py` | singleton marker for an absent field; identity equality, hashable, copy/deepcopy/pickle stable (incl. separate processes and external-sort chunk files); never equal to `None`, `"MISSING"` or ordinary objects |
| `fromdicts(..., preserve_missing=False)` | `petl/io/json.py` | absent key -> `MISSING`; explicit `None`/null stays `None`; other values unchanged |
| `fromjson(..., preserve_missing=False)` (array and `lines`) | `petl/io/json.py` | same as above; `preserve_missing=True` + non-`None` `missing` raises `ArgumentError` |
| `tojson(..., omit_missing=False)` (array and `lines`) | `petl/io/json.py` | omit only marker cells; `None` still written `null`; without the flag a marker raises `TypeError` |
| `join/leftjoin/rightjoin/outerjoin/antijoin/lookupjoin(..., match_missing=False)` | `petl/transform/joins.py` | default: any key containing a marker never matches; `True`: marker equals only marker in the same position; `None` semantics unchanged |
| `hashjoin/hashleftjoin/hashrightjoin/hashantijoin/hashlookupjoin(..., match_missing=False)` | `petl/transform/hashjoins.py` | same contract via raw-key hash lookup; marker keys skipped before probing when disabled |
| `petl.sort` | `petl/comparison.py` | marker sorts before `None` ascending; reverse inverts; compound/componentwise and stable; no new flag needed |

Outer-join `missing=` still controls the padded side and may itself be
`petl.MISSING`; genuinely absent input cells are never overwritten by the
default `missing=None`. All Table chain methods (`Table.join`, `.sort`,
`.tojson`, ...) pick the new keyword arguments up automatically.

## Design decision (two options compared)

1. **In-cell sentinel (chosen).** The marker is an ordinary cell value in
   plain tuple rows. One helper (`key_contains_missing`) recognises it in
   raw/compound keys; `Comparable` defines its sort position; join/hash
   plans branch on the flag once per key group or lookup probe; pickle's
   `__reduce__` makes external sort caches and pickle tables round-trip the
   singleton with no new file format.
2. **Parallel presence bitmap (rejected).** Would require a second data
   plane through every transform, bespoke sort/pickle representations, and
   would break the plain-tuple row contract for all tables, sparse or not.

Costs of the chosen design:

* concentrated incomplete keys: dedicated MISSING branches keep relational
  decisions at O(N + M) - no cartesian enumeration is ever built and
  discarded (see scale probe, inner pairs = 0 at 1k/10k/100k);
* external intermediate representation: unchanged - marker pickles as a
  singleton reconstruction call, existing sort-cache lifecycle untouched;
* cache reuse: hash joins keep the existing right-table lookup cache keyed
  on raw row keys; `cache=False` rebuilds per pass, independent views and
  iterators keep independent matching state;
* plain-row compatibility: only cell values were added; headers, tuple
  rows and all ordinary accessors are unchanged.

## Files changed

* `petl/missing.py` (new) - marker, `key_contains_missing`
* `petl/__init__.py` - export `MISSING`
* `petl/comparison.py` - MISSING ordering in `Comparable`
* `petl/io/json.py` - `preserve_missing` on read, `omit_missing` on write
* `petl/transform/joins.py` - `match_missing` in sorted/merge join,
  anti-join and lookup join; explicit end-of-iterator sentinels so
  MISSING/`None` keys cannot be confused with exhaustion
* `petl/transform/hashjoins.py` - `match_missing` in all hash joins;
  internal raw-key lookup builders
* `docs/missing.rst` (new), `docs/index.rst`, `docs/changes.rst`

## Tests added (75 tests)

* `petl/test/test_missing.py` - identity, copy/deepcopy, pickle (all
  protocols), separate-process pickle, dict-key use, same-named string
* `petl/test/io/test_json_missing.py` - sparse reads (array/lines,
  sampled/explicit header, empty, generator cache), `ArgumentError`,
  omit/`TypeError`/key-set expectations, full round trip, legacy defaults
* `petl/test/transform/test_joins_missing.py` - hand-written expected rows
  for inner/left/right/outer across sorted, presorted, spilled and hash
  paths; both `match_missing` modes; compound partial-missing keys;
  duplicate complete keys (cartesian multiplicity); both sides fully
  missing; different key names and prefixes; `missing` fills of
  `None`/`MISSING`/custom value; anti/lookup joins; single-pass counting,
  `cache=True/False`, independent iterators, source failure and early close
* `petl/test/transform/test_sorts_missing.py` - MISSING-before-None,
  reverse, stability, compound keys, external spill identity, pickle

## Reproduction commands

```bash
# unit tests (new + existing; DB suites excluded as they need optional deps)
python3 -m pytest petl/test -q \
  --ignore=petl/test/io/test_db.py \
  --ignore=petl/test/io/test_db_server.py \
  --ignore=petl/test/io/test_sqlalchemy.py

# only the new suites
python3 -m pytest petl/test/test_missing.py petl/test/io/test_json_missing.py \
  petl/test/transform/test_joins_missing.py \
  petl/test/transform/test_sorts_missing.py -q

# doctests of the changed modules
python3 -m pytest --doctest-modules petl/io/json.py petl/transform/joins.py \
  petl/transform/hashjoins.py petl/comparison.py petl/missing.py -q

# end-to-end pipeline demo (objects -> cut -> sort -> joins -> JSON -> re-read)
PYTHONPATH=. python3 scratch_missing/e2e_demo.py

# scale probes at 1000 / 10000 / 100000 all-MISSING keys
PYTHONPATH=. python3 scratch_missing/scale_probe.py
```

## Observed results

Test run: **1393 tests collected, 1383 passed, 21 skipped (optional
dependencies unavailable), 0 failures** across the whole available suite,
including the pre-existing json, comparison, sorts, joins, lookups and
`fromcsvcheckpoint` regressions and the original (non-omitting) JSON output
tests.

Scale probe (all-MISSING keys, no complete keys, rows streamed and not
retained):

| N=M | match_missing | sorted inner pairs | hash inner pairs | hash-left rows | sorted time | hash time |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | False | 0 | 0 | 1000 | 0.005 s | 0.001 s |
| 10000 | False | 0 | 0 | 10000 | 0.049 s | 0.008 s |
| 100000 | False | 0 | 0 | 100000 | 3.45 s | 0.09 s |

Recorded merge comparisons at the counted sizes grow linearly:
`(lt, eq) = (1998, 999)` at 1k and `(19998, 9999)` at 10k for the sorted
path; the hash path performs zero `Comparable` ordering calls. With
`match_missing=True` and a 10-row right side, the product is exactly
N x 10 (10000 / 100000 / 1000000 emitted pairs), i.e. the opt-in cartesian
behaviour only activates when explicitly requested.

## Not verified here

* Database/remote/optional-format suites (`test_db*`, `test_sqlalchemy`,
  avro/xls/pandas/numpy etc.) were not run because those optional
  dependencies are not installed; none of their code paths were modified.
* Sphinx documentation build was not executed (docs toolchain absent);
  the new `docs/missing.rst` uses the same autodomain directives as
  existing pages and was added to the index toctree.
* Python 2 compatibility is retained stylistically (`__future__` imports)
  but cannot be exercised in this Python 3.11-only environment, matching
  the task's stated runtime.
