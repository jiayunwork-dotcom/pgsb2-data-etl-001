# NaN keys: sort order vs join matching — implementation notes and verification

## Problem

With `n = float('nan')` and

```
left  = [('k','l'), (1,'L1'), (n,'LN'), (2,'L2'), (3,'L3')]
right = [('k','r'), (1,'R1'), (2,'R2'), (3,'R3'), (n,'RN')]
```

the pre-fix code showed two complementary defects from the same numeric-key
handling:

* **sorted joins** (`join`/`leftjoin`/`outerjoin`, any buffersize): native
  `nan < x` is always False, so NaN had no defined sort position. The merge
  advance (`a > b` implemented as `not (a < b or a == b)`) treated NaN as
  greater than everything and everything as greater than NaN, silently
  dropping key 3: `join` returned only keys 1 and 2, `leftjoin` padded L3,
  `outerjoin` split L3/R3 into unmatched rows.
* **hash joins** (`hashjoin`/`hashleftjoin`): the right-table dict probe
  `k in rlookup` hits the interpreter's identity fast path when the *same*
  NaN object appears on both sides, pairing LN with RN.

Expected (now implemented): keys 1, 2, 3 match everywhere; LN/RN never pair;
outer joins keep them as unmatched rows on their own side.

## Design decision (two options compared)

The task is to carry two facts through the shared machinery (comparison,
external-sort spill/readback, merge advance, hash lookup): **"co-equal for
sorting"** and **"relationally unmatchable"**.

### Option A — total preorder in `Comparable` + separate unmatchable predicate (chosen)

* Sort information travels inside the existing `Comparable` wrapper. A
  per-value `isnan` flag is computed once when a key is wrapped and stored
  in a new slot. `__lt__` places NaN after `None` and before all other
  values; `__eq__` treats any two quiet NaNs as co-equal **for ordering**,
  so the stability machinery sees a consistent total preorder: the
  `heapq.merge` `[key, order, next]` tie-break, `max()` first-win in the
  reverse shortlist merge, and `itertools.groupby` all behave. (With
  `__eq__` left as native `nan != nan`, the heap-merge chunk tie-break
  never fires for NaN and cross-chunk stability is lost.)
* Match information travels as an explicit predicate
  `petl.comparison.key_contains_nan` (mirroring the existing
  `key_contains_missing`). The merge joins evaluate it once per key group
  before any comparison; the hash joins evaluate it once per row before
  any dict probe. NaN groups/rows are consumed or emitted by the
  outer-join rules without ever being compared or probed.
* This is exactly the architecture the MISSING feature already
  established in this codebase, so sort, spill, merge and hash paths stay
  consistent by construction.

### Option B — canonicalize NaN out of the key at extraction time (rejected)

Replace each NaN component in extracted keys with a sentinel. Two
sub-variants, both inferior:

* *Shared sentinel*: sorts fine, but the sentinel equals itself, so merge
  equality and hash lookup would match NaN keys — the unmatchable
  predicate is still needed on top, now against a second key
  representation. The sentinel would also need a pickle contract for the
  external-sort chunk files and a defined order inside `Comparable`
  anyway.
* *Per-row unique sentinel*: never matches (good) but destroys sort
  co-equality — each NaN becomes its own rank, so "co-equal keeps input
  order" stability has to be re-implemented, groupby shatters NaN runs
  into singletons, and `presorted=True` callers can no longer be given a
  single order to prepare. Both variants allocate O(rows) extra objects
  per pass.

Option A keeps one key representation, reuses the existing wrapper, and
needs no new serialization contract.

## Cost analysis

* **Compound-key repeated comparisons**: NaN-ness is computed once per
  wrapped value (`Comparable.__init__`) and cached in a slot; the
  O(n log n) sort comparisons and repeated component-wise compound-key
  comparisons just read the slot. No per-comparison type dispatch beyond
  the pre-existing checks.
* **External sort serialization**: unchanged. Rows are pickled as-is;
  float NaN sign and Decimal quiet-NaN sign+payload survive the pickle
  round-trip (verified). No marker objects are introduced, so chunk files
  keep the existing format and lifecycle; identity of NaN objects is not
  preserved across spill (pickle semantics), only category/sign/payload —
  which is what the contract checks.
* **Hash index**: unchanged structure and cost — O(M) inserts for the
  right table. NaN keys on the indexed side are inert entries (each
  distinct NaN object its own dict entry; a shared object one entry);
  probes are skipped before `__contains__`, so the identity fast path can
  never fire. Probe-side guard is O(1) per row (scalar) or O(components)
  per compound key.
* **NaN hot key**: with N left and M right NaN rows, the merge join makes
  O(1) group-level decisions per NaN run (all consecutive NaN rows form
  one co-equal group) and never enumerates pairs; the hash join makes N
  guard checks and **zero** dict probes. Total pairing-decision work is
  O(N + M); see the scale probe below — inner pairs emitted = 0 at
  K = 256 / 4096 / 65536 while a hypothetical N*M would be up to
  8.6·10⁹. Sorting remains O(R log R) comparisons (plus chunk I/O when
  spilling); the hash index remains O(M). No whole-left-table or
  whole-result caches were added.

## Semantics preserved

* Ordinary values keep native comparison and matching: strings,
  arbitrary-precision ints (exact — no float-conversion merge;
  `2**53+1 != 2**53`), finite floats/Decimals, ±inf
  (`Decimal('Infinity') == float('inf')` still matches), bool, ±0
  (`0.0 == -0.0 == Decimal('-0')` match). `1 == 1.0 == Decimal('1') ==
  True` still match, in both merge and hash paths.
* Duplicate complete keys keep their cartesian product and multiplicity
  (control probe: K×2K pairs for a shared finite key).
* `None` keys still match `None`; MISSING rules unchanged;
  `match_missing=True` lets MISSING match MISSING only — never NaN.
* Signaling NaN is out of scope (not treated as NaN; native behavior,
  including possible `InvalidOperation` under default traps, unchanged).
* Output cells keep original values and types; NaN verified by category,
  sign and Decimal payload. No internal markers leak into output.
* Decimal context: all NaN handling avoids arithmetic and ordering
  comparisons on Decimal NaN, so it works with `InvalidOperation` trapped
  (default) or not, and never mutates the caller's context, traps or
  flags (tested under both regimes with flag inspection before/after).

## Files changed

* `petl/comparison.py` — `_isnan`, `key_contains_nan`; `Comparable`
  gains an `isnan` slot, NaN sort position in `__lt__`, NaN co-equality
  in `__eq__`; docstring documents the order and the sort/match split.
* `petl/transform/joins.py` — `iterjoin`, `iterantijoin`,
  `iterlookupjoin` consume NaN-keyed groups via the unmatchable branch
  (same shape as the MISSING branch, independent of `match_missing`);
  `join` docstring documents the NaN contract.
* `petl/transform/hashjoins.py` — `iterhashjoin`, `iterhashleftjoin`,
  `iterhashrightjoin`, `iterhashantijoin`, `iterhashlookupjoin` guard
  NaN keys before any dict/set probe; `hashjoin` docstring updated.
* `petl/transform/sorts.py` — `sort` docstring documents the order.
* `petl/test/helpers.py` — additive NaN-aware helpers (`nan_token`,
  `canonical_rows`, `ieq_nan`, `row_multiset`).
* `docs/nan.rst` (new), `docs/index.rst`, `docs/changes.rst`.

Public signatures unchanged; `Table` chain methods pick the behavior up
automatically (tested).

## Tests added (47)

* `petl/test/test_comparison_nan.py` (9) — `_isnan`/`key_contains_nan`
  coverage (float ±NaN, Decimal quiet NaN signs/payloads, sNaN excluded,
  non-values); sort position MISSING < None < NaN < rest; cross-category
  co-equality and total-preorder consistency; compound component-wise
  comparison; ordinary equalities (1/1.0/Decimal('1')/True, ±0, inf) and
  big-int exactness; wrapper stores the original object; decimal context
  and flags untouched under both trap regimes.
* `petl/test/transform/test_sorts_nan.py` (10) — hand-written full order
  across memory/spill/cache paths; reverse (groups inverted, co-equal
  rows keep input order — Python stable-reverse semantics); stability
  with many NaN keys and duplicate finite keys through buffersize 1/2/3;
  compound keys with NaN in each position; empty and all-NaN tables;
  output value/type preservation through the pickle spill; non-key NaN
  pass-through (identity on the memory path); decimal context; mergesort
  (sorted and presorted=True, reverse); issorted.
* `petl/test/transform/test_joins_nan.py` (28) — hand-written expected
  rows and multiplicities for: the original repro (join/leftjoin/
  rightjoin/outerjoin/antijoin/lookupjoin + hash variants + Table
  chaining); NaN on left/right/both sides; same object reused, distinct
  objects, signs, Decimal payloads; compound keys with NaN in each
  position and mixed with None/MISSING; empty tables; all-NaN tables
  (exact padded-row counts); duplicate finite keys with NaN present
  (2×3=6 cartesian rows verified); mixed numerics; big ints; infinities;
  signed zero; non-key NaN; None/MISSING under both `match_missing`
  modes; decimal context end to end; cross-path multiset consistency
  (memory, buffersize=2 spill, cache re-traversal ×2, presorted, hash,
  hash cache ×2); iterator independence; source exception propagation;
  explicit close.

All expectations are hand-written literals; NaN cells are compared by
category/sign/payload via `petl.test.helpers.nan_token`, never via the
comparator under test or another join.

## Scale probe

`scratch_nan/scale_probe_nan.py` — left K / right 2K all-NaN rows,
K ∈ {256, 4096, 65536}; results consumed row by row (counters only);
decisions counted by monkeypatching `Comparable.__lt__`,
`key_contains_nan` and `_hash_lookup` from the probe (no library
instrumentation); wall time = best of 3 without tracemalloc, peak =
separate traced run.

```
K=256    N+2N rows=768     hypothetical N*M pairs=131072
  merge inner : rows=1        lt=766       kcn=1
  merge outer : rows=769      lt=766       kcn=1
  hash  inner : rows=1        kcn=256       contains=0 getitem=0
  hash  left  : rows=257      kcn=256       contains=0 getitem=0
  costs: sort(both) 0.000s/0.0MB | join mem 0.000s/0.0MB | join presorted(pairing only) 0.000s/0.0MB | join spill 0.000s/0.0MB
         hash index 0.000s/0.1MB | hashjoin 0.000s/0.1MB | hash probe share 0.000s

K=4096   N+2N rows=12288    hypothetical N*M pairs=33554432
  merge inner : rows=1        lt=12286     kcn=1
  merge outer : rows=12289    lt=12286     kcn=1
  hash  inner : rows=1        kcn=4096      contains=0 getitem=0
  hash  left  : rows=4097     kcn=4096      contains=0 getitem=0
  costs: sort(both) 0.002s/0.6MB | join mem 0.003s/0.6MB | join presorted(pairing only) 0.001s/0.0MB | join spill 0.008s/0.1MB
         hash index 0.001s/1.3MB | hashjoin 0.001s/1.3MB | hash probe share 0.001s

K=65536  N+2N rows=196608   hypothetical N*M pairs=8589934592
  merge inner : rows=1        lt=196605    kcn=1
  merge outer : rows=196609   lt=196605    kcn=1
  hash  inner : rows=1        kcn=65536     contains=0 getitem=0
  hash  left  : rows=65537    kcn=65536     contains=0 getitem=0
  costs: sort(both) 0.144s/7.2MB | join mem 0.082s/7.8MB | join presorted(pairing only) 0.013s/0.0MB | join spill 0.172s/1.1MB
         hash index 0.019s/21.0MB | hashjoin 0.030s/21.0MB | hash probe share 0.011s

control: finite duplicate keys 1 x K x 2K -> 131072 pairs (cartesian product preserved)
```

Reading: merge-join NaN decisions are O(1) per side (one co-equal group),
hash guards are exactly K with **zero** dict probes, inner joins emit zero
pairs, outer joins emit exactly 3K rows; the pure pairing phase
(presorted=True) is 0.013s at K=65536 and the hash probe share 0.011s —
both linear. Sorting dominates the merge join (0.144s standalone for both
tables; the inner join total is lower because the abandoned side's merge
read-back is cut short by laziness); the hash index build (O(M), 21MB for
131k entries) dominates the hash join. `lt` counts are ~3K because an
all-co-equal input sorts in n−1 comparisons (single run); mixed data is
the usual O(n log n).

## Reproduction commands

```bash
# new NaN suites
python3 -m pytest petl/test/test_comparison_nan.py \
  petl/test/transform/test_sorts_nan.py \
  petl/test/transform/test_joins_nan.py -q

# required existing suites: comparison, sorts, joins (merge + hash),
# lookups, Table wiring, MISSING + CSV checkpoint regressions
python3 -m pytest petl/test/test_comparison.py \
  petl/test/transform/test_sorts.py petl/test/transform/test_sorts_missing.py \
  petl/test/transform/test_joins.py petl/test/transform/test_joins_missing.py \
  petl/test/util/test_lookups.py petl/test/test_fluent.py \
  petl/test/test_method_shadow.py petl/test/test_missing.py \
  petl/test/io/test_json_missing.py petl/test/io/test_csvcheckpoint.py \
  petl/test/io/test_csv.py petl/test/io/test_csv_unicode.py -q

# full suite (DB suites excluded: need optional deps)
python3 -m pytest petl/test -q \
  --ignore=petl/test/io/test_db.py \
  --ignore=petl/test/io/test_db_server.py \
  --ignore=petl/test/io/test_sqlalchemy.py

# doctests of changed modules
python3 -m pytest --doctest-modules petl/comparison.py \
  petl/transform/joins.py petl/transform/hashjoins.py \
  petl/transform/sorts.py petl/missing.py -q

# scale probe
PYTHONPATH=. python3 scratch_nan/scale_probe_nan.py

# pre-existing MISSING probes still pass
PYTHONPATH=. python3 scratch_missing/e2e_demo.py
PYTHONPATH=. python3 scratch_missing/scale_probe.py
```

## Raw results (this environment, Python 3.11.2)

* new suites: `47 passed in 0.09s`
* required existing suites listed above: `357 passed`
* full suite: `1430 passed, 21 skipped` — all 21 skips are missing
  optional dependencies (fastavro, bcolz, sqlalchemy, numpy, pandas,
  gspread, tables, whoosh, xlrd, openpyxl, intervaltree, pymysql,
  psycopg2, fsspec, smbclient); no failures.
* doctests of changed modules: `12 passed`
* MISSING probes: `ALL END-TO-END CHECKS PASSED`, `SCALE PROBES COMPLETE`

## Baseline failures and unverified parts

* Baseline (pre-fix) had no failing tests in the runnable scope; the same
  21 dependency skips apply. The pre-fix defects were behavioral (wrong
  join results), not test failures — they are pinned by the repro test.
* Not verified here: DB-backed tests (test_db, test_db_server,
  test_sqlalchemy) and optional-dependency suites listed above; signaling
  NaN semantics (explicitly out of scope); `dedup`/`lookups` NaN-key
  behavior, which intentionally keeps its pre-existing raw-key semantics
  (not part of the join/sort contract fixed here); Sphinx doc build (no
  sphinx in this environment — `docs/nan.rst` was validated by review
  only).
