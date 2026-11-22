import sys
sys.path.insert(0, '/workspace')
import math, decimal
from decimal import Decimal, localcontext
import petl as etl
from petl.comparison import Comparable, _isnan, key_contains_nan

# 1. spill-path stability with interleaved NaN keys (buffersize=2 forces chunks)
n1, n2, n3, n4 = float('nan'), float('nan'), float('nan'), float('nan')
tbl = [('k', 'v'), (n1, 'a'), (1, 'x'), (n2, 'b'), (2, 'y'), (n3, 'c'), (n4, 'd'), (1, 'z')]
out = list(etl.sort(tbl, 'k', buffersize=2, cache=False))
print("spill sort:", [(('nan' if _isnan(k) else k), v) for k, v in out[1:]])
# expect NaN bucket a,b,c,d in input order, then 1:x,z stable, then 2

# 2. reverse stability, also spilled
out_r = list(etl.sort(tbl, 'k', buffersize=2, cache=False, reverse=True))
print("reverse spill:", [(('nan' if _isnan(k) else k), v) for k, v in out_r[1:]])

# 3. compound keys, NaN in 2nd position
t2 = [('a', 'b', 'v'), (1, n1, 'p'), (1, 5, 'q'), (0, n2, 'r'), (1, n3, 's')]
out2 = list(etl.sort(t2, key=('a', 'b'), buffersize=2, cache=False))
print("compound:", [(a, 'nan' if _isnan(b) else b, v) for a, b, v in out2[1:]])

# 4. Decimal context flags untouched by sort + join
dl = [('k', 'v'), (Decimal('NaN'), 'LN'), (Decimal('1'), 'L1'), (Decimal('2'), 'L2')]
dr = [('k', 'w'), (Decimal('1'), 'R1'), (Decimal('-NaN7'), 'RN'), (Decimal('2'), 'R2')]
ctx = decimal.getcontext()
for trap in (True, False):
    with localcontext() as lc:
        lc.traps[decimal.InvalidOperation] = trap
        lc.clear_flags()
        j = list(etl.join(dl, dr, key='k', buffersize=2, cache=False))
        s = list(etl.sort(dl, 'k', buffersize=2, cache=False))
        flags_after = {k.__name__: v for k, v in lc.flags.items() if v}
        print(f"trap={trap} join_rows={len(j)-1} sort_rows={len(s)-1} flags_set={flags_after}")
        assert not flags_after, flags_after
print("caller ctx flags still:", {k.__name__: v for k, v in ctx.flags.items() if v})

# 5. groupby co-equal NaN: duplicates() style sanity via rowgroupby not needed; check issorted
print("issorted:", etl.issorted(out, 'k'), etl.issorted(out_r, 'k', reverse=True))

# 6. hashjoin with same object reused both sides + compound
n = float('nan')
l3 = [('k', 'v'), (n, 'L'), (1, 'L1')]
r3 = [('k', 'w'), (n, 'R'), (1, 'R1')]
print("hashjoin same-obj:", list(etl.hashjoin(l3, r3, key='k', cache=False))[1:])
l4 = [('a', 'b', 'v'), (1, n, 'L'), (1, 2, 'L1')]
r4 = [('a', 'b', 'w'), (1, n, 'R'), (1, 2, 'R1')]
print("hashjoin compound same-obj:", list(etl.hashjoin(l4, r4, key=('a', 'b'), cache=False))[1:])
print("mergejoin compound same-obj:", list(etl.join(l4, r4, key=('a', 'b'), buffersize=2, cache=False))[1:])
