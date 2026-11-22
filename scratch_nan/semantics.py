import decimal, math, pickle
from decimal import Decimal, localcontext

# 1. Decimal NaN == quiet? flags?
ctx = decimal.getcontext()
ctx.clear_flags()
print("trap InvalidOperation:", ctx.traps[decimal.InvalidOperation])
try:
    r = Decimal('NaN') == Decimal('NaN')
    print("NaN==NaN ->", r, "flags:", [k.__name__ for k,v in ctx.flags.items() if v])
except Exception as e:
    print("NaN==NaN raised", e)
ctx.clear_flags()
try:
    r = Decimal('NaN') < 1
    print("NaN<1 ->", r)
except decimal.InvalidOperation as e:
    print("NaN<1 raised InvalidOperation, flags:", [k.__name__ for k,v in ctx.flags.items() if v])
ctx.clear_flags()
print("is_qnan:", Decimal('NaN').is_qnan(), "flags:", [k.__name__ for k,v in ctx.flags.items() if v])
print("sNaN is_qnan:", Decimal('sNaN').is_qnan(), "is_nan:", Decimal('sNaN').is_nan())
ctx.clear_flags()
r = Decimal('NaN') != Decimal('NaN')
print("NaN!=NaN ->", r, "flags:", [k.__name__ for k,v in ctx.flags.items() if v])

# 2. hashes / dict identity
n1 = float('nan'); n2 = float('nan')
print("hash(float nan):", hash(n1), "hash(Decimal NaN):", hash(Decimal('NaN')), "hash(-NaN dec):", hash(Decimal('-NaN')))
d = {n1: 'x'}
print("same obj lookup:", n1 in d, "diff obj lookup:", n2 in d)
t1 = (1, n1); t2 = (1, n1); t3 = (1, n2)
print("tuple eq same nan obj:", t1 == t2, "diff nan obj:", t1 == t3)

# 3. pickle round trip: float nan sign, decimal payload+sign
import struct
nf = float('nan')
nn = -float('nan')
p = pickle.loads(pickle.dumps(nn))
print("neg float nan roundtrip sign:", math.copysign(1.0, p))
dp = Decimal('-NaN123')
q = pickle.loads(pickle.dumps(dp))
print("decimal roundtrip:", q, "signed:", q.is_signed(), "payload:", q.as_tuple().digits, "qnan:", q.is_qnan())

# 4. nan != nan check on float costs no context
print("float nan != itself:", n1 != n1)

# 5. tuple __eq__ identity fast path with Decimal NaN
dn = Decimal('NaN')
print("dec tuple same obj:", (dn,) == (dn,), "diff obj:", (dn,) == (Decimal('NaN'),))

# 6. comparison Decimal NaN vs float nan (native, traps off)
with localcontext() as lc:
    lc.traps[decimal.InvalidOperation] = False
    print("traps off: decNaN < 1 ->", Decimal('NaN') < 1, "decNaN==decNaN ->", Decimal('NaN') == Decimal('NaN'))
