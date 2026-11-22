import sys
sys.path.insert(0, '/workspace')
from decimal import Decimal
import petl as etl

n = float('nan')
left = [('k','l'),(1,'L1'),(n,'LN'),(2,'L2'),(3,'L3')]
right = [('k','r'),(1,'R1'),(2,'R2'),(3,'R3'),(n,'RN')]

print("== join (sort-merge, buffersize=2, cache=False) ==")
for row in etl.join(left, right, key='k', buffersize=2, cache=False):
    print(row)
print("== leftjoin ==")
for row in etl.leftjoin(left, right, key='k', buffersize=2, cache=False):
    print(row)
print("== outerjoin ==")
for row in etl.outerjoin(left, right, key='k', buffersize=2, cache=False):
    print(row)
print("== hashjoin (cache=False) ==")
for row in etl.hashjoin(left, right, key='k', cache=False):
    print(row)
print("== hashleftjoin (cache=False) ==")
for row in etl.hashleftjoin(left, right, key='k', cache=False):
    print(row)
print("== sort ==")
for row in etl.sort(left, 'k', buffersize=2, cache=False):
    print(row)
