from __future__ import absolute_import, division, print_function

import operator

from petl.compat import next, text_type
from petl.comparison import key_contains_nan
from petl.missing import key_contains_missing
from petl.transform.basics import stack
from petl.transform.joins import keys_from_args, _nonkey_indices
from petl.util.base import Table, asindices, iterpeek, rowgetter


def _hash_lookup(table, key):
    """Build a right-table lookup keyed on raw (unwrapped) row keys.

    Mirrors :func:`petl.util.lookups.lookup` but lives here so the hash joins
    can keep a single internal construction path as the MISSING contract
    evolves. Rows are grouped under their raw tuple/scalar key, which keeps
    :data:`petl.MISSING` identity intact through hashing.
    """
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    kindices = asindices(hdr, key)
    getkey = operator.itemgetter(*kindices)
    getvalue = rowgetter(*range(len(hdr)))
    dictionary = dict()
    for row in it:
        k = getkey(row)
        v = getvalue(row)
        if k in dictionary:
            dictionary[k].append(v)
        else:
            dictionary[k] = [v]
    return dictionary


def _hash_lookupone(table, key):
    """First-wins variant of :func:`_hash_lookup`."""
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    kindices = asindices(hdr, key)
    getkey = operator.itemgetter(*kindices)
    getvalue = rowgetter(*range(len(hdr)))
    dictionary = dict()
    for row in it:
        k = getkey(row)
        if k not in dictionary:
            dictionary[k] = getvalue(row)
    return dictionary


def hashjoin(left, right, key=None, lkey=None, rkey=None, cache=True,
             lprefix=None, rprefix=None, missing=None, match_missing=False):
    """Alternative implementation of :func:`petl.transform.joins.join`, where
    the join is executed by constructing an in-memory lookup for the right
    hand table, then iterating over rows from the left hand table.
    
    May be faster and/or more resource efficient where the right table is small
    and the left table is large.
    
    By default data from right hand table is cached to improve performance
    (only available when `key` is given).

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    `match_missing` has the same meaning as for
    :func:`petl.transform.joins.join`: by default a key containing
    :data:`petl.MISSING` never matches anything, while ``True`` makes
    MISSING equal only to MISSING in the same key position. ``None`` keys
    are unaffected either way.

    A key containing a NaN value (float NaN or Decimal quiet NaN, in any
    key component) never matches anything - not even the identical NaN
    object - regardless of `match_missing`; such rows are dropped by the
    inner join and padded by the outer variants without ever probing the
    lookup. NaN values in non-key fields do not affect matching.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up and rows will be filled with the value if `missing` keyword
        argument before joining to ensure correct results.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return HashJoinView(left, right, lkey=lkey, rkey=rkey, cache=cache,
                        lprefix=lprefix, rprefix=rprefix, missing=missing,
                        match_missing=match_missing)


Table.hashjoin = hashjoin


class HashJoinView(Table):

    def __init__(self, left, right, lkey, rkey, cache=True, lprefix=None,
                 rprefix=None, missing=None, match_missing=False):
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        self.lkey = lkey
        self.rkey = rkey
        self._cache = cache
        self.rlookup = None
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_missing = match_missing

    def __iter__(self):
        if not self._cache or self.rlookup is None:
            self.rlookup = _hash_lookup(self.right, self.rkey)
        return iterhashjoin(self.left, self.right, self.lkey, self.rkey,
                            self.rlookup, self.lprefix, self.rprefix,
                            match_missing=self.match_missing)


def iterhashjoin(left, right, lkey, rkey, rlookup, lprefix, rprefix,
                 match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)
    
    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    
    # construct functions to extract key values from left table
    lgetk = operator.itemgetter(*lkind)
    
    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)
    
    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f))
                  for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f)) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join rows
    def joinrows(_lrow, _rrows):
        for rrow in _rrows:
            # start with the left row
            _outrow = list(_lrow)
            # extend with non-key values from the right row
            _outrow.extend(rgetv(rrow))
            yield tuple(_outrow)

    for lrow in lit:
        k = lgetk(lrow)
        # NaN keys never match anything - not even the identical NaN
        # object, which a dict probe would find via the interpreter's
        # identity fast path - and MISSING components are unmatchable
        # unless explicitly enabled; skip the probe entirely instead of
        # risking a hash collision or identity match.
        if key_contains_nan(k) \
                or (not match_missing and key_contains_missing(k)):
            continue
        if k in rlookup:
            rrows = rlookup[k]
            for outrow in joinrows(lrow, rrows):
                yield outrow


def hashleftjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
                 cache=True, lprefix=None, rprefix=None,
                 match_missing=False):
    """Alternative implementation of :func:`petl.transform.joins.leftjoin`,
    where the join is executed by constructing an in-memory lookup for the
    right hand table, then iterating over rows from the left hand table.
    
    May be faster and/or more resource efficient where the right table is small
    and the left table is large.
    
    By default data from right hand table is cached to improve performance
    (only available when `key` is given).

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return HashLeftJoinView(left, right, lkey, rkey, missing=missing,
                            cache=cache, lprefix=lprefix, rprefix=rprefix,
                            match_missing=match_missing)


Table.hashleftjoin = hashleftjoin


class HashLeftJoinView(Table):

    def __init__(self, left, right, lkey, rkey, missing=None, cache=True,
                 lprefix=None, rprefix=None, match_missing=False):
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self._cache = cache
        self.rlookup = None
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_missing = match_missing

    def __iter__(self):
        if not self._cache or self.rlookup is None:
            self.rlookup = _hash_lookup(self.right, self.rkey)
        return iterhashleftjoin(self.left, self.right, self.lkey, self.rkey,
                                self.missing, self.rlookup, self.lprefix,
                                self.rprefix,
                                match_missing=self.match_missing)


def iterhashleftjoin(left, right, lkey, rkey, missing, rlookup, lprefix,
                     rprefix, match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)
    
    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    
    # construct functions to extract key values from left table
    lgetk = operator.itemgetter(*lkind)
    
    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)
    
    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f))
                  for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f)) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join rows
    def joinrows(_lrow, _rrows):
        for rrow in _rrows:
            # start with the left row
            _outrow = list(_lrow)
            # extend with non-key values from the right row
            _outrow.extend(rgetv(rrow))
            yield tuple(_outrow)

    for lrow in lit:
        k = lgetk(lrow)
        # without match_missing a MISSING-bearing key is always unmatched,
        # and a NaN-bearing key is always unmatched regardless, so emit
        # the left-padded outer row directly (no lookup probe)
        if key_contains_nan(k) \
                or (not match_missing and key_contains_missing(k)) \
                or k not in rlookup:
            outrow = list(lrow)  # start with the left row
            # extend with missing values in place of the right row
            outrow.extend([missing] * len(rvind))
            yield tuple(outrow)
        else:
            rrows = rlookup[k]
            for outrow in joinrows(lrow, rrows):
                yield outrow
        
        
def hashrightjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
                  cache=True, lprefix=None, rprefix=None,
                  match_missing=False):
    """Alternative implementation of :func:`petl.transform.joins.rightjoin`,
    where the join is executed by constructing an in-memory lookup for the
    left hand table, then iterating over rows from the right hand table.
    
    May be faster and/or more resource efficient where the left table is small
    and the right table is large.
    
    By default data from right hand table is cached to improve performance
    (only available when `key` is given).

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return HashRightJoinView(left, right, lkey, rkey, missing=missing,
                             cache=cache, lprefix=lprefix, rprefix=rprefix,
                             match_missing=match_missing)


Table.hashrightjoin = hashrightjoin


class HashRightJoinView(Table):

    def __init__(self, left, right, lkey, rkey, missing=None, cache=True,
                 lprefix=None, rprefix=None, match_missing=False):
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self._cache = cache
        self.llookup = None
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_missing = match_missing

    def __iter__(self):
        if not self._cache or self.llookup is None:
            self.llookup = _hash_lookup(self.left, self.lkey)
        return iterhashrightjoin(self.left, self.right, self.lkey, self.rkey,
                                 self.missing, self.llookup, self.lprefix,
                                 self.rprefix,
                                 match_missing=self.match_missing)


def iterhashrightjoin(left, right, lkey, rkey, missing, llookup, lprefix,
                      rprefix, match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)
    
    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    
    # construct functions to extract key values from left table
    rgetk = operator.itemgetter(*rkind)
    
    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)
    
    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f))
                  for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f))
                       for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join rows
    def joinrows(_rrow, _lrows):
        for lrow in _lrows:
            # start with the left row
            _outrow = list(lrow)
            # extend with non-key values from the right row
            _outrow.extend(rgetv(_rrow))
            yield tuple(_outrow)

    for rrow in rit:
        k = rgetk(rrow)
        # a NaN-bearing right key can never match, and a MISSING-bearing
        # right key can only match when match_missing is on; otherwise
        # the row is emitted as a right-only outer row and its key cells
        # (NaN/MISSING included) are back-filled unchanged
        if key_contains_nan(k) \
                or (not match_missing and key_contains_missing(k)) \
                or k not in llookup:
            # start with missing values in place of the left row
            outrow = [missing] * len(lhdr)
            # set key values
            for li, ri in zip(lkind, rkind):
                outrow[li] = rrow[ri]
            # extend with non-key values from the right row
            outrow.extend(rgetv(rrow))
            yield tuple(outrow)
        else:
            lrows = llookup[k]
            for outrow in joinrows(rrow, lrows):
                yield outrow
        
        
def hashantijoin(left, right, key=None, lkey=None, rkey=None,
                 match_missing=False):
    """Alternative implementation of :func:`petl.transform.joins.antijoin`,
    where the join is executed by constructing an in-memory set for all keys
    found in the right hand table, then iterating over rows from the left
    hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    With `match_missing=False` (the default), left rows whose key contains a
    :data:`petl.MISSING` component are always retained, as they cannot match
    anything.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return HashAntiJoinView(left, right, lkey, rkey,
                            match_missing=match_missing)


Table.hashantijoin = hashantijoin


class HashAntiJoinView(Table):

    def __init__(self, left, right, lkey, rkey, match_missing=False):
        self.left = left
        self.right = right
        self.lkey = lkey
        self.rkey = rkey
        self.match_missing = match_missing

    def __iter__(self):
        return iterhashantijoin(self.left, self.right, self.lkey, self.rkey,
                                match_missing=self.match_missing)


def iterhashantijoin(left, right, lkey, rkey, match_missing=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)
    yield tuple(lhdr)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = operator.itemgetter(*lkind)
    rgetk = operator.itemgetter(*rkind)

    rkeys = set()
    for rrow in rit:
        rk = rgetk(rrow)
        rkeys.add(rk)

    for lrow in lit:
        lk = lgetk(lrow)
        if key_contains_nan(lk):
            # NaN keys are unmatchable by contract, so the anti-join
            # keeps the row (no probe against the right key set)
            yield tuple(lrow)
        elif not match_missing and key_contains_missing(lk):
            # unmatchable by contract, so the anti-join keeps the row
            yield tuple(lrow)
        elif lk not in rkeys:
            yield tuple(lrow)


def hashlookupjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
                   lprefix=None, rprefix=None, match_missing=False):
    """Alternative implementation of :func:`petl.transform.joins.lookupjoin`,
    where the join is executed by constructing an in-memory lookup for the
    right hand table, then iterating over rows from the left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return HashLookupJoinView(left, right, lkey, rkey, missing=missing,
                              lprefix=lprefix, rprefix=rprefix,
                              match_missing=match_missing)


Table.hashlookupjoin = hashlookupjoin


class HashLookupJoinView(Table):

    def __init__(self, left, right, lkey, rkey, missing=None, lprefix=None,
                 rprefix=None, match_missing=False):
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_missing = match_missing

    def __iter__(self):
        return iterhashlookupjoin(self.left, self.right, self.lkey, self.rkey,
                                  self.missing, self.lprefix, self.rprefix,
                                  match_missing=self.match_missing)


def iterhashlookupjoin(left, right, lkey, rkey, missing, lprefix, rprefix,
                       match_missing=False):
    lit = iter(left)
    lhdr = next(lit)

    rhdr, rit = iterpeek(right)  # need the whole lot to pass to lookup
    rlookup = _hash_lookupone(rit, rkey)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from left table
    lgetk = operator.itemgetter(*lkind)

    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)

    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f))
                  for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f))
                       for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join rows
    def joinrows(_lrow, _rrow):
        # start with the left row
        _outrow = list(_lrow)
        # extend with non-key values from the right row
        _outrow.extend(rgetv(_rrow))
        return tuple(_outrow)

    for lrow in lit:
        k = lgetk(lrow)
        if key_contains_nan(k) \
                or (not match_missing and key_contains_missing(k)) \
                or k not in rlookup:
            outrow = list(lrow)  # start with the left row
            # extend with missing values in place of the right row
            outrow.extend([missing] * len(rvind))
            yield tuple(outrow)
        else:
            rrow = rlookup[k]
            yield joinrows(lrow, rrow)
