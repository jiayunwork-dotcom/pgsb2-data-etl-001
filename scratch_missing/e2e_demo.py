"""End-to-end demonstration of the petl.MISSING field-presence contract.

Scenario
--------
An upstream observation feed yields sparse Python objects: some sensors omit
fields entirely (MISSING), others report an explicit ``null`` (``None``).
A small dimension table describes the sensors, and is itself incomplete.

The pipeline is:

    objects -> fromdicts(preserve_missing) -> cut -> sort
            -> leftjoin / outerjoin (sorted and hash paths)
            -> tojson(omit_missing) -> fromjson(preserve_missing)
            -> cell-by-cell existence and value verification

Run from the repository root::

    PYTHONPATH=. python3 scratch_missing/e2e_demo.py
"""

from __future__ import print_function

import json
import os
import tempfile

import petl as etl
from petl import MISSING


# --------------------------------------------------------------------------
# 1. Sparse "observation detail" objects straight from an upstream system
# --------------------------------------------------------------------------
observations = [
    # sensor_id present, temperature reported explicitly
    {'sensor_id': 's1', 'temperature': 21.5, 'note': 'ok'},
    # explicit null temperature: the field exists, its value is None
    {'sensor_id': 's2', 'temperature': None, 'note': 'null-sensor'},
    # temperature key absent entirely: must stay distinguishable from None
    {'sensor_id': 's3', 'note': 'no-temp-field'},
    # sensor_id key absent
    {'temperature': 18.0, 'note': 'anonymous-reading'},
    # both optional fields absent
    {'sensor_id': 's5'},
    # literal string "MISSING" must never be mistaken for the marker
    {'sensor_id': 's6', 'temperature': 'MISSING', 'note': 'string-value'},
]

dimension = [
    {'sensor_id': 's1', 'site': 'north'},
    {'sensor_id': 's2', 'site': 'south'},
    # s3 has no dimension row at all
    {'sensor_id': 's5', 'site': None},     # explicit null site
    {'site': 'roaming'},                   # dimension row missing its key
]


def banner(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    workdir = tempfile.mkdtemp(prefix='petl-missing-demo-')
    print('working directory:', workdir)

    # ------------------------------------------------------------------
    # 2. Read objects while preserving field presence
    # ------------------------------------------------------------------
    banner('2. sparse read: absent key -> MISSING, explicit null -> None')
    obs = etl.fromdicts(
        observations,
        header=['sensor_id', 'temperature', 'note'],
        preserve_missing=True,
    )
    dim = etl.fromdicts(
        dimension,
        header=['sensor_id', 'site'],
        preserve_missing=True,
    )
    print(obs.lookall())

    # Table's ordinary access methods still work; rows are plain tuples
    assert isinstance(next(iter(obs)), tuple)
    assert obs.header() == ('sensor_id', 'temperature', 'note')

    # ------------------------------------------------------------------
    # 3. Select columns and sort: MISSING sorts before None
    # ------------------------------------------------------------------
    banner('3. cut + sort (MISSING precedes None, stable order)')
    obs_sel = obs.cut('sensor_id', 'temperature')
    obs_sorted = obs_sel.sort('sensor_id')
    print(obs_sorted.lookall())
    ordered_keys = [r[0] for r in etl.data(obs_sorted)]
    assert ordered_keys[0] is MISSING          # anonymous row first
    assert MISSING in ordered_keys
    assert None not in ordered_keys            # no None key here, only MISSING

    # ------------------------------------------------------------------
    # 4. Relational joins, both execution paths, both match_missing modes
    # ------------------------------------------------------------------
    banner('4. joins: sorted vs hash, match_missing on/off')
    for mm in (False, True):
        lj = etl.leftjoin(obs_sorted, dim, key='sensor_id',
                          match_missing=mm)
        hl = etl.hashleftjoin(obs_sorted, dim, key='sensor_id',
                              match_missing=mm)
        assert etl.util.materialise.tupleoftuples(lj) == \
            etl.util.materialise.tupleoftuples(hl)
        print('--- match_missing=%s (left join, both paths identical) ---' % mm)
        print(lj.lookall())

    # full outer join with the fill side explicitly using the marker itself
    banner('5. outer join: missing=MISSING keeps absence through back-fill')
    # match_missing=True here: the two MISSING-key rows pair positionally
    outer = etl.outerjoin(obs_sorted, dim, key='sensor_id',
                          missing=MISSING, match_missing=True)
    print(outer.lookall())

    # ------------------------------------------------------------------
    # 6. Export both JSON layouts, omitting only MISSING cells
    # ------------------------------------------------------------------
    banner('6. export JSON (array and lines); None -> null, MISSING -> omit')
    arr_path = os.path.join(workdir, 'observations.json')
    lines_path = os.path.join(workdir, 'observations.jsonl')
    etl.tojson(outer, arr_path, omit_missing=True, sort_keys=True)
    etl.tojson(outer, lines_path, omit_missing=True, lines=True,
               sort_keys=True)
    raw_arr = json.load(open(arr_path))
    print('array file key sets per row:')
    for obj in raw_arr:
        print('  ', sorted(obj))
    print('\nfirst JSON lines record:')
    print(open(lines_path).readline().strip())

    # explicit None must survive on the wire as null
    s2_rows = [o for o in raw_arr if o.get('sensor_id') == 's2']
    assert s2_rows and s2_rows[0]['temperature'] is None
    # the MISSING temperature key must be absent, not null
    s3_rows = [o for o in raw_arr if o.get('sensor_id') == 's3']
    assert s3_rows and 'temperature' not in s3_rows[0]
    # the string "MISSING" survives as an ordinary string value
    s6_rows = [o for o in raw_arr if o.get('sensor_id') == 's6']
    assert s6_rows and s6_rows[0]['temperature'] == 'MISSING'

    # ------------------------------------------------------------------
    # 7. Read the exports back and verify every cell's presence + value
    # ------------------------------------------------------------------
    banner('7. re-read and verify presence and values cell by cell')
    back_arr = etl.fromjson(arr_path,
                            header=['sensor_id', 'temperature', 'site'],
                            preserve_missing=True)
    back_lines = etl.fromjson(lines_path, lines=True,
                              header=['sensor_id', 'temperature', 'site'],
                              preserve_missing=True)

    def cell_map(view):
        return {
            (r[0] if r[0] is not MISSING else '<absent>'):
                {'temperature_present': r[1] is not MISSING,
                 'temperature': r[1],
                 'site_present': r[2] is not MISSING,
                 'site': r[2]}
            for r in etl.data(view)
        }

    from_arr = cell_map(back_arr)
    from_lines = cell_map(back_lines)
    assert from_arr == from_lines
    for label, cells in sorted(from_arr.items(), key=lambda kv: str(kv[0])):
        print('%-10r %s' % (label, cells))

    # precise assertions on the re-read (array) content
    # the anonymous reading (MISSING key) matched the roaming dimension row
    # (also MISSING key) because this outer join used match_missing=True
    assert from_arr['<absent>']['temperature_present'] is True
    assert from_arr['<absent>']['temperature'] == 18.0
    assert from_arr['<absent>']['site_present'] is True
    assert from_arr['<absent>']['site'] == 'roaming'

    assert from_arr['s3']['temperature_present'] is False
    assert from_arr['s3']['temperature'] is MISSING
    assert from_arr['s3']['site_present'] is False      # unmatched dimension

    assert from_arr['s2']['temperature_present'] is True
    assert from_arr['s2']['temperature'] is None         # explicit null
    assert from_arr['s2']['site'] == 'south'

    assert from_arr['s5']['site_present'] is True
    assert from_arr['s5']['site'] is None                # explicit null site

    assert from_arr['s6']['temperature'] == 'MISSING'    # string, not marker

    # and demonstrate the default contract: with match_missing=False the two
    # MISSING-key rows must never pair, even sitting in the same key group
    banner('7b. default match_missing=False: MISSING keys never pair')
    outer_default = etl.outerjoin(
        obs_sorted, dim, key='sensor_id', missing=MISSING,
        match_missing=False,
    )
    default_rows = [
        r for r in etl.data(outer_default) if r[0] is MISSING
    ]
    # one left-only row (roaming site cell filled) and one right-only row
    # (temperature kept, site from dimension), never a merged row
    assert len(default_rows) == 2
    signatures = {tuple('MISSING' if c is MISSING else c for c in r)
                  for r in default_rows}
    assert (MISSING.__str__(), 18.0, 'MISSING') in signatures
    assert (MISSING.__str__(), 'MISSING', 'roaming') in signatures
    print(outer_default.lookall())

    # ------------------------------------------------------------------
    # 8. The non-omitting writer refuses MISSING instead of hiding it
    # ------------------------------------------------------------------
    banner('8. default JSON writer raises TypeError on MISSING')
    try:
        etl.tojson(outer, os.path.join(workdir, 'should-not-exist.json'))
    except TypeError as exc:
        print('got expected TypeError:', exc)
    else:
        raise AssertionError('MISSING was silently encoded')

    print('\nALL END-TO-END CHECKS PASSED')
    print('artifacts left in:', workdir)


if __name__ == '__main__':
    main()
