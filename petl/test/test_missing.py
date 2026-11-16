# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import copy
import os
import pickle
import subprocess
import sys
import textwrap

import pytest

from petl import MISSING
from petl.missing import _MissingMarker, key_contains_missing


def test_missing_public_singleton():
    import petl
    assert petl.MISSING is MISSING
    assert isinstance(MISSING, _MissingMarker)


def test_missing_distinct_from_none_strings_and_objects():
    # identity based equality: nothing else compares equal
    assert MISSING is not None
    assert not (MISSING == None)  # noqa: E711
    assert MISSING != None  # noqa: E711
    assert MISSING != 'MISSING'
    assert MISSING != ''
    assert MISSING != 0
    assert MISSING != False  # noqa: E712
    assert MISSING != object()

    class UserObject(object):
        pass

    # ordinary user objects stay distinct because the marker keeps identity
    # semantics on its own side
    assert not (MISSING == UserObject())
    assert MISSING != UserObject()
    assert UserObject() != MISSING


def test_missing_is_falsy():
    assert not MISSING
    assert bool(MISSING) is False


def test_missing_usable_as_dict_key():
    d = dict()
    d[MISSING] = 'marker'
    assert d[MISSING] == 'marker'
    # compound tuple keys, as produced by itemgetter over several fields
    compound = (MISSING, 1, None)
    d2 = {compound: 'x'}
    assert d2[(MISSING, 1, None)] == 'x'
    assert (MISSING, 1, None) != (None, 1, None)
    assert hash(MISSING) == hash(MISSING)
    assert len({MISSING, MISSING, None, 'MISSING'}) == 3


def test_missing_copy_roundtrip():
    assert copy.copy(MISSING) is MISSING
    assert copy.deepcopy(MISSING) is MISSING
    container = {'k': (1, MISSING, [MISSING, None])}
    dup = copy.deepcopy(container)
    assert dup['k'][1] is MISSING
    assert dup['k'][2][0] is MISSING
    assert dup['k'][2][1] is None


def test_missing_pickle_roundtrip_in_process():
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(MISSING, protocol=protocol))
        assert restored is MISSING
    restored = pickle.loads(pickle.dumps((MISSING, [None, MISSING])))
    assert restored[0] is MISSING
    assert restored[1][1] is MISSING
    assert restored[1][0] is None


def test_missing_pickle_roundtrip_separate_process():
    code = textwrap.dedent("""
        import pickle, sys
        from petl import MISSING
        payload = pickle.loads(bytes.fromhex(sys.argv[1]))
        assert payload is MISSING
        assert (MISSING, 1) == (payload, 1)
        print('ok')
    """)
    hex_payload = pickle.dumps(MISSING).hex()
    env = dict(os.environ)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, '-c', code, hex_payload],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == 'ok'


def test_missing_named_string_is_not_the_marker():
    # data containing the literal text "MISSING" must stay an ordinary string
    table = [('f',), ('MISSING',), (MISSING,), (None,)]
    values = [row[0] for row in table[1:]]
    assert values[0] == 'MISSING'
    assert values[0] is not MISSING
    assert values[1] is MISSING
    assert values[2] is None
    assert values[0] != values[1]


def test_missing_repr_str():
    assert repr(MISSING) == 'MISSING'
    assert str(MISSING) == 'MISSING'


def test_key_contains_missing():
    assert key_contains_missing(MISSING) is True
    assert key_contains_missing(None) is False
    assert key_contains_missing('MISSING') is False
    assert key_contains_missing((MISSING, 1)) is True
    assert key_contains_missing((1, MISSING, None)) is True
    assert key_contains_missing((1, None, 'x')) is False
    assert key_contains_missing((1, (2, 3))) is False
    assert key_contains_missing(()) is False


def test_missing_is_picklable_inside_rows_via_file(tmpdir):
    import petl as etl
    table = [('a', 'b'), (MISSING, None), (1, MISSING)]
    path = str(tmpdir.join('rows.p'))
    etl.topickle(table, path)
    restored = list(etl.frompickle(path))
    assert restored[1][0] is MISSING
    assert restored[1][1] is None
    assert restored[2][1] is MISSING


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
