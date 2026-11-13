"""Tiny pytest stand-in for running petl tests without pytest installed.

Only implements the handful of features exercised by the test modules run
in this environment (approx, raises with excinfo, skip, parametrize,
marks/fixtures).
"""
import contextlib
import unittest


class _Approx:
    def __init__(self, expected, abs=None, rel=None):
        self.expected = expected
        self.abs = abs
        self.rel = rel

    def _tol(self, expected):
        tol = self.abs
        if tol is None:
            tol = 1e-6 * max(1.0, abs(expected))
        return tol

    def __eq__(self, actual):
        if isinstance(self.expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) \
                    or len(actual) != len(self.expected):
                return False
            return all(
                abs(a - e) <= self._tol(e)
                for a, e in zip(actual, self.expected))
        return abs(actual - self.expected) <= self._tol(self.expected)

    def __repr__(self):
        return 'approx(%r)' % self.expected


def approx(expected, abs=None, rel=None):
    return _Approx(expected, abs=abs, rel=rel)


class _Raises:
    def __init__(self, exc, match=None):
        self.exc = exc
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, tb):
        if etype is None:
            raise AssertionError('expected %s' % self.exc.__name__)
        if not issubclass(etype, self.exc):
            return False
        if self.match is not None and self.match not in str(evalue):
            raise AssertionError(
                'pattern %r not found in %r' % (self.match, str(evalue)))
        self.value = evalue
        return True


def raises(exc, match=None):
    return _Raises(exc, match=match)


def skip(reason='', allow_module_level=False, **kwargs):
    raise unittest.SkipTest(reason)


def importorskip(modname, minversion=None, reason=None):
    try:
        return __import__(modname)
    except ImportError:
        raise unittest.SkipTest(
            'missing optional dependency %s' % modname)


def fail(msg=''):
    raise AssertionError(msg)


class _Parametrize:
    def __init__(self, argnames, argvalues):
        self.argnames = argnames if isinstance(argnames, str) else argnames
        self.argvalues = argvalues

    def __call__(self, fn):
        names = [n.strip() for n in self.argnames.split(',')] \
            if isinstance(self.argnames, str) else list(self.argnames)
        new_cases = []
        n_names = len(names)
        for values in self.argvalues:
            if not isinstance(values, tuple) or len(values) != n_names:
                values = (values,)
            new_cases.append(dict(zip(names, values)))
        existing = getattr(fn, '_param_cases', None)
        if existing:
            # stacked parametrize decorators: Cartesian product
            merged = []
            for old in existing:
                for new in new_cases:
                    merged.append(dict(old, **new))
            fn._param_cases = merged
        else:
            fn._param_cases = new_cases
        return fn


class _Marker:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        # supports both @pytest.mark.foo and @pytest.mark.foo(condition...)
        if args and callable(args[0]) and len(args) == 1:
            return args[0]

        def deco(obj):
            return obj
        return deco


class _Mark:
    parametrize = staticmethod(lambda argnames, argvalues:
                               _Parametrize(argnames, argvalues))

    def __getattr__(self, name):
        return _Marker(name)

    def __call__(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1:
            return args[0]

        def deco(obj):
            return obj
        return deco


mark = _Mark()


def fixture(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]

    def deco(fn):
        return fn
    return deco


class _Config:
    @staticmethod
    def getini(name):
        return ''


config = _Config()
