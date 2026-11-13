"""Minimal pytest-compatible runner for environments without pytest.

Discovers ``test_*`` functions in the given modules/directories, runs them
and reports failures. Supports the small subset of pytest features used by
the petl test suite (plain functions, parametrize, simple assertions and
raises helpers).
"""
import importlib
import inspect
import logging
import os
import shutil
import tempfile
import sys
import traceback
import unittest

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tmp_scripts')


class _TmpPath:
    """Minimal py.path.local stand-in."""

    def __init__(self, path):
        self.path = path

    def join(self, *parts):
        return _TmpPath(os.path.join(self.path, *parts))

    def mkdir(self, *parts):
        p = os.path.join(self.path, *parts)
        os.makedirs(p, exist_ok=True)
        return _TmpPath(p)

    def write(self, content=''):
        with open(self.path, 'w') as f:
            f.write(content)

    def read(self):
        with open(self.path) as f:
            return f.read()

    def open(self, *args, **kwargs):
        return open(self.path, *args, **kwargs)

    def __str__(self):
        return self.path

    def __fspath__(self):
        return self.path


def _resolve_fixture(name, module, stack=None):
    """Resolve a fixture by name, recursively resolving any fixtures it
    depends on (function-scoped, so a fresh chain per test case)."""
    if stack is None:
        stack = set()
    if name == 'tmp_root':
        d = tempfile.mkdtemp()
        root = os.path.join(d, 'dataset')
        return root, lambda: shutil.rmtree(d, ignore_errors=True)
    if name == 'tmpdir':
        d = tempfile.mkdtemp()
        return _TmpPath(d), lambda: shutil.rmtree(d, ignore_errors=True)
    if name == 'tmp_path':
        d = tempfile.mkdtemp()
        return d, lambda: shutil.rmtree(d, ignore_errors=True)
    if name == 'monkeypatch':
        saved = []

        class Monkey:
            def setattr(self, target, attr_or_value, value=None):
                if isinstance(target, str):
                    module_path, _, attr = target.rpartition('.')
                    obj = importlib.import_module(module_path)
                    self.setattr(obj, attr, attr_or_value)
                    return
                obj, attr, value = target, attr_or_value, value
                old = getattr(obj, attr, None)
                saved.append((obj, attr, old, hasattr(obj, attr)))
                setattr(obj, attr, value)

            def delattr(self, obj, attr):
                saved.append((obj, attr, None, True))
                delattr(obj, attr)

            def chdir(self, path):
                cwd = os.getcwd()
                os.chdir(str(path))
                saved.append(('chdir', cwd))

            def setenv(self, name, value, prepend=None):
                old = os.environ.get(name)
                saved.append(('env', name, old, old is not None))
                os.environ[name] = value

            def delenv(self, name, raising=True):
                old = os.environ.get(name)
                saved.append(('env', name, old, False))
                os.environ.pop(name, None)

            def context(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.undo()
                return False

            def undo(self):
                while saved:
                    item = saved.pop()
                    if item[0] == 'chdir':
                        os.chdir(item[1])
                    elif item[0] == 'env':
                        _, name, old, existed = item
                        if existed:
                            os.environ[name] = old
                        else:
                            os.environ.pop(name, None)
                    else:
                        obj, attr, old, existed = item
                        if existed:
                            setattr(obj, attr, old)
                        else:
                            delattr(obj, attr)
        m = Monkey()
        return m, m.undo
    if name in ('capfd', 'capsys'):
        import types as types_module
        captured = {}

        class Cap:
            def readouterr(self):
                out = captured.get('out', '')
                err = captured.get('err', '')
                captured['out'] = ''
                captured['err'] = ''
                return types_module.SimpleNamespace(out=out, err=err)
        return Cap(), lambda: None
    if name == 'request':
        import types as types_module
        req = types_module.SimpleNamespace(
            param=None,
            node=types_module.SimpleNamespace(callspec=None),
            config=types_module.SimpleNamespace(
                getini=lambda *_a, **_k: ''))
        return req, lambda: None
    # fixtures defined locally in the test module
    if module is not None and hasattr(module, name):
        obj = getattr(module, name)
        if callable(obj):
            deps = [p for p in inspect.signature(obj).parameters]
            kwargs = {}
            cleanups = []
            for dep in deps:
                value, cleanup = _resolve_fixture(dep, module, stack)
                kwargs[dep] = value
                cleanups.append(cleanup)
            if inspect.isgeneratorfunction(obj):
                gen = obj(**kwargs)
                value = next(gen)

                def cleanup_gen(gen=gen, cleanups=cleanups):
                    try:
                        next(gen)
                    except StopIteration:
                        pass
                    for c in reversed(cleanups):
                        c()
                return value, cleanup_gen
            return obj(**kwargs), (lambda cleanups=cleanups:
                                   [c() for c in reversed(cleanups)])
    raise ValueError('unknown fixture %s' % name)


# Backwards-compatible alias used by earlier call sites
_fixture = _resolve_fixture


def _pytest_configure():
    # mirror petl/test/conftest.py
    org = logging.Logger.debug

    def debug(self, msg, *args, **kwargs):
        org(self, str(msg), *args, **kwargs)

    logging.Logger.debug = debug


def iter_test_modules(paths):
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _dirs, files in os.walk(path):
                for name in sorted(files):
                    if not (name.startswith('test_')
                            and name.endswith('.py')):
                        continue
                    pkg = os.path.relpath(dirpath, '/workspace').replace(
                        os.sep, '.')
                    try:
                        yield importlib.import_module(
                            pkg + '.' + name[:-3])
                    except unittest.SkipTest:
                        print('skipped module',
                              pkg + '.' + name[:-3])
                        continue
        else:
            mod = os.path.splitext(os.path.basename(path))[0]
            pkg = os.path.relpath(os.path.dirname(path),
                                  '/workspace').replace(os.sep, '.')
            try:
                yield importlib.import_module(pkg + '.' + mod)
            except unittest.SkipTest:
                print('skipped module', pkg + '.' + mod)
                continue


def main(paths):
    _pytest_configure()
    failures = []
    passed = 0
    for module in iter_test_modules(paths):
        funcs = [getattr(module, n) for n in dir(module)
                 if n.startswith('test_') and callable(getattr(module, n))]
        for fn in funcs:
            cases = getattr(fn, '_param_cases', None) or [{}]
            params = set(inspect.signature(fn).parameters)
            for kwargs in cases:
                # fixtures are function scoped in pytest: fresh instance
                # per parametrized case
                fixtures = {}
                cleanups = []
                for name in params - set(kwargs):
                    value, cleanup = _fixture(name, module)
                    fixtures[name] = value
                    cleanups.append(cleanup)
                try:
                    fn(**dict(fixtures, **kwargs))
                    passed += 1
                except Exception as e:
                    if e.__class__.__name__ == 'SkipTest':
                        continue
                    failures.append(
                        (module.__name__ + '.' + fn.__name__
                         + (str(kwargs) if kwargs else ''),
                         traceback.format_exc()))
                finally:
                    for cleanup in reversed(cleanups):
                        try:
                            cleanup()
                        except Exception:
                            pass
    print('passed=%d failed=%d' % (passed, len(failures)))
    for name, tb in failures:
        print('=' * 70)
        print('FAIL:', name)
        print(tb)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
