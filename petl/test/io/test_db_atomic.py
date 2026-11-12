# -*- coding: utf-8 -*-
"""Regression tests for atomic sqlite3 loads via todb()/appenddb().

The atomic contract is specific to the standard library sqlite3 driver, so
these tests exercise real sqlite3 connections in memory and on disk.
"""
from __future__ import absolute_import, print_function, division


import os
import sqlite3
from tempfile import NamedTemporaryFile

import pytest

import petl as etl
import petl.io.db as db
from petl.errors import ArgumentError
from petl.test.helpers import ieq


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

def make_database(autocommit=False, existing=True):
    connection = sqlite3.connect(
        ':memory:', isolation_level=None if autocommit else '')
    connection.execute('CREATE TABLE target (foo TEXT, bar INTEGER UNIQUE)')
    if existing:
        connection.executemany(
            'INSERT INTO target VALUES (?, ?)',
            [('old1', 1), ('old2', 2)])
        connection.commit()
    connection.execute('CREATE TABLE side (n INTEGER)')
    connection.commit()
    return connection


@pytest.fixture
def connection():
    connection = make_database()
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def db_file():
    f = NamedTemporaryFile(delete=False)
    f.close()
    connection = sqlite3.connect(f.name)
    connection.execute('CREATE TABLE target (foo TEXT, bar INTEGER UNIQUE)')
    connection.executemany(
        'INSERT INTO target VALUES (?, ?)', [('old1', 1), ('old2', 2)])
    connection.commit()
    connection.close()
    try:
        yield f.name
    finally:
        if os.path.exists(f.name):
            os.unlink(f.name)


OLD_ROWS = [('old1', 1), ('old2', 2)]
NEW_ROWS = (('foo', 'bar'), ('a', 10), ('b', 20))


class CountingTable(object):
    """Records how many times the source is iterated."""

    def __init__(self, rows):
        self.rows = rows
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.rows)


def rows_with_source_failure(failure):
    yield ('foo', 'bar')
    yield ('a', 10)
    raise failure


def rows_with_constraint_failure():
    yield ('foo', 'bar')
    yield ('a', 10)
    yield ('b', 10)  # duplicate bar value violates the UNIQUE constraint


def assert_old_data_intact(connection):
    ieq(OLD_ROWS, connection.execute('SELECT foo, bar FROM target '
                                     'ORDER BY bar'))


# ---------------------------------------------------------------------------
# success paths across the four supported entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('autocommit', [False, True])
@pytest.mark.parametrize('entry', ['connection', 'cursor', 'factory', 'path'])
def test_atomic_replace_success_entries(entry, autocommit, db_file):
    if entry == 'path':
        dbo = db_file
        etl.todb(NEW_ROWS, dbo, 'target', atomic=True)
        connection = sqlite3.connect(db_file)
        try:
            ieq([('a', 10), ('b', 20)],
                connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
        finally:
            connection.close()
        return

    connection = make_database(autocommit=autocommit)
    try:
        if entry == 'connection':
            dbo = connection
        elif entry == 'cursor':
            dbo = connection.cursor()
        else:
            dbo = connection.cursor
        etl.todb(NEW_ROWS, dbo, 'target', atomic=True)
        assert not connection.in_transaction
        ieq([('a', 10), ('b', 20)],
            connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
        if entry == 'cursor':
            # borrowed cursor remains usable
            assert dbo.execute('SELECT 42').fetchone() == (42,)
            dbo.close()
    finally:
        connection.close()


@pytest.mark.parametrize('entry', ['connection', 'cursor', 'factory', 'path'])
def test_atomic_append_success_entries(entry, db_file):
    if entry == 'path':
        etl.appenddb((('foo', 'bar'), ('n1', 30)), db_file, 'target',
                     atomic=True)
        connection = sqlite3.connect(db_file)
        try:
            ieq([('old1', 1), ('old2', 2), ('n1', 30)],
                connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
        finally:
            connection.close()
        return

    connection = make_database()
    try:
        dbo = {'connection': connection,
               'cursor': connection.cursor(),
               'factory': connection.cursor}[entry]
        etl.appenddb((('foo', 'bar'), ('n1', 30)), dbo, 'target',
                     atomic=True)
        ieq([('old1', 1), ('old2', 2), ('n1', 30)],
            connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
        if entry == 'cursor':
            dbo.close()
    finally:
        connection.close()


def test_atomic_chained_table_methods(connection):
    table = etl.wrap(NEW_ROWS)
    table.todb(connection, 'target', atomic=True)
    ieq([('a', 10), ('b', 20)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    etl.wrap((('foo', 'bar'), ('c', 30))).appenddb(
        connection, 'target', atomic=True)
    ieq([('a', 10), ('b', 20), ('c', 30)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))


def test_atomic_empty_rows_replaces_all(connection):
    etl.todb(iter([('foo', 'bar')]), connection, 'target', atomic=True)
    assert connection.execute('SELECT count(*) FROM target').fetchone() == (0,)
    assert not connection.in_transaction


def test_atomic_load_honours_schema_argument(connection):
    connection.execute('ATTACH DATABASE \':memory:\' AS aux')
    connection.execute('CREATE TABLE aux.scoped (a INTEGER UNIQUE)')
    connection.execute('INSERT INTO aux.scoped VALUES (1)')
    connection.commit()
    etl.todb((('a',), (2,), (3,)), connection, 'scoped', schema='aux',
             atomic=True)
    assert connection.execute('SELECT * FROM aux.scoped').fetchall() \
        == [(2,), (3,)]
    with pytest.raises(sqlite3.IntegrityError):
        etl.todb((('a',), (4,), (4,)), connection, 'scoped', schema='aux',
                 atomic=True)
    assert connection.execute('SELECT * FROM aux.scoped').fetchall() \
        == [(2,), (3,)]


def test_atomic_append_empty_rows_is_noop(connection):
    etl.appenddb(iter([('foo', 'bar')]), connection, 'target', atomic=True)
    assert_old_data_intact(connection)
    assert not connection.in_transaction


# ---------------------------------------------------------------------------
# failure paths: source errors and constraint errors restore old data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('entry', ['connection', 'cursor', 'factory'])
@pytest.mark.parametrize('replace', [True, False])
def test_atomic_source_failure_rolls_back_and_preserves_error(
        connection, entry, replace):
    failure = ValueError('source iteration failed')
    dbo = {'connection': connection,
           'cursor': connection.cursor(),
           'factory': connection.cursor}[entry]
    load = etl.todb if replace else etl.appenddb
    try:
        with pytest.raises(ValueError) as caught:
            load(rows_with_source_failure(failure), dbo, 'target',
                 atomic=True)
        assert caught.value is failure
        assert not connection.in_transaction
        assert_old_data_intact(connection)
        # connection is still usable for another batch
        etl.appenddb((('foo', 'bar'), ('retry', 30)), connection, 'target',
                     atomic=True)
        ieq(OLD_ROWS + [('retry', 30)],
            connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        if entry == 'cursor':
            dbo.close()


@pytest.mark.parametrize('entry', ['connection', 'cursor', 'factory'])
@pytest.mark.parametrize('replace', [True, False])
def test_atomic_constraint_failure_rolls_back(connection, entry, replace):
    dbo = {'connection': connection,
           'cursor': connection.cursor(),
           'factory': connection.cursor}[entry]
    load = etl.todb if replace else etl.appenddb
    try:
        with pytest.raises(sqlite3.IntegrityError):
            load(rows_with_constraint_failure(), dbo, 'target', atomic=True)
        assert not connection.in_transaction
        assert_old_data_intact(connection)
    finally:
        if entry == 'cursor':
            dbo.close()


def test_atomic_file_path_failure_leaves_committed_data(db_file):
    with pytest.raises(sqlite3.IntegrityError):
        etl.todb(rows_with_constraint_failure(), db_file, 'target',
                 atomic=True)
    connection = sqlite3.connect(db_file)
    try:
        ieq(OLD_ROWS, connection.execute(
            'SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        connection.close()


def test_repeated_failures_then_success_on_same_connection(connection):
    for _ in range(3):
        with pytest.raises(sqlite3.IntegrityError):
            etl.todb(rows_with_constraint_failure(), connection, 'target',
                     atomic=True)
        assert not connection.in_transaction
    etl.todb(NEW_ROWS, connection, 'target', atomic=True)
    ieq([('a', 10), ('b', 20)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))


# ---------------------------------------------------------------------------
# streaming: one single pass, earlier rows visible before later ones exist
# ---------------------------------------------------------------------------

def test_atomic_load_consumes_source_exactly_once(connection):
    source = CountingTable(NEW_ROWS)
    etl.todb(source, connection, 'target', atomic=True)
    assert source.iterations == 1
    source = CountingTable((('foo', 'bar'), ('c', 30)))
    etl.appenddb(source, connection, 'target', atomic=True)
    assert source.iterations == 1


def test_atomic_load_streams_rows_without_materialising(connection):
    # while later rows have not yet been produced, the earlier rows are
    # already written inside this load's transaction
    seen = {}

    def rows():
        yield ('foo', 'bar')
        yield ('first', 10)
        seen['inside_transaction'] = connection.in_transaction
        seen['rows_visible_so_far'] = connection.execute(
            'SELECT count(*) FROM target').fetchone()[0]
        seen['new_row_visible'] = connection.execute(
            'SELECT count(*) FROM target WHERE foo = ?',
            ('first',)).fetchone()[0]
        yield ('second', 20)

    etl.todb(rows(), connection, 'target', atomic=True, commit=False)
    assert seen['inside_transaction'] is True
    # truncate removed the two old rows, the first new row is already in
    assert seen['rows_visible_so_far'] == 1
    assert seen['new_row_visible'] == 1
    connection.rollback()


# ---------------------------------------------------------------------------
# caller-owned outer transactions and nested atomic loads (savepoints)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('commit', [False, True])
def test_atomic_load_inside_outer_transaction_success(connection, commit):
    connection.execute('BEGIN')
    connection.execute('INSERT INTO side VALUES (1)')
    etl.todb(NEW_ROWS, connection, 'target', atomic=True, commit=commit)
    # petl must never commit the caller's transaction, whatever commit says
    assert connection.in_transaction
    ieq([('a', 10), ('b', 20)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    assert connection.execute('SELECT * FROM side').fetchall() == [(1,)]
    connection.rollback()
    # caller rollback undoes the load as well
    assert_old_data_intact(connection)
    assert connection.execute('SELECT * FROM side').fetchall() == []


def test_atomic_load_inside_outer_transaction_failure(connection):
    connection.execute('BEGIN')
    connection.execute('INSERT INTO side VALUES (1)')
    with pytest.raises(sqlite3.IntegrityError):
        etl.todb(rows_with_constraint_failure(), connection, 'target',
                 atomic=True)
    # only the load is undone; the outer transaction and its writes remain
    assert connection.in_transaction
    assert_old_data_intact(connection)
    assert connection.execute('SELECT * FROM side').fetchall() == [(1,)]
    connection.commit()
    assert connection.execute('SELECT * FROM side').fetchall() == [(1,)]


def test_atomic_load_preserves_caller_savepoint(connection):
    connection.execute('BEGIN')
    connection.execute('SAVEPOINT caller_sp')
    connection.execute('INSERT INTO side VALUES (1)')
    with pytest.raises(sqlite3.IntegrityError):
        etl.todb(rows_with_constraint_failure(), connection, 'target',
                 atomic=True)
    assert connection.in_transaction
    # caller savepoint still works
    connection.execute('ROLLBACK TO caller_sp')
    connection.execute('RELEASE caller_sp')
    assert connection.execute('SELECT * FROM side').fetchall() == []
    connection.rollback()


def test_nested_atomic_loads_use_distinct_savepoints(connection):
    connection.execute('BEGIN')
    connection.execute('INSERT INTO side VALUES (1)')

    def outer_rows():
        yield ('foo', 'bar')
        yield ('outer1', 30)
        # inner load appends inside the outer transaction
        etl.appenddb((('foo', 'bar'), ('inner', 31)), connection, 'target',
                     atomic=True)
        yield ('outer2', 32)

    etl.todb(outer_rows(), connection, 'target', atomic=True)
    assert connection.in_transaction
    ieq([('outer1', 30), ('inner', 31), ('outer2', 32)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    connection.rollback()
    assert_old_data_intact(connection)


def test_outer_failure_undoes_successful_inner_load(connection):
    connection.execute('BEGIN')

    def outer_rows():
        yield ('foo', 'bar')
        yield ('outer1', 30)
        etl.appenddb((('foo', 'bar'), ('inner', 31)), connection, 'target',
                     atomic=True)
        yield ('outer2', 30)  # duplicate bar -> outer constraint failure

    with pytest.raises(sqlite3.IntegrityError):
        etl.todb(outer_rows(), connection, 'target', atomic=True)
    assert connection.in_transaction
    # nothing from either load survives
    assert_old_data_intact(connection)
    connection.rollback()


def test_inner_failure_does_not_roll_back_outer_writes(connection):
    connection.execute('BEGIN')
    connection.execute('INSERT INTO side VALUES (1)')

    def outer_rows():
        yield ('foo', 'bar')
        yield ('outer1', 30)
        with pytest.raises(sqlite3.IntegrityError):
            etl.appenddb(rows_with_constraint_failure(), connection, 'target',
                         atomic=True)
        yield ('outer2', 32)

    etl.todb(outer_rows(), connection, 'target', atomic=True)
    assert connection.in_transaction
    ieq([('outer1', 30), ('outer2', 32)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    assert connection.execute('SELECT * FROM side').fetchall() == [(1,)]
    connection.rollback()
    assert_old_data_intact(connection)


def test_nested_atomic_loads_from_same_connection_source(connection):
    # the source iterator itself reads from the same connection while the
    # outer atomic load is writing, so boundaries must not overwrite each
    # other
    connection.execute('CREATE TABLE source_table (foo TEXT, bar INTEGER)')
    connection.executemany('INSERT INTO source_table VALUES (?, ?)',
                           [('s1', 1), ('s2', 2), ('s3', 3)])
    connection.commit()
    connection.execute('BEGIN')

    def outer_rows():
        # fromdb() includes a header row; data() strips it
        inner = etl.fromdb(connection,
                           'SELECT foo, bar FROM source_table').data()
        yield ('foo', 'bar')
        for row in inner:
            yield tuple(row)

    etl.todb(outer_rows(), connection, 'target', atomic=True)
    assert connection.execute(
        'SELECT foo, bar FROM target ORDER BY bar').fetchall() \
        == [('s1', 1), ('s2', 2), ('s3', 3)]
    connection.rollback()
    assert_old_data_intact(connection)


# ---------------------------------------------------------------------------
# no transaction on entry: commit control in default and autocommit modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('autocommit', [False, True])
def test_atomic_commit_false_leaves_open_transaction(connection, autocommit):
    connection.close()
    connection = make_database(autocommit=autocommit)
    try:
        etl.appenddb((('foo', 'bar'), ('a', 10)), connection, 'target',
                     atomic=True, commit=False)
        assert connection.in_transaction
        assert connection.execute(
            'SELECT count(*) FROM target WHERE foo = ?', ('a',)).fetchone() \
            == (1,)
        connection.rollback()
        assert_old_data_intact(connection)
        assert not connection.in_transaction
    finally:
        connection.close()


@pytest.mark.parametrize('autocommit', [False, True])
def test_atomic_failure_without_outer_transaction_restores_entry_state(
        autocommit):
    connection = make_database(autocommit=autocommit)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            etl.todb(rows_with_constraint_failure(), connection, 'target',
                     atomic=True, commit=False)
        assert not connection.in_transaction
        assert_old_data_intact(connection)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# cursor and connection resource ownership
# ---------------------------------------------------------------------------

class TrackedCursor(sqlite3.Cursor):
    def __init__(self, connection):
        super(TrackedCursor, self).__init__(connection)
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        return super(TrackedCursor, self).close()


class TrackedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super(TrackedConnection, self).__init__(*args, **kwargs)
        self.created_cursors = []

    def cursor(self, *args, **kwargs):
        kwargs['factory'] = TrackedCursor
        cursor = super(TrackedConnection, self).cursor(*args, **kwargs)
        self.created_cursors.append(cursor)
        return cursor


def _tracked_connection():
    connection = sqlite3.connect(':memory:', factory=TrackedConnection)
    connection.execute('CREATE TABLE target (foo TEXT, bar INTEGER UNIQUE)')
    connection.executemany('INSERT INTO target VALUES (?, ?)',
                           [('old1', 1), ('old2', 2)])
    connection.commit()
    return connection


@pytest.mark.parametrize('use_factory', [False, True])
@pytest.mark.parametrize('failure', [None, 'source', 'constraint'])
def test_atomic_owned_cursors_are_closed(use_factory, failure):
    connection = _tracked_connection()
    try:
        dbo = connection.cursor if use_factory else connection
        if failure == 'source':
            source = rows_with_source_failure(ValueError('boom'))
            expected = ValueError
        elif failure == 'constraint':
            source = rows_with_constraint_failure()
            expected = sqlite3.IntegrityError
        else:
            source = NEW_ROWS
            expected = None
        if expected is None:
            etl.todb(source, dbo, 'target', atomic=True)
        else:
            with pytest.raises(expected):
                etl.todb(source, dbo, 'target', atomic=True)
        assert connection.created_cursors
        for cursor in connection.created_cursors:
            assert cursor.close_calls == 1
            with pytest.raises(sqlite3.ProgrammingError):
                cursor.execute('SELECT 1')
        # connection itself remains usable
        assert connection.execute('SELECT 42').fetchone() == (42,)
    finally:
        connection.close()


def test_atomic_borrowed_cursor_stays_open_on_success_and_failure(connection):
    cursor = connection.cursor()
    try:
        etl.todb(NEW_ROWS, cursor, 'target', atomic=True)
        assert cursor.execute('SELECT 42').fetchone() == (42,)
        ieq([('a', 10), ('b', 20)],
            connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
        with pytest.raises(sqlite3.IntegrityError):
            etl.todb(rows_with_constraint_failure(), cursor, 'target',
                     atomic=True)
        # borrowed cursor still usable; the failed load rolled back to the
        # state committed by the successful load
        assert cursor.execute('SELECT 42').fetchone() == (42,)
        ieq([('a', 10), ('b', 20)],
            connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        cursor.close()


def test_atomic_factory_cursor_rejected_type_is_closed():
    connection = make_database()
    created = []

    def factory():
        cursor = connection.cursor()
        created.append(cursor)
        return _PlainProxy(cursor)

    try:
        with pytest.raises(ArgumentError):
            etl.todb(NEW_ROWS, factory, 'target', atomic=True)
        # the proxied sqlite3 cursor was created by this call and must be
        # released; the proxy delegates close to it
        with pytest.raises(sqlite3.ProgrammingError):
            created[0].execute('SELECT 1')
        assert_old_data_intact(connection)
    finally:
        connection.close()


class _PlainProxy(object):
    """Non-sqlite3 cursor lookalike returned by a misconfigured factory."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __getattr__(self, item):
        return getattr(self._cursor, item)


# ---------------------------------------------------------------------------
# rejection branches: no writes or row consumption before the error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('load', [etl.todb, etl.appenddb])
def test_atomic_rejects_file_path_with_commit_false(db_file, load):
    source = CountingTable(NEW_ROWS)
    with pytest.raises(ArgumentError):
        load(source, db_file, 'target', atomic=True, commit=False)
    # rejected before opening the connection or consuming the source
    assert source.iterations == 0
    connection = sqlite3.connect(db_file)
    try:
        ieq(OLD_ROWS, connection.execute(
            'SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        connection.close()


def test_atomic_rejects_create_before_any_write(connection):
    with pytest.raises(ArgumentError):
        etl.todb(NEW_ROWS, connection, 'new_table', atomic=True, create=True)
    with pytest.raises(ArgumentError):
        etl.todb(NEW_ROWS, connection, 'new_table', atomic=True, create=True,
                 drop=True)
    # nothing was created or written
    assert connection.execute(
        "SELECT count(*) FROM sqlite_master WHERE name = 'new_table'"
    ).fetchone() == (0,)
    assert_old_data_intact(connection)


def test_atomic_rejects_non_sqlite3_dbo(connection):
    class DuckConnection(object):
        def cursor(self):
            return self

    with pytest.raises(ArgumentError):
        etl.todb(NEW_ROWS, DuckConnection(), 'target', atomic=True)
    assert_old_data_intact(connection)


def test_atomic_rejects_sqlalchemy_like_dbo(monkeypatch):
    # SQLAlchemy objects are deliberately unsupported even when sqlite3 is
    # the backing driver; the rejection must happen before row consumption
    source = CountingTable(NEW_ROWS)
    sa_like = object()
    monkeypatch.setattr(db, '_is_sqlalchemy_connection', lambda v: True)
    with pytest.raises(ArgumentError):
        etl.todb(source, sa_like, 'target', atomic=True)
    assert source.iterations == 0


def test_atomic_rejects_real_sqlalchemy_objects():
    sqlalchemy = pytest.importorskip('sqlalchemy')
    orm = pytest.importorskip('sqlalchemy.orm')
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text('CREATE TABLE t (a INTEGER)'))
        for dbo in (engine, engine.connect()):
            with pytest.raises(ArgumentError):
                etl.todb((('a',), (2,)), dbo, 't', atomic=True)
            if hasattr(dbo, 'close'):
                dbo.close()
        session = orm.Session(engine)
        try:
            with pytest.raises(ArgumentError):
                etl.appenddb((('a',), (2,)), session, 't', atomic=True)
        finally:
            session.close()
    finally:
        engine.dispose()


def test_atomic_rejects_non_sqlite3_factory(connection):
    source = CountingTable(NEW_ROWS)

    def factory():
        return _PlainProxy(connection.cursor())

    with pytest.raises(ArgumentError):
        etl.todb(source, factory, 'target', atomic=True)
    assert source.iterations == 0
    assert_old_data_intact(connection)


def test_atomic_rejection_does_not_consume_rows(connection):
    source = CountingTable(NEW_ROWS)
    with pytest.raises(ArgumentError):
        etl.todb(source, object(), 'target', atomic=True)
    assert source.iterations == 0


def test_atomic_header_failure_still_closes_owned_cursors():
    connection = _tracked_connection()
    try:
        def broken_source():
            raise ValueError('cannot even produce a header')
            yield  # pragma: no cover - make this a generator

        with pytest.raises(ValueError):
            etl.todb(broken_source(), connection, 'target', atomic=True)
        # the data cursor was created before the source was touched
        assert connection.created_cursors
        for cursor in connection.created_cursors:
            assert cursor.close_calls == 1
        assert connection.execute('SELECT 42').fetchone() == (42,)
    finally:
        connection.close()


def test_atomic_savepoint_failure_does_not_leave_cursors_open(monkeypatch):
    connection = _tracked_connection()
    base_cursor = sqlite3.Connection.cursor
    connection.created_cursors = []

    class BrokenTxCursor(TrackedCursor):
        def execute(self, statement, *args, **kwargs):
            if statement.startswith('SAVEPOINT'):
                raise sqlite3.OperationalError('savepoint rejected')
            return super(BrokenTxCursor, self).execute(
                statement, *args, **kwargs)

    calls = {'n': 0}

    def cursor_factory(*args, **kwargs):
        calls['n'] += 1
        # second cursor created by the load is its transaction cursor
        factory = BrokenTxCursor if calls['n'] == 2 else TrackedCursor
        cursor = base_cursor(connection, factory=factory)
        connection.created_cursors.append(cursor)
        return cursor

    monkeypatch.setattr(connection, 'cursor', cursor_factory)
    # pretend a transaction is already open so the SAVEPOINT branch is taken
    monkeypatch.setattr(db, '_sqlite3_in_transaction', lambda value: True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            etl.todb(NEW_ROWS, connection, 'target', atomic=True)
        assert connection.created_cursors
        for cursor in connection.created_cursors:
            assert cursor.close_calls == 1
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# commit-stage failures: deferred foreign keys and database locks
#
# A deferred foreign key (DEFERRABLE INITIALLY DEFERRED with
# PRAGMA foreign_keys=ON) is not checked while rows are written; the failure
# only appears when the load's transaction is finalised (COMMIT, or the
# RELEASE of the outermost SAVEPOINT). Another connection holding a read lock
# similarly makes the COMMIT report "database is locked". In both cases the
# load must be undone and the original sqlite3 exception must still propagate.
# ---------------------------------------------------------------------------

def make_fk_database(autocommit=False):
    connection = sqlite3.connect(
        ':memory:', isolation_level=None if autocommit else '')
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('CREATE TABLE parent (id INTEGER PRIMARY KEY)')
    connection.execute('CREATE TABLE child ('
                       'id INTEGER PRIMARY KEY, '
                       'pid INTEGER REFERENCES parent(id) '
                       'DEFERRABLE INITIALLY DEFERRED)')
    connection.execute('CREATE TABLE side (n INTEGER)')
    connection.executemany('INSERT INTO parent VALUES (?)',
                           [(1,), (2,)])
    connection.execute('INSERT INTO child VALUES (?, ?)', (10, 1))
    connection.commit()
    return connection


FK_OLD_ROWS = [(10, 1)]
# parent 99 does not exist: the deferred FK only fails at finalisation time
FK_BAD_ROWS = (('id', 'pid'), (20, 99))
FK_GOOD_ROWS = (('id', 'pid'), (20, 2))


def assert_fk_old_data_intact(connection):
    ieq(FK_OLD_ROWS,
        connection.execute('SELECT id, pid FROM child ORDER BY id'))


def _fk_dbo(connection, entry):
    if entry == 'connection':
        return connection
    if entry == 'cursor':
        return connection.cursor()
    if entry == 'factory':
        return connection.cursor
    raise AssertionError('unknown entry %r' % entry)


@pytest.mark.parametrize('autocommit', [False, True])
@pytest.mark.parametrize('entry', ['connection', 'cursor', 'factory'])
@pytest.mark.parametrize('replace', [True, False])
def test_atomic_deferred_fk_commit_failure_restores_data(
        entry, replace, autocommit):
    connection = make_fk_database(autocommit=autocommit)
    dbo = _fk_dbo(connection, entry)
    load = etl.todb if replace else etl.appenddb
    try:
        with pytest.raises(sqlite3.IntegrityError) as caught:
            load(FK_BAD_ROWS, dbo, 'child', atomic=True)
        # the caller receives the native sqlite3 exception type and message
        assert type(caught.value) is sqlite3.IntegrityError
        assert 'FOREIGN KEY constraint failed' in str(caught.value)
        # no active transaction is left behind and old data is back
        assert not connection.in_transaction
        assert_fk_old_data_intact(connection)
        # the same connection serves another, successful, atomic load
        etl.appenddb(FK_GOOD_ROWS, connection, 'child', atomic=True)
        ieq(FK_OLD_ROWS + [(20, 2)],
            connection.execute('SELECT id, pid FROM child ORDER BY id'))
    finally:
        if entry == 'cursor':
            dbo.close()
        connection.close()


@pytest.mark.parametrize('autocommit', [False, True])
def test_atomic_deferred_fk_commit_failure_file_path(autocommit, tmp_path,
                                                     monkeypatch):
    path = str(tmp_path / 'fk.db')
    setup = sqlite3.connect(path)
    setup.execute('PRAGMA foreign_keys=ON')
    setup.execute('CREATE TABLE parent (id INTEGER PRIMARY KEY)')
    setup.execute('CREATE TABLE child ('
                  'id INTEGER PRIMARY KEY, '
                  'pid INTEGER REFERENCES parent(id) '
                  'DEFERRABLE INITIALLY DEFERRED)')
    setup.executemany('INSERT INTO parent VALUES (?)', [(1,), (2,)])
    setup.execute('INSERT INTO child VALUES (?, ?)', (10, 1))
    setup.commit()
    setup.close()

    # petl opens its own connection for a path argument; foreign_keys is a
    # per-connection setting that defaults to OFF, so the connection factory
    # must enable it for deferred enforcement to be active
    real_connect = sqlite3.connect

    def connect_with_fk(dbo, *args, **kwargs):
        connection = real_connect(
            dbo, isolation_level=None if autocommit else '')
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    monkeypatch.setattr(db.sqlite3, 'connect', connect_with_fk)

    with pytest.raises(sqlite3.IntegrityError):
        etl.todb(FK_BAD_ROWS, path, 'child', atomic=True)

    check = sqlite3.connect(path)
    try:
        ieq(FK_OLD_ROWS, check.execute('SELECT id, pid FROM child'))
    finally:
        check.close()


def test_atomic_deferred_fk_failure_inside_outer_transaction():
    connection = make_fk_database()
    try:
        connection.execute('BEGIN')
        connection.execute('INSERT INTO side VALUES (7)')
        for load, rows in ((etl.todb, FK_BAD_ROWS),
                           (etl.appenddb, FK_BAD_ROWS)):
            with pytest.raises(sqlite3.IntegrityError) as caught:
                load(rows, connection, 'child', atomic=True)
            assert type(caught.value) is sqlite3.IntegrityError
            # only this load is undone; the caller's transaction and writes
            # survive and the caller can still commit successfully
            assert connection.in_transaction
            assert_fk_old_data_intact(connection)
            assert connection.execute('SELECT * FROM side').fetchall() \
                == [(7,)]
        connection.commit()
        assert not connection.in_transaction
        assert connection.execute('SELECT * FROM side').fetchall() == [(7,)]
        assert_fk_old_data_intact(connection)
        # connection reuse inside a fresh outer transaction
        connection.execute('BEGIN')
        etl.appenddb(FK_GOOD_ROWS, connection, 'child', atomic=True)
        connection.commit()
        ieq(FK_OLD_ROWS + [(20, 2)],
            connection.execute('SELECT id, pid FROM child ORDER BY id'))
    finally:
        connection.close()


def test_atomic_deferred_fk_failure_keeps_caller_savepoint_usable():
    connection = make_fk_database()
    try:
        connection.execute('BEGIN')
        connection.execute('SAVEPOINT caller_sp')
        connection.execute('INSERT INTO side VALUES (8)')
        with pytest.raises(sqlite3.IntegrityError):
            etl.todb(FK_BAD_ROWS, connection, 'child', atomic=True)
        assert connection.in_transaction
        assert_fk_old_data_intact(connection)
        # caller's own savepoint is still operational
        connection.execute('ROLLBACK TO caller_sp')
        connection.execute('RELEASE caller_sp')
        assert connection.execute('SELECT * FROM side').fetchall() == []
        connection.rollback()
    finally:
        connection.close()


def test_atomic_deferred_fk_nested_loads_independent_rollback():
    # an outer atomic load contains an inner atomic load; the inner deferred
    # FK failure must undo only the inner load, and the outer load then
    # completes inside the caller's transaction
    connection = make_fk_database()
    try:
        connection.execute('BEGIN')

        def outer_rows():
            yield ('id', 'pid')
            yield (30, 1)
            with pytest.raises(sqlite3.IntegrityError):
                etl.appenddb(FK_BAD_ROWS, connection, 'child', atomic=True)
            yield (31, 2)

        etl.todb(outer_rows(), connection, 'child', atomic=True)
        assert connection.in_transaction
        ieq([(30, 1), (31, 2)],
            connection.execute('SELECT id, pid FROM child ORDER BY id'))
        connection.rollback()
        assert_fk_old_data_intact(connection)
    finally:
        connection.close()


def test_atomic_preexisting_fk_violation_still_commits():
    # native COMMIT only checks the current transaction's changes; rows that
    # already violated the FK before the transaction began are tolerated. The
    # atomic load must match that behaviour rather than failing on them.
    connection = sqlite3.connect(':memory:')
    try:
        connection.execute('CREATE TABLE parent (id INTEGER PRIMARY KEY)')
        connection.execute('CREATE TABLE child ('
                           'id INTEGER PRIMARY KEY, '
                           'pid INTEGER REFERENCES parent(id) '
                           'DEFERRABLE INITIALLY DEFERRED)')
        connection.execute('CREATE TABLE target (n INTEGER)')
        # poison the schema with a dangling row written while FKs are off
        connection.execute('INSERT INTO child VALUES (?, ?)', (1, 99))
        connection.commit()
        connection.execute('PRAGMA foreign_keys=ON')
        # the load touches a different table; the pre-existing violation
        # must not be mistaken for one introduced by this load
        etl.todb((('n',), (5,)), connection, 'target', atomic=True)
        assert connection.execute('SELECT * FROM target').fetchall() == [(5,)]
        assert connection.execute('SELECT * FROM child').fetchall() \
            == [(1, 99)]
        assert not connection.in_transaction
        # and the same tolerance applies on the savepoint path, where petl
        # itself compares before/after foreign_key_check snapshots
        connection.execute('BEGIN')
        etl.appenddb((('n',), (6,)), connection, 'target', atomic=True)
        connection.commit()
        assert connection.execute('SELECT * FROM target ORDER BY n') \
            .fetchall() == [(5,), (6,)]
    finally:
        connection.close()


def test_atomic_commit_false_no_outer_tx_defers_finalisation():
    # with commit=False there is nothing for petl to finalise, so the
    # deferred failure remains the caller's to hit on its own commit, and
    # the transaction is intentionally left open (historical semantics)
    fk_connection = make_fk_database()
    try:
        etl.todb(FK_BAD_ROWS, fk_connection, 'child', atomic=True,
                 commit=False)
        assert fk_connection.in_transaction
        with pytest.raises(sqlite3.IntegrityError):
            fk_connection.commit()
        # after the failed caller commit the transaction is still open;
        # rolling back restores the pre-load contents
        fk_connection.rollback()
        assert not fk_connection.in_transaction
        assert_fk_old_data_intact(fk_connection)
    finally:
        fk_connection.close()


def _locked_database(tmp_path, timeout=0.0):
    path = str(tmp_path / 'locked.db')
    setup = sqlite3.connect(path)
    setup.execute('CREATE TABLE target (foo TEXT, bar INTEGER)')
    setup.executemany('INSERT INTO target VALUES (?, ?)',
                      [('old1', 1), ('old2', 2)])
    setup.commit()
    setup.close()
    # a second connection with an open read transaction holds a shared lock,
    # so the writer cannot upgrade to the exclusive lock needed at COMMIT
    reader = sqlite3.connect(path, timeout=timeout)
    reader.execute('BEGIN')
    reader.execute('SELECT * FROM target').fetchall()
    writer = sqlite3.connect(path, timeout=timeout)
    return path, reader, writer


LOCK_OLD_ROWS = [('old1', 1), ('old2', 2)]
LOCK_NEW_ROWS = (('foo', 'bar'), ('new', 9))


@pytest.mark.parametrize('replace', [True, False])
def test_atomic_commit_lock_error_rolls_back_and_reuses(tmp_path, replace):
    path, reader, writer = _locked_database(tmp_path)
    try:
        load = etl.todb if replace else etl.appenddb
        with pytest.raises(sqlite3.OperationalError) as caught:
            load(LOCK_NEW_ROWS, writer, 'target', atomic=True)
        assert type(caught.value) is sqlite3.OperationalError
        assert 'database is locked' in str(caught.value)
        # no half-batch and no active transaction left for the caller
        assert not writer.in_transaction
        ieq(LOCK_OLD_ROWS,
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
        # once the reader lets go, the same connection loads again
        reader.rollback()
        reader.close()
        etl.todb(LOCK_NEW_ROWS, writer, 'target', atomic=True)
        ieq([('new', 9)],
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
        assert not writer.in_transaction
    finally:
        writer.close()
        if reader:
            reader.close()


def test_atomic_commit_lock_error_with_borrowed_cursor(tmp_path):
    path, reader, writer = _locked_database(tmp_path)
    cursor = writer.cursor()
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            etl.todb(LOCK_NEW_ROWS, cursor, 'target', atomic=True)
        assert 'database is locked' in str(caught.value)
        assert not writer.in_transaction
        ieq(LOCK_OLD_ROWS,
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
        # borrowed cursor remains usable
        assert cursor.execute('SELECT 42').fetchone() == (42,)
    finally:
        cursor.close()
        writer.close()
        reader.rollback()
        reader.close()


def test_atomic_commit_lock_error_with_cursor_factory(tmp_path):
    path, reader, writer = _locked_database(tmp_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            etl.appenddb(LOCK_NEW_ROWS, writer.cursor, 'target', atomic=True)
        assert not writer.in_transaction
        ieq(LOCK_OLD_ROWS,
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        writer.close()
        reader.rollback()
        reader.close()


def test_atomic_commit_lock_error_then_repeated_failures_and_success(tmp_path):
    path, reader, writer = _locked_database(tmp_path)
    try:
        for _ in range(2):
            with pytest.raises(sqlite3.OperationalError):
                etl.todb(LOCK_NEW_ROWS, writer, 'target', atomic=True)
            assert not writer.in_transaction
            ieq(LOCK_OLD_ROWS,
                writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
        reader.rollback()
        reader.close()
        reader = None
        etl.todb(LOCK_NEW_ROWS, writer, 'target', atomic=True)
        ieq([('new', 9)],
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        writer.close()
        if reader:
            reader.close()


def test_atomic_commit_lock_error_within_outer_transaction(tmp_path):
    # the lock is only hit when the *caller* later commits; petl's savepoint
    # release does not finalise the transaction, so the load itself succeeds
    # and the caller's earlier writes remain intact
    path, reader, writer = _locked_database(tmp_path)
    try:
        writer.execute('BEGIN')
        etl.todb(LOCK_NEW_ROWS, writer, 'target', atomic=True)
        assert writer.in_transaction
        with pytest.raises(sqlite3.OperationalError):
            writer.commit()
        # the failed commit leaves the transaction open; the caller rolls it
        # back and gets the pre-load contents
        writer.rollback()
        assert not writer.in_transaction
        ieq(LOCK_OLD_ROWS,
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        writer.close()
        reader.rollback()
        reader.close()


def test_atomic_commit_lock_error_autocommit_connection(tmp_path):
    path, reader, _ = _locked_database(tmp_path)
    writer = sqlite3.connect(path, timeout=0.0, isolation_level=None)
    try:
        with pytest.raises(sqlite3.OperationalError):
            etl.todb(LOCK_NEW_ROWS, writer, 'target', atomic=True)
        assert not writer.in_transaction
        ieq(LOCK_OLD_ROWS,
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
        reader.rollback()
        reader.close()
        reader = None
        etl.appenddb((('foo', 'bar'), ('again', 5)), writer, 'target',
                     atomic=True)
        ieq(LOCK_OLD_ROWS + [('again', 5)],
            writer.execute('SELECT foo, bar FROM target ORDER BY bar'))
    finally:
        writer.close()
        if reader:
            reader.close()


def test_atomic_deferred_fk_and_lock_close_owned_cursors(tmp_path):
    # commit-stage failures must release every cursor petl created, just like
    # statement-stage failures do
    fk_connection = _tracked_fk_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            etl.todb(FK_BAD_ROWS, fk_connection, 'child', atomic=True)
        assert fk_connection.created_cursors
        for cursor in fk_connection.created_cursors:
            assert cursor.close_calls == 1
            with pytest.raises(sqlite3.ProgrammingError):
                cursor.execute('SELECT 1')
        assert fk_connection.execute('SELECT 42').fetchone() == (42,)
    finally:
        fk_connection.close()

    path, reader, tracked = _tracked_locked_database(tmp_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            etl.appenddb(LOCK_NEW_ROWS, tracked, 'target', atomic=True)
        assert tracked.created_cursors
        for cursor in tracked.created_cursors:
            assert cursor.close_calls == 1
    finally:
        tracked.close()
        reader.rollback()
        reader.close()


def _tracked_fk_connection():
    connection = sqlite3.connect(':memory:', factory=TrackedConnection)
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('CREATE TABLE parent (id INTEGER PRIMARY KEY)')
    connection.execute('CREATE TABLE child ('
                       'id INTEGER PRIMARY KEY, '
                       'pid INTEGER REFERENCES parent(id) '
                       'DEFERRABLE INITIALLY DEFERRED)')
    connection.executemany('INSERT INTO parent VALUES (?)', [(1,), (2,)])
    connection.execute('INSERT INTO child VALUES (?, ?)', (10, 1))
    connection.commit()
    return connection


def _tracked_locked_database(tmp_path):
    path = str(tmp_path / 'tracked_lock.db')
    setup = sqlite3.connect(path)
    setup.execute('CREATE TABLE target (foo TEXT, bar INTEGER)')
    setup.executemany('INSERT INTO target VALUES (?, ?)',
                      [('old1', 1), ('old2', 2)])
    setup.commit()
    setup.close()
    reader = sqlite3.connect(path, timeout=0.0)
    reader.execute('BEGIN')
    reader.execute('SELECT * FROM target').fetchall()
    writer = sqlite3.connect(path, timeout=0.0, factory=TrackedConnection)
    return path, reader, writer


def test_atomic_immediate_constraint_failure_still_works(connection):
    # statement-time failures must keep their existing rollback contract
    with pytest.raises(sqlite3.IntegrityError) as caught:
        etl.todb(rows_with_constraint_failure(), connection, 'target',
                 atomic=True)
    assert type(caught.value) is sqlite3.IntegrityError
    assert not connection.in_transaction
    assert_old_data_intact(connection)


def test_atomic_source_failure_after_commit_stage_fix(connection):
    failure = RuntimeError('source blew up')
    with pytest.raises(RuntimeError) as caught:
        etl.todb(rows_with_source_failure(failure), connection, 'target',
                 atomic=True)
    assert caught.value is failure
    assert not connection.in_transaction
    assert_old_data_intact(connection)


class _RollbackBrokenCursor(sqlite3.Cursor):
    def execute(self, statement, *args, **kwargs):
        if statement == 'ROLLBACK':
            raise sqlite3.OperationalError('rollback broken')
        return super(_RollbackBrokenCursor, self).execute(
            statement, *args, **kwargs)


class _RollbackBrokenConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):
        kwargs['factory'] = _RollbackBrokenCursor
        return super(_RollbackBrokenConnection, self).cursor(
            *args, **kwargs)


def test_atomic_commit_failure_cleanup_error_does_not_mask_original():
    # if the post-COMMIT rollback itself errors, the original sqlite3
    # exception must still reach the caller and cursors must still close
    connection = sqlite3.connect(':memory:',
                                 factory=_RollbackBrokenConnection)
    try:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('CREATE TABLE parent (id INTEGER PRIMARY KEY)')
        connection.execute('CREATE TABLE child ('
                           'id INTEGER PRIMARY KEY, '
                           'pid INTEGER REFERENCES parent(id) '
                           'DEFERRABLE INITIALLY DEFERRED)')
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError) as caught:
            etl.todb((('id', 'pid'), (1, 99)), connection, 'child',
                     atomic=True)
        assert type(caught.value) is sqlite3.IntegrityError
        assert 'FOREIGN KEY constraint failed' in str(caught.value)
    finally:
        connection.close()


class _ReleaseBrokenCursor(sqlite3.Cursor):
    def execute(self, statement, *args, **kwargs):
        if statement.startswith('RELEASE'):
            raise sqlite3.OperationalError('release broken')
        return super(_ReleaseBrokenCursor, self).execute(
            statement, *args, **kwargs)


class _ReleaseBrokenConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):
        kwargs['factory'] = _ReleaseBrokenCursor
        return super(_ReleaseBrokenConnection, self).cursor(
            *args, **kwargs)


def test_atomic_savepoint_release_error_propagates_unchanged():
    connection = sqlite3.connect(':memory:',
                                 factory=_ReleaseBrokenConnection)
    try:
        connection.execute('CREATE TABLE target (foo TEXT, bar INTEGER)')
        connection.commit()
        connection.execute('BEGIN')
        with pytest.raises(sqlite3.OperationalError) as caught:
            etl.todb((('foo', 'bar'), ('a', 10)), connection, 'target',
                     atomic=True)
        assert str(caught.value) == 'release broken'
        connection.rollback()
        assert not connection.in_transaction
    finally:
        connection.close()


def test_atomic_no_false_positive_when_foreign_keys_disabled():
    # with PRAGMA foreign_keys left at its default OFF, a dangling reference
    # must load successfully on both the commit and savepoint paths
    connection = sqlite3.connect(':memory:')
    try:
        connection.execute('CREATE TABLE parent (id INTEGER PRIMARY KEY)')
        connection.execute('CREATE TABLE child ('
                           'id INTEGER PRIMARY KEY, '
                           'pid INTEGER REFERENCES parent(id) '
                           'DEFERRABLE INITIALLY DEFERRED)')
        connection.commit()
        etl.todb((('id', 'pid'), (1, 99)), connection, 'child', atomic=True)
        assert connection.execute('SELECT * FROM child').fetchall() \
            == [(1, 99)]
        connection.execute('BEGIN')
        etl.appenddb((('id', 'pid'), (2, 98)), connection, 'child',
                     atomic=True)
        connection.commit()
        assert connection.execute('SELECT * FROM child ORDER BY id') \
            .fetchall() == [(1, 99), (2, 98)]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# default behaviour is unchanged when atomic is not requested
# ---------------------------------------------------------------------------

def test_default_non_atomic_behaviour_unchanged(connection):
    # historical behaviour: a failed load leaves a half-batch in the
    # connection's transaction and does not commit
    with pytest.raises(sqlite3.IntegrityError):
        etl.todb(rows_with_constraint_failure(), connection, 'target')
    # the truncate already happened inside the still-open transaction
    assert connection.in_transaction
    ieq([('a', 10)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
    connection.rollback()
    assert_old_data_intact(connection)


def test_atomic_false_routes_through_historical_dispatch(monkeypatch,
                                                         connection):
    calls = []

    def fake_loader(table, dbo, tablename, schema=None, commit=True,
                    truncate=False):
        calls.append((truncate,))

    monkeypatch.setattr(db, '_todb_dbapi_connection', fake_loader)
    etl.todb(NEW_ROWS, connection, 'target')
    etl.appenddb(NEW_ROWS, connection, 'target')
    assert calls == [(True,), (False,)]


def test_explicit_atomic_false_matches_default_path(monkeypatch, connection):
    calls = []

    def fake_loader(table, dbo, tablename, schema=None, commit=True,
                    truncate=False, atomic=False):
        calls.append((truncate, atomic))

    monkeypatch.setattr(db, '_todb_dbapi_connection', fake_loader)
    etl.todb(NEW_ROWS, connection, 'target', atomic=False)
    etl.appenddb(NEW_ROWS, connection, 'target', atomic=False)
    assert calls == [(True, False), (False, False)]


def test_atomic_true_skips_historical_dispatch(monkeypatch, connection):
    def fail_loader(*args, **kwargs):
        raise AssertionError('historical DB-API loader must not be used '
                             'in atomic mode')

    monkeypatch.setattr(db, '_todb_dbapi_connection', fail_loader)
    etl.todb(NEW_ROWS, connection, 'target', atomic=True)
    ieq([('a', 10), ('b', 20)],
        connection.execute('SELECT foo, bar FROM target ORDER BY bar'))
