# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# standard library dependencies
import itertools
import logging
import sqlite3
import threading
from petl.compat import next, text_type, string_types


# internal dependencies
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
    _is_sqlalchemy_connection, _is_sqlalchemy_engine, _is_sqlalchemy_session, \
    _is_clikchouse_dbapi_connection, _quote, _placeholders, \
    _execute_sqlalchemy, _sqlalchemy_transaction
from petl.io.db_create import drop_table, create_table


logger = logging.getLogger(__name__)
debug = logger.debug
warning = logger.warning


# Counter used to give every atomic load unique SAVEPOINT names, so that
# nested atomic loads on the same connection cannot overwrite each other's
# savepoints.
_savepoint_counter = itertools.count(1)
_savepoint_lock = threading.Lock()


def _next_savepoint_name():
    with _savepoint_lock:
        number = next(_savepoint_counter)
    return 'petl_sp_%d' % number


def _is_sqlite3_connection(dbo):
    try:
        import sqlite3
    except ImportError:
        return False
    return isinstance(dbo, sqlite3.Connection)


def _is_sqlite3_cursor(dbo):
    try:
        import sqlite3
    except ImportError:
        return False
    return isinstance(dbo, sqlite3.Cursor)


def fromdb(dbo, query, *args, **kwargs):
    """Provides access to data from any DB-API 2.0 connection via a given query.
    E.g., using :mod:`sqlite3`::

        >>> import petl as etl
        >>> import sqlite3
        >>> connection = sqlite3.connect('example.db')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example', ownsdb=True)

    E.g., using :mod:`psycopg2` (assuming you've installed it first)::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    E.g., using :mod:`pymysql` (assuming you've installed it first)::

        >>> import petl as etl
        >>> import pymysql
        >>> connection = pymysql.connect(password='moonpie', database='thangs')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    The `dbo` argument may also be a function that creates a cursor. N.B., each
    call to the function should return a new cursor. E.g.::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> mkcursor = lambda: connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        >>> table = etl.fromdb(mkcursor, 'SELECT * FROM example')

    The parameter `dbo` may also be an SQLAlchemy engine, session or
    connection object.

    The parameter `dbo` may also be a string, in which case it is interpreted as
    the name of a file containing an :mod:`sqlite3` database.

    When parameter `ownsdb` is True, the database connection will be closed when the
    cursor is out of scope and is being garbage collected.

    Note that the default behaviour of most database servers and clients is for
    the entire result set for each query to be sent from the server to the
    client. If your query returns a large result set this can result in
    significant memory usage at the client. Some databases support server-side
    cursors which provide a means for client libraries to fetch result sets
    incrementally, reducing memory usage at the client.

    To use a server-side cursor with a PostgreSQL database, e.g.::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> table = etl.fromdb(lambda: connection.cursor(name='arbitrary'),
        ...                    'SELECT * FROM example')

    For more information on server-side cursors see the following links:

        * http://initd.org/psycopg/docs/usage.html#server-side-cursors
        * http://mysql-python.sourceforge.net/MySQLdb.html#using-and-extending

    """

    ownsdb = kwargs.pop('ownsdb', False)
    # convenience for working with sqlite3
    if isinstance(dbo, string_types):
        import sqlite3
        dbo = sqlite3.connect(dbo)
        ownsdb = True

    return DbView(dbo, query, *args, ownsdb=ownsdb, **kwargs)


class DbView(Table):

    def __init__(self, dbo, query, *args, **kwargs):
        self.dbo = dbo
        self.query = query
        self.args = args
        self.kwargs = kwargs
        self.ownsdb = kwargs.pop('ownsdb', False)

    def __del__(self):
        if self.ownsdb:
            try:
                self.dbo.close()
                self.dbo = None
            except Exception:
                pass

    def __iter__(self):

        # does it quack like a standard DB-API 2.0 connection?
        if _is_dbapi_connection(self.dbo):
            debug('assuming %r is standard DB-API 2.0 connection', self.dbo)
            _iter = _iter_dbapi_connection

        # does it quack like a standard DB-API 2.0 cursor?
        elif _is_dbapi_cursor(self.dbo):
            debug('assuming %r is standard DB-API 2.0 cursor')
            warning('using a DB-API cursor with fromdb() is not recommended '
                    'and may lead to unexpected results, a DB-API connection '
                    'is better')
            _iter = _iter_dbapi_cursor

        # does it quack like an SQLAlchemy engine?
        elif _is_sqlalchemy_engine(self.dbo):
            debug('assuming %r instance of sqlalchemy.engine.base.Engine',
                  self.dbo)
            _iter = _iter_sqlalchemy_engine

        # does it quack like an SQLAlchemy session?
        elif _is_sqlalchemy_session(self.dbo):
            debug('assuming %r instance of sqlalchemy.orm.session.Session',
                  self.dbo)
            _iter = _iter_sqlalchemy_session

        # does it quack like an SQLAlchemy connection?
        elif _is_sqlalchemy_connection(self.dbo):
            debug('assuming %r instance of sqlalchemy.engine.base.Connection',
                  self.dbo)
            _iter = _iter_sqlalchemy_connection

        elif callable(self.dbo):
            debug('assuming %r is a function returning a cursor', self.dbo)
            _iter = _iter_dbapi_mkcurs

        # some other sort of duck...
        else:
            raise ArgumentError('unsupported database object type: %r' % self.dbo)

        return _iter(self.dbo, self.query, *self.args, **self.kwargs)


def _iter_dbapi_mkcurs(mkcurs, query, *args, **kwargs):
    cursor = mkcurs()
    try:
        for row in _iter_dbapi_cursor(cursor, query, *args, **kwargs):
            yield row
    finally:
        cursor.close()


def _iter_dbapi_connection(connection, query, *args, **kwargs):
    cursor = connection.cursor()
    try:
        for row in _iter_dbapi_cursor(cursor, query, *args, **kwargs):
            yield row
    finally:
        cursor.close()


def _iter_dbapi_cursor(cursor, query, *args, **kwargs):
    cursor.execute(query, *args, **kwargs)
    # fetch one row before iterating, to force population of cursor.description
    # which may be postponed if using server-side cursors
    # not all database drivers populate cursor after execute so we call fetchall
    try:
        it = iter(cursor)
    except TypeError:
        it = iter(cursor.fetchall())
    try:
        first_row = next(it)
    except StopIteration:
        first_row = None
    # fields should be available now
    hdr = [d[0] for d in cursor.description]
    yield tuple(hdr)
    if first_row is None:
        return
    yield first_row
    for row in it:
        yield row  # don't wrap, return whatever the database engine returns


def _iter_sqlalchemy_engine(engine, query, *args, **kwargs):
    with engine.connect() as connection:
        iterator = _iter_sqlalchemy_connection(connection, query, *args, **kwargs)
        try:
            for row in iterator:
                yield row
        finally:
            iterator.close()


def _iter_sqlalchemy_connection(connection, query, *args, **kwargs):
    debug('connection: %r', connection)
    results = _execute_sqlalchemy(connection, query, *args, **kwargs)
    try:
        yield tuple(results.keys())
        for row in results:
            yield row
    finally:
        results.close()


def _iter_sqlalchemy_session(session, query, *args, **kwargs):
    if isinstance(query, string_types):
        from sqlalchemy import text
        query = text(query)
    results = session.execute(query, *args, **kwargs)
    try:
        yield tuple(results.keys())
        for row in results:
            yield row
    finally:
        results.close()


def todb(table, dbo, tablename, schema=None, commit=True,
         create=False, drop=False, constraints=True, metadata=None,
         dialect=None, sample=1000, truncate=True, atomic=False):
    """
    Load data into an existing database table via a DB-API 2.0
    connection or cursor. By default the database table will be truncated,
    i.e., all existing rows will be deleted prior to inserting the new data.
    Pass ``truncate=False`` to keep the existing rows and append instead, as
    :func:`petl.io.db.appenddb` does. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'],
        ...          ['a', 1],
        ...          ['b', 2],
        ...          ['c', 2]]
        >>> # using sqlite3
        ... import sqlite3
        >>> connection = sqlite3.connect('example.db')
        >>> # assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')
        >>> # using psycopg2
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> # assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')
        >>> # using pymysql
        >>> import pymysql
        >>> connection = pymysql.connect(password='moonpie', database='thangs')
        >>> # tell MySQL to use standard quote character
        ... connection.cursor().execute('SET SQL_MODE=ANSI_QUOTES')
        >>> # load data, assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')

    N.B., for MySQL the statement ``SET SQL_MODE=ANSI_QUOTES`` is required to
    ensure MySQL uses SQL-92 standard quote characters.

    A cursor can also be provided instead of a connection, e.g.::

        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> cursor = connection.cursor()
        >>> etl.todb(table, cursor, 'foobar')

    The parameter `dbo` may also be an SQLAlchemy engine, session or
    connection object.

    Cursors created to truncate or insert rows are closed on success or failure.
    A cursor supplied directly by the caller is left open.

    The parameter `dbo` may also be a string, in which case it is interpreted
    as the name of a file containing an :mod:`sqlite3` database.

    If ``create=True`` this function will attempt to automatically create a
    database table before loading the data. This functionality requires
    `SQLAlchemy <http://www.sqlalchemy.org/>`_ to be installed.

    **Keyword arguments:**

    table : table container
        Table data to load
    dbo : database object
        DB-API 2.0 connection, callable returning a DB-API 2.0 cursor, or
        SQLAlchemy connection, engine or session
    tablename : string
        Name of the table in the database
    schema : string
        Name of the database schema to find the table in
    commit : bool
        If True commit the changes
    create : bool
        If True attempt to create the table before loading, inferring types
        from a sample of the data (requires SQLAlchemy)
    drop : bool or 'if_exists'
        If True attempt to drop the table before recreating (only relevant if
        create=True). If 'if_exists', do not raise an error if the table does
        not exist.
    constraints : bool
        If True use length and nullable constraints (only relevant if
        create=True)
    metadata : sqlalchemy.MetaData
        Custom table metadata (only relevant if create=True)
    dialect : string
        One of {'access', 'sybase', 'sqlite', 'informix', 'firebird', 'mysql',
        'oracle', 'maxdb', 'postgresql', 'mssql'} (only relevant if
        create=True)
    sample : int
        Number of rows to sample when inferring types etc. Set to 0 to use the
        whole table (only relevant if create=True)
    truncate : bool
        If True (the default) truncate the table before loading, deleting any
        existing rows. Set to False to keep existing rows and append the new
        data instead.
    atomic : bool
        If True, wrap the truncate and all inserts in a single SQLite
        transaction so that a source iteration error, constraint violation or
        SQL failure rolls the whole load back, restoring the previous table
        contents. Only standard library :mod:`sqlite3` objects are supported:
        a ``sqlite3.Connection`` (including subclasses), a borrowed
        ``sqlite3.Cursor``, a callable returning a ``sqlite3.Cursor``, or a
        string path to a SQLite database file. See the "Atomic SQLite loads"
        section below for the full transaction ownership contract. Defaults
        to False, which preserves the historical non-atomic behaviour.

    Atomic SQLite loads
    ^^^^^^^^^^^^^^^^^^^

    Pass ``atomic=True`` to make the load atomic when using a standard
    library :mod:`sqlite3` database::

        >>> import sqlite3
        >>> import petl as etl
        >>> connection = sqlite3.connect('example.db')
        >>> # table "foobar" already exists and may contain data
        ... etl.todb(table, connection, 'foobar', atomic=True)

    The truncate (``todb`` only) and every insert run inside one SQLite
    transaction boundary. If iterating the source table raises, a unique or
    NOT NULL constraint fails, or any SQL statement errors, all writes made
    by this load are undone and the original error propagates unchanged; bad
    rows are never skipped and rows are never committed one at a time. The
    source is still consumed lazily in a single pass, so the full batch is
    never materialised in memory.

    Failures that SQLite only reports when the transaction is finalised are
    handled as well. In particular, a foreign key declared
    ``DEFERRABLE INITIALLY DEFERRED`` with ``PRAGMA foreign_keys=ON`` is not
    checked while rows are written: the violation only appears at
    ``COMMIT`` (or when the outermost ``SAVEPOINT`` is released), and another
    connection holding a read lock can make the same ``COMMIT`` report
    "database is locked". In these cases the load is still rolled back in
    full: the target table returns to its pre-load contents, the original
    :class:`sqlite3.IntegrityError` or :class:`sqlite3.OperationalError`
    reaches the caller unchanged, and a connection that had no transaction on
    entry is left without an active transaction rather than holding a
    half-finished batch. When the load runs inside a caller-owned
    transaction, deferred foreign key violations introduced by this load are
    detected before the savepoint is released so that only the load is
    undone; violations that already existed before the load are tolerated,
    matching SQLite's own commit behaviour.

    Transaction ownership follows the state of the connection on entry:

    * If the connection already has an open transaction (including
      ``isolation_level=None`` autocommit connections after an explicit
      ``BEGIN``, or a caller-created SAVEPOINT), the load runs inside a new
      SAVEPOINT. On success the SAVEPOINT is released but nothing is
      committed, regardless of ``commit``; on failure only the load is rolled
      back to the SAVEPOINT, leaving the caller's earlier writes and
      SAVEPOINTs intact. Nested atomic loads are supported and use distinct
      SAVEPOINTs, so an outer rollback also undoes inner loads.
    * If the connection has no open transaction, petl opens one for the load.
      With ``commit=True`` (the default) it is committed after a successful
      load and rolled back on failure, leaving the connection without an
      active transaction again. With ``commit=False`` the transaction is left
      open on success for the caller to commit or roll back; on failure it is
      rolled back so the connection returns to its entry state. This applies
      to both the default transaction handling and ``isolation_level=None``.

    Cursors created by petl are closed on both success and failure; a cursor
    supplied directly by the caller and the connection itself are left open
    and usable. Cleanup errors never replace the original load exception.

    When ``dbo`` is a file path string, petl opens and owns the connection and
    closes it again when the call returns. Because an uncommitted load would
    be silently lost when that connection closes, ``atomic=True`` with
    ``commit=False`` is rejected with an :class:`petl.errors.ArgumentError`
    before any data is read.

    ``atomic=True`` cannot be combined with ``create=True`` (including
    ``drop=True``) and is not supported for non-sqlite3 DB-API drivers or
    SQLAlchemy objects; such calls raise :class:`petl.errors.ArgumentError`
    before any rows are consumed or written.

    .. note::

        This function is in principle compatible with any DB-API 2.0
        compliant database driver. However, at the time of writing some DB-API
        2.0 implementations, including cx_Oracle and MySQL's
        Connector/Python, are not compatible with this function, because they
        only accept a list argument to the cursor.executemany() function
        called internally by :mod:`petl`. This can be worked around by
        proxying the cursor objects, e.g.::

            >>> import cx_Oracle
            >>> connection = cx_Oracle.Connection(...)
            >>> class CursorProxy(object):
            ...     def __init__(self, cursor):
            ...         self._cursor = cursor
            ...     def executemany(self, statement, parameters, **kwargs):
            ...         # convert parameters to a list
            ...         parameters = list(parameters)
            ...         # pass through to proxied cursor
            ...         return self._cursor.executemany(statement, parameters, **kwargs)
            ...     def __getattr__(self, item):
            ...         return getattr(self._cursor, item)
            ...
            >>> def get_cursor():
            ...     return CursorProxy(connection.cursor())
            ...
            >>> import petl as etl
            >>> etl.todb(tbl, get_cursor, ...)

        Note however that this does imply loading the entire table into
        memory as a list prior to inserting into the database.

    """

    needs_closing = False

    # atomic mode is only implemented for the standard library sqlite3 driver
    if atomic:
        if create:
            raise ArgumentError(
                'atomic=True is not supported with create=True (including '
                'drop=True)')
        # a connection owned by petl is closed at the end of this call, so an
        # uncommitted atomic load would be silently lost
        if isinstance(dbo, string_types) and not commit:
            raise ArgumentError(
                'atomic=True with commit=False is not supported when dbo is '
                'a SQLite file path: petl closes the connection it opens, so '
                'uncommitted data would be lost; pass commit=True or supply '
                'your own sqlite3 connection')

    # convenience for working with sqlite3
    if isinstance(dbo, string_types):
        import sqlite3
        dbo = sqlite3.connect(dbo)
        needs_closing = True

    try:
        if create:
            if drop:
                if_exists = drop == 'if_exists'
                drop_table(dbo, tablename, schema=schema, commit=commit,
                           if_exists=if_exists)
            create_table(table, dbo, tablename, schema=schema, commit=commit,
                         constraints=constraints, metadata=metadata,
                         dialect=dialect, sample=sample)
        _todb(table, dbo, tablename, schema=schema, commit=commit,
              truncate=truncate, atomic=atomic)

    finally:
        if needs_closing:
            dbo.close()


Table.todb = todb


def _todb(table, dbo, tablename, schema=None, commit=True, truncate=False,
          atomic=False):

    # atomic mode only supports the standard library sqlite3 driver
    if atomic:
        _todb_sqlite3_atomic(table, dbo, tablename, schema=schema,
                             commit=commit, truncate=truncate)
        return

    # need to deal with polymorphic dbo argument
    # what sort of duck is it?

    if _is_clikchouse_dbapi_connection(dbo):
        debug('assuming %r is clickhosue DB-API 2.0 connection', dbo)
        _todb_clikchouse_dbapi_connection(table, dbo, tablename, schema=schema,
                               commit=commit, truncate=truncate)

    # does it quack like a standard DB-API 2.0 connection?
    elif _is_dbapi_connection(dbo):
        debug('assuming %r is standard DB-API 2.0 connection', dbo)
        _todb_dbapi_connection(table, dbo, tablename, schema=schema,
                               commit=commit, truncate=truncate)

    # does it quack like a standard DB-API 2.0 cursor?
    elif _is_dbapi_cursor(dbo):
        debug('assuming %r is standard DB-API 2.0 cursor')
        _todb_dbapi_cursor(table, dbo, tablename, schema=schema, commit=commit,
                           truncate=truncate)

    # does it quack like an SQLAlchemy engine?
    elif _is_sqlalchemy_engine(dbo):
        debug('assuming %r instance of sqlalchemy.engine.base.Engine', dbo)
        _todb_sqlalchemy_engine(table, dbo, tablename, schema=schema,
                                commit=commit, truncate=truncate)

    # does it quack like an SQLAlchemy session?
    elif _is_sqlalchemy_session(dbo):
        debug('assuming %r instance of sqlalchemy.orm.session.Session', dbo)
        _todb_sqlalchemy_session(table, dbo, tablename, schema=schema,
                                 commit=commit, truncate=truncate)

    # does it quack like an SQLAlchemy connection?
    elif _is_sqlalchemy_connection(dbo):
        debug('assuming %r instance of sqlalchemy.engine.base.Connection', dbo)
        _todb_sqlalchemy_connection(table, dbo, tablename, schema=schema,
                                    commit=commit, truncate=truncate)

    elif callable(dbo):
        debug('assuming %r is a function returning standard DB-API 2.0 cursor '
              'objects', dbo)
        _todb_dbapi_mkcurs(table, dbo, tablename, schema=schema, commit=commit,
                           truncate=truncate)

    # some other sort of duck...
    else:
        raise ArgumentError('unsupported database object type: %r' % dbo)


SQL_TRUNCATE_QUERY = 'DELETE FROM %s'
SQL_INSERT_QUERY = 'INSERT INTO %s (%s) VALUES (%s)'


def _close_load_cursor(cursor, load_failed):
    try:
        cursor.close()
    except Exception:
        if not load_failed:
            raise
        debug('could not close cursor after failed load', exc_info=True)


def _todb_sqlite3_atomic(table, dbo, tablename, schema=None, commit=True,
                         truncate=False):
    """Atomic load for standard library sqlite3 connections only.

    The truncate and the whole insert are wrapped in a single transaction
    boundary: a SAVEPOINT when the caller already owns a transaction, or a
    fresh transaction otherwise. Any failure rolls back exactly this load
    without touching the caller's outer transaction.
    """

    owns_cursor = False
    if _is_sqlite3_connection(dbo):
        connection = dbo
        cursor = connection.cursor()
        owns_cursor = True
    elif _is_sqlite3_cursor(dbo):
        cursor = dbo
        connection = cursor.connection
    elif callable(dbo):
        # the factory contract is one new cursor per call
        cursor = dbo()
        if not _is_sqlite3_cursor(cursor):
            # reject before any writes or row consumption, but release the
            # resource that this call just created, if any
            if cursor is not None and hasattr(cursor, 'close'):
                try:
                    cursor.close()
                except Exception:
                    debug('could not close non-sqlite3 factory cursor',
                          exc_info=True)
            raise ArgumentError(
                'atomic=True requires a standard library sqlite3 cursor, '
                'factory returned: %r' % cursor)
        connection = cursor.connection
        owns_cursor = True
    else:
        raise ArgumentError(
            'atomic=True is only supported for standard library sqlite3 '
            'connections, cursors, cursor factories and SQLite file paths, '
            'unsupported database object type: %r' % dbo)

    # sanitise table name
    quoted_tablename = _quote(tablename)
    if schema is not None:
        quoted_tablename = _quote(schema) + '.' + quoted_tablename
    debug('tablename: %r', quoted_tablename)

    # decide where the transaction boundary sits before touching any data
    outer_transaction = _sqlite3_in_transaction(connection)
    savepoint = None
    tx_cursor = None
    load_failed = True
    try:
        # sanitise field names (may itself fail on a broken source; the
        # cursors we created must still be released)
        it = iter(table)
        hdr = next(it)
        flds = list(map(text_type, hdr))
        colnames = [_quote(n) for n in flds]
        debug('column names: %r', colnames)

        # sqlite3 uses the qmark paramstyle
        placeholders = _placeholders(connection, colnames)
        debug('placeholders: %r', placeholders)

        tx_cursor = connection.cursor()
        if outer_transaction:
            # nest inside the caller's transaction; never commit or roll it
            # back, and use a unique name so nested loads do not collide
            savepoint = _next_savepoint_name()
            debug('atomic load inside existing transaction, savepoint %r',
                  savepoint)
            tx_cursor.execute('SAVEPOINT %s' % savepoint)
            # a deferred foreign key violation only surfaces when the
            # outermost savepoint is released, and by then SQLite has popped
            # the savepoint, so the load can no longer be undone on its own.
            # Snapshot the violations already present so that, before the
            # RELEASE, we only react to violations introduced by this load
            # (native COMMIT likewise tolerates violations that predate the
            # transaction).
            fk_snapshot = _sqlite3_foreign_key_violations(tx_cursor)
        else:
            # open a transaction ourselves; works both with the default
            # isolation handling and with isolation_level=None
            debug('atomic load opens a new transaction')
            tx_cursor.execute('BEGIN')
            fk_snapshot = None

        try:
            if truncate:
                truncatequery = SQL_TRUNCATE_QUERY % quoted_tablename
                debug('truncate the table via query %r', truncatequery)
                cursor.execute(truncatequery)

            insertcolnames = ', '.join(colnames)
            insertquery = SQL_INSERT_QUERY % (quoted_tablename,
                                              insertcolnames, placeholders)
            debug('insert data via query %r' % insertquery)
            # executemany consumes the source iterator lazily in one pass
            cursor.executemany(insertquery, it)
        except BaseException:
            # undo exactly this load, preserving any outer transaction;
            # a rollback failure must not mask the original load error
            _sqlite3_undo_load(tx_cursor, savepoint)
            raise

        if savepoint is not None:
            # detect deferred FK failures before the RELEASE pops the
            # outermost savepoint, so this load can still be rolled back on
            # its own without touching the caller's outer transaction
            new_violations = _sqlite3_new_foreign_key_violations(
                tx_cursor, fk_snapshot)
            if new_violations:
                debug('atomic load introduced deferred foreign key '
                      'violations: %r', new_violations)
                _sqlite3_undo_load(tx_cursor, savepoint)
                raise sqlite3.IntegrityError('FOREIGN KEY constraint failed')
            # leave the caller's transaction open for its own
            # commit/rollback; commit is intentionally ignored here
            debug('release savepoint %r after successful atomic load',
                  savepoint)
            try:
                tx_cursor.execute('RELEASE %s' % savepoint)
            except BaseException:
                # defensive: if the RELEASE itself fails after the checks
                # above, undo the load where a boundary still exists; the
                # original RELEASE exception is never replaced
                _sqlite3_undo_load(tx_cursor, savepoint, raise_on_error=False)
                raise
        elif commit:
            debug('commit transaction after successful atomic load')
            try:
                tx_cursor.execute('COMMIT')
            except BaseException:
                # a COMMIT can fail after every row was written, e.g. a
                # deferred constraint check or "database is locked"; the
                # transaction is still open at this point, so roll it back
                # to restore the pre-load contents and return the
                # connection to its entry state (no active transaction)
                debug('commit of atomic load failed, rolling back',
                      exc_info=True)
                _sqlite3_undo_load(tx_cursor, None)
                raise
        # else: leave the transaction we opened uncommitted for the caller
        load_failed = False
    finally:
        # always release the transaction cursor we created; a borrowed data
        # cursor and the connection stay open for the caller
        if tx_cursor is not None:
            try:
                tx_cursor.close()
            except Exception:
                if not load_failed:
                    raise
                debug('could not close transaction cursor after failed load',
                      exc_info=True)
        if owns_cursor:
            debug('close the load cursor')
            _close_load_cursor(cursor, load_failed)


def _sqlite3_undo_load(tx_cursor, savepoint, raise_on_error=False):
    try:
        if savepoint is not None:
            debug('roll back savepoint %r after failed atomic load',
                  savepoint)
            tx_cursor.execute('ROLLBACK TO %s' % savepoint)
            tx_cursor.execute('RELEASE %s' % savepoint)
        else:
            debug('roll back transaction after failed atomic load')
            tx_cursor.execute('ROLLBACK')
    except BaseException:
        if raise_on_error:
            raise
        debug('could not roll back failed atomic load', exc_info=True)


def _sqlite3_foreign_key_enforced(tx_cursor):
    # PRAGMA foreign_keys is per-connection; only a 1 means deferred foreign
    # key enforcement can make a COMMIT or the outermost SAVEPOINT release
    # fail at transaction finalisation time
    row = tx_cursor.execute('PRAGMA foreign_keys').fetchone()
    return bool(row and row[0])


def _sqlite3_foreign_key_violations(tx_cursor):
    if not _sqlite3_foreign_key_enforced(tx_cursor):
        return frozenset()
    # PRAGMA foreign_key_check scans every attached database and works
    # inside an open transaction, seeing this connection's uncommitted writes
    return frozenset(tx_cursor.execute('PRAGMA foreign_key_check'))


def _sqlite3_new_foreign_key_violations(tx_cursor, earlier_violations):
    if earlier_violations is None:
        return frozenset()
    return _sqlite3_foreign_key_violations(tx_cursor) - earlier_violations


def _sqlite3_in_transaction(connection):
    # sqlite3.Connection.in_transaction reports whether the driver currently
    # holds an open transaction (including a caller-created SAVEPOINT)
    in_transaction = getattr(connection, 'in_transaction', None)
    if in_transaction is None:
        return False
    return bool(in_transaction)


def _todb_dbapi_connection(table, connection, tablename, schema=None,
                           commit=True, truncate=False):

    # sanitise table name
    tablename = _quote(tablename)
    if schema is not None:
        tablename = _quote(schema) + '.' + tablename
    debug('tablename: %r', tablename)

    # sanitise field names
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    colnames = [_quote(n) for n in flds]
    debug('column names: %r', colnames)

    # determine paramstyle and build placeholders string
    placeholders = _placeholders(connection, colnames)
    debug('placeholders: %r', placeholders)

    # get a cursor
    cursor = connection.cursor()

    load_failed = True
    try:
        if truncate:
            # TRUNCATE is not supported in some databases and causing locks with
            # MySQL used via SQLAlchemy, fall back to DELETE FROM for now
            truncatequery = SQL_TRUNCATE_QUERY % tablename
            debug('truncate the table via query %r', truncatequery)
            cursor.execute(truncatequery)
            # just in case, close and resurrect cursor
            closing_cursor, cursor = cursor, None
            closing_cursor.close()
            cursor = connection.cursor()

        insertcolnames = ', '.join(colnames)
        insertquery = SQL_INSERT_QUERY % (tablename, insertcolnames, placeholders)
        debug('insert data via query %r' % insertquery)
        cursor.executemany(insertquery, it)
        load_failed = False
    finally:
        if cursor is not None:
            debug('close the cursor')
            _close_load_cursor(cursor, load_failed)

    if commit:
        debug('commit transaction')
        connection.commit()


def _todb_clikchouse_dbapi_connection(table, connection, tablename, schema=None,
                           commit=True, truncate=False):

    # sanitise table name
    tablename = _quote(tablename)
    if schema is not None:
        tablename = _quote(schema) + '.' + tablename
    debug('tablename: %r', tablename)

    # sanitise field names
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    colnames = [_quote(n) for n in flds]
    debug('column names: %r', colnames)

    # determine paramstyle and build placeholders string
    placeholders = _placeholders(connection, colnames)
    debug('placeholders: %r', placeholders)

    # get a cursor
    cursor = connection.cursor()

    load_failed = True
    try:
        if truncate:
            # TRUNCATE is not supported in some databases and causing locks with
            # MySQL used via SQLAlchemy, fall back to DELETE FROM for now
            truncatequery = 'TRUNCATE TABLE IF EXISTS %s' % tablename
            debug('truncate the table via query %r', truncatequery)
            cursor.execute(truncatequery)
            # just in case, close and resurrect cursor
            closing_cursor, cursor = cursor, None
            closing_cursor.close()
            cursor = connection.cursor()

        insertcolnames = ', '.join(colnames)
        insertquery = 'INSERT INTO %s (%s) VALUES' % (tablename, insertcolnames)
        debug('insert data via query %r' % insertquery)
        cursor.executemany(insertquery, it)
        load_failed = False
    finally:
        if cursor is not None:
            debug('close the cursor')
            _close_load_cursor(cursor, load_failed)

    if commit:
        debug('commit transaction')
        connection.commit()


def _todb_dbapi_mkcurs(table, mkcurs, tablename, schema=None, commit=True,
                       truncate=False):

    # sanitise table name
    tablename = _quote(tablename)
    if schema is not None:
        tablename = _quote(schema) + '.' + tablename
    debug('tablename: %r', tablename)

    # sanitise field names
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    colnames = [_quote(n) for n in flds]
    debug('column names: %r', colnames)

    debug('obtain cursor and connection')
    cursor = mkcurs()

    load_failed = True
    try:
        # N.B., we depend on this optional DB-API 2.0 attribute being implemented
        assert hasattr(cursor, 'connection'), \
            'could not obtain connection via cursor'
        connection = cursor.connection

        # determine paramstyle and build placeholders string
        placeholders = _placeholders(connection, colnames)
        debug('placeholders: %r', placeholders)

        if truncate:
            # TRUNCATE is not supported in some databases and causing locks with
            # MySQL used via SQLAlchemy, fall back to DELETE FROM for now
            truncatequery = SQL_TRUNCATE_QUERY % tablename
            debug('truncate the table via query %r', truncatequery)
            cursor.execute(truncatequery)
            # N.B., may be server-side cursor, need to resurrect
            closing_cursor, cursor = cursor, None
            closing_cursor.close()
            cursor = mkcurs()

        insertcolnames = ', '.join(colnames)
        insertquery = SQL_INSERT_QUERY % (tablename, insertcolnames, placeholders)
        debug('insert data via query %r' % insertquery)
        cursor.executemany(insertquery, it)
        load_failed = False
    finally:
        if cursor is not None:
            _close_load_cursor(cursor, load_failed)

    if commit:
        debug('commit transaction')
        connection.commit()


def _todb_dbapi_cursor(table, cursor, tablename, schema=None, commit=True,
                       truncate=False):

    # sanitise table name
    tablename = _quote(tablename)
    if schema is not None:
        tablename = _quote(schema) + '.' + tablename
    debug('tablename: %r', tablename)

    # sanitise field names
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    colnames = [_quote(n) for n in flds]
    debug('column names: %r', colnames)

    debug('obtain connection via cursor')
    # N.B., we depend on this optional DB-API 2.0 attribute being implemented
    assert hasattr(cursor, 'connection'), \
        'could not obtain connection via cursor'
    connection = cursor.connection

    # determine paramstyle and build placeholders string
    placeholders = _placeholders(connection, colnames)
    debug('placeholders: %r', placeholders)

    if truncate:
        # TRUNCATE is not supported in some databases and causing locks with
        # MySQL used via SQLAlchemy, fall back to DELETE FROM for now
        truncatequery = SQL_TRUNCATE_QUERY % tablename
        debug('truncate the table via query %r', truncatequery)
        cursor.execute(truncatequery)

    insertcolnames = ', '.join(colnames)
    insertquery = SQL_INSERT_QUERY % (tablename, insertcolnames, placeholders)
    debug('insert data via query %r' % insertquery)
    cursor.executemany(insertquery, it)

    # N.B., don't close the cursor, leave that to the application

    if commit:
        debug('commit transaction')
        connection.commit()


def _todb_sqlalchemy_engine(table, engine, tablename, schema=None, commit=True,
                            truncate=False):

    with engine.connect() as connection:
        _todb_sqlalchemy_connection(table, connection, tablename,
                                    schema=schema, commit=commit, truncate=truncate)


def _todb_sqlalchemy_connection(table, connection, tablename, schema=None,
                                commit=True, truncate=False):

    debug('connection: %r', connection)

    # sanitise table name
    tablename = _quote(tablename)
    if schema is not None:
        tablename = _quote(schema) + '.' + tablename
    debug('tablename: %r', tablename)

    # sanitise field names
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    colnames = [_quote(n) for n in flds]
    debug('column names: %r', colnames)

    # N.B., we need to obtain a reference to the underlying DB-API connection so
    # we can import the module and determine the paramstyle
    proxied_raw_connection = connection.connection
    if hasattr(proxied_raw_connection, 'driver_connection'):
        actual_raw_connection = proxied_raw_connection.driver_connection
    else:
        actual_raw_connection = proxied_raw_connection.connection

    # determine paramstyle and build placeholders string
    placeholders = _placeholders(actual_raw_connection, colnames)
    debug('placeholders: %r', placeholders)

    with _sqlalchemy_transaction(connection, commit):
        if truncate:
            # TRUNCATE is not supported in some databases and causes locks.
            truncatequery = SQL_TRUNCATE_QUERY % tablename
            debug('truncate the table via query %r', truncatequery)
            _execute_sqlalchemy(connection, truncatequery).close()

        insertcolnames = ', '.join(colnames)
        insertquery = SQL_INSERT_QUERY % (tablename, insertcolnames, placeholders)
        debug('insert data via query %r' % insertquery)
        for row in it:
            _execute_sqlalchemy(connection, insertquery, tuple(row)).close()

    # N.B., don't close connection, leave that to the application


def _todb_sqlalchemy_session(table, session, tablename, schema=None,
                             commit=True, truncate=False):

    with _sqlalchemy_transaction(session, commit):
        _todb_sqlalchemy_connection(table, session.connection(), tablename,
                                    schema=schema, commit=False,
                                    truncate=truncate)


def appenddb(table, dbo, tablename, schema=None, commit=True, atomic=False):
    """
    Load data into an existing database table via a DB-API 2.0
    connection or cursor. As :func:`petl.io.db.todb` except that the database
    table will be appended, i.e., the new data will be inserted into the
    table, and any existing rows will remain.

    Pass ``atomic=True`` for an atomic load backed by standard library
    :mod:`sqlite3`; see :func:`petl.io.db.todb` for the full transaction
    ownership, supported object and failure contracts. Defaults to False,
    preserving the historical non-atomic behaviour.

    """

    needs_closing = False

    if atomic and isinstance(dbo, string_types) and not commit:
        # a connection owned by petl is closed at the end of this call, so an
        # uncommitted atomic load would be silently lost
        raise ArgumentError(
            'atomic=True with commit=False is not supported when dbo is '
            'a SQLite file path: petl closes the connection it opens, so '
            'uncommitted data would be lost; pass commit=True or supply '
            'your own sqlite3 connection')

    # convenience for working with sqlite3
    if isinstance(dbo, string_types):
        import sqlite3
        dbo = sqlite3.connect(dbo)
        needs_closing = True

    try:
        _todb(table, dbo, tablename, schema=schema, commit=commit,
              truncate=False, atomic=atomic)

    finally:
        if needs_closing:
            dbo.close()


Table.appenddb = appenddb
