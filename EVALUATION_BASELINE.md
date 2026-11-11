# Evaluation baseline provenance

This is an isolated snapshot of petl, a real Python ETL library.

- Upstream: https://github.com/petl-developers/petl
- Fixed upstream commit: 9cc6c751e0a68c034b270478599cb1be351c530e
- Retrieved as the upstream source tarball; no upstream Git history is claimed.
- Upstream MIT license is retained in LICENSE.txt.
- Task: add an opt-in SQLite atomic-loading transaction contract to the existing database writer.
- petl/version.py is a local packaging shim for this source archive, which lacks generated SCM version metadata. It is not a change to the database writer.

Run database tests from the repository root with Python 3 and pytest; SQLAlchemy is needed by some existing optional database tests. No external database server is required for the SQLite scope.
