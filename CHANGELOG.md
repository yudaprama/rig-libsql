# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial `rig-libsql` companion crate: a Rig vector store backed by libSQL's
  built-in native vector support. Provides `LibsqlVectorStore`,
  `LibsqlVectorIndex`, the `LibsqlVectorStoreTable` schema trait, the
  `LibsqlSearchFilter` filter type, and `LibsqlDistanceMetric` (cosine /
  euclidean). No native extension is required — pass an asynchronous
  `libsql::Connection` (local, embedded replica, or remote/Turso).
