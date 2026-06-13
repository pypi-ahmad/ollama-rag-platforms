# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- AST-based SQL validation with `sqlglot`.
- Real end-to-end MCP transport tests (`stdio`, `streamable-http`, auth on/off).
- OSS governance docs: `LICENSE`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, `SECURITY`.

### Changed

- Timeout handling now interrupts DuckDB execution and enforces wall-clock bounds.
- Query pagination uses single-pass counting path with fallback for empty high-offset pages.
- Catalog scanning now caches by TTL and skips unsafe path-escape entries safely.
- Tool handlers now use Pydantic request models consistently and standardized error mapping.
- README updated to OSS-standard structure.
