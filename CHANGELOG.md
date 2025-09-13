# Changelog

## [0.1.2] - 2025-09-13
- Tests: +unit (rate_quote, memory) and gated online integration test (@slow, skipped in CI)
- Coverage: ~45% locally; CI gate stays at 30–35%
- Docs: README Day 10 final polish + offline mode & testing notes
- Robustness: backend.main soft-degrades when model load fails (offline path)


## [0.1.1] - 2025-09-12
### Added
- CI caches (pip, HF, torch) and deterministic env flags.
- Unit tests for `parse_tracking` and `settings`, smoke tests for `/chat` + streaming.
- Docker smoke job on Ubuntu.
- README CI badge.

### Changed
- Offline-safe implementations for `backend/search.py` and `backend/tools/search_kb.py` (lazy loading, no import-time downloads).
- Fixed mypy stability in `backend/generation.py` (scoped ignores) and cleaned tokenizer calls.
- Import cleanup and formatting (ruff/black), mypy configuration adjustments.

### Fixed
- mypy errors in `main.py` (Callable import and alias redefinition).
- CI failures from HF downloads by short-circuiting in offline mode and stubbing in tests.

## [0.1.0] - 2025-09-11
- Initial stable CI + smoke tests baseline.
