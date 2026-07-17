# Changelog

All notable changes to bbg-fetch are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.3] - 2026-07-17

### Changed
- `fetch_div_yields()` now returns a 3-tuple `(divs, divs_1y, divs_yield)`;
  previously `(divs, divs_1y)`. Callers unpacking two values must update. The
  new `divs_yield` is the trailing-twelve-month dividend divided by the
  dividend-UNADJUSTED price (`PX_LAST` fetched with cash adjustment off), so the
  denominator is the actual traded price that drops on each ex-date, not a
  total-return price — dividing by a cash-adjusted price would understate the
  yield. Returned as a decimal (`0.03 == 3%`) on the daily price grid, columns
  matching `divs`.

## [2.0.2] - 2026-05-02

Tags the same commit as 2.0.1.

### Added
- CI test workflow and docstrings across `core`.

### Changed
- README rewritten for the direct-`blpapi` interface.

## [2.0.1] - 2026-05-02

### Added
- Extended fetching of volatility data.

## [2.0.0] - 2026-03-25

### Changed
- Removed the `xbbg` dependency in favour of a direct `blpapi` interface.

---

Versions prior to 2.0.0 predate this changelog. Run `git log --tags --oneline`
for earlier history.
