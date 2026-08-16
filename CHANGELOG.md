# Changelog

All notable changes to bbg-fetch are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Migrated the import package from the repository root to `src/bbg_fetch/` while preserving the
  `bbg_fetch` import name and public API. CI now builds and installs the wheel before running the
  terminal-free suite, so repository-root import shadowing cannot hide packaging defects.
- Established `examples/` at repository root as the authoritative example surface, with a
  deterministic terminal-free installation check and clearly labelled live Bloomberg scripts.
- Aligned the package description, README opening, Python/platform support, and citation metadata
  on the current 2.3.0 request/response Desktop API identity. Installation guidance now points to
  Bloomberg's current API Library instead of pinning a stale wheel-version range.
- Added a Sphinx documentation foundation with installation, first-success, API, and
  troubleshooting pages, warning/link checks in CI, and an opt-in GitHub Pages deployment
  workflow. The package metadata will gain a Documentation URL only after the site is deployed.

### Fixed
- Every `axis=1` `pd.concat` in `bbg_fetch` states `sort=` explicitly, all three `sort=True`.
  Each series here arrives from its own Bloomberg request with its own history, so these are
  unions of differing DatetimeIndexes: pandas 2.2 sorted them whatever the argument said,
  pandas 3.0 honours an explicit `sort=False`, and pandas 4 drops the implicit sort as well. An
  unsorted price panel leaving this leaf reaches every package downstream, where it either
  raises deep inside a pandas resample or forward-fills a later price onto an earlier date and
  raises nothing. `fetch_vol_timeseries_per_ticker` (`core.py:522`), the rate join in
  `core.py:549` and the frame assembly in `_blp_api.py:342` are the three sites; the third was
  already followed by `sort_index()`.

### Added
- `tests/test_concat_sort_convention.py`: an `axis=1` `pd.concat` in `bbg_fetch` without an
  explicit `sort=` fails the suite. The check reads the source with `ast` and imports nothing,
  so it runs without a Bloomberg terminal.

## [2.3.0] - 2026-07-24

### Added
- `fetch_vol_surface()` returns the implied-vol surface for one date as a DataFrame indexed by
  tenor with moneyness columns (the OVDV moneyness grid), reshaping the same
  `{tenor}_IMPVOL_{mny}%MNY_DF` fields `fetch_vol_timeseries` uses. Each cell is the last quote
  on or before `value_date`. The default covers the five standard BVOL tenors
  (`30d, 60d, 3m, 6m, 12m`) at nine moneyness points; pass further tenor dicts to widen it.

## [2.2.0] - 2026-07-24

### Added
- New module `bbg_fetch.option_chain` (re-exported from the package top level) for option
  chain retrieval and put-call parity recovery.
- `fetch_option_chain()` returns a listed option chain for an underlying, one row per
  option indexed by ticker over the `OPTION_CHAIN_FIELDS` set. Strikes are selected
  either by `num_strikes_per_side` (a window around the ATM strike) or by an explicit
  `strike_grid` (the listed strike nearest each target value is kept); selection happens
  before the per-option `bdp`, bounding the hit count. `num_strikes_per_side=None` with
  no `strike_grid` fetches the full chain. `expiry` is validated as a `YYYYMMDD` date,
  so a malformed value raises instead of silently resolving to a different expiry.
- `recover_option_forward()` recovers the implied forward and rate from put-call parity
  (`C(K) - P(K) = exp(-r T)(F - K)`), returning `forward`, `rate`, `r2`,
  `num_strikes_used`. The forward is well determined; the rate is only indicative at
  short maturity (a 1e-3 error in the parity slope moves it by order 1%), so prefer a
  money-market curve for the rate when precision matters.
- `run()` fetches a chain and recovers the forward and rate in one call, inferring spot
  and the year fraction from the chain (`opt_undl_px`, `opt_expire_dt`), and returns an
  `OptionChainResult` snapshot.
- `OptionPriceSource` enum (`MID`, `LAST`), the `OptionChainResult` dataclass with
  `to_csv(path)` / `OptionChainResult.read_csv(path)` for a dependency-free single-file
  round trip (scalars as a commented header, the chain below), and the
  `OPTION_CHAIN_FIELDS` default field set.

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
