# AGENTS.md

Guidance for AI coding agents working in the **BloombergFetch** repository.

## Project overview

`bbg-fetch` wraps the Bloomberg Desktop API (`blpapi`) and returns prices, implied
volatilities, fundamentals, and index constituents as analysis-ready pandas DataFrames.
It is a thin request/response layer with no streaming support and only two runtime
dependencies (numpy, pandas) besides `blpapi` itself.

Distribution name `bbg-fetch`; import name `bbg_fetch`. Licensed MIT (`LICENSE.txt`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
bbg_fetch/
  core.py       public fetch functions returning DataFrames
  _blp_api.py   direct blpapi session handling (private)
  tests/        tests that require a Bloomberg connection
tests/
  test_pure.py  tests that run without a terminal
examples/       runnable examples
```

## Commands

```bash
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
pip install -e ".[dev]"
pytest tests/          # terminal-free tests only, as CI runs them
ruff check .           # lint
```

Supported Python is >= 3.9; CI runs 3.12.

## Conventions

- Terminal-free tests go in the top-level `tests/` directory and are named `test_*.py`.
  Tests that need a live Bloomberg session go in `bbg_fetch/tests/`.
- Line length is not enforced (`ruff` rules `E`, `F`, `W` with `E501` ignored) because
  existing code has many long field-name lines.
- Public functions return pandas objects with a `DatetimeIndex`; Bloomberg field names
  are passed through rather than renamed, so callers can match them to the terminal.
- `blpapi` access is confined to `_blp_api.py`. Public API lives in `core.py`.

## Constraints — do not do these

- Do not mock `blpapi` to make terminal-dependent tests pass in CI. Tests that need a
  session belong in `bbg_fetch/tests/` and are not run by CI by design.
- Do not add streaming or subscription support: this package is request/response only.
- Do not add runtime dependencies. Anything beyond numpy, pandas and `blpapi` needs a
  strong justification — `xbbg` was deliberately removed in favour of direct `blpapi`.
- Do not commit fetched Bloomberg data, credentials, or terminal output. Bloomberg data
  is licensed and must not enter the repository.
- Do not hardcode tickers, field names, or entitlement assumptions into library code.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.
