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
- **Two invariants are enforced by ruff rather than written down**, both green on the package, so
  a violation is always something you just introduced:
  - `TID251` fails any import of `qis`, `optimalportfolios`, `factorlasso`, `trendfollowing` or
    `privateassets`. `bbg-fetch` is a leaf with no stack dependencies in either direction: a
    consumer imports this package, and this package imports nothing from the stack. Analytics on
    fetched data belong in the consumer, not here.
  - `ICN` pins `import numpy as np` and `import pandas as pd`.
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

<!-- ===== SHARED AGENT CORE (standalone variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-08-08, agent core v1.2. -->

## Dependency surface

This package is a leaf: it imports nothing from the stack (see Conventions, `TID251`), and its
runtime surface — numpy, pandas and `blpapi` — is a design constraint, not a preference. Ask
before adding any dependency.

**Never invent a symbol.** If a function, class, or keyword argument is not in the export
surface of this package or of a dependency, it does not exist. Check in one line —
`python -c "import bbg_fetch; print([n for n in dir(bbg_fetch) if not n.startswith('_')])"`
— and say a symbol is missing rather than producing code that calls it.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean. Verify against a reference computed a different way, and say which.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Roadmap execution

Feature roadmaps live at the repository root as `ROADMAP_<feature>.md`. An execution request
names the file and the stage. A stage is complete when its stated verification command passes;
its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.
