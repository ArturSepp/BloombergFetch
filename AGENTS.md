## Python environment (mandatory)

- Never create, use, or install packages into a Python virtual environment anywhere under `C:\Users\artur\OneDrive`.
- Keep this repository's environment outside OneDrive at `C:\Python\BloombergFetch312`.
- Use `C:\Python\BloombergFetch312\Scripts\python.exe` for Python, tests, linters, and package installation.
- If it is missing, create it with `py -3.12 -m venv C:\Python\BloombergFetch312`.
- Never run plain `uv sync` or plain `uv run` from this checkout: uv otherwise creates `<repo>\.venv` even when uv was launched through a Python executable under `C:\Python`.
- If a uv project operation is required, first set `UV_PROJECT_ENVIRONMENT=C:\Python\BloombergFetch312`; for pip-style operations prefer `uv pip ... --python C:\Python\BloombergFetch312\Scripts\python.exe`.
- If any OneDrive-local environment already exists, do not use it; report it for removal.

# AGENTS.md

Guidance for AI coding agents working in the **BloombergFetch** repository.

## Project overview

`bbg-fetch` wraps the Bloomberg Desktop API (`blpapi`) and returns prices, implied
volatilities, fundamentals, and index constituents as analysis-ready pandas DataFrames.
It is a thin request/response layer with no streaming support and only two runtime
dependencies (numpy, pandas) besides `blpapi` itself.

Distribution name `bbg-fetch`; import name `bbg_fetch`. Licensed MIT (`LICENSE.txt`).

## Ecosystem position

This package is one of ten public Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Check the owning package before
adding a capability or copying code between repositories.

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | performance analytics, backtesting, and factsheet reporting |
| `optimalportfolios` | OptimalPortfolios | portfolio construction and rolling backtesting |
| `factorlasso` | FactorLasso | sparse factor-model estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data in pandas DataFrames |
| `stochvolmodels` | StochVolModels | stochastic-volatility pricing and calibration |
| `trendfollowing` | TrendFollowingSystems | closed-form trend-following analytics |
| `privateassets` | PrivateAssets | multi-factor PME for private assets |
| `goal-based-allocation` | GoalBasedAllocation | goal-based allocation under regime-switching jump-diffusions |
| `vanilla-option-pricers` | VanillaOptionPricers | Numba-vectorised BSM and Bachelier pricing |
| `option-chain-analytics` | OptionChainAnalytics | point-in-time option-chain data and queries |

Core dependency edges: `optimalportfolios` consumes `qis` and `factorlasso`;
`trendfollowing` and `privateassets` consume `qis`; `stochvolmodels` consumes
`vanilla-option-pricers`; `option-chain-analytics` consumes `qis` and
`vanilla-option-pricers`. The remaining packages have no core stack dependencies.

Optional edges: PrivateAssets' `factors` extra adds `factorlasso`; StochVolModels'
`research` extra adds `qis` and `option-chain-analytics`; OCA's `bloomberg` and `all`
extras add `bbg-fetch`. Core imports must work without optional dependencies.
OCA never imports StochVolModels or the private SigmaStrats consumer. Exact
maintainer-tool exceptions are recorded in `.github/stack-policy.json`; they do
not authorise adding those dependencies to core or importing them at package root.

## Repository layout

```
src/bbg_fetch/
  core.py         public fetch functions returning DataFrames
  option_chain.py option-chain fetching and parity recovery
  _blp_api.py     direct blpapi session handling (private)
  tests/          live-Bloomberg pytest modules (*_test.py)
  run_local/      source-checkout development runners (*_run.py; no __init__.py)
tests/
  test_pure.py    tests that run without a terminal
examples/         authoritative runnable examples at repository root
```

## Commands

```bash
uv sync --locked --group test
uv pip install --python .venv --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
uv run --no-sync pytest                    # terminal-free tests only, as CI runs them
uv run --locked --only-group lint ruff check src/bbg_fetch tests examples
```

Supported Python is >= 3.10; CI runs Linux 3.10–3.14 plus Windows and macOS 3.12.

## Conventions

- Terminal-free tests go in the top-level `tests/` directory and are named `test_*.py`.
  Tests that need a live Bloomberg session go in `src/bbg_fetch/tests/` and are named
  `*_test.py`. Every pytest module collects tests and has no executable main guard.
- Component development diagnostics live in `src/bbg_fetch/run_local/<subject>_run.py`, expose
  `Locals` and `run_local(local=...)`, and are excluded from built distributions. The
  `run_local` folder intentionally has no `__init__.py`; Python treats it as an implicit namespace
  for explicit source-checkout execution with `python -m`.
- Broader repository examples remain under `examples/` and use the same `Locals` /
  `run_local(local=...)` dispatcher names. Production modules and public `__init__.py` files never
  import `run_local`.
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
  session belong in `src/bbg_fetch/tests/` and are not run by CI by design.
- Do not add streaming or subscription support: this package is request/response only.
- Do not add runtime dependencies. Anything beyond numpy, pandas and `blpapi` needs a
  strong justification — `xbbg` was deliberately removed in favour of direct `blpapi`.
- Do not commit fetched Bloomberg data, credentials, or terminal output. Bloomberg data
  is licensed and must not enter the repository.
- Do not hardcode tickers, field names, or entitlement assumptions into library code.

<!-- ===== SHARED AGENT CORE (standalone variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-09-08, agent core v1.5 -->

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

For an authorized publication: commit, tag that exact main-reachable commit as
`v<version>`, then build, verify and publish its artifacts. Frequent PyPI updates are
supported. A GitHub Release page is optional and created only when requested; it is not
required for a local build, pip installation or routine package publication. Development
versions on main may be ahead of PyPI. Do not publish or bump a version for unrelated work.

## Temporary workspace hygiene

- Pytest's cache is configured outside OneDrive at `~/.cache/bbg-fetch/pytest`.
- Never create `tmp/`, `.pytest*`, or `.codex*` work directories inside the checkout.
- Use a task-specific directory below `tempfile.gettempdir()` under
  `BloombergFetch/<task-id>`, and remove it after verification.
