# Contributing to BloombergFetch

Thanks for your interest in `bbg-fetch`. The package is a small request/response wrapper around
Bloomberg's Desktop API, so a narrow dependency surface and a clean separation between
terminal-free checks and licensed-data diagnostics are part of its contract.

## Scope

In scope:

- Bug fixes in request construction, response parsing, DataFrame assembly, or option-chain helpers
- Compatibility improvements for supported Python and `blpapi` releases
- Documentation, deterministic examples, packaging, and terminal-free tests
- Clearer errors for connection, entitlement, field, or response-shape failures

Please open an issue before writing code for changes that add streaming/subscription support,
runtime dependencies, public-API breaks, or terminal-specific behavior. Do not submit fetched
Bloomberg data, credentials, entitlement details, terminal output, or tests that require a paid
session in the ordinary CI lane.

## Reporting a bug

Use the bug-report template and include the `bbg-fetch` version, Python version, operating system,
installation command, a minimal reproducer, and the full traceback. Replace licensed data with a
small synthetic DataFrame wherever possible. If the failure requires a live terminal, describe the
request shape and error without attaching proprietary responses.

## Development setup

Bloomberg distributes `blpapi` from its own public index rather than PyPI, so it is installed after
the locked project environment:

```bash
git clone https://github.com/ArturSepp/BloombergFetch.git
cd BloombergFetch
uv sync --locked --group test
uv pip install --python .venv --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/bbg_fetch tests examples
```

The default pytest configuration runs only `tests/`, the terminal-free suite. Live Bloomberg
assertions remain under `src/bbg_fetch/tests/` and are maintainer-run with a suitable Desktop API
session; do not mock the terminal to move them into ordinary CI. Source-checkout diagnostics under
`src/bbg_fetch/run_local/` are also excluded from built distributions.

Build documentation with the same warning gate used in CI:

```bash
uv sync --locked --extra docs
uv pip install --python .venv --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
uv run --no-sync python -m sphinx -E -W --keep-going -b html docs docs/_build/html
uv run --no-sync python -m sphinx -E -W -b linkcheck docs docs/_build/linkcheck
```

## Pull requests

- Keep one focused topic per pull request.
- Add a terminal-free regression test for changed behavior.
- Preserve Bloomberg field names, pandas return conventions, and lazy session creation.
- Do not commit licensed data, credentials, local paths, generated output, or terminal diagnostics.
- Run the terminal-free suite, Ruff, and any relevant docs or wheel checks before submitting.
- Do not bump package or citation versions; releases are handled separately.
- Call out any public-signature or dependency change explicitly.

## Conduct and licence

Be civil and assume good faith. By contributing, you agree that your contribution is licensed
under this project's MIT licence.
