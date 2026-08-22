## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact terminal-free commands run and their results. Do not paste licensed Bloomberg data
or terminal output.

## Checklist

- [ ] Tests cover the changed public behavior or defect without requiring a live terminal.
- [ ] Bloomberg field names, pandas return conventions, and lazy session creation remain intact.
- [ ] No streaming support or runtime dependency has been added implicitly.
- [ ] No credentials, proprietary data, local paths, generated outputs, or terminal diagnostics are included.
- [ ] `uv run --no-sync pytest` and the relevant lint/docs/wheel checks pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] New dependencies or public-signature changes are called out explicitly.
