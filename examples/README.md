# Runnable examples

The scripts in this repository-root directory are the authoritative `bbg-fetch` examples. They
use the package's public API and are deliberately kept outside `src/bbg_fetch/` so they are not
installed as package modules. Each script exposes `Locals` and `run_local(local=...)`; its main
guard selects one explicit default workflow.

Install Bloomberg's Python API and the package before running any script:

```bash
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
python -m pip install bbg-fetch
```

## No Bloomberg Terminal required

| Script | Purpose |
|---|---|
| `quickstart_no_terminal.py` | Verifies installation and recovers a known forward/rate from a deterministic synthetic option chain |

```bash
python examples/quickstart_no_terminal.py
```

This checks that `blpapi` and `bbg-fetch` import correctly and exercises public package logic. It
does **not** prove that a Bloomberg session, data entitlement, or live request is available.

## Bloomberg Terminal required

The Terminal must be running on the same machine and logged in with the necessary data
entitlements. These are local diagnostics and are never run by CI.

| Script | Purpose | How to select the request |
|---|---|---|
| `diagnose_terminal.py` | One scalar request with redacted status, dimensions, schema, and failure category | Pass `--ticker` and `--field` |
| `fetch_core_data.py` | High-level price, reference, volatility, futures, fixed-income, and constituent fetchers | Select a `Locals` member in the `__main__` call |
| `fetch_div_history.py` | Trailing dividend yield for one instrument | Edit `tickers` in `run_local()` |
| `fetch_option_chain.py` | Option-chain fetch and put-call-parity recovery | Pass a current listed expiry with `--expiry YYYYMMDD` |

For example:

```bash
python examples/diagnose_terminal.py
python examples/diagnose_terminal.py --ticker "IBM US Equity" --field "SECURITY_NAME"
python examples/fetch_option_chain.py --expiry 20261231
```

The expiry above is illustrative and must be replaced with an expiry listed for the selected
underlying. The scripts print results to the terminal only. Do not redirect or commit Bloomberg
responses, CSV snapshots, credentials, or terminal output.
