"""BLOOMBERG TERMINAL REQUIRED: run one redacted Desktop API diagnostic."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

try:
    import bbg_fetch
except ModuleNotFoundError as error:
    if error.name != "blpapi":
        raise
    print("diagnostic: FAIL")
    print("category: import")
    print("next_step: install blpapi from Bloomberg's official package index")
    raise SystemExit(1) from error


DEFAULT_TICKER = "AAPL US Equity"
DEFAULT_FIELD = "PX_LAST"


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Make one Bloomberg Desktop API scalar request and print no returned value. "
            "Bloomberg Professional must be open, logged in, and entitled."
        )
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Bloomberg security identifier (default: {DEFAULT_TICKER!r})",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help=f"Bloomberg scalar field mnemonic (default: {DEFAULT_FIELD!r})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one live request and report only status, dimensions, and schema."""
    args = _parser().parse_args(argv)
    try:
        frame = bbg_fetch.bdp(args.ticker, args.field)
    except ConnectionError:
        print("diagnostic: FAIL")
        print("category: session")
        print("next_step: open and log in to Bloomberg Professional, then check localhost:8194")
        return 1
    except TimeoutError as error:
        partial = len(getattr(error, "partial_messages", ()))
        print("diagnostic: FAIL")
        print("category: timeout")
        print(f"partial_message_count: {partial}")
        print("next_step: retry this one-security/one-field request before a larger request")
        return 1
    except Exception as error:
        print("diagnostic: FAIL")
        print("category: unexpected")
        print(f"exception_type: {type(error).__name__}")
        print("next_step: use the exception type and redacted request shape in a bug report")
        return 1
    finally:
        bbg_fetch.disconnect()

    has_data = not frame.empty and bool(frame.notna().any().any())
    print(f"diagnostic: {'PASS' if has_data else 'NO DATA'}")
    print(f"shape: rows={len(frame.index)}; columns={len(frame.columns)}")
    print(
        "schema: "
        f"columns={frame.columns.to_list()}; "
        f"index_name={frame.index.name!r}; "
        f"index_type={type(frame.index).__name__}"
    )
    if not has_data:
        print("category: empty/field/security/entitlement")
        print("next_step: verify the ticker, field, request type, and user entitlement")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
