"""BLOOMBERG TERMINAL REQUIRED: fetch a trailing dividend yield."""

from enum import Enum

from bbg_fetch import fetch_div_yields


class Locals(Enum):
    """Available dividend-history example workflows."""

    DIVIDEND_YIELD = 1


def run_local(local: Locals) -> None:
    """Fetch and print the trailing dividend yield for the selected tickers."""
    if local == Locals.DIVIDEND_YIELD:
        tickers = ["FFASIAY LX Equity"]
        _, _, dividend_yield = fetch_div_yields(tickers=tickers)
        print(dividend_yield)
    else:
        raise NotImplementedError(f"unsupported local: {local}")


if __name__ == "__main__":
    run_local(local=Locals.DIVIDEND_YIELD)
