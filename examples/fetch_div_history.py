"""BLOOMBERG TERMINAL REQUIRED: fetch a trailing dividend yield."""

from bbg_fetch import fetch_div_yields


def main() -> None:
    """Fetch and print the trailing dividend yield for the selected tickers."""
    tickers = ["FFASIAY LX Equity"]
    _, _, dividend_yield = fetch_div_yields(tickers=tickers)
    print(dividend_yield)


if __name__ == "__main__":
    main()
