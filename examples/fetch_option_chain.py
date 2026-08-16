"""BLOOMBERG TERMINAL REQUIRED: fetch an option chain and recover its forward.

Run on the Bloomberg machine. bbg_fetch.option_chain.run() does the work — fetch the
chain, infer spot and the year fraction from it, recover the parity forward and rate —
and returns an OptionChainResult. Pass a currently listed expiry on the command line.
"""

import argparse

import numpy as np

from bbg_fetch import OptionPriceSource, run


def _parse_args() -> argparse.Namespace:
    """Parse the live request parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default="KOSPI2 Index")
    parser.add_argument("--expiry", required=True, help="currently listed expiry in YYYYMMDD")
    parser.add_argument("--strike-min", type=float, default=600.0)
    parser.add_argument("--strike-max", type=float, default=2300.0)
    parser.add_argument("--num-strikes", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    """Run the entitled option-chain request selected on the command line."""
    args = _parse_args()
    strike_grid = np.linspace(args.strike_min, args.strike_max, args.num_strikes)
    result = run(
        underlying=args.underlying,
        expiry=args.expiry,
        strike_grid=strike_grid,
        price_source=OptionPriceSource.LAST,
    )

    print(f"rows={len(result.chain.index)} columns={result.chain.columns.to_list()}")
    print(
        f"spot={result.spot:.2f} year_fraction={result.year_fraction:.4f} "
        f"forward={result.forward:.2f} rate={result.rate:.4%} "
        f"r2={result.r2:.4f} num_strikes_used={result.num_strikes_used}"
    )


if __name__ == "__main__":
    main()
