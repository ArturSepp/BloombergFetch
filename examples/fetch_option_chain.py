"""
fetch a KOSPI2 option chain for one expiry and recover the implied forward and rate.

Run on the Bloomberg machine. bbg_fetch.option_chain.run() does the work — fetch the
chain, infer spot and the year fraction from it, recover the parity forward and rate —
and returns an OptionChainResult. This script calls it and prints the result. Choose the
strikes either by num_strikes_per_side (an ATM window) or by an explicit strike_grid.
"""
# packages
import numpy as np
# bbg
from bbg_fetch.option_chain import run, OptionPriceSource


if __name__ == '__main__':
    strike_grid = np.linspace(700, 2200, 31)
    print(strike_grid)
    result = run(underlying='KOSPI2 Index',
                 expiry='20260803',
                 strike_grid=np.linspace(600, 2300, 35),
                 price_source=OptionPriceSource.LAST)

    print(f"{len(result.chain.index)} options")
    print(result.chain.to_string())
    print(f"\nspot={result.spot:.2f}  year_fraction={result.year_fraction:.4f}")
    print(f"forward={result.forward:.2f}  rate={result.rate:.4%}  "
          f"r2={result.r2:.4f}  num_strikes_used={result.num_strikes_used}")
    result.to_csv('kospi2_20260803.csv')