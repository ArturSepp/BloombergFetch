"""NO TERMINAL: verify installation with a deterministic public-API workflow."""

import platform
from enum import Enum
from importlib.metadata import version

import numpy as np
import pandas as pd

import bbg_fetch


SPOT = 100.0
FORWARD = 102.0
RATE = 0.03
YEAR_FRACTION = 0.25


class Locals(Enum):
    """Available terminal-free example workflows."""

    QUICKSTART = 1


def _synthetic_option_chain() -> pd.DataFrame:
    """Create call/put prices satisfying put-call parity exactly."""
    discount = np.exp(-RATE * YEAR_FRACTION)
    rows = []
    for strike in np.array([90.0, 95.0, 100.0, 105.0, 110.0]):
        put = 12.0 + 0.02 * np.square(strike - SPOT)
        call = put + discount * (FORWARD - strike)
        rows.extend((
            {"opt_put_call": "Call", "opt_strike_px": strike, "px_last": call},
            {"opt_put_call": "Put", "opt_strike_px": strike, "px_last": put},
        ))
    return pd.DataFrame(rows)


def run_local(local: Locals) -> None:
    """Run the terminal-free installation and public-API check."""
    if local != Locals.QUICKSTART:
        raise NotImplementedError(f"unsupported local: {local}")
    recovered = bbg_fetch.recover_option_forward(
        option_chain=_synthetic_option_chain(),
        spot=SPOT,
        year_fraction=YEAR_FRACTION,
        price_source=bbg_fetch.OptionPriceSource.LAST,
    )

    if not np.isclose(recovered["forward"], FORWARD, atol=1e-10):
        raise RuntimeError(f"unexpected forward: {recovered['forward']}")
    if not np.isclose(recovered["rate"], RATE, atol=1e-10):
        raise RuntimeError(f"unexpected rate: {recovered['rate']}")

    print("installation/API check: PASS")
    print(
        f"versions: python={platform.python_version()}; "
        f"bbg-fetch={bbg_fetch.__version__}; blpapi={version('blpapi')}"
    )
    print(
        f"synthetic forward={recovered['forward']:.2f}; "
        f"rate={recovered['rate']:.2%}; "
        f"strikes={int(recovered['num_strikes_used'])}"
    )
    print("Bloomberg connection: NOT TESTED")


if __name__ == "__main__":
    run_local(local=Locals.QUICKSTART)
