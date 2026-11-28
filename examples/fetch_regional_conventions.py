"""Retrieve regional conventions, local OIS proxies and independent total-return indices.

This isolated BloombergFetch example writes private inputs only. Zero-curve and
surface discount data may be unavailable; the manifest preserves that limitation.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import pandas as pd


class Locals(Enum):
    """Provider evidence operation."""

    FETCH = "fetch"


MARKETS = {
    "sx5e": ("SX5E Index", "EUR", "SX5GT Index", "EESWE1 Curncy", "S0133Z 1Y BLC Curncy"),
    "ukx": ("UKX Index", "GBP", "TUKXG Index", "BPSWS1 BGN Curncy", "S0141Z 1Y BLC Curncy"),
    "nky": ("NKY Index", "JPY", "NKYTR Index", "JYSO1 BGN Curncy", "S0195Z 1Y BLC Curncy"),
}


def run_local(local=Locals.FETCH):
    """Freeze verified identifiers, tenor/moneyness checks and unfilled histories."""
    import bbg_fetch as bbg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2026-11-18")
    args = parser.parse_args()
    if local != Locals.FETCH:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        tickers = list(dict.fromkeys(t for cfg in MARKETS.values() for t in (cfg[0], *cfg[2:])))
        metadata = bbg.bdp(tickers, ["NAME", "CRNCY", "SECURITY_TYP", "DAY_CNT_DES"])
        metadata.to_csv(args.output / "security_metadata.csv")
        for asset, (ticker, currency, total, ois, zero) in MARKETS.items():
            assert metadata.loc[ticker, "crncy"] == currency
            assert metadata.loc[total, "crncy"] == currency
            assert metadata.loc[ois, "crncy"] == currency
            history = bbg.bdh(
                [total, ois, zero], "PX_LAST", start_date=args.start, end_date=args.end
            )
            history.columns = history.columns.get_level_values(0)
            history = history.rename(
                columns={
                    total: "total_return_check",
                    ois: "ois_1y_percent",
                    zero: "zero_1y_percent",
                }
            )
            history.to_csv(args.output / f"{asset}_carry_returns.csv")
            checks = []
            # Sample early history, normal conditions, crisis and recent history.
            for end in ("2010-11-01", "2015-06-30", "2020-03-31", args.end):
                for moneyness in (80.0, 100.0, 120.0):
                    fixed = next(
                        k
                        for k, label in bbg.IMPVOL_FIELDS_MNY_12M.items()
                        if label == f"12m{moneyness:.1f}"
                    )
                    values = bbg.bdh(
                        ticker,
                        ["IVOL_MONEYNESS", fixed],
                        start_date=pd.Timestamp(end) - pd.Timedelta(days=7),
                        end_date=end,
                        IVOL_MATURITY="MATURITY_360D",
                        IVOL_MONEYNESS_LEVEL=f"MONEY_LVL_{moneyness:.1f}".replace(".", "_"),
                    )
                    values.columns = values.columns.get_level_values(1)
                    values.to_csv(args.output / f"{asset}_tenor_{end}_{int(moneyness)}.csv")
                    pair = values.dropna()
                    error = (
                        float((pair.IVOL_MONEYNESS - pair[fixed]).abs().max())
                        if len(pair)
                        else None
                    )
                    checks.append(
                        {
                            "end": end,
                            "spot_moneyness_percent": moneyness,
                            "n": len(pair),
                            "max_difference_vol_percent": error,
                            "within_rounding": bool(error is not None and error <= 0.00501),
                        }
                    )
            surface = bbg.bds(ticker, "IVOL_SURFACE_MONEYNESS", IVOL_TIME_RANGE="1/1/12")
            surface.to_csv(args.output / f"{asset}_live_moneyness.csv", index=False)
            ratio = float(
                (surface.strike / surface.underlying_price - surface.moneyness).abs().max()
            )
            bulk = []
            for field, overrides in [
                ("IMP_FORWARD_PRICE", {"IVOL_SURFACE_SOURCE": "LIVE"}),
                ("END_OF_DAY_IMPLIED_FORWARD_PRICE", {}),
            ]:
                frame = bbg.bds(ticker, field, **overrides)
                frame.to_csv(args.output / f"{asset}_{field}.csv", index=False)
                bulk.append({"field": field, "rows": len(frame), "columns": list(frame.columns)})
            record = {
                "asset": asset,
                "ticker": ticker,
                "currency": currency,
                "total_return_ticker": total,
                "ois_ticker": ois,
                "ois_name": metadata.loc[ois, "name"],
                "ois_nonmissing": int(history.ois_1y_percent.notna().sum()),
                "zero_nonmissing": int(history.zero_1y_percent.notna().sum()),
                "total_return_nonmissing": int(history.total_return_check.notna().sum()),
                "moneyness_ratio_error": ratio,
                "tenor_checks": checks,
                "discount_bulk": bulk,
                "moneyness_error_index_points": float(
                    (surface.strike - surface.moneyness * surface.underlying_price).abs().max()
                ),
                "moneyness_tolerance_index_points": 0.01,
                "conventions_pass": bool(
                    (surface.strike - surface.moneyness * surface.underlying_price).abs().max()
                    <= 0.01
                    and all(c["within_rounding"] for c in checks)
                ),
            }
            records.append(record)
            print(
                f"{asset}: conventions={record['conventions_pass']}; OIS={record['ois_nonmissing']}; zero={record['zero_nonmissing']}; TR={record['total_return_nonmissing']}",
                flush=True,
            )
        manifest = {
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "markets": records,
            "start": args.start,
            "end": args.end,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "price_policy": "provider returned observations, no local forward filling",
            "rate_policy": "1Y local OIS par rates are proxy inputs, not recovered LIVE discount factors",
            "history_note": "EUR generic currently identifies ESTR; generic historical series can include legacy EONIA. No undocumented splice adjustment.",
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.output.glob("*.csv")
            },
        }
        (args.output / "conventions_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    finally:
        bbg.disconnect()


if __name__ == "__main__":
    run_local(local=Locals.FETCH)
