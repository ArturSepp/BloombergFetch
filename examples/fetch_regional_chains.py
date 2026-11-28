"""Archive metadata-verified near-year regional listed options through public Bloomberg API.

Current listings are a discovery snapshot, never a point-in-time historical universe.
Pricing analysis belongs in the consuming research repository.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re

import bbg_fetch as bbg
import pandas as pd


class Locals(Enum):
    """Available acquisition tasks."""

    SNAPSHOT = "snapshot"


def acquire(output, date):
    """Save listings, actual contract metadata and same-day historical bid/ask quotes."""
    output.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp(date)
    manifest = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                "quote_date": date, "universe": "current OPT_CHAIN; not historical PIT",
                "markets": []}
    try:
        for asset in ("sx5e", "ukx", "nky"):
            chain = bbg.bds(f"{asset.upper()} Index", "OPT_CHAIN")
            chain.to_csv(output / f"{asset}_opt_chain.csv", index=False)
            parsed = []
            if not chain.empty:
                for value in chain.iloc[:, 0].astype(str):
                    match = re.search(r" (\d{2}/\d{2}/\d{2}) ([CP])(\d+(?:\.\d+)?)", value)
                    if match:
                        expiry = pd.to_datetime(match[1], format="%m/%d/%y")
                        parsed.append({"ticker": value, "expiry_hint": expiry,
                                       "side_hint": match[2], "strike_hint": float(match[3])})
            listing = pd.DataFrame(parsed)
            if listing.empty:
                manifest["markets"].append({"asset": asset, "status": "no_parseable_listings"})
                continue
            expiries = listing.expiry_hint.unique()
            eligible = [pd.Timestamp(x) for x in expiries if 270 <= (pd.Timestamp(x)-asof).days <= 550]
            if not eligible:
                manifest["markets"].append({"asset": asset, "status": "no_near_year_listing"})
                continue
            expiry = min(eligible, key=lambda x: (abs((x-asof).days-365), x))
            candidates = listing[listing.expiry_hint == expiry].copy()
            tickers = candidates.ticker.tolist()
            fields = ["OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPT_STRIKE_PX", "CRNCY",
                      "OPT_UNDL_TICKER", "OPT_EXER_TYP", "SECURITY_DES"]
            pieces = []
            quotes = []
            for start in range(0, len(tickers), 100):
                batch = tickers[start:start+100]
                pieces.append(bbg.bdp(batch, fields))
                quotes.append(bbg.bdh(batch, ["PX_BID", "PX_ASK", "PX_LAST", "OPEN_INT"],
                                      start_date=date, end_date=date))
            metadata = pd.concat(pieces)
            metadata.to_csv(output / f"{asset}_metadata.csv")
            history = pd.concat(quotes, axis=1)
            history.to_csv(output / f"{asset}_quotes.csv")
            entry = {"asset": asset, "listings": len(chain), "candidate_contracts": len(tickers),
                     "expiry_hint": str(expiry.date()), "days": (expiry-asof).days,
                     "metadata_columns": list(metadata.columns), "quote_dates": len(history)}
            # Metadata is authoritative: report mismatches instead of quietly relabeling.
            field = next((c for c in metadata if c.lower() == "opt_expire_dt"), None)
            actual = pd.to_datetime(metadata[field], errors="coerce") if field else pd.Series(dtype="datetime64[ns]")
            entry["verified_expiries"] = bool(len(actual) == len(tickers) and (actual == expiry).all())
            # Active-contract history is only an availability probe, not an empirical panel.
            sample = candidates.groupby("side_hint").apply(
                lambda x: x.iloc[len(x)//2], include_groups=False).ticker.tolist()
            probe = bbg.bdh(sample, ["PX_BID", "PX_ASK"], start_date="2025-11-18", end_date=date)
            probe.to_csv(output / f"{asset}_active_contract_history_probe.csv")
            entry["active_history_rows"] = len(probe)
            manifest["markets"].append(entry)
            (output / "listed_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(entry), flush=True)
    finally:
        bbg.disconnect()
    return manifest


def run_local(local=Locals.SNAPSHOT):
    """Run acquisition with explicit portable input/output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    if local == Locals.SNAPSHOT:
        acquire(args.output, args.date)


if __name__ == "__main__":
    run_local(local=Locals.SNAPSHOT)
