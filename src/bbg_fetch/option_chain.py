"""
option chain retrieval and put-call parity recovery.

Fetch a listed option chain for an underlying (fetch_option_chain), recover the implied
forward and rate from put-call parity (recover_option_forward), or do both in one call
(run). Depends only on the bdp/bds interface, numpy and pandas.
"""
# packages
import re
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
# bbg
from bbg_fetch._blp_api import bdp, bds


OPTION_CHAIN_FIELDS: Tuple[str, ...] = ('security_des',
                                        'opt_put_call',
                                        'opt_strike_px',
                                        'opt_expire_dt',
                                        'opt_cont_size',
                                        'opt_undl_px',
                                        'px_bid', 'px_ask', 'px_last',
                                        'ivol_bid', 'ivol_ask', 'ivol_mid',
                                        'delta_mid', 'gamma_mid', 'vega_mid', 'theta_mid',
                                        'px_volume', 'open_int')


class OptionPriceSource(str, Enum):
    """which quote drives put-call parity."""
    MID = 'mid'    # 0.5 * (px_bid + px_ask), NaN unless both sides quote
    LAST = 'last'  # px_last, may be stale


_OPTION_STRIKE_RE = re.compile(r'[CP](\d+(?:\.\d+)?)\b')  # option token: C/P then strike, e.g. 'C337.5'


def _fetch_spot(underlying: str) -> float:
    """last price of the underlying, used to locate the ATM strike."""
    df = bdp(tickers=[underlying], flds='px_last')
    if df.empty or df.iloc[:, 0].isna().all():
        raise ValueError(f"no px_last for underlying {underlying!r}")
    return float(df.iloc[0, 0])


def _fetch_option_chain_tickers(underlying: str,
                                expiry: Optional[str] = None,  # CHAIN_EXP_DT_OVRD, YYYYMMDD
                                put_call: Optional[str] = None,  # 'C', 'P', or None for both
                                points: int = 10000,  # CHAIN_POINTS_OVRD upper bound on strikes per side
                                yellow_key: str = ' Index',  # suffix appended to each chain ticker
                                ) -> List[str]:
    """
    listed option tickers for an underlying via the CHAIN_TICKERS bulk field.

    CHAIN_TICKERS returns tickers without a yellow key ('KOSPI2 09/10/26 C335'), which
    bdp will not resolve; yellow_key is appended here. put_call=None issues two bulk
    requests, one per side.
    """
    if points <= 0:
        raise ValueError(f"points must be positive, got {points!r}")

    sides = ('C', 'P') if put_call is None else (put_call,)
    tickers: List[str] = []
    for side in sides:
        overrides: Dict[str, str] = {'CHAIN_PUT_CALL_TYPE_OVRD': side,
                                     'CHAIN_POINTS_OVRD': str(points)}
        if expiry is not None:
            overrides['CHAIN_EXP_DT_OVRD'] = expiry
            overrides['CHAIN_EXP_MATCH_OVRD'] = 'E'  # exact match, not nearest
        df = bds(underlying, 'CHAIN_TICKERS', **overrides)
        if df.empty:
            continue
        column = df.columns[0]  # bulk sub-element name, normalised by bbg_fetch.bds
        tickers.extend(f"{x}{yellow_key}" for x in df[column].dropna().astype(str))
    return tickers


def _parse_option_strikes(tickers: Sequence[str]) -> List[Tuple[str, float]]:
    """
    (ticker, strike) pairs, the strike parsed from each ticker's option token.

    Assumes the OMON convention with the strike embedded after C/P ('...C337.5 Index'
    -> 337.5); a ticker with no parseable strike raises rather than being dropped.
    """
    parsed: List[Tuple[str, float]] = []
    for ticker in tickers:
        match = _OPTION_STRIKE_RE.search(ticker)
        if match is None:
            raise ValueError(f"cannot parse strike from ticker {ticker!r}")
        parsed.append((ticker, float(match.group(1))))
    return parsed


def _select_strikes_near_atm(tickers: Sequence[str],
                             spot: float,  # underlying price locating the ATM strike
                             num_strikes_per_side: int,  # listed strikes kept each side of the ATM strike
                             ) -> List[str]:
    """
    trim a chain ticker list to a strike window centred on the ATM strike.

    The ATM strike is the listed strike nearest spot; num_strikes_per_side listed strikes
    below and above it are kept, calls and puts alike. The window spans at most
    2 * num_strikes_per_side + 1 strikes. No Bloomberg call is spent to size the window.
    """
    if num_strikes_per_side <= 0:
        raise ValueError(f"num_strikes_per_side must be positive, got {num_strikes_per_side!r}")

    parsed = _parse_option_strikes(tickers)
    strikes = np.array(sorted({strike for _, strike in parsed}))
    if len(strikes) == 0:
        return []
    atm_index = int(np.argmin(np.abs(strikes - spot)))
    lo = max(0, atm_index - num_strikes_per_side)
    hi = min(len(strikes), atm_index + num_strikes_per_side + 1)
    keep = set(strikes[lo:hi].tolist())
    return [ticker for ticker, strike in parsed if strike in keep]


def _select_strikes_on_grid(tickers: Sequence[str],
                            strike_grid: Sequence[float],  # target strikes; nearest listed strike is kept
                            ) -> List[str]:
    """
    keep the listed strikes nearest each target strike in strike_grid.

    For every value in strike_grid the nearest listed strike is selected — a grid that
    does not fall on the listing is snapped to it — and duplicates are collapsed. Both
    legs at each selected strike are returned. Unlike the ATM window this needs no spot.
    """
    if len(strike_grid) == 0:
        raise ValueError("strike_grid must be non-empty")

    parsed = _parse_option_strikes(tickers)
    listed = np.array(sorted({strike for _, strike in parsed}))
    if len(listed) == 0:
        return []
    grid = np.asarray(strike_grid, dtype=float)
    nearest = listed[np.argmin(np.abs(listed[np.newaxis, :] - grid[:, np.newaxis]), axis=1)]
    keep = set(nearest.tolist())
    return [ticker for ticker, strike in parsed if strike in keep]


def _bdp_in_batches(tickers: Sequence[str],
                    flds: Sequence[str],
                    batch_size: int = 100,  # tickers per bdp request
                    ) -> pd.DataFrame:
    """
    bdp over a long ticker list, split into batches to bound the request size.

    A ReferenceDataRequest carrying an entire index chain in one call is the failure
    mode this avoids.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")

    frames: List[pd.DataFrame] = []
    for start in range(0, len(tickers), batch_size):
        batch = list(tickers[start:start + batch_size])
        frames.append(bdp(tickers=batch, flds=list(flds)))
    if len(frames) == 0:
        return pd.DataFrame(columns=list(flds))
    return pd.concat(frames, axis=0)


def _split_option_legs(option_chain: pd.DataFrame,
                       price_source: 'OptionPriceSource',
                       ) -> Tuple[pd.Series, pd.Series]:
    """
    call and put price series indexed by strike, NaNs dropped.

    MID uses 0.5 * (px_bid + px_ask); LAST uses px_last.
    """
    required = ['opt_strike_px', 'opt_put_call']
    missing = [c for c in required if c not in option_chain.columns]
    if len(missing) > 0:
        raise ValueError(f"missing columns {missing}, got {option_chain.columns.to_list()}")

    if price_source == OptionPriceSource.MID:
        if not {'px_bid', 'px_ask'}.issubset(option_chain.columns):
            raise ValueError(f"MID needs px_bid and px_ask, got {option_chain.columns.to_list()}")
        price = 0.5 * (option_chain['px_bid'].astype(float) + option_chain['px_ask'].astype(float))
    elif price_source == OptionPriceSource.LAST:
        if 'px_last' not in option_chain.columns:
            raise ValueError(f"LAST needs px_last, got {option_chain.columns.to_list()}")
        price = option_chain['px_last'].astype(float)
    else:
        raise NotImplementedError(f"{price_source}")

    table = pd.DataFrame({'strike': option_chain['opt_strike_px'].astype(float),
                          'put_call': option_chain['opt_put_call'].astype(str).str.upper().str[0],
                          'price': price}).dropna()
    calls = table[table['put_call'] == 'C'].set_index('strike')['price'].sort_index()
    puts = table[table['put_call'] == 'P'].set_index('strike')['price'].sort_index()
    return calls, puts


def fetch_option_chain(underlying: str = 'KOSPI2 Index',
                       expiry: Optional[str] = None,  # CHAIN_EXP_DT_OVRD, YYYYMMDD; None = every listed expiry
                       num_strikes_per_side: Optional[int] = 20,  # strikes each side of ATM; None = full chain
                       strike_grid: Optional[Sequence[float]] = None,  # explicit strikes; overrides num_strikes_per_side
                       put_call: Optional[str] = None,  # 'C', 'P', or None for both
                       fields: Sequence[str] = OPTION_CHAIN_FIELDS,
                       yellow_key: str = ' Index',  # suffix appended to each chain ticker
                       spot: Optional[float] = None,  # ATM reference; None fetches PX_LAST on underlying
                       points: int = 10000,  # CHAIN_POINTS_OVRD upper bound on strikes per side
                       batch_size: int = 100,  # tickers per bdp request
                       ) -> pd.DataFrame:
    """
    listed option chain for an underlying, one row per option, trimmed to a chosen set
    of strikes.

    CHAIN_TICKERS enumerates the listed strikes in one bulk request; the strike set is
    then chosen by parsing the strike from each ticker before the per-option bdp, so the
    number of bdp hits is bounded by the selection, not the full chain. Strikes are
    selected one of two ways:

        strike_grid          keep the listed strike nearest each target value (both legs)
        num_strikes_per_side keep a window of listed strikes each side of the ATM strike

    strike_grid takes precedence when both are given, and needs no spot. The deep
    in-the-money strikes dropped by either mode carry only a stale px_last.

    Parameters
    ----------
    underlying : str
        Bloomberg underlying ticker, e.g. 'KOSPI2 Index'.
    expiry : str, optional
        CHAIN_EXP_DT_OVRD value in YYYYMMDD. None returns every listed expiry.
    num_strikes_per_side : int, optional
        Listed strikes kept each side of the ATM strike. None with no strike_grid fetches
        the full chain, which can be hundreds of strikes — mind the data limit.
    strike_grid : sequence of float, optional
        Explicit target strikes; the listed strike nearest each value is kept, both legs.
        Takes precedence over num_strikes_per_side and needs no spot.
    put_call : str, optional
        CHAIN_PUT_CALL_TYPE_OVRD value: 'C', 'P', or None for both.
    fields : sequence of str
        Bloomberg fields fetched per option. Default is the OMON row set.
    yellow_key : str
        Suffix appended to each chain ticker, e.g. ' Index' for an index option.
    spot : float, optional
        Underlying price locating the ATM strike. None fetches PX_LAST on underlying.
        Ignored when strike_grid is given or num_strikes_per_side is None.
    points : int
        CHAIN_POINTS_OVRD value, an upper bound on strikes per side. The default is
        deliberately larger than any real chain.
    batch_size : int
        Tickers per bdp request.

    Returns
    -------
    pd.DataFrame
        One row per option, indexed by option ticker (with yellow key), columns the
        normalised fields. Empty DataFrame with the field columns if the chain is empty.

    Raises
    ------
    ValueError
        If expiry is not a valid YYYYMMDD date, num_strikes_per_side or points is not
        positive, strike_grid is empty, or a chain ticker carries no parseable strike.
    """
    if expiry is not None:
        try:
            pd.to_datetime(expiry, format='%Y%m%d')
        except (ValueError, TypeError) as error:
            raise ValueError(f"expiry must be a valid YYYYMMDD date, got {expiry!r}") from error

    tickers = _fetch_option_chain_tickers(underlying=underlying,
                                          expiry=expiry,
                                          put_call=put_call,
                                          points=points,
                                          yellow_key=yellow_key)
    if len(tickers) == 0:
        return pd.DataFrame(columns=list(fields))

    if strike_grid is not None:
        tickers = _select_strikes_on_grid(tickers=tickers, strike_grid=strike_grid)
    elif num_strikes_per_side is not None:
        if spot is None:
            spot = _fetch_spot(underlying=underlying)
        tickers = _select_strikes_near_atm(tickers=tickers,
                                           spot=spot,
                                           num_strikes_per_side=num_strikes_per_side)
    return _bdp_in_batches(tickers=tickers, flds=fields, batch_size=batch_size)


def recover_option_forward(option_chain: pd.DataFrame,
                           spot: float,  # underlying price S
                           year_fraction: float,  # T to expiry, in years
                           price_source: OptionPriceSource = OptionPriceSource.LAST,
                           num_strikes: int = 20,  # strikes nearest spot used in the regression
                           ) -> Dict[str, float]:
    """
    implied forward and rate from put-call parity by OLS on the call-put price spread.

        C(K) - P(K) = exp(-r T) (F - K)

    Regressing the spread on strike gives slope = -exp(-r T) and intercept =
    exp(-r T) F, hence F = -intercept / slope and r = -ln(-slope) / T.

    The forward is well determined; the rate is not. Over a short maturity the slope pins
    F to a fraction of a point, but the discount factor -exp(-r T) is ~ -1 and a 1e-3
    error in the slope moves r by order 1%, so r (and any dividend backed out of it as
    q = r - ln(F / S) / T) is only indicative. r2 near 1 is no comfort: it measures line
    straightness, not that the slope sits at -1. Prefer a money-market curve for the rate
    when precision matters.

    Parameters
    ----------
    option_chain : pd.DataFrame
        Output of fetch_option_chain: one row per option with opt_strike_px,
        opt_put_call, and the price columns required by price_source.
    spot : float
        Underlying price S, used to select the strikes nearest the money.
    year_fraction : float
        Year fraction T to expiry.
    price_source : OptionPriceSource
        MID uses 0.5 * (px_bid + px_ask); LAST uses px_last. LAST spans the full strike
        grid; MID drops strikes quoted one-sided.
    num_strikes : int
        Strikes nearest spot, quoted on both legs, used in the regression.

    Returns
    -------
    Dict[str, float]
        forward, rate, r2, num_strikes_used.

    Raises
    ------
    ValueError
        If year_fraction is not positive, fewer than three strikes are quoted on both
        legs, or the fitted parity slope is non-negative.
    """
    if year_fraction <= 0.0:
        raise ValueError(f"year_fraction must be positive, got {year_fraction!r}")

    calls, puts = _split_option_legs(option_chain=option_chain, price_source=price_source)
    common = calls.index.intersection(puts.index)
    if len(common) < 3:
        raise ValueError(f"need at least 3 strikes quoted on both legs, got {len(common)}")

    common = pd.Index(sorted(common, key=lambda k: abs(k - spot))[:num_strikes]).sort_values()
    strikes = common.to_numpy(dtype=float)
    spread = (calls[common] - puts[common]).to_numpy(dtype=float)

    slope, intercept = np.polyfit(strikes, spread, deg=1)
    if slope >= 0.0:
        raise ValueError(f"parity slope must be negative, got {slope!r}")

    fitted = slope * strikes + intercept
    ss_res = float(np.sum(np.square(spread - fitted)))
    ss_tot = float(np.sum(np.square(spread - np.mean(spread))))
    return dict(forward=-intercept / slope,
                rate=-np.log(-slope) / year_fraction,
                r2=1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan,
                num_strikes_used=float(len(strikes)))


@dataclass(frozen=True)
class OptionChainResult:
    """
    immutable snapshot: the option chain with the parity forward and rate.

    spot and year_fraction are read from the chain (opt_undl_px and opt_expire_dt); rate
    is the parity rate and is only indicative at short maturity (see recover_option_forward).
    """
    chain: pd.DataFrame
    spot: float             # opt_undl_px, the underlying price used for the options
    year_fraction: float    # (opt_expire_dt - today) / 365, actual/365
    forward: float          # implied forward from put-call parity
    rate: float             # parity rate, indicative only at short maturity
    r2: float               # parity regression fit
    num_strikes_used: int   # strikes in the regression

    def to_csv(self, path: str) -> None:
        """
        write the snapshot to one self-contained CSV: the scalars as commented header
        lines, then the chain. Read back with OptionChainResult.read_csv. Pandas only.
        """
        scalar_fields = ('spot', 'year_fraction', 'forward', 'rate', 'r2', 'num_strikes_used')
        with open(path, 'w', newline='') as file:
            for name in scalar_fields:
                file.write(f"# {name}={getattr(self, name)}\n")
            self.chain.to_csv(file)

    @classmethod
    def read_csv(cls, path: str) -> 'OptionChainResult':
        """
        read back a snapshot written by to_csv.

        The commented header carries the scalars; the chain follows. opt_expire_dt is
        re-parsed to datetime, which CSV stores as text.
        """
        scalars: Dict[str, float] = {}
        with open(path) as file:
            for line in file:
                if not line.startswith('#'):
                    break
                key, _, value = line[1:].strip().partition('=')
                scalars[key.strip()] = float(value)
        chain = pd.read_csv(path, comment='#', index_col=0)
        if 'opt_expire_dt' in chain.columns:
            chain['opt_expire_dt'] = pd.to_datetime(chain['opt_expire_dt'])
        return cls(chain=chain,
                   spot=scalars['spot'],
                   year_fraction=scalars['year_fraction'],
                   forward=scalars['forward'],
                   rate=scalars['rate'],
                   r2=scalars['r2'],
                   num_strikes_used=int(scalars['num_strikes_used']))


def run(underlying: str = 'KOSPI2 Index',
        expiry: Optional[str] = '20260910',  # CHAIN_EXP_DT_OVRD, YYYYMMDD; a single expiry
        num_strikes_per_side: Optional[int] = 20,  # strikes each side of ATM; None = full chain
        strike_grid: Optional[Sequence[float]] = None,  # explicit strikes; overrides num_strikes_per_side
        price_source: OptionPriceSource = OptionPriceSource.LAST,
        ) -> OptionChainResult:
    """
    fetch an option chain and recover the implied forward and rate, in one call.

    Choose strikes by num_strikes_per_side (ATM window) or strike_grid (explicit); when
    both are given strike_grid wins. Spot and the year fraction are inferred from the
    returned chain: spot = opt_undl_px, year_fraction = (opt_expire_dt - today) / 365
    (actual/365). A single expiry is required, since parity mixes expiries otherwise.

    Parameters
    ----------
    underlying : str
        Bloomberg underlying ticker, e.g. 'KOSPI2 Index'.
    expiry : str, optional
        CHAIN_EXP_DT_OVRD value in YYYYMMDD. Must resolve to a single listed expiry.
    num_strikes_per_side : int, optional
        Listed strikes kept each side of the ATM strike. None with no strike_grid fetches
        the full chain.
    strike_grid : sequence of float, optional
        Explicit target strikes; the listed strike nearest each value is kept.
    price_source : OptionPriceSource
        Quote driving the parity recovery. LAST spans the full strike grid.

    Returns
    -------
    OptionChainResult
        chain, spot, year_fraction, forward, rate, r2, num_strikes_used.

    Raises
    ------
    ValueError
        If the chain is empty or does not resolve to a single expiry.
    """
    chain = fetch_option_chain(underlying=underlying,
                               expiry=expiry,
                               num_strikes_per_side=num_strikes_per_side,
                               strike_grid=strike_grid)
    if chain.empty:
        raise ValueError(f"empty chain for {underlying} {expiry}")

    expiries = pd.to_datetime(chain['opt_expire_dt'].dropna()).unique()
    if len(expiries) != 1:
        raise ValueError(f"forward recovery needs a single expiry, got {len(expiries)}; pass expiry=")

    spot = float(chain['opt_undl_px'].dropna().iloc[0])
    expiry_date = pd.Timestamp(expiries[0])
    year_fraction = (expiry_date - pd.Timestamp.now().normalize()).days / 365.0

    params = recover_option_forward(option_chain=chain,
                                    spot=spot,
                                    year_fraction=year_fraction,
                                    price_source=price_source)
    return OptionChainResult(chain=chain,
                             spot=spot,
                             year_fraction=year_fraction,
                             forward=params['forward'],
                             rate=params['rate'],
                             r2=params['r2'],
                             num_strikes_used=int(params['num_strikes_used']))
