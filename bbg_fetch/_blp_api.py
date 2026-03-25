"""
Direct blpapi interface for Bloomberg data access.

Replaces xbbg dependency with thin wrappers around blpapi's //blp/refdata service.
Implements bdp(), bdh(), bds() with DataFrame output contracts matching the xbbg signatures
used by bbg_fetch.

Install blpapi:
    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
"""

import logging
import datetime
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import blpapi
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session singleton (thread-safe, lazy-init)
# ---------------------------------------------------------------------------

_session: Optional[blpapi.Session] = None
_session_lock = threading.Lock()

REFDATA_SERVICE = "//blp/refdata"


def _get_session(host: str = "localhost", port: int = 8194) -> blpapi.Session:
    """Return a shared blpapi Session, creating one on first call."""
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        opts = blpapi.SessionOptions()
        opts.setServerHost(host)
        opts.setServerPort(port)
        session = blpapi.Session(opts)
        if not session.start():
            raise ConnectionError("Failed to start blpapi session")
        if not session.openService(REFDATA_SERVICE):
            raise ConnectionError(f"Failed to open {REFDATA_SERVICE}")
        _session = session
        return _session


def disconnect() -> None:
    """Explicitly stop and discard the shared session."""
    global _session
    with _session_lock:
        if _session is not None:
            try:
                _session.stop()
            except Exception:
                pass
            _session = None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _as_list(x: Any) -> list:
    """Coerce scalar or sequence to list."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]


def _normalize_name(name: str) -> str:
    """Normalize Bloomberg field/sub-element names: lowercase, spaces and hyphens to underscores."""
    return name.lower().replace(' ', '_').replace('-', '_')


def _element_to_value(elem: blpapi.Element) -> Any:
    """Convert a scalar blpapi Element to a Python value."""
    if elem.numValues() == 0:
        return None

    dtype = elem.datatype()
    try:
        if dtype in (blpapi.DataType.FLOAT32, blpapi.DataType.FLOAT64):
            return elem.getValueAsFloat()
        if dtype in (blpapi.DataType.INT32, blpapi.DataType.INT64):
            return elem.getValueAsInteger()
        if dtype == blpapi.DataType.STRING:
            return elem.getValueAsString()
        if dtype in (blpapi.DataType.DATE, blpapi.DataType.DATETIME, blpapi.DataType.TIME):
            return elem.getValueAsDatetime()
        if dtype == blpapi.DataType.BOOL:
            return elem.getValueAsBool()
        return elem.getValueAsString()
    except Exception:
        return None


def _set_overrides(request: blpapi.Request, overrides: Dict[str, Any]) -> None:
    """Append override key-value pairs to a blpapi request."""
    if not overrides:
        return
    ov_elem = request.getElement("overrides")
    for key, val in overrides.items():
        ov = ov_elem.appendElement()
        ov.setElement("fieldId", key)
        ov.setElement("value", str(val))


def _collect_responses(session: blpapi.Session,
                       timeout_ms: int = 60_000
                       ) -> List[blpapi.Message]:
    """Drain the event queue until a final RESPONSE event; return all data messages.

    Collects messages from both PARTIAL_RESPONSE and RESPONSE events that
    contain a 'securityData' element (i.e., actual data messages).
    Admin, session-status, and service-status events are silently skipped.
    """
    messages: List[blpapi.Message] = []
    done = False
    while not done:
        ev = session.nextEvent(timeout_ms)
        ev_type = ev.eventType()

        if ev_type in (blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE):
            for msg in ev:
                if msg.hasElement("securityData"):
                    messages.append(msg)
            if ev_type == blpapi.Event.RESPONSE:
                done = True

        elif ev_type == blpapi.Event.TIMEOUT:
            logger.warning("blpapi event loop timed out")
            done = True

        # else: ADMIN, SESSION_STATUS, SERVICE_STATUS, etc. — skip

    return messages


# ---------------------------------------------------------------------------
# bdp  —  Bloomberg Data Point  (ReferenceDataRequest)
# ---------------------------------------------------------------------------

def bdp(tickers: Union[str, Sequence[str]],
        flds: Union[str, Sequence[str]],
        **overrides: Any,
        ) -> pd.DataFrame:
    """
    Point-in-time reference data — equivalent to BDP in Excel.

    Returns
    -------
    pd.DataFrame
        Index = tickers, columns = normalized field names.
    """
    tickers_list = _as_list(tickers)
    flds_list = _as_list(flds)

    session = _get_session()
    service = session.getService(REFDATA_SERVICE)
    request = service.createRequest("ReferenceDataRequest")

    for t in tickers_list:
        request.getElement("securities").appendValue(t)
    for f in flds_list:
        request.getElement("fields").appendValue(f)

    _set_overrides(request, overrides)
    session.sendRequest(request)

    records: Dict[str, Dict[str, Any]] = {t: {} for t in tickers_list}
    for msg in _collect_responses(session):
        sec_data_array = msg.getElement("securityData")
        for i in range(sec_data_array.numValues()):
            sec_data = sec_data_array.getValueAsElement(i)
            ticker = sec_data.getElementAsString("security")

            if sec_data.hasElement("securityError"):
                logger.warning("Security error for %s: %s",
                               ticker, sec_data.getElement("securityError"))
                continue

            field_data = sec_data.getElement("fieldData")
            row: Dict[str, Any] = {}
            for j in range(field_data.numElements()):
                elem = field_data.getElement(j)
                name = str(elem.name())
                if elem.isArray():
                    continue
                row[_normalize_name(name)] = _element_to_value(elem)
            if ticker in records:
                records[ticker].update(row)

    df = pd.DataFrame.from_dict(records, orient="index")
    expected_cols = [_normalize_name(f) for f in flds_list]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df.reindex(columns=expected_cols)
    df.index.name = None
    return df


# ---------------------------------------------------------------------------
# bdh  —  Bloomberg Data History  (HistoricalDataRequest)
# ---------------------------------------------------------------------------

def bdh(tickers: Union[str, Sequence[str]],
        flds: Union[str, Sequence[str]],
        start_date: Any = None,
        end_date: Any = None,
        CshAdjNormal: bool = False,
        CshAdjAbnormal: bool = False,
        CapChg: bool = False,
        **overrides: Any,
        ) -> pd.DataFrame:
    """
    Historical end-of-day data — equivalent to BDH in Excel.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, MultiIndex columns = (ticker, field).
    """
    tickers_list = _as_list(tickers)
    flds_list = _as_list(flds)

    session = _get_session()
    service = session.getService(REFDATA_SERVICE)
    request = service.createRequest("HistoricalDataRequest")

    for t in tickers_list:
        request.getElement("securities").appendValue(t)
    for f in flds_list:
        request.getElement("fields").appendValue(f)

    def _fmt_date(d: Any) -> str:
        if d is None:
            return datetime.date.today().strftime("%Y%m%d")
        if isinstance(d, str):
            return pd.Timestamp(d).strftime("%Y%m%d")
        if isinstance(d, (pd.Timestamp, datetime.datetime, datetime.date)):
            return d.strftime("%Y%m%d")
        return str(d)

    request.set("startDate", _fmt_date(start_date))
    request.set("endDate", _fmt_date(end_date))

    if CshAdjNormal:
        request.set("adjustmentNormal", True)
    if CshAdjAbnormal:
        request.set("adjustmentAbnormal", True)
    if CapChg:
        request.set("adjustmentSplit", True)

    _set_overrides(request, overrides)
    session.sendRequest(request)

    per_ticker: Dict[str, pd.DataFrame] = {}
    for msg in _collect_responses(session):
        sec_data = msg.getElement("securityData")
        ticker = sec_data.getElementAsString("security")

        if sec_data.hasElement("securityError"):
            logger.warning("Security error for %s: %s",
                           ticker, sec_data.getElement("securityError"))
            continue

        field_data_array = sec_data.getElement("fieldData")
        rows = []
        for k in range(field_data_array.numValues()):
            row_elem = field_data_array.getValueAsElement(k)
            row: Dict[str, Any] = {}
            for j in range(row_elem.numElements()):
                elem = row_elem.getElement(j)
                name = str(elem.name())
                row[name] = _element_to_value(elem)
            rows.append(row)

        if rows:
            df_t = pd.DataFrame(rows)
            if "date" in df_t.columns:
                df_t["date"] = pd.to_datetime(df_t["date"])
                df_t = df_t.set_index("date")
            per_ticker[ticker] = df_t

    # assemble MultiIndex columns (ticker, field)
    fld_map = {_normalize_name(f): f for f in flds_list}

    if not per_ticker:
        cols = pd.MultiIndex.from_product([tickers_list, flds_list])
        return pd.DataFrame(columns=cols)

    frames = []
    for t in tickers_list:
        if t in per_ticker:
            df_t = per_ticker[t]
            df_t.columns = [fld_map.get(_normalize_name(c), c) for c in df_t.columns]
            for f in flds_list:
                if f not in df_t.columns:
                    df_t[f] = np.nan
            df_t = df_t[flds_list]
            df_t.columns = pd.MultiIndex.from_product([[t], df_t.columns])
            frames.append(df_t)
        else:
            idx = frames[0].index if frames else pd.DatetimeIndex([])
            empty = pd.DataFrame(
                np.nan, index=idx,
                columns=pd.MultiIndex.from_product([[t], flds_list])
            )
            frames.append(empty)

    result = pd.concat(frames, axis=1)
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    return result


# ---------------------------------------------------------------------------
# bds  —  Bloomberg Data Set  (ReferenceDataRequest with bulk field)
# ---------------------------------------------------------------------------

def bds(tickers: Union[str, Sequence[str]],
        flds: Union[str, Sequence[str]],
        **overrides: Any,
        ) -> pd.DataFrame:
    """
    Bulk reference data — equivalent to BDS in Excel.

    Returns
    -------
    pd.DataFrame
        Rows from the bulk data set, with ticker as (repeated) index
        and normalized sub-element names as columns.
    """
    tickers_list = _as_list(tickers)
    flds_list = _as_list(flds)

    session = _get_session()
    service = session.getService(REFDATA_SERVICE)
    request = service.createRequest("ReferenceDataRequest")

    for t in tickers_list:
        request.getElement("securities").appendValue(t)
    for f in flds_list:
        request.getElement("fields").appendValue(f)

    _set_overrides(request, overrides)
    session.sendRequest(request)

    all_rows: List[Dict[str, Any]] = []
    all_tickers: List[str] = []

    msgs = _collect_responses(session)
    for msg in msgs:
        sec_data_array = msg.getElement("securityData")
        for i in range(sec_data_array.numValues()):
            sec_data = sec_data_array.getValueAsElement(i)
            ticker = sec_data.getElementAsString("security")

            if sec_data.hasElement("securityError"):
                logger.warning("Security error for %s: %s",
                               ticker, sec_data.getElement("securityError"))
                continue

            field_data = sec_data.getElement("fieldData")
            for j in range(field_data.numElements()):
                bulk_elem = field_data.getElement(j)
                if not bulk_elem.isArray():
                    continue
                for k in range(bulk_elem.numValues()):
                    row_elem = bulk_elem.getValueAsElement(k)
                    row: Dict[str, Any] = {}
                    for m in range(row_elem.numElements()):
                        sub = row_elem.getElement(m)
                        row[_normalize_name(str(sub.name()))] = _element_to_value(sub)
                    all_rows.append(row)
                    all_tickers.append(ticker)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.index = all_tickers
    df.index.name = None
    return df