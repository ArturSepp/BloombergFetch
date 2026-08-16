.. meta::
   :description: Request Bloomberg reference, historical, and bulk data with bbg-fetch and understand the exact pandas shapes.

Request reference, historical, and bulk data
=============================================

The problem
-----------

Bloomberg exposes scalar reference values, dated histories, and repeated-row
bulk datasets through different request types. ``bbg-fetch`` provides direct
``bdp``/``bdh``/``bds`` wrappers plus high-level functions that reshape common
research requests. Choose the lowest level whose output contract matches the
task; a bulk field sent to ``bdp`` does not become a bulk table.

Prerequisites
-------------

Complete :doc:`task_install_connect_diagnose`. Every example below is a live
Desktop API request and depends on the logged-in user's security and field
entitlements. Verify field mnemonics in Bloomberg Professional. The examples
print only shape and schema.

Choose the request layer
------------------------

``bbg_fetch.bdp(tickers, flds, **overrides)``
   Point-in-time scalar reference data. Use for last price, name, sector, or
   another scalar field. Both arguments accept one string or a sequence.

``bbg_fetch.bdh(tickers, flds, start_date, end_date, ...)``
   Historical end-of-day data. Both ticker and field inputs accept a string or
   sequence. Dates accept strings, ``datetime`` values, or pandas timestamps.

``bbg_fetch.bds(tickers, flds, **overrides)``
   Repeated-row bulk reference data, such as ``INDX_MEMBERS`` or
   ``DVD_HIST_ALL``. Both inputs accept a string or sequence, although the
   resulting rows are combined into one table.

Prefer high-level functions when their reshaping is the desired contract:

* ``bbg_fetch.fetch_field_timeseries_per_tickers`` returns one field across
  many tickers as a wide, sorted ``DatetimeIndex`` DataFrame. A ticker dict
  maps Bloomberg tickers to output labels; optional ``freq`` resamples and
  forward-fills.
* ``bbg_fetch.fetch_fields_timeseries_per_ticker`` returns many fields for one
  ticker, ordered as requested.
* ``bbg_fetch.fetch_fundamentals`` accepts ticker and field sequences or dicts
  and preserves their requested order and labels.
* ``bbg_fetch.fetch_dividend_history`` and
  ``bbg_fetch.fetch_index_members_weights`` are shaped BDS workflows.

Exact output contracts
----------------------

``bdp``
   A DataFrame with one row per requested ticker and one column per requested
   field. Response field names are lowercased and spaces or hyphens become
   underscores: ``PX_LAST`` becomes ``px_last``. Requested field columns are
   present in requested order; missing scalar values remain ``NaN``.

``bdh``
   A sorted ``DatetimeIndex`` DataFrame. Columns are a two-level pandas
   ``MultiIndex`` of ``(ticker, requested_field)`` in input order. The field
   spelling in this MultiIndex is the spelling supplied by the caller. If no
   security responds, the result has the requested MultiIndex columns and no
   rows.

``bds``
   One row per bulk element, with the source ticker repeated as the index.
   Bulk sub-element names are lowercased and spaces or hyphens become
   underscores. If no bulk rows arrive, the result is an empty DataFrame with
   no guaranteed columns.

Minimal live example
--------------------

.. code-block:: python

   import pandas as pd

   import bbg_fetch

   end = pd.Timestamp.today().normalize()
   start = end - pd.Timedelta(days=10)

   try:
       reference = bbg_fetch.bdp(
           ["AAPL US Equity", "MSFT US Equity"],
           ["PX_LAST", "SECURITY_NAME"],
       )
       history = bbg_fetch.bdh(
           "SPX Index",
           "PX_LAST",
           start_date=start,
           end_date=end,
       )
       members = bbg_fetch.bds("SPX Index", "INDX_MEMBERS")

       print("reference", reference.shape, reference.columns.to_list())
       print("history", history.shape, history.columns.names)
       print("bulk", members.shape, members.columns.to_list())
   finally:
       bbg_fetch.disconnect()

When entitled, ``reference`` has shape ``(2, 2)`` and normalized columns;
``history`` has an observation-dependent row count and one two-level column;
``members`` has an index-length and schema determined by Bloomberg's current
bulk dataset. These are shape assertions, not promises about returned values.

Adjustments, overrides, and dates
---------------------------------

``bdh`` exposes three Boolean adjustment flags: ``CshAdjNormal`` maps to
normal cash-dividend adjustment, ``CshAdjAbnormal`` to special-dividend
adjustment, and ``CapChg`` to split/capital-change adjustment. All three
default to ``False`` at the low level but to ``True`` in the two high-level
historical fetchers. State the choice explicitly when raw quotes versus
adjusted equity history matters; futures and rates commonly use all three as
``False``.

Additional keyword arguments on ``bdp``, ``bdh``, and ``bds`` are sent as
Bloomberg request overrides after conversion to strings. Examples include
``END_DATE_OVERRIDE="20260815"`` where a field supports it. The package does
not validate field-specific override semantics; use the Bloomberg Terminal to
confirm them. ``bdh`` defaults a missing start or end date to today, so a
meaningful history request should always pass both bounds.

Batching and request size
-------------------------

The direct wrappers and general high-level fetchers do not automatically
split large ticker/field universes. Split large requests into bounded chunks
and concatenate locally while preserving the documented index/column order.
``bbg_fetch.fetch_option_chain`` is the exception: its ``batch_size`` controls
the per-option BDP batches after strike selection.

The event collector waits up to 60 seconds for a final response. That timeout
is internal and is not a public argument. A ``TimeoutError`` may contain
``partial_messages`` for local debugging, but no partial DataFrame is returned.
Reduce to one ticker and one field before treating a large-request timeout as
a package defect.

Missing data and failure modes
------------------------------

* A security error is logged and its scalar row may remain all ``NaN``.
* An invalid or unentitled field may produce a missing column value rather
  than a Python exception.
* A BDS request can be empty because the request type, security, field,
  date/override, or entitlement is wrong; emptiness does not identify which.
* The high-level historical functions return ``None`` when a response cannot
  be parsed into their promised one-level shape. Check for ``None`` before
  using the DataFrame.
* Resampling with ``freq`` forward-fills by design; it does not create new
  Bloomberg observations.

Session lifetime and shutdown
-----------------------------

All requests share one lazy, process-local session. Call
``bbg_fetch.disconnect()`` when a long-lived process should stop and discard
it; the package also registers shutdown at interpreter exit. The next request
creates a new session. Do not assume that this singleton makes concurrent
request batching or streaming available.

Examples, API, and source
-------------------------

The labelled `root core example
<https://github.com/ArturSepp/BloombergFetch/blob/main/examples/fetch_core_data.py>`_
shows high-level price, reference, bulk, and research requests. The
`dividend example
<https://github.com/ArturSepp/BloombergFetch/blob/main/examples/fetch_div_history.py>`_
covers the shaped dividend workflow. See :doc:`api`, the high-level
`source <https://github.com/ArturSepp/BloombergFetch/blob/main/src/bbg_fetch/core.py>`_,
and the low-level `request source
<https://github.com/ArturSepp/BloombergFetch/blob/main/src/bbg_fetch/_blp_api.py>`_.

Non-goals
---------

This layer does not discover Bloomberg field mnemonics, infer entitlements,
supply proprietary fixtures, stream subscriptions, provide intraday bars, or
perform portfolio and performance analytics owned by sibling packages.
