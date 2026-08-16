.. meta::
   :description: Use bbg-fetch option-chain, volatility, futures, fixed-income, and index-constituent research workflows.

Specialised research workflows
==============================

The problem
-----------

These workflows combine Bloomberg requests with narrowly scoped pandas
shaping. They make request structure reproducible; they do not replace the
Terminal's field definitions or add a general analytics layer. Every workflow
below requires the local setup in :doc:`task_install_connect_diagnose` and the
logged-in user's applicable entitlements.

Option chains and put-call parity
---------------------------------

``bbg_fetch.fetch_option_chain`` first requests ``CHAIN_TICKERS``, selects an
explicit strike grid or an at-the-money window, then BDP-fetches option fields
in bounded batches. ``expiry`` is a calendar date string in ``YYYYMMDD`` form;
``None`` asks for every listed expiry and can create a large request.
``strike_grid`` takes precedence over ``num_strikes_per_side``.

The result has one row per option ticker and normalized field columns. Default
fields include strike, put/call, expiry, underlying price, bid, ask, and last;
their units remain Bloomberg's instrument units. No currency or contract
multiplier conversion is applied.

.. code-block:: python

   import bbg_fetch

   try:
       chain = bbg_fetch.fetch_option_chain(
           underlying="SPX Index",
           expiry="20260918",  # replace with a currently listed expiry
           num_strikes_per_side=5,
           batch_size=25,
       )
       print(chain.shape, chain.columns.to_list())
   finally:
       bbg_fetch.disconnect()

For an entitled listed expiry, expect up to eleven distinct strikes per leg
around the selected ATM strike, subject to listing and returned data. An empty
chain means no rows were returned for the chosen request; it does not prove an
entitlement cause.

``bbg_fetch.recover_option_forward`` is terminal-free: it takes a chain,
``spot`` in the underlying's quoted price units, and a positive
``year_fraction`` in years. It returns a dict with forward in spot units,
continuously compounded annual ``rate`` as a decimal, regression ``r2``, and
``num_strikes_used``. ``bbg_fetch.run`` combines fetch and recovery into an
``bbg_fetch.OptionChainResult`` using actual/365 to expiry. The recovered rate
is indicative, especially at short maturities; this is not a yield-curve
builder or option pricer. Use the deterministic root quickstart to verify the
calculation without a Terminal, and the `live option example
<https://github.com/ArturSepp/BloombergFetch/blob/main/examples/fetch_option_chain.py>`_
for the combined request.

Common failures are a stale/non-listed expiry, a ticker suffix that does not
match ``yellow_key``, fewer than three common call/put strikes, one-sided
quotes under ``OptionPriceSource.MID``, and request size from a full chain.

Volatility time series and surfaces
-----------------------------------

``bbg_fetch.fetch_vol_timeseries`` returns a sorted ``DatetimeIndex``
DataFrame. It accepts a field-to-label dict, a list of those dicts (one request
per tenor), or a list of raw field strings. With the default ``scaler=0.01``,
Bloomberg percentage values such as implied vol, dividend yield, and rate are
converted to decimals. ``scaler=None`` keeps terminal display units.
``add_underlying`` adds ``spot_price``, ``div_yield``, and ``rf_rate``;
``add_forwards`` additionally computes forward and discount-factor columns
from those inputs for the configured tenor year fractions.

``bbg_fetch.fetch_vol_surface`` takes the last quote on or before
``value_date`` within ``lookback_days`` and pivots it to tenor rows and numeric
moneyness-percent columns. With default fields the intended shape is five
tenors by nine moneyness points; missing field observations remain ``NaN`` and
no data in the window returns an empty DataFrame.

.. code-block:: python

   import bbg_fetch

   try:
       surface = bbg_fetch.fetch_vol_surface(
           ticker="SPX Index",
           scaler=0.01,
           lookback_days=10,
       )
       print(surface.shape, surface.index.to_list(), surface.columns.to_list())
   finally:
       bbg_fetch.disconnect()

The implied-vol fields are passed to Bloomberg as defined by the exported
field maps; this guide does not reproduce Bloomberg's catalogue. Field
availability is instrument- and entitlement-specific. This workflow does not
interpolate a continuous surface, calibrate a volatility model, or price
options. See the `high-level source
<https://github.com/ArturSepp/BloombergFetch/blob/main/src/bbg_fetch/core.py>`_.

Futures chains and generic histories
------------------------------------

``bbg_fetch.fetch_futures_contract_table`` requests ``FUT_CHAIN`` and then
the selected scalar fields per listed contract. It returns a DataFrame indexed
by contract ticker, with a ``ticker`` column for the input generic/active
ticker. Defaults add ``gen_number``, a UTC request ``timestamp``, a combined
timezone-aware ``update``, and ``an_carry``.

``fut_days_expire`` is in days as returned by Bloomberg. ``an_carry`` is the
annualized decimal roll yield computed from adjacent bid/ask mids and day
differences; the front contract and missing/two-sided quote cases are ``NaN``.
Other prices, sizes, contract values, point values, currencies, and dates are
passed through without unit conversion.

.. code-block:: python

   import bbg_fetch

   try:
       contracts = bbg_fetch.fetch_futures_contract_table(
           ticker="ES1 Index",
           add_timestamp=True,
           add_gen_number=True,
           add_carry=True,
       )
       print(contracts.shape, contracts.columns.to_list())
   finally:
       bbg_fetch.disconnect()

``bbg_fetch.fetch_active_futures`` instead returns a tuple of two historical
price Series for adjacent generic numbers, retrying each up to
``max_attempts``. The generic ticker needs Terminal ``GFUT`` settings
consistent with the intended unadjusted roll convention. The pure helpers
``bbg_fetch.instrument_to_active_ticker`` and
``bbg_fetch.contract_to_instrument`` only transform ticker strings.

An empty chain, unavailable bid/ask, inconsistent timestamp field types, or a
large chain can change the schema/population. This workflow does not define a
roll schedule, build a continuous contract, normalize contract notionals, or
backtest futures.

Fixed-income and issuer reference data
--------------------------------------

``bbg_fetch.fetch_bonds_info`` accepts a sequence of bond ISINs, queries each
as ``"<ISIN> corp"``, and returns a DataFrame reindexed to the original ISIN
order. Unresolved securities remain ``NaN`` rows. Default columns cover
identity, currency, amount outstanding, price, and selected YAS fields;
Bloomberg's field units are passed through. ``END_DATE_OVERRIDE`` is an
optional ``YYYYMMDD`` string for fields that support an as-of override.

.. code-block:: python

   import bbg_fetch

   try:
       bonds = bbg_fetch.fetch_bonds_info(
           isins=["US03522AAJ97"],
           fields=["ID_BB", "CRNCY", "PX_LAST", "YAS_MOD_DUR"],
       )
       print(bonds.shape, bonds.index.to_list(), bonds.columns.to_list())
   finally:
       bbg_fetch.disconnect()

``bbg_fetch.fetch_cds_info`` resolves an equity ticker to the requested CDS
spread ticker; ``bbg_fetch.fetch_issuer_isins_from_bond_isins`` performs two
reference lookups from bond ISIN to ultimate-parent equity ISIN. Both can
contain missing entries when the relationship or entitlement is unavailable.
``bbg_fetch.fetch_balance_data`` is a shaped issuer-fundamental reference
request, not a fixed-income pricing function.

These functions do not calculate accrued interest, cash flows, yield curves,
credit curves, spread risk, or valuations. YAS field definitions and units
must be checked in Bloomberg Professional; the package does not reinterpret
them.

Index constituents and weights
------------------------------

``bbg_fetch.fetch_index_members_weights`` sends one bulk request and indexes
the returned DataFrame by its first Bloomberg sub-element. Use
``field="INDX_MWEIGHT"`` for members with weights where supported,
``"INDX_MEMBERS"`` for ticker membership, or ``"INDX_MEMBERS3"`` for the
additional metadata Bloomberg returns. Remaining column names and units are
the normalized BDS sub-elements; weight scaling is not changed.

.. code-block:: python

   import bbg_fetch

   try:
       members = bbg_fetch.fetch_index_members_weights(
           index="SPX Index",
           field="INDX_MEMBERS",
       )
       print(members.shape, members.index.name, members.columns.to_list())
   finally:
       bbg_fetch.disconnect()

Pass ``END_DATE_OVERRIDE="YYYYMMDD"`` only when the selected bulk field and
entitlement support historical membership. The function raises ``ValueError``
when Bloomberg returns no bulk rows. It does not distinguish invalid fields,
unsupported history, entitlement failures, or genuinely empty datasets, and
it does not calculate returns, rebalance portfolios, or repair survivorship
bias.

Request-size and failure discipline
-----------------------------------

Start every workflow with one instrument, the smallest field set, a short
history, and (for options) a small strike window. Increase one dimension at a
time. A 60-second final-response timeout applies to the underlying request
layer; general workflows do not auto-batch. Option chains alone expose
``batch_size`` after strike selection.

Treat missing/empty output as ambiguous until the security, field, date or
expiry, override, and user entitlement have been checked independently in the
Terminal. Record local diagnostics as dimensions, labels, dtypes, and
non-null flags—not licensed values. Always call ``bbg_fetch.disconnect()`` in
a ``finally`` block for short diagnostic scripts.

API, examples, and non-goals
----------------------------

All named symbols are listed in :doc:`api`. Root scripts are indexed in the
`examples README
<https://github.com/ArturSepp/BloombergFetch/blob/main/examples/README.md>`_,
and option implementation details are in the `option-chain source
<https://github.com/ArturSepp/BloombergFetch/blob/main/src/bbg_fetch/option_chain.py>`_.

Out of scope are new Bloomberg request types, streaming, intraday bars,
proprietary data fixtures, copied Bloomberg field documentation, speculative
entitlement claims, and analytics that belong in the maintainer's sibling
portfolio, performance, or model packages.
