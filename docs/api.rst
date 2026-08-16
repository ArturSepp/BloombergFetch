.. meta::
   :description: Top-level bbg_fetch Python API reference for Bloomberg request/response workflows.

Top-level API reference
=======================

The inventory below is checked against the actual non-underscore names bound on
``bbg_fetch``. The package does not currently define ``__all__``; consequently,
the two imported module names are part of the observable top-level surface too.

.. currentmodule:: bbg_fetch

.. code-block:: text
   :class: export-inventory

   DEFAULT_START_DATE
   DEFAULT_TENOR_YEARS
   FX_DICT
   IMPVOL_FIELDS_DELTA
   IMPVOL_FIELDS_MNY_12M
   IMPVOL_FIELDS_MNY_30DAY
   IMPVOL_FIELDS_MNY_3MTH
   IMPVOL_FIELDS_MNY_60DAY
   IMPVOL_FIELDS_MNY_6MTH
   OPTION_CHAIN_FIELDS
   OptionChainResult
   OptionPriceSource
   VOLS_START_DATE
   bdh
   bdp
   bds
   contract_to_instrument
   core
   disconnect
   fetch_active_futures
   fetch_balance_data
   fetch_bonds_info
   fetch_cds_info
   fetch_div_yields
   fetch_dividend_history
   fetch_field_timeseries_per_tickers
   fetch_fields_timeseries_per_ticker
   fetch_fundamentals
   fetch_futures_contract_table
   fetch_index_members_weights
   fetch_issuer_isins_from_bond_isins
   fetch_last_prices
   fetch_option_chain
   fetch_tickers_from_isins
   fetch_vol_surface
   fetch_vol_timeseries
   instrument_to_active_ticker
   option_chain
   recover_option_forward
   run

Detailed reference
------------------

The reference is generated from the installed package and includes imported
top-level functions, classes, enums, and constants.

.. automodule:: bbg_fetch
   :members:
   :imported-members:
   :member-order: bysource
   :show-inheritance:

Top-level modules
-----------------

``bbg_fetch.core`` and ``bbg_fetch.option_chain`` are also bound on the package
when it is imported. Their supported user-facing callables are documented above
through the top-level namespace.

.. automodule:: bbg_fetch.core

.. automodule:: bbg_fetch.option_chain
