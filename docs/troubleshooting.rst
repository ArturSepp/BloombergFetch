.. meta::
   :description: Diagnose bbg-fetch installation, Bloomberg session, entitlement, field, and empty-response failures.

Troubleshooting
===============

Author: `Artur Sepp <https://github.com/ArturSepp>`_ / First recorded:
`2026-08-16 <https://github.com/ArturSepp/BloombergFetch/commit/0832b666ead0e2b695113beb98f3428f051ee150>`_.

Implemented in `bbg-fetch <https://github.com/ArturSepp/BloombergFetch>`_.
Software citation: `CITATION.cff <https://github.com/ArturSepp/BloombergFetch/blob/main/CITATION.cff>`_.

``No module named blpapi``
----------------------------

Install Bloomberg's package from its separate index as shown in
:doc:`installation`. Installing ``bbg-fetch`` from PyPI does not install
``blpapi`` for you.

Connection or session startup fails
-----------------------------------

Confirm that Bloomberg Professional is running and logged in on the same
Windows machine. The Desktop API normally uses ``localhost:8194``. A successful
terminal-free example only proves the installed computation/API path; it does
not test this connection.

If local security software or policy blocks the port or Desktop API, work with
your organisation's Bloomberg administrator. Do not place credentials in code
or repository configuration.

A request times out
-------------------

Retry a small request after confirming the session. Large security/field sets,
slow Bloomberg responses, and connection interruptions can all surface as
timeouts. Reduce the request to one entitled security and one known field before
investigating batching or higher-level workflows.

The result is empty
-------------------

Check all of the following independently:

* the instrument includes its Bloomberg market-sector suffix, for example
  ``AAPL US Equity``;
* the field mnemonic is valid for that instrument;
* the date or option expiry is current and available;
* the logged-in user is entitled to the security and field;
* a bulk field is requested with ``bds`` rather than ``bdp`` or ``bdh``.

An empty DataFrame is not evidence that any one of these causes is responsible.
Verify the security and field in the Bloomberg Terminal before changing code.

A field or security error is returned
-------------------------------------

Use Bloomberg's Terminal field search and security lookup to verify identifiers.
``bbg-fetch`` normalises response column labels for pandas use, but it does not
translate an arbitrary concept into a Bloomberg field or bypass field-specific
permissions.

The package imports but a live request fails
---------------------------------------------

Import success, session success, and data entitlement are three separate
checkpoints. Run the terminal-free example first, then one small local live
diagnostic. Print only schema, dimensions, and success state when recording a
diagnostic; do not commit Bloomberg values or terminal output.

If the problem persists, open an `issue
<https://github.com/ArturSepp/BloombergFetch/issues>`_ with the package version,
Python version, exception type, and a redacted request shape. Do not include
credentials or proprietary response values.
