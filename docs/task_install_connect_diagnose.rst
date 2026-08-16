.. meta::
   :description: Install bbg-fetch and Bloomberg blpapi, connect to the local Desktop API, and diagnose failures safely.

Install, connect, and diagnose
==============================

The problem
-----------

A working ``bbg-fetch`` setup has three independent layers: Python imports,
a local Bloomberg Desktop API session, and entitlements for the requested
security and field. This guide verifies each layer separately so an import
success is not mistaken for data access.

Prerequisites and supported environment
---------------------------------------

The package metadata supports Python 3.10 and newer and classifies the live
platform as Microsoft Windows. The repository CI independently builds the
wheel, imports the package, runs tests, and builds the docs on Python 3.12;
CI cannot validate a Bloomberg session or entitlement.

For live data, Bloomberg Professional must be running and logged in on the
same machine. The request layer opens ``//blp/refdata`` on
``localhost:8194``. It does not accept or store Bloomberg credentials.
Entitlements belong to the logged-in Bloomberg user and can differ by
security, field, bulk dataset, and analytics function.

Install the two packages
------------------------

Bloomberg distributes ``blpapi`` separately; it is deliberately not a
``bbg-fetch`` runtime dependency. In a fresh virtual environment:

.. code-block:: powershell

   python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
   python -m pip install bbg-fetch

Behind a corporate proxy, use an organisation-approved browser or package
mirror to obtain a compatible wheel from Bloomberg's official index, then
install that local file:

.. code-block:: powershell

   python -m pip install C:\path\to\blpapi-wheel.whl

Do not use an unofficial ``blpapi`` package source. Bloomberg's current
platform and download guidance is available from its `API Library
<https://professional.bloomberg.com/support/api-library/>`_.

Verify without opening a Bloomberg session
-------------------------------------------

From a repository clone, run the authoritative terminal-free example:

.. code-block:: powershell

   python examples/quickstart_no_terminal.py

It imports ``blpapi`` and ``bbg_fetch``, calls the public
``bbg_fetch.recover_option_forward`` computation on deterministic synthetic
data, and prints ``Bloomberg connection: NOT TESTED``. Its source is included
on :doc:`first_success` and is exercised by CI. This verifies imports and a
public calculation only; it neither connects nor proves an entitlement.

Run one redacted live diagnostic
--------------------------------

With Bloomberg Professional open and logged in, request one field for one
publicly identifiable instrument. Record shape and schema, never returned
values. The authoritative root script is
`examples/diagnose_terminal.py
<https://github.com/ArturSepp/BloombergFetch/blob/main/examples/diagnose_terminal.py>`_:

.. code-block:: powershell

   python examples/diagnose_terminal.py

Choose another entitled security or scalar field without editing the script:

.. code-block:: powershell

   python examples/diagnose_terminal.py --ticker "IBM US Equity" --field "SECURITY_NAME"

The expected contract is a one-row DataFrame indexed by the requested ticker
with one normalized column, ``px_last``. A successful run reports ``PASS``,
the dimensions, the normalized column name, and the index metadata. No
licensed value is printed. A no-data result is classified separately because
an invalid identifier, invalid field, or missing entitlement can have the same
observable shape.

The script classifies import, session, timeout, no-data/field/security/
entitlement, and unexpected failures; it always closes the shared session.

Classify failures
-----------------

``ModuleNotFoundError: blpapi``
   The Bloomberg wheel is absent from this Python environment. Reinstall it
   from Bloomberg's index or an approved local wheel. Installing
   ``bbg-fetch`` alone does not install it.

``ConnectionError`` while starting the session
   The package could not start the local session or open ``//blp/refdata``.
   Confirm that Bloomberg Professional is running and logged in, then check
   local policy around ``localhost:8194``. Import success does not test this.

``TimeoutError`` after 60 seconds
   The internal response collector waited 60 seconds without a final
   response. Any partial messages are attached to the exception as
   ``partial_messages`` for local diagnosis. Retry a one-security/one-field
   request before investigating large request sizes or connectivity.

Security or field warning, ``NaN``, or an empty DataFrame
   These are not uniquely diagnostic. Verify the full market-sector ticker,
   field mnemonic, date or expiry, request type, and user entitlement in the
   Terminal. ``bdp`` preserves the requested ticker row and expected field
   column with ``NaN`` when no scalar value arrives; ``bds`` can return a
   completely empty DataFrame.

Proxy or TLS installation error
   This affects package download, not the localhost Desktop API session. Use
   the approved wheel path above or ask the organisation's package/Bloomberg
   administrator; do not disable TLS verification or commit proxy secrets.

What can be checked without a Terminal
--------------------------------------

You can verify Python compatibility, wheel installation, imports, package
version, top-level symbol presence, ticker string helpers, and the synthetic
put-call-parity calculation. You cannot verify session startup, field
validity, security resolution, freshness, or entitlement without the local
Terminal context.

Failure-report checklist
------------------------

A useful issue contains the ``bbg-fetch`` and Python versions, operating
system, exception type, request shape (counts and field names), and whether
the one-field diagnostic passed. Exclude credentials, proprietary values,
screenshots containing licensed data, and raw responses.

Non-goals and next links
------------------------

The package does not configure Bloomberg Professional, manage users or
entitlements, bypass network policy, or provide streaming/subscription data.
Continue with :doc:`task_request_data`, use :doc:`troubleshooting` for the
short failure index, inspect the :doc:`api`, or review the request layer
`source <https://github.com/ArturSepp/BloombergFetch/blob/main/src/bbg_fetch/_blp_api.py>`_.
