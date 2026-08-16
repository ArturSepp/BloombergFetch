.. meta::
   :description: Install bbg-fetch and Bloomberg blpapi, then check Desktop API prerequisites.

Installation and connection prerequisites
=========================================

What you need
-------------

``bbg-fetch`` supports Python 3.10 and newer. The documented live workflow is
for Windows with Bloomberg Professional running on the same machine. Access to
data depends on the logged-in user's Bloomberg entitlements; the package does
not provide credentials or market data.

A Terminal is not needed for the deterministic installation check, but
Bloomberg's Python package must still be importable because ``bbg_fetch`` loads
the request layer at import time.

Install Bloomberg's Python API
------------------------------

Bloomberg distributes ``blpapi`` through its own package index. In a fresh
virtual environment, run:

.. code-block:: powershell

   python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

If a corporate proxy blocks that index, download a wheel from the same official
index in an approved browser session and install the local file:

.. code-block:: powershell

   python -m pip install C:\path\to\blpapi-wheel.whl

See Bloomberg's `API Library
<https://professional.bloomberg.com/support/api-library/>`_ for the current
distribution and platform guidance.

Install bbg-fetch
-----------------

.. code-block:: powershell

   python -m pip install bbg-fetch

Verify the imports and package version:

.. code-block:: powershell

   python -c "import blpapi, bbg_fetch; print(bbg_fetch.__version__)"

This proves that the packages import. It does not prove that a Bloomberg
session or a particular entitlement is available. Continue with
:doc:`first_success` for the terminal-free public-API check.

Live Desktop API boundary
-------------------------

For a live request, Bloomberg Professional must be running and logged in on
the same machine. The Desktop API normally connects to ``localhost:8194``.
Successful session startup still does not guarantee that a requested security
or field is entitled.

The library covers request/response workflows. It does not expose streaming or
intraday subscriptions, manage Bloomberg credentials, or bypass entitlements.
For connection and response failures, see :doc:`troubleshooting`.
