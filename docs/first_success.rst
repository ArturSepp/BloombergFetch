.. meta::
   :description: Run the deterministic bbg-fetch installation and public-API check without a Bloomberg Terminal.

First success without a Terminal
================================

*Author:* `Artur Sepp <https://github.com/ArturSepp>`_ / First recorded:
`2026-08-16 <https://github.com/ArturSepp/BloombergFetch/commit/0832b666ead0e2b695113beb98f3428f051ee150>`_.

Source: `BloombergFetch <https://github.com/ArturSepp/BloombergFetch>`_.
Software citation: `CITATION.cff <https://github.com/ArturSepp/BloombergFetch/blob/main/CITATION.cff>`_.

Reproducible released workflow
------------------------------

The tested package floor is **bbg-fetch 3.2.1** on Python 3.12, not a historical
introduction version. The released example was verified with **blpapi 3.26.9.1**.
The genuine Bloomberg SDK is required even for import, although it
is absent from PyPI runtime dependencies. A PyPI-only installation is insufficient.
Install the SDK from its official index; no mock SDK is used. Use an environment and working
folder outside OneDrive and other synchronized directories:

.. code-block:: powershell

   python -m pip install "bbg-fetch==3.2.1"
   python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi==3.26.9.1
   python -c "from importlib.metadata import version; import bbg_fetch as p; print(version('bbg-fetch')); print(version('blpapi')); print(p.__file__)"
   Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/ArturSepp/BloombergFetch/059f3e49ba6fc45db10107ca33a2e6e6921ac9eb/examples/quickstart_no_terminal.py' -OutFile quickstart_no_terminal.py
   python quickstart_no_terminal.py

Setup/downloads need internet; execution is offline. Record SDK/package versions and measure
setup separately from execution. No universal runtime is promised. Inputs are a synthetic
ten-row call/put table over five strikes, spot 100, forward 102, continuously compounded annual
rate 0.03 and year fraction 0.25. The script checks the recovered forward/rate, writes no files,
opens no Terminal session and requires no credentials. This cannot verify connectivity,
entitlement or licensed-data correctness.

`Stable documentation <https://bloombergfetch.readthedocs.io/en/stable/>`_ is the release channel;
`latest <https://bloombergfetch.readthedocs.io/en/latest/>`_ follows main and may be unreleased.
Check the build's source version; a stable alias does not prove the latest PyPI upload.

The authoritative first-success script lives at repository root in
``examples/quickstart_no_terminal.py``. Run it from a clone after installing
``blpapi`` and ``bbg-fetch``:

.. code-block:: powershell

   python examples/quickstart_no_terminal.py

Expected evidence
-----------------

The script prints the Python, ``bbg-fetch``, and ``blpapi`` versions, a
deterministic synthetic forward and rate, the number of strikes used, and this explicit
boundary:

.. code-block:: text

   Bloomberg connection: NOT TESTED

It exercises the public ``recover_option_forward`` function with a compact
synthetic option chain satisfying put-call parity. It does not open a session,
fetch licensed values, or write output to disk.

Authoritative script
--------------------

The source below is included mechanically from the root example so the docs do
not maintain a second implementation.

.. literalinclude:: ../examples/quickstart_no_terminal.py
   :language: python
   :linenos:

Next steps
----------

After this check passes, review :doc:`installation` for the live Desktop API
boundary and choose a labelled local-only script in the `root examples index
<https://github.com/ArturSepp/BloombergFetch/tree/main/examples>`_. Never use a
live script as a CI or hosted-notebook connection test.
