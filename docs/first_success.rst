.. meta::
   :description: Run the deterministic bbg-fetch installation and public-API check without a Bloomberg Terminal.

First success without a Terminal
================================

The authoritative first-success script lives at repository root in
``examples/quickstart_no_terminal.py``. Run it from a clone after installing
``blpapi`` and ``bbg-fetch``:

.. code-block:: powershell

   python examples/quickstart_no_terminal.py

Expected evidence
-----------------

The script prints the installed ``bbg-fetch`` version, a deterministic
synthetic forward and rate, the number of strikes used, and this explicit
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
