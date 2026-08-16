.. meta::
   :description: Bloomberg Desktop API request/response data in pandas DataFrames for quantitative research.
   :google-site-verification: WJen7v3RzYStpnJNMjZL5X35cuWl__U-MBvZtgN65-g

bbg-fetch
=========

``bbg-fetch`` provides Bloomberg Desktop API request/response data in pandas
DataFrames for quantitative research.

It wraps BDP-, BDH-, and BDS-style requests and selected research workflows.
Live requests require a running Bloomberg Terminal, suitable entitlements, and
Bloomberg's separately installed ``blpapi``. Streaming and intraday
subscriptions are outside this project's scope.

Task guides
-----------

* :doc:`task_install_connect_diagnose` — install the two packages, establish the
  local Desktop API boundary, and isolate failures without exposing data.
* :doc:`task_request_data` — choose between high-level fetchers and
  ``bdp``/``bdh``/``bds``, with their exact pandas output contracts.
* :doc:`task_research_workflows` — use option, volatility, futures,
  fixed-income, and index-constituent workflows within their stated limits.

Choosing a client
-----------------

* :doc:`comparison` — a dated, source-linked comparison with direct Bloomberg
  ``blpapi``, ``xbbg``, and ``blp``; each option has use cases it serves best.

Start here
----------

* :doc:`installation` — install ``blpapi`` and ``bbg-fetch`` and understand the
  local Terminal boundary.
* :doc:`first_success` — verify the installed public API without claiming a
  Bloomberg connection.
* :doc:`api` — inspect the real top-level request, shaping, and research
  workflow exports.
* :doc:`troubleshooting` — diagnose installation, session, entitlement, field,
  and empty-response failures.

Project links
-------------

* `Source <https://github.com/ArturSepp/BloombergFetch>`_
* `Issues <https://github.com/ArturSepp/BloombergFetch/issues>`_
* `PyPI <https://pypi.org/project/bbg-fetch/>`_
* `Changelog <https://github.com/ArturSepp/BloombergFetch/blob/main/CHANGELOG.md>`_
* `License <https://github.com/ArturSepp/BloombergFetch/blob/main/LICENSE.txt>`_
* `Citation metadata <https://github.com/ArturSepp/BloombergFetch/blob/main/CITATION.cff>`_

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   task_install_connect_diagnose
   task_request_data
   task_research_workflows
   comparison
   first_success
   api
   troubleshooting
