.. meta::
   :description: Bloomberg Desktop API request/response data in pandas DataFrames for quantitative research.

bbg-fetch
=========

``bbg-fetch`` provides Bloomberg Desktop API request/response data in pandas
DataFrames for quantitative research.

It wraps BDP-, BDH-, and BDS-style requests and selected research workflows.
Live requests require a running Bloomberg Terminal, suitable entitlements, and
Bloomberg's separately installed ``blpapi``. Streaming and intraday
subscriptions are outside this project's scope.

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
   first_success
   api
   troubleshooting
