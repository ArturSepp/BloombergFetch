.. meta::
   :description: A neutral, dated comparison of bbg-fetch, Bloomberg blpapi, xbbg, and blp for Python data workflows.

Choosing a Bloomberg Python client
==================================

Audit date: 2026-08-16.

This page compares stable releases visible on the audit date: `bbg-fetch 2.3.0
<https://pypi.org/project/bbg-fetch/2.3.0/>`_, `Bloomberg blpapi 3.26.7.1
<https://blpapi.bloomberg.com/repository/releases/python/simple/blpapi/>`_,
`xbbg 1.4.6 <https://pypi.org/project/xbbg/1.4.6/>`_, and `blp 0.0.4
<https://pypi.org/project/blp/0.0.4/>`_. Versions, interfaces, and maintenance
status can change after this date; follow the linked primary sources before a
new adoption decision.

No universal recommendation
---------------------------

These projects sit at different layers. Direct ``blpapi`` maximizes SDK
control. ``xbbg`` offers the broadest high-level surface among the compared
wrappers. ``blp`` keeps session, parsing, aggregation, and streaming concepts
visible in Python. ``bbg-fetch`` stays deliberately narrow around pandas
request/response data and selected quantitative-research shapes. The right
choice follows the required Bloomberg services, transport, output contract,
and amount of infrastructure the application should own.

All four choices still require an appropriately licensed and authorized
Bloomberg environment. A Python package does not provide Bloomberg data,
credentials, or entitlements; compare the `Bloomberg SDK documentation
<https://bloomberg.github.io/blpapi-docs/>`_, the `xbbg access statement
<https://github.com/xbbg-org/xbbg>`_, the `blp project scope
<https://github.com/matthewgilbert/blp>`_, and :doc:`bbg-fetch connection
prerequisites <task_install_connect_diagnose>`.

Side-by-side scope
------------------

.. list-table::
   :header-rows: 1
   :widths: 14 21 21 22 22

   * - Dimension
     - ``bbg-fetch 2.3.0``
     - direct ``blpapi 3.26.7.1``
     - ``xbbg 1.4.6``
     - ``blp 0.0.4``
   * - Intended layer
     - Compact pandas request/response client plus opinionated option,
       volatility, futures, fixed-income, dividend, and constituent helpers
       (:doc:`task workflows <task_research_workflows>`; `source
       <https://github.com/ArturSepp/BloombergFetch/tree/main/src/bbg_fetch>`_).
     - Bloomberg's SDK primitives: the application constructs sessions,
       services, requests, correlation IDs, and event handling (`Session
       documentation
       <https://bloomberg.github.io/blpapi-docs/python/3.26.6/_autosummary/blpapi.Session.html>`_;
       `Request documentation
       <https://bloomberg.github.io/blpapi-docs/python/3.26.5/_autosummary/blpapi.Request.html>`_).
     - Broad client backed by a Rust execution/parsing engine, with high-level
       helpers and a generic request escape hatch (`current project scope
       <https://github.com/xbbg-org/xbbg>`_).
     - Pythonic wrapper designed around explicit separation of session
       management, event parsing, and aggregation (`project README
       <https://github.com/matthewgilbert/blp>`_).
   * - Reference/history/bulk
     - Public ``bdp``, ``bdh``, and ``bds`` functions return documented pandas
       shapes (:doc:`request contracts <task_request_data>`).
     - Available by creating the appropriate Bloomberg service requests and
       consuming partial/final response events; there are no Excel-named
       DataFrame convenience functions in the SDK (`Request and Session
       documentation
       <https://bloomberg.github.io/blpapi-docs/python/3.26.6/>`_).
     - ``bdp``, ``bdh``, and ``bds`` are part of its common request surface
       (`1.4.6 API overview <https://pypi.org/project/xbbg/1.4.6/>`_).
     - ``BlpQuery`` exposes ``bdp``, ``bdh``, and ``bds`` returning pandas
       DataFrames (`API reference
       <https://matthewgilbert.github.io/blp/generated/blp.blp.html>`_).
   * - Beyond BDP/BDH/BDS
     - Selected research shaping over the reference-data service; no intraday
       bars/ticks, BQL, screening, or generic-service API (:doc:`API <api>` and
       :doc:`research scope <task_research_workflows>`).
     - The SDK exposes service, schema, request, provider, authorization, and
       subscription primitives; the application supplies service-specific
       request and output logic (`class inventory
       <https://bloomberg.github.io/blpapi-docs/python/3.26.6/classes.html>`_).
     - Documents intraday bars/ticks, BQL, screening/search, analytics,
       metadata, and generic requests (`common API surface
       <https://github.com/xbbg-org/xbbg>`_).
     - Documents intraday bars/ticks, BEQS, BQL, and lower-level query creation
       alongside BDP/BDH/BDS (`quickstart
       <https://matthewgilbert.github.io/blp/quickstart.html>`_; `current
       BlpQuery source
       <https://github.com/matthewgilbert/blp/blob/master/src/blp/blp.py>`_).
   * - Streaming
     - Not implemented; the project scope is request/response only
       (:doc:`project boundary <task_install_connect_diagnose>`).
     - Native synchronous or asynchronous ``Session`` subscriptions; the
       application owns event dispatch and correlation lifetimes (`Session
       documentation
       <https://bloomberg.github.io/blpapi-docs/python/3.26.6/_autosummary/blpapi.Session.html>`_).
     - Provides subscription/stream helpers and isolated subscription sessions
       (`real-time surface
       <https://github.com/xbbg-org/xbbg>`_; `engine model
       <https://github.com/xbbg-org/xbbg>`_).
     - ``BlpStream`` wraps an asynchronous Bloomberg session and yields parsed
       event dictionaries (`current source
       <https://github.com/matthewgilbert/blp/blob/master/src/blp/blp.py>`_).
   * - Output objects
     - pandas DataFrames/Series with workflow-specific index and column
       contracts (:doc:`request contracts <task_request_data>`).
     - ``Event``, ``Message``, and ``Element`` SDK objects; conversion and
       domain shaping belong to the application (`SDK class inventory
       <https://bloomberg.github.io/blpapi-docs/python/3.26.6/classes.html>`_).
     - Defaults to a Narwhals DataFrame and documents native Arrow, PyArrow,
       pandas, Polars, and DuckDB paths (`output backends
       <https://github.com/xbbg-org/xbbg>`_).
     - Query helpers return pandas DataFrames; streaming events are converted
       to dictionaries (`API reference
       <https://matthewgilbert.github.io/blp/generated/blp.blp.html>`_; `stream
       conversion source
       <https://github.com/matthewgilbert/blp/blob/master/src/blp/blp.py>`_).
   * - Dependency model
     - Runtime dependencies are NumPy and pandas; Bloomberg ``blpapi`` is a
       separate installation and is imported directly (`project metadata
       <https://github.com/ArturSepp/BloombergFetch/blob/main/pyproject.toml>`_;
       :doc:`installation`).
     - Bloomberg's Python package plus the compatible Bloomberg SDK/runtime;
       it adds no third-party DataFrame abstraction (`official Python source
       installation <https://github.com/msitt/blpapi-python>`_).
     - Native Rust package; the Python ``blpapi`` wheel is not required as a
       dependency, but Bloomberg's shared runtime is still required. Frame
       conversions are optional extras (`1.4.6 installation
       <https://pypi.org/project/xbbg/1.4.6/>`_).
     - Imports and wraps both ``blpapi`` and pandas (`current implementation
       <https://github.com/matthewgilbert/blp/blob/master/src/blp/blp.py>`_).
   * - Session/transport assumptions
     - One lazy, process-local shared session fixed to ``localhost:8194`` and
       ``//blp/refdata``; ``disconnect`` resets it (`request source
       <https://github.com/ArturSepp/BloombergFetch/blob/main/src/bbg_fetch/_blp_api.py>`_).
     - The application configures ``SessionOptions`` and chooses synchronous or
       asynchronous operation, DAPI/SAPI mode, addresses, authentication, TLS,
       and other supported SDK options (`SessionOptions documentation
       <https://bloomberg.github.io/blpapi-docs/python/3.26.6/_autosummary/blpapi.SessionOptions.html>`_).
     - Defaults to local DAPI but documents configurable DAPI, SAPI/B-PIPE,
       ZFP, TLS, failover, SOCKS5, and worker pools (`configuration and engines
       <https://github.com/xbbg-org/xbbg>`_).
     - ``BlpQuery`` and ``BlpStream`` accept host/port and forward session
       options; users explicitly start/stop or use context managers (`current
       session source
       <https://github.com/matthewgilbert/blp/blob/master/src/blp/blp.py>`_).

When each choice is a better fit
--------------------------------

Choose ``bbg-fetch`` when
~~~~~~~~~~~~~~~~~~~~~~~~~

You need pandas-native BDP/BDH/BDS output plus the package's existing
option-chain/parity, volatility-surface, futures-chain/carry, bond/issuer,
dividend, or constituent shaping, and a local Windows Desktop API session is
the intended boundary. Its advantage is the maintained workflow contract, not
general Bloomberg API breadth. Confirm the exact functions in :doc:`api` and
their limits in :doc:`task_research_workflows`.

Choose direct Bloomberg ``blpapi`` when
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your application needs SDK-level control over an unusual Bloomberg service,
authorization identity, provider/publisher behavior, correlation/event
lifecycle, or transport configuration and you are prepared to own parsing and
output schemas. The official `SDK class inventory
<https://bloomberg.github.io/blpapi-docs/python/3.26.6/classes.html>`_ and
`Session documentation
<https://bloomberg.github.io/blpapi-docs/python/3.26.6/_autosummary/blpapi.Session.html>`_
show that lower-level surface.

Choose ``xbbg`` when
~~~~~~~~~~~~~~~~~~~~

You need high-level intraday, streaming, async, BQL/search/analytics,
enterprise connection modes, or non-pandas output backends in one client.
Those are explicit parts of the `1.4.6 scope
<https://pypi.org/project/xbbg/1.4.6/>`_. The tradeoff is adopting
a much broader engine and configuration surface than a small request/response
pandas wrapper.

Choose ``blp`` when
~~~~~~~~~~~~~~~~~~~

You want a pandas-first Python wrapper with BDP/BDH/BDS, intraday, BQL/BEQS,
and streaming, while keeping session, parser, collector, and event concepts
available for Python-level extension. Its `project design
<https://github.com/matthewgilbert/blp>`_, `quickstart
<https://matthewgilbert.github.io/blp/quickstart.html>`_, and `stream source
<https://github.com/matthewgilbert/blp/blob/master/src/blp/blp.py>`_
support that fit. The documentation site still labels itself 0.0.3 while PyPI
serves 0.0.4, so release-specific documentation parity is unknown and should
be checked against the installed source.

Why ``pdblp`` is not a current comparison target
------------------------------------------------

The roadmap named ``pdblp`` as a candidate subject to source verification.
Its own `repository notice <https://github.com/matthewgilbert/pdblp>`_ says it
is no longer under active development and has been superseded by ``blp``.
Accordingly this dated guide compares the maintained successor rather than
presenting ``pdblp`` as a current peer.

Decision checklist
------------------

Before adopting any client, write down:

* required Bloomberg services: reference/history/bulk only, or also intraday,
  subscriptions, BQL/search, authorization, or provider operations;
* required transport: local Desktop API or firm-managed SAPI/B-PIPE/ZFP;
* required output: fixed pandas shapes, another frame backend, or raw SDK
  messages;
* ownership boundary: whether the library or the application should manage
  sessions, retries, event loops, parsing, and schema normalization;
* local entitlement and data-handling constraints, which no wrapper removes;
* version compatibility verified in the actual Bloomberg environment.

This comparison intentionally makes no speed, scale, or reliability ranking:
no equivalent live benchmark across the four choices was performed for U5.
