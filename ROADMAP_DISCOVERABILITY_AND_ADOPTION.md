# bbg-fetch discoverability and adoption roadmap

Version 1.0, 2026-08-16

Status: proposed execution contract. Adapted from the maintainer's Unified OSS discoverability
and adoption roadmap v1.0 (2026-08-15). No stage is complete merely because it appears here.

## Outcome

Make `bbg-fetch` easier for qualified users to find, evaluate, install, and use without implying
that Bloomberg data or credentials are included. The target user is a quantitative researcher or
developer with access to the Bloomberg Desktop API who wants request/response data returned as
pandas objects without maintaining a session and event-loop wrapper.

The canonical identity sentence is:

> `bbg-fetch` — Bloomberg Desktop API request/response data in pandas DataFrames for quantitative
> research.

The required boundary sentence is:

> It wraps BDP-, BDH-, and BDS-style requests and selected research workflows; live requests still
> require a running Bloomberg Terminal, suitable entitlements, and Bloomberg's `blpapi`, while
> streaming and intraday subscriptions are out of scope.

Success means qualified discovery followed by an honest first result. Raw traffic, stars, and
downloads are secondary signals, not the objective.

## Package adaptation profile

| Field | Package-specific answer |
|---|---|
| Distribution name | `bbg-fetch` |
| Import name | `bbg_fetch` |
| Current source version | `2.3.0` in `pyproject.toml` and `bbg_fetch.__version__` on 2026-08-16 |
| One-sentence role | Bloomberg Desktop API request/response data in pandas DataFrames for quantitative research |
| Primary user | Quantitative researcher or Python developer with Bloomberg Terminal access |
| Priority task 1 | Install `blpapi` and `bbg-fetch`, validate the environment, and understand terminal/entitlement prerequisites |
| Priority task 2 | Fetch historical, reference, and bulk data through public high- and low-level functions |
| Priority task 3 | Use specialised option-chain, volatility, futures, fixed-income, and constituent workflows with correct output schemas |
| Differentiating workflow | A thin direct-`blpapi` request/response layer plus analysis-ready pandas shaping, with no stack dependency |
| Canonical repository | `https://github.com/ArturSepp/BloombergFetch` |
| Canonical documentation | None yet; U3 must establish one approved HTTPS root before metadata points to it |
| Package index | `https://pypi.org/project/bbg-fetch/` |
| Documentation system | None yet; select and record a static Sphinx host in U0/U3 |
| First-success archetype | Credentialed service library |
| First-success constraint | `blpapi` is installed separately; a local Bloomberg Terminal session and entitlements are required only for the live diagnostic |
| Release authority | Maintainer with PyPI, GitHub, and documentation-host credentials |
| Existing analytics | PyPI/GitHub public signals and a Pepy badge; Search Console and documentation analytics are unknown |
| Scientific publication boundary | No publication work is part of this roadmap |

## Binding decisions

1. **M0.1, migration to a `src/` layout, is mandatory.** Adoption work must not institutionalise
   the current flat package layout. The migration preserves the `bbg_fetch` import name, public
   call signatures, runtime behaviour, and numerical outputs.
2. **M0.2, authoritative examples at repository-root `examples/`, is mandatory.** The directory
   already exists and stays at root through the layout migration. Examples are made deliberate,
   runnable entry points rather than copied into the package or documentation.
3. This is a credentialed connector. CI and hosted examples never claim to prove a Bloomberg
   connection. Live calls remain separately labelled local diagnostics.
4. Do not mock `blpapi`, commit Bloomberg data, record terminal output, embed entitlements, or add
   streaming support. Synthetic public-API examples may exercise terminal-free computation.
5. Do not add runtime dependencies. Documentation and build tools may be optional development
   dependencies; `blpapi` remains documented as a separate Bloomberg-index installation unless a
   packaging investigation proves a better supported route.
6. Do not change public signatures or results under this roadmap. Any such need stops the stage
   and gets a separate approved roadmap, changelog entry, and version decision.
7. Public execution roadmaps live at repository root. Operational Markdown reports and working
   notes live under ignored `agents/`; conclusions users need are promoted into tracked docs.
8. A release is never inferred. Publishing to PyPI, tagging, or changing external project
   settings requires explicit maintainer approval.

## Target repository structure

```text
src/
  bbg_fetch/
    __init__.py
    _blp_api.py
    core.py
    option_chain.py
    tests/                  live Bloomberg integration diagnostics; preserved for compatibility
tests/                      terminal-free CI tests
examples/                   authoritative runnable examples; mandatory repository-root location
docs/                       tracked user documentation source, if U0 approves the docs investment
agents/                     ignored operational Markdown roadmaps and reports
ROADMAP_DISCOVERABILITY_AND_ADOPTION.md
```

`examples/` must not move under `src/bbg_fetch/`. The package's internal live-test location is not
changed by M0 unless a separate compatibility review authorises it.

## Stage overview and order

| Order | Stage | Deliverable | Gate |
|---:|---|---|---|
| 1 | M0.1 | Migrate the import package to `src/bbg_fetch/` | Mandatory |
| 2 | M0.2 | Establish root `examples/` as the authoritative example surface | Mandatory |
| 3 | U0 | Package profile and proceed/defer decision | Mandatory |
| 4 | U1 | Dated discovery and conversion baseline | Mandatory before public changes |
| 5 | U2 | One canonical package identity and consistent metadata | Selected |
| 6 | U3 | Documentation foundation and technical discoverability | Conditional on U0 proceed |
| 7 | Gate A | External repository/docs/Search Console alignment | Maintainer action |
| 8 | U4 | Task-oriented documentation for the three priority tasks | Selected if U3 proceeds |
| 9 | U5 | Neutral comparison and choice guide | Selected after core docs are stable |
| 10 | U6 | Single-source first-success workflow | Selected; root examples are authoritative |
| 11 | Gate B/U7 | Hosted notebook decision and optional thin notebook | Default recommendation: skip |
| 12 | U8 | Explicit release/deployment and trust-surface verification | Maintainer approval required |
| 13 | U9 | Approximately 30/60/90-day measurement | Starts after public deployment |

Stages execute in order. M0 may be released separately from adoption content if that produces a
smaller review. U1 must be captured before U2-U6 change public conversion paths.

---

## M0.1 — Mandatory migration to `src/` layout

**Deliverable:** one packaging-only change that moves `bbg_fetch/` to `src/bbg_fetch/` and updates
all repository-owned path assumptions.

Required work:

- move the package without changing module names or content;
- set setuptools package discovery to `src`, including an explicit package-dir mapping;
- update Ruff per-file paths, coverage configuration, CI commands, AGENTS layout notes, and the
  README package tree;
- preserve top-level `tests/`, root `examples/`, and `src/bbg_fetch/tests/` live diagnostics;
- add a repository-layout regression check that fails on the pre-migration tree and passes only
  when `src/bbg_fetch/__init__.py` exists and root `bbg_fetch/` does not;
- build both wheel and sdist and inspect their contents for the expected package and exclusions;
- install the wheel into a clean environment outside the checkout and prove `import bbg_fetch`
  resolves from the installed artefact rather than the working tree.

**Acceptance:** the wheel contains `bbg_fetch`, no import package remains at repository root, the
public import/export surface matches the pre-migration snapshot, terminal-free tests pass, and no
numerical output or live request construction changes.

**Verification:** record exact versions and results in `agents/SRC_LAYOUT_MIGRATION_REPORT.md`.
The execution stage must run at least:

```bash
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
python -m pip install -e ".[dev]"
python -m pytest tests/
python -m ruff check src/bbg_fetch tests examples
python -m build
python -c "from pathlib import Path; assert Path('src/bbg_fetch/__init__.py').is_file(); assert not Path('bbg_fetch').exists()"
```

CI must additionally install the built wheel in a clean environment and run the terminal-free
suite. Capture the expected pre-migration failure of the new layout check before moving the
package, then restore the migration and capture the pass.

**Out of scope:** module refactors, public API cleanup, integration-test relocation, dependency
changes, version bump, and live Bloomberg calls.

## M0.2 — Mandatory authoritative examples at repository root

**Deliverable:** a curated `examples/` directory at repository root with an index and two clearly
separated classes of example.

The root example contract is:

- `examples/README.md` states prerequisites and labels every script `NO TERMINAL` or
  `BLOOMBERG TERMINAL REQUIRED`;
- one deterministic terminal-free installation example uses only public `bbg_fetch` symbols and
  compact synthetic inputs, and prints stable evidence without claiming a live connection;
- live examples cover the common price/reference request and one specialised workflow while
  remaining small enough to inspect and edit;
- existing `fetch_core_data.py`, `fetch_div_history.py`, and `fetch_option_chain.py` are retained,
  renamed, or split only after checking all inbound README/docs links;
- no example writes fetched data, CSV snapshots, figures, credentials, or terminal output into
  the repository;
- README and docs link to these scripts or mechanically include them; they do not maintain a
  second drifting copy of the same workflow;
- packaging keeps `examples/` at repository root and does not install it as `bbg_fetch.examples`.

**Acceptance:** a new user can identify which examples are safe without a terminal; the
terminal-free example succeeds against the built wheel; every live script compiles and has an
explicit local-only entry point; `examples/` remains at root after M0.1.

**Verification:** record results in `agents/EXAMPLES_REPORT.md` and run:

```bash
python -m compileall -q examples
python examples/quickstart_no_terminal.py
python -m ruff check examples
```

Run live scripts manually only on an entitled Bloomberg machine and report pass/failure without
capturing proprietary response values.

**Out of scope:** notebooks, downloaded fixtures, generated output, analytics using sibling
packages, and any example that makes a live Bloomberg call in CI.

---

## U0 — Triage and adapt

**Deliverable:** `agents/ADOPTION_PROFILE.md` with a dated proceed/defer decision.

Validate the profile above against the released artefact, top-level exports, issues/support
questions, root examples, package-index page, repository settings, and any existing documentation
traffic. Estimate the maintenance cost of a documentation site and live examples. Confirm the
three priority tasks or revise them using evidence.

Resolve these known local inconsistencies before marking U0 complete:

- `pyproject.toml` and `bbg_fetch.__version__` say 2.3.0, while `CITATION.cff` says 2.0.3 and the
  README BibTeX block says 2.0.1;
- `requires-python` says 3.10+, while current prose and classifiers describe different ranges;
- README installation claims about available `blpapi` wheels must be checked against Bloomberg's
  current official distribution rather than copied forward;
- the repository has no canonical documentation URL or documentation build today;
- the README's opening example uses `pd.Timestamp` without importing pandas and therefore is not
  a self-contained first-success snippet.

**Acceptance:** the package has a distinct role, three evidence-backed tasks, a feasible
credentialed-service first-success contract, a chosen docs-host direction or explicit defer, and a
maintainer-time estimate.

**Verification:** cross-check local metadata, the built wheel, public package/repository surfaces,
and official Bloomberg installation guidance. Label external or credentialed unknowns as unknown.

**Out of scope:** changing files while measuring, adding features, or committing to a publication
venue.

## U1 — Establish the baseline

**Deliverable:** `agents/DISCOVERABILITY_BASELINE.md`.

Record one dated snapshot with the following headings:

- `Indexing`: PyPI version/metadata, GitHub About, current docs absence or status, links, HTTP
  results, and any canonical/robots/sitemap state;
- `Queries`: exact `bbg-fetch`/`bbg_fetch` treatment plus three fixed non-branded task queries;
- `Conversion path`: search or package page → repository/docs → installation → terminal-free
  check → live local diagnostic;
- `Adoption signals`: trailing-period downloads, stars/forks/watchers/dependents, issues,
  citations, referrals if available, with definitions;
- `Limitations`: personalisation, geography, fresh Search Console properties, privacy thresholds,
  proprietary access, and unavailable analytics.

Search spot checks are observations, not rank measurements. Never record Bloomberg response data
or a private analytics export.

**Acceptance:** every value has a date, source family, and definition; missing credentialed data
is explicitly unknown; the three task queries and priority pages are frozen for U9.

**Verification:** confirm every required heading and baseline definition is present before any U2
change.

**Out of scope:** changing GitHub, PyPI, docs, metadata, examples, or Search Console while taking
the baseline.

## U2 — Establish one canonical package identity

**Deliverable:** one small tracked change aligning repository-owned identity and trust surfaces.

Use the canonical and boundary sentences from this roadmap unless U0 records an evidence-based
revision. Align:

- `[project].description`, keywords, supported-Python declarations, and project URLs;
- README title, opening, requirements, and first code block;
- documentation landing title once U3 establishes the site;
- `CITATION.cff`, the README citation, changelog, and all version claims;
- repository About text through Gate A, not through an unauthorised local edit.

Replace claims such as “production-ready,” “same result,” or precise boilerplate line counts unless
they are narrowly defined and reproducibly supported. Preserve the distribution/import tokens and
state the Bloomberg Terminal, `blpapi`, and entitlement boundary near installation and first use.

**Acceptance:** every primary surface describes the same request/response package, current version,
Python support, and access prerequisites without keyword lists or superiority claims.

**Verification:** run tests and lint, build wheel/sdist, inspect wheel `METADATA` `Summary:` and
project URLs, render the README as PyPI will, and build docs warning-free once U3 exists.

**Out of scope:** GitHub settings, release, custom domain, public signature changes, or API growth.

## U3 — Establish documentation and technical discoverability

**Deliverable:** a minimal tracked documentation tree plus deployed technical audit, conditional
on U0 approving ongoing docs maintenance.

Default proposal: Sphinx source under root `docs/`, with its tools isolated in a documentation or
development extra, deployed on an approved static host. U0 may choose another static system, but
must record why and keep the same acceptance contract. Establish one HTTPS canonical root before
adding a Documentation project URL.

The initial site contains only:

- landing page with package boundary and routes to the three user tasks;
- installation/connection page;
- first-success page sourced from root `examples/`;
- API reference for actual top-level public exports;
- troubleshooting page;
- changelog, source, issues, PyPI, license, and citation links.

Audit the deployed root and priority pages for HTTP status, canonical HTTPS URLs, accidental
`noindex`, robots exclusions, redirect loops, server-rendered titles/descriptions, internal
navigation, and sitemap validity. Use the host's native sitemap if sound; add no sitemap extension
speculatively.

**Acceptance:** warning-free docs build; all priority pages are reachable without JavaScript-only
navigation; canonical URLs and public links agree; deployed robots/canonical/sitemap behaviour is
recorded in `agents/DISCOVERABILITY_AUDIT.md`.

**Verification:** commands are finalised when the docs system is selected. For Sphinx, the minimum
local gate is:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
```

Follow with scripted deployed HTTP checks after the host rebuilds.

**Out of scope:** custom-domain migration, paid SEO tools, analytics cookies by default, AI crawler
files, or documenting Bloomberg's proprietary field catalogue.

## Maintainer gate A — External identity and indexing

After the docs URL is deployed and stable, the maintainer:

1. updates GitHub About to the canonical sentence and sets repository, docs, and package links;
2. creates or verifies the documentation property in Google Search Console;
3. uses a persistent public verification method appropriate to the host without committing a
   private credential;
4. submits or confirms the canonical sitemap;
5. inspects the landing, installation, first-success, and API pages;
6. records a redacted summary under `agents/` and keeps raw exports elsewhere.

Normal crawl latency is recorded as such. Gate evidence includes property type, verification date,
sitemap state, and point-in-time priority-page statuses.

## U4 — Publish task-oriented documentation

**Deliverable:** focused pages for the three priority tasks.

### Task 1: install, connect, and diagnose

Document the separate Bloomberg `blpapi` installation, supported Python/OS evidence, Desktop API
localhost/session prerequisite, entitlements, expected import check, connection failure classes,
timeouts, empty responses, proxy installation path, and what can be verified without a terminal.

### Task 2: request historical, reference, and bulk data

Route users between high-level fetch functions and `bdp`/`bdh`/`bds`. State accepted input forms,
DatetimeIndex and column behaviour, field-name normalisation, adjustment flags, batching, missing
data, timeouts, and session shutdown. Examples use public symbols and link to root scripts.

### Task 3: specialised research workflows

Cover option chains/parity, volatility surfaces, futures, fixed income, and index constituents as
separate concise workflows. State units, shapes, Bloomberg fields passed through, date/expiry
conventions, entitlements, request-size risks, and important non-goals. Do not turn this into a
copied Bloomberg field catalogue.

Every task page explains the problem, prerequisites, exact inputs/outputs, minimal example,
expected shape rather than licensed values, failure modes, non-goals, and links to API/source.

**Acceptance:** each priority task has one clear entry page; every named public symbol exists; all
terminal-free snippets execute; live snippets are maintainer-checked without recording values.

**Verification:** warning-free docs and link builds, a top-level export coverage check, execution
of root terminal-free examples, compile checks for live examples, and one redacted local-terminal
diagnostic per live workflow class.

**Out of scope:** new request types, analytics owned by sibling packages, streaming, proprietary
data fixtures, and speculative entitlement claims.

## U5 — Publish a neutral comparison and choice guide

**Deliverable:** one dated page comparing `bbg-fetch` with direct Bloomberg `blpapi` and two
maintained Python alternatives such as `xbbg` and `pdblp`, subject to U5 source verification.

Compare design audience and workflow: direct API exposure, request/response coverage, streaming,
output objects, dependency model, session assumptions, platform/access requirements, and project
scope. Include at least one use case that favours each alternative. Use current stable versions and
primary official documentation/repositories at execution time; mark unknowns.

**Acceptance:** every nontrivial competitor claim links to a primary source, versions and audit
date are stated, and no package is universally recommended.

**Verification:** docs/link builds plus manual claim-to-source audit.

**Out of scope:** favourable performance theatre, popularity as technical evidence, copying code,
or adding comparison targets as dependencies.

## U6 — Create one source of truth for first success

**Deliverable:** the authoritative root example from M0.2 plus a docs page that includes or
mechanically checks it.

The first-success funnel has two honest checkpoints:

1. **No terminal:** install `blpapi` and the built `bbg-fetch` wheel, import the package, report
   versions, run deterministic public helpers or synthetic option-parity recovery, and print
   compact stable evidence. Label this an installation/API check, not a Bloomberg connection test.
2. **Local terminal diagnostic:** make one small entitled request, print only schema/dimensions and
   success state, explain how to change ticker/field, and surface session, timeout, empty-result,
   field, and entitlement failures. This is never a CI gate or hosted-notebook promise.

Test the released or built wheel from outside the checkout. README and docs point to the same root
scripts instead of copying them. Do not write results to disk.

**Acceptance:** a new user can distinguish successful installation from successful Bloomberg
access and reach a meaningful DataFrame with stated local prerequisites.

**Verification:** terminal-free wheel execution on CI's supported platform, compilation of every
live example, and one maintainer-recorded redacted run on an entitled Bloomberg machine.

**Out of scope:** mocking `blpapi`, live services in CI, checked-in response data, notebook extras,
or a second example implementation in docs.

## Maintainer gate B and U7 — Hosted notebook decision

**Default recommendation: skip.** A hosted notebook cannot access the user's local Bloomberg
Desktop API session and would make the central workflow look more accessible than it is.

Only approve U7 if the terminal-free synthetic workflow has independent educational value and the
notebook prominently states that it does not test or provide Bloomberg access. If approved, it is
thin, output-free, installs the released package, reports its version, derives from the U6 script,
and has a mechanical drift check. It never embeds Bloomberg data.

Record `SKIPPED` with the credentialed-service reason after the maintainer confirms the default;
do not leave the decision implicit.

## U8 — Release, deploy, and align trust surfaces

**Deliverable:** either an explicitly approved package release or a docs-only deployment record.

Before a release:

- reconcile `pyproject.toml`, `bbg_fetch.__version__`, `CITATION.cff`, README BibTeX, changelog, tag,
  and release date/version rules;
- run terminal-free tests, lint touched files, docs/link builds, wheel/sdist builds, artefact-content
  inspection, and the clean-wheel first-success check;
- verify examples and docs describe the version to be published;
- keep M0 migration and adoption edits separable if review or rollback benefits;
- obtain explicit maintainer approval before tagging or publishing.

After release/deployment, directly inspect PyPI's rendered README and metadata, GitHub tag/release,
the intended docs commit/build, canonical URLs, robots, sitemap, and priority pages. Record commit,
tag, package URL, release URL, docs build, and results in `agents/RELEASE_DEPLOYMENT_REPORT.md`.

**Acceptance:** all public surfaces expose the intended version, package identity, prerequisites,
and canonical links; successful upload alone is not completion.

**Out of scope:** bundling unrelated features or numerical changes into the adoption release.

## U9 — Measure at approximately 30, 60, and 90 days

**Deliverable:** `agents/DISCOVERABILITY_90_DAY_REPORT.md`, updated at each checkpoint.

Fix definitions at U1:

- Search Console window: latest 28 complete days ending at least two days before observation;
- branded query: contains `bbg-fetch` or `bbg_fetch`, case-insensitive;
- non-branded queries: the three task queries frozen at baseline;
- priority pages: fixed in U1;
- downloads: same trailing-period source, never interpreted as unique users;
- exact-name search treatment: dated text observation in the maintainer's normal context;
- missing, delayed, privacy-suppressed, or unavailable data stays missing.

At each checkpoint compare index/canonical state, branded and non-branded impressions/clicks/CTR,
documentation entry and first-success paths where observable, package downloads, GitHub signals,
and attributable issues/references. Search Console properties created during this roadmap have no
historical D0 data; state the limitation and use later like-for-like windows.

At 90 days recommend exactly one: deepen a performing topic, repair a demonstrated conversion
failure, improve external distribution, or stop investing in a channel. Use multiple observations
and do not infer causality from one metric.

**Acceptance:** every checkpoint uses frozen definitions, missing data is explicit, and the final
recommendation follows from the evidence.

**Scheduling:** after U8 deployment, create one task-attached follow-up covering all three dates in
the maintainer's Europe/Zurich timezone. Scheduling starts U9; it does not complete it.

## Operational documents under ignored `agents/`

Expected local Markdown artefacts:

```text
agents/README.md
agents/ADOPTION_PROFILE.md
agents/DISCOVERABILITY_BASELINE.md
agents/SRC_LAYOUT_MIGRATION_REPORT.md
agents/EXAMPLES_REPORT.md
agents/DISCOVERABILITY_AUDIT.md
agents/RELEASE_DEPLOYMENT_REPORT.md
agents/DISCOVERABILITY_90_DAY_REPORT.md
```

Additional stage notes may use `agents/ROADMAP_<TOPIC>.md` or `agents/<STAGE>_REPORT.md`. Keep raw
analytics exports, credentials, browser/session state, proprietary data, fetched responses, and
terminal output outside the repository entirely. The ignored directory is for Markdown decisions
and redacted evidence, not a loophole for sensitive data.

## Status log

Append one line for every completed, skipped, or blocked stage:

```text
YYYY-MM-DD · stage · branch/commit · PASS|SKIPPED|BLOCKED · concise verification result
```

Use `PASS-LOCAL` only temporarily before a required deployment check, then replace it with `PASS`.

2026-08-16 · M0.1 · main/uncommitted · PASS · src-layout guard failed before migration, then 25 terminal-free tests, scoped Ruff, build, clean-wheel import, and artefact inspection passed.

2026-08-16 · M0.2 · main/uncommitted · PASS · root example index, deterministic terminal-free quickstart, labelled live scripts, compile check, CI gate, and redacted live BDP diagnostic passed.

2026-08-16 · U0 · main/uncommitted · PASS · credentialed-service profile validated; proceed with Sphinx/GitHub Pages direction and a 2–4 hour monthly maintenance budget.

2026-08-16 · U1 · main/uncommitted · PASS · dated GitHub/PyPI/search/conversion baseline recorded with fixed queries, priority URLs, public signals, and explicit analytics unknowns.

2026-08-16 · U2 · main/uncommitted · PASS · canonical identity, Python/platform support, citation metadata, tests, Ruff, distributions, and wheel METADATA agree on 2.3.0.

2026-08-16 · U3 · main/f2a1a99 · PASS · CI and Pages deployment passed; five priority pages, robots, sitemap, canonical metadata, and links return successfully over public HTTPS.

2026-08-16 · Gate A · main/f2a1a99 · PASS · GitHub About aligned; Search Console URL-prefix ownership verified persistently; sitemap submitted; priority pages inspected and correctly recorded as newly unknown to Google.

2026-08-16 · U4 · main/052302b · PASS · three task guides, navigation/sitemap coverage, symbol/snippet checks, terminal-free example, strict docs/link builds, and redacted live workflow diagnostics passed.

2026-08-16 · U5 · main/c49fa5c · PASS · dated neutral comparison published for blpapi, xbbg, blp, and bbg-fetch; primary-source claim audit, strict docs/link builds, CI, Pages, and public HTTPS checks passed.

2026-08-16 · U6 · main/4657a6a · PASS · authoritative terminal-free and live scripts, built-wheel execution outside the checkout, example compilation, 36 tests, CI, Pages, public HTTPS checks, and a redacted live BDP diagnostic passed.

2026-08-16 · Gate B/U7 · main/3935527 · SKIPPED · maintainer confirmed the default: a hosted notebook cannot access the local Bloomberg Desktop API, while the authoritative U6 scripts already cover honest terminal-free and entitled-machine checkpoints without duplication.

## Definition of complete

Implementation is complete when both mandatory M0 stages pass, selected U0-U8 stages pass, public
deployments and maintainer gates are recorded, and U9 follow-ups are scheduled. The roadmap itself
is complete only after the final U9 observation and recommendation. Version publication,
scientific submission, API expansion, and streaming support remain separate maintainer-approved
work.
