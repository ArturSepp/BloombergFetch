"""Sphinx configuration for the bbg-fetch documentation."""

import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 metadata tests use the compatible parser.
    import tomli as tomllib

project = "bbg-fetch"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
release = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
]

root_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build"]
# _templates/base.html titles pages other than the homepage "<page title> - bbg-fetch".
templates_path = ["_templates"]

html_theme = "furo"
html_title = "bbg-fetch - Bloomberg data in pandas DataFrames"


def _consolidate_stable(url: str) -> str:
    """Return the canonical base URL with the moving ``stable`` alias replaced by ``latest``.

    Read the Docs builds ``stable`` from the newest release tag and ``latest`` from ``main``, so
    both serve the same pages. Left alone, each copy names itself canonical and search engines see
    every page twice. Numbered versions keep their own canonical URL.
    """
    return re.sub(r"(\.readthedocs\.io/en/)stable(/|$)", r"\1latest\2", url)


html_baseurl = _consolidate_stable(
    os.environ.get(
        "READTHEDOCS_CANONICAL_URL",
        "https://bloombergfetch.readthedocs.io/en/latest/",
    )
)
html_show_sourcelink = True

autodoc_member_order = "bysource"
autodoc_mock_imports = ["blpapi"]
autodoc_typehints = "description"


def _use_root_canonical(app, pagename, templatename, context, doctree) -> None:
    """Use the HTTPS site root, rather than index.html, as the landing canonical."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("html-page-context", _use_root_canonical)


linkcheck_ignore = [
    r"http://localhost:8194/.*",
]
linkcheck_retries = 2
linkcheck_timeout = 15
