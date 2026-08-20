"""Sphinx configuration for the bbg-fetch documentation."""

import os

project = "bbg-fetch"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
version = "3.0"
release = "3.0.0"

extensions = [
    "sphinx.ext.autodoc",
]

root_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_title = "bbg-fetch documentation"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://bloombergfetch.readthedocs.io/en/latest/",
)
html_show_sourcelink = True
html_extra_path = ["robots.txt", "sitemap.xml"]

autodoc_member_order = "bysource"
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
