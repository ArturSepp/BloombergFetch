"""Sphinx configuration for the bbg-fetch documentation."""

import os
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin
from xml.etree import ElementTree

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
html_title = f"bbg-fetch {release} - Bloomberg data in pandas DataFrames"


# Each channel describes its own source revision and keeps its canonical URL.
html_baseurl = (
    os.environ.get('READTHEDOCS_CANONICAL_URL', 'https://bloombergfetch.readthedocs.io/en/latest/')
)
html_show_sourcelink = True

autodoc_member_order = "bysource"
autodoc_mock_imports = ["blpapi"]
autodoc_typehints = "description"


def _use_root_canonical(app, pagename, templatename, context, doctree) -> None:
    """Use the HTTPS site root, rather than index.html, as the landing canonical."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl



class _CanonicalHead(HTMLParser):
    """Read canonical and indexing metadata from the HTML actually being published."""

    def __init__(self):
        super().__init__()
        self.canonicals = []
        self.noindex = False
        self.redirect = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonicals.append(attributes.get("href"))
        if tag == "meta" and attributes.get("name", "").lower() == "robots":
            self.noindex |= "noindex" in attributes.get("content", "").lower()
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            self.redirect = True


def _write_canonical_sitemap(app, exception) -> None:
    """Offer only rendered, self-canonical pages from this build's own channel."""
    if exception is not None or app.builder.name != "html" or not app.config.html_baseurl:
        return
    base = app.config.html_baseurl.rstrip("/") + "/"
    urls = set()
    for docname in app.env.found_docs:
        head = _CanonicalHead()
        file = Path(app.builder.get_outfilename(docname))
        head.feed(file.read_text(encoding="utf-8").split("</head>")[0])
        if head.noindex or head.redirect or len(head.canonicals) != 1:
            continue
        uri = app.builder.get_target_uri(docname)
        expected = urljoin(base, quote("" if uri == "index.html" else uri, safe="/%"))
        if head.canonicals == [expected]:
            urls.add(expected)
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    root = ElementTree.Element(f"{{{namespace}}}urlset")
    for url in sorted(urls):
        entry = ElementTree.SubElement(root, f"{{{namespace}}}url")
        ElementTree.SubElement(entry, f"{{{namespace}}}loc").text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / "sitemap.xml", encoding="utf-8", xml_declaration=True,
        default_namespace=namespace,
    )


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("html-page-context", _use_root_canonical)
    app.connect("build-finished", _write_canonical_sitemap)


linkcheck_ignore = [
    r"http://localhost:8194/.*",
]
linkcheck_retries = 2
linkcheck_timeout = 15
