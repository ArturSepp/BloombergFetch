"""Repository checks for the Sphinx documentation foundation."""

import re
import textwrap
from pathlib import Path
from xml.etree import ElementTree

import bbg_fetch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def _read(relative_path: str) -> str:
    """Read one repository text file as UTF-8."""
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_required_documentation_pages_and_single_source_example_exist() -> None:
    """Keep the minimal user journey and root example linkage intact."""
    required = {
        "conf.py",
        "index.rst",
        "installation.rst",
        "first_success.rst",
        "task_install_connect_diagnose.rst",
        "task_request_data.rst",
        "task_research_workflows.rst",
        "api.rst",
        "troubleshooting.rst",
        "robots.txt",
        "sitemap.xml",
    }

    assert required <= {path.name for path in DOCS_ROOT.iterdir() if path.is_file()}
    assert "literalinclude:: ../examples/quickstart_no_terminal.py" in _read(
        "docs/first_success.rst"
    )
    assert 'html_baseurl = "https://artursepp.github.io/BloombergFetch/"' in _read(
        "docs/conf.py"
    )
    assert "docs = [" in _read("pyproject.toml")
    assert ":google-site-verification:" in _read("docs/index.rst")


def test_sitemap_covers_the_canonical_priority_pages() -> None:
    """Keep the small static sitemap aligned with the user-facing docs surface."""
    namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ElementTree.fromstring(_read("docs/sitemap.xml"))
    urls = {node.text for node in root.findall("sitemap:url/sitemap:loc", namespace)}
    base_url = "https://artursepp.github.io/BloombergFetch/"

    assert urls == {
        base_url,
        f"{base_url}installation.html",
        f"{base_url}first_success.html",
        f"{base_url}task_install_connect_diagnose.html",
        f"{base_url}task_request_data.html",
        f"{base_url}task_research_workflows.html",
        f"{base_url}api.html",
        f"{base_url}troubleshooting.html",
    }
    assert f"Sitemap: {base_url}sitemap.xml" in _read("docs/robots.txt")


def test_api_inventory_matches_the_observable_top_level_surface() -> None:
    """Fail if a public top-level name appears or disappears without docs review."""
    api_reference = _read("docs/api.rst")
    inventory_match = re.search(
        r":class: export-inventory\n\n(?P<inventory>(?:   [A-Za-z]\w*\n)+)",
        api_reference,
    )
    assert inventory_match is not None

    documented = {
        line.strip() for line in inventory_match.group("inventory").splitlines() if line.strip()
    }
    observable = {name for name in vars(bbg_fetch) if not name.startswith("_")}

    assert observable == documented
    assert ".. automodule:: bbg_fetch" in api_reference
    assert ":imported-members:" in api_reference


def test_task_guides_only_name_real_top_level_symbols() -> None:
    """Reject task-guide examples that invent or bypass the public surface."""
    task_guides = "\n".join(
        _read(f"docs/{name}.rst")
        for name in (
            "task_install_connect_diagnose",
            "task_request_data",
            "task_research_workflows",
        )
    )
    named = set(re.findall(r"bbg_fetch\.([A-Za-z]\w*)", task_guides))

    assert named
    assert named <= {name for name in vars(bbg_fetch) if not name.startswith("_")}


def test_task_guide_python_blocks_compile() -> None:
    """Compile every live guide snippet without opening a Bloomberg session."""
    pattern = re.compile(
        r"\.\. code-block:: python\n(?:   :[^\n]+\n)*\n"
        r"(?P<body>(?:(?:   .*|)\n)+?)(?=\n?\S|\Z)"
    )
    compiled = 0
    for name in (
        "task_install_connect_diagnose",
        "task_request_data",
        "task_research_workflows",
    ):
        source = _read(f"docs/{name}.rst")
        for match in pattern.finditer(source):
            compile(textwrap.dedent(match.group("body")), f"docs/{name}.rst", "exec")
            compiled += 1

    assert compiled == 7


def test_task_guides_are_in_the_primary_navigation() -> None:
    """Keep all three priority tasks reachable from the landing page."""
    index = _read("docs/index.rst")
    for page in (
        "task_install_connect_diagnose",
        "task_request_data",
        "task_research_workflows",
    ):
        assert f"   {page}" in index


def test_ci_builds_and_link_checks_the_documentation() -> None:
    """Keep warning and link checks in the pull-request gate."""
    workflow = _read(".github/workflows/ci.yml")

    assert "-b html docs docs/_build/html" in workflow
    assert "-b linkcheck docs docs/_build/linkcheck" in workflow
