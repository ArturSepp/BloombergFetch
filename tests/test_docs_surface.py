"""Repository checks for the Sphinx documentation foundation."""

import json
import re
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import bbg_fetch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
CANONICAL_DOCS_URL = "https://bloombergfetch.readthedocs.io/en/latest/"


def _read(relative_path: str) -> str:
    """Read one repository text file as UTF-8."""
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_required_documentation_pages_and_single_source_example_exist(monkeypatch) -> None:
    """Keep the minimal user journey and root example linkage intact."""
    required = {
        "conf.py",
        "index.rst",
        "installation.rst",
        "first_success.rst",
        "task_install_connect_diagnose.rst",
        "task_request_data.rst",
        "task_research_workflows.rst",
        "comparison.rst",
        "api.rst",
        "troubleshooting.rst",
    }

    assert required <= {path.name for path in DOCS_ROOT.iterdir() if path.is_file()}
    assert "literalinclude:: ../examples/quickstart_no_terminal.py" in _read(
        "docs/first_success.rst"
    )
    install_guide = _read("docs/task_install_connect_diagnose.rst")
    assert "python examples/diagnose_terminal.py" in install_guide
    assert "bbg_fetch.bdp(" not in install_guide
    readme = _read("README.md")
    for example in ("quickstart_no_terminal.py", "diagnose_terminal.py"):
        assert f"examples/{example}" in readme
    monkeypatch.delenv("READTHEDOCS_CANONICAL_URL", raising=False)
    assert runpy.run_path(str(DOCS_ROOT / "conf.py"))["html_baseurl"] == CANONICAL_DOCS_URL
    workflow = _read(".github/workflows/ci.yml")
    assert "python -m compileall -q examples" in workflow
    assert "working-directory: ${{ runner.temp }}" in workflow
    assert 'python "$GITHUB_WORKSPACE/examples/quickstart_no_terminal.py"' in workflow
    assert "docs = [" in _read("pyproject.toml")
    assert ":google-site-verification:" in _read("docs/index.rst")


def test_legacy_redirects_cover_the_canonical_priority_pages(tmp_path) -> None:
    """Keep every priority page reachable after moving discovery to Read the Docs."""
    source, output = tmp_path / "rendered", tmp_path / "redirects"
    source.mkdir()
    pages = {
        "index",
        "installation",
        "first_success",
        "task_install_connect_diagnose",
        "task_request_data",
        "task_research_workflows",
        "comparison",
        "api",
        "troubleshooting",
    }
    for page in pages:
        assert (DOCS_ROOT / f"{page}.rst").is_file()
        (source / f"{page}.html").write_text("Original page content", encoding="utf-8")
    redirect = runpy.run_path(str(REPOSITORY_ROOT / ".github/scripts/build_docs_redirects.py"))
    assert redirect["build_redirects"](
        source, output, CANONICAL_DOCS_URL, "/BloombergFetch/"
    ) == len(pages)
    for page in pages:
        target = CANONICAL_DOCS_URL if page == "index" else f"{CANONICAL_DOCS_URL}{page}.html"
        document = (output / f"{page}.html").read_text(encoding="utf-8")
        assert f'href="{target}"' in document
        assert "noindex,follow" in document
        assert "Original page content" not in document
    assert "path: docs/_build/redirects" in _read(".github/workflows/docs.yml")


@pytest.mark.parametrize(
    ("service_url", "canonical_url"),
    [
        # stable and latest serve the same pages, so both name latest as canonical
        ("https://bloombergfetch.readthedocs.io/en/stable/", CANONICAL_DOCS_URL),
        (
            "https://bloombergfetch.readthedocs.io/en/3.2.0/",
            "https://bloombergfetch.readthedocs.io/en/3.2.0/",
        ),
    ],
)
def test_sphinx_uses_readthedocs_canonical_override(monkeypatch, service_url, canonical_url) -> None:
    """Use RTD's version URL, with stable folded into latest, and collapse index.html to the root."""
    monkeypatch.setenv("READTHEDOCS_CANONICAL_URL", service_url)
    config = runpy.run_path(str(DOCS_ROOT / "conf.py"))
    context = {"pageurl": f"{canonical_url}index.html"}

    config["_use_root_canonical"](
        SimpleNamespace(config=SimpleNamespace(html_baseurl=config["html_baseurl"])),
        "index",
        "page.html",
        context,
        None,
    )

    assert config["html_baseurl"] == canonical_url
    assert context["pageurl"] == canonical_url


def test_built_pages_carry_short_titles(monkeypatch, tmp_path) -> None:
    """Furo would end every page title with the full html_title, which search results cut off."""
    for module in ("sphinx", "furo"):
        pytest.importorskip(module)
    monkeypatch.delenv("READTHEDOCS_CANONICAL_URL", raising=False)
    config = runpy.run_path(str(DOCS_ROOT / "conf.py"))
    templates = [str(DOCS_ROOT / path) for path in config.get("templates_path", [])]
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        "import runpy\n"
        f"_site = runpy.run_path({str(DOCS_ROOT / 'conf.py')!r})\n"
        "html_theme = 'furo'\n"
        f"templates_path = {templates!r}\n"
        "for _key in ('project', 'html_title', 'html_baseurl'):\n"
        "    globals()[_key] = _site[_key]\n"
        "setup = _site['setup']\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "Home\n====\n\n.. toctree::\n\n   troubleshooting\n", encoding="utf-8"
    )
    (source / "troubleshooting.rst").write_text(
        "Troubleshooting\n===============\n\nText.\n", encoding="utf-8"
    )
    output = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", "-b", "html", str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    def titles(name: str) -> list:
        """Return the title elements of a built page's head."""
        head = (output / f"{name}.html").read_text(encoding="utf-8").split("</head>")[0]
        return re.findall(r"<title>(.*?)</title>", head)

    assert titles("index") == [config["html_title"]]
    assert titles("troubleshooting") == ["Troubleshooting - bbg-fetch"]


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

    assert compiled == 6


def test_task_guides_are_in_the_primary_navigation() -> None:
    """Keep all three priority tasks reachable from the landing page."""
    index = _read("docs/index.rst")
    for page in (
        "task_install_connect_diagnose",
        "task_request_data",
        "task_research_workflows",
    ):
        assert f"   {page}" in index


def test_comparison_is_dated_neutral_and_primary_sourced() -> None:
    """Keep the choice guide auditable and prevent an unqualified winner claim."""
    comparison = _read("docs/comparison.rst")

    assert "Audit date: 2026-08-16" in comparison
    for version in ("bbg-fetch 3.0.0", "blpapi 3.26.7.1", "xbbg 1.4.6", "blp 0.0.4"):
        assert version in comparison
    for primary_source in (
        "https://blpapi.bloomberg.com/repository/releases/python/simple/blpapi/",
        "https://bloomberg.github.io/blpapi-docs/",
        "https://github.com/xbbg-org/xbbg",
        "https://pypi.org/project/xbbg/1.4.6/",
        "https://github.com/matthewgilbert/blp",
        "https://pypi.org/project/blp/0.0.4/",
        "https://github.com/matthewgilbert/pdblp",
    ):
        assert primary_source in comparison

    assert "No universal recommendation" in comparison
    assert "no longer under active development" in comparison
    assert "popularity" not in comparison.lower()
    assert "   comparison" in _read("docs/index.rst")


def test_docs_workflow_builds_and_link_checks_the_documentation() -> None:
    """Keep strict shared HTML builds and the separate link-health checks."""
    workflow = _read(".github/workflows/docs.yml")

    profile = json.loads(_read(".github/oss-checks.json"))
    assert [
        "-m",
        "sphinx",
        "-E",
        "-W",
        "--keep-going",
        "-b",
        "html",
        "docs",
        "{output}/html",
    ] in profile["docs"]
    assert "python .github/oss_checks.py docs --working-tree" in workflow
    assert "-b linkcheck docs docs/_build/linkcheck" in workflow
