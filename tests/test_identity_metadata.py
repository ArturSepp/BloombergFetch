"""Repository-owned package identity and release-metadata checks."""

from pathlib import Path
from datetime import date
import re
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DESCRIPTION = (
    "Bloomberg Desktop API request/response data in pandas DataFrames for quantitative research"
)
DOCUMENTATION_URL = "https://bloombergfetch.readthedocs.io"


def _read(relative_path: str) -> str:
    """Read one repository text file as UTF-8."""
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_canonical_identity_is_consistent() -> None:
    """Keep source metadata and the README on one factual package identity."""
    pyproject = _read("pyproject.toml")
    readme = _read("README.md")
    normalized_readme = " ".join(readme.replace("`", "").split())

    assert f'description = "{CANONICAL_DESCRIPTION}"' in pyproject
    assert CANONICAL_DESCRIPTION in normalized_readme
    assert "production-ready" not in readme.lower()
    assert "40–60 lines" not in readme
    assert "Same result. One line." not in readme
    assert f'Documentation = "{DOCUMENTATION_URL}"' in pyproject
    assert DOCUMENTATION_URL in readme


def test_supported_python_and_platform_are_consistent() -> None:
    """Keep support prose aligned with the package metadata."""
    pyproject = _read("pyproject.toml")
    agents = _read("AGENTS.md")
    readme = _read("README.md")

    assert 'requires-python = ">=3.10"' in pyproject
    assert '"Programming Language :: Python :: 3.9"' not in pyproject
    assert '"Operating System :: OS Independent"' not in pyproject
    assert '"Operating System :: Microsoft :: Windows"' in pyproject
    assert "Supported Python is >= 3.10" in agents
    assert "Python 3.10+" in readme


def test_release_metadata_and_first_snippet_are_current() -> None:
    """Keep citations current and the opening README example self-contained."""
    pyproject = _read("pyproject.toml")
    package_init = _read("src/bbg_fetch/__init__.py")
    citation = _read("CITATION.cff")
    readme = _read("README.md")
    changelog = _read("CHANGELOG.md")

    version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE).group(1)
    citation_version = re.search(r'^version:\s*([^\n]+)', citation, re.MULTILINE)
    release_date = re.search(r'^date-released:\s*([^\n]+)', citation, re.MULTILINE)
    assert citation_version.group(1).strip('"\'') == version
    release_date = release_date.group(1).strip('"\'')
    assert date.fromisoformat(release_date) <= date.today()
    assert f'__version__ = "{version}"' in package_init
    assert f"version = {{{version}}}" in readme
    assert changelog.index("## [Unreleased]") < changelog.index(
        f"## [{version}] - {release_date}"
    )
    assert readme.index("import pandas as pd") < readme.index("pd.Timestamp")


def test_documentation_version_follows_package_metadata() -> None:
    """Execute the documentation configuration rather than pinning a release literal."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    project = tomllib.loads(_read("pyproject.toml"))["project"]
    sphinx = runpy.run_path(str(REPOSITORY_ROOT / "docs/conf.py"))
    assert sphinx["release"] == project["version"]
    assert sphinx["version"] == ".".join(project["version"].split(".")[:2])
