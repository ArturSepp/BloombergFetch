"""Repository-owned package identity and release-metadata checks."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DESCRIPTION = (
    "Bloomberg Desktop API request/response data in pandas DataFrames for quantitative research"
)


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
    citation = _read("CITATION.cff")
    readme = _read("README.md")

    assert "version: 2.3.0" in citation
    assert 'date-released: "2026-07-24"' in citation
    assert "version = {2.3.0}" in readme
    assert readme.index("import pandas as pd") < readme.index("pd.Timestamp")
