"""Repository-only packaging layout checks."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_import_package_uses_src_layout() -> None:
    """Keep the import package isolated from the repository root."""
    package_root = REPOSITORY_ROOT / "src" / "bbg_fetch"

    assert (package_root / "__init__.py").is_file()
    assert not (REPOSITORY_ROOT / "bbg_fetch").exists()


def test_examples_stay_at_repository_root() -> None:
    """Keep runnable examples out of the installed import package."""
    examples_root = REPOSITORY_ROOT / "examples"

    assert (examples_root / "README.md").is_file()
    assert (examples_root / "quickstart_no_terminal.py").is_file()
    assert not (REPOSITORY_ROOT / "src" / "bbg_fetch" / "examples").exists()
