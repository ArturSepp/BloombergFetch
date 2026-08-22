"""Contracts separating pytest modules, development runners, and examples."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "bbg_fetch"
TESTS_ROOT = REPOSITORY_ROOT / "tests"
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
EXPECTED_LIVE_TESTS = {"tests/bbg_adj_price_vs_tri_test.py"}
EXPECTED_RUNNERS = {
    "run_local/adj_price_vs_tri_run.py",
    "run_local/core_run.py",
}
LEGACY_DISPATCHERS = {
    "LocalTest",
    "LocalTests",
    "UnitTests",
    "local_test",
    "run_local_test",
    "run_unit_test",
    "unit_test",
}


def _tree(path: Path) -> ast.Module:
    """Parse one Python module."""
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    """Return top-level class and function names."""
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_test_candidate(path: Path) -> bool:
    """Return whether a module defines a pytest-collectable test."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_tree(path))
    )


def _is_main_guard(node: ast.AST) -> bool:
    """Return whether a node is an executable main guard."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _has_main_guard(path: Path) -> bool:
    """Return whether a module has a top-level executable main guard."""
    return any(_is_main_guard(node) for node in _tree(path).body)


def _main_calls_run_local_directly(path: Path) -> bool:
    """Return whether the sole main statement directly calls ``run_local``."""
    guards = [node for node in _tree(path).body if _is_main_guard(node)]
    if len(guards) != 1 or len(guards[0].body) != 1:
        return False
    statement = guards[0].body[0]
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "run_local"
    ):
        return False
    return any(
        keyword.arg == "local"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "Locals"
        for keyword in statement.value.keywords
    )


def _imports_run_local(path: Path) -> bool:
    """Return whether production code imports development-only code."""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            if any("run_local" in alias.name.split(".") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if "run_local" in parts or any(alias.name == "run_local" for alias in node.names):
                return True
    return False


def test_pytest_modules_use_their_automated_lanes() -> None:
    """Terminal-free and live tests remain pure pytest modules in their own lanes."""
    top_level_tests = sorted(
        path
        for path in TESTS_ROOT.glob("*.py")
        if path.name not in {"__init__.py", "conftest.py"}
    )
    live_tests = sorted(
        path
        for path in (PACKAGE_ROOT / "tests").glob("*.py")
        if path.name not in {"__init__.py", "conftest.py"}
    )
    failures = []
    for path in top_level_tests:
        if not path.name.startswith("test_"):
            failures.append(f"{path.name}: expected test_*.py")
        if not _has_test_candidate(path):
            failures.append(f"{path.name}: no pytest test candidate")
        if _has_main_guard(path):
            failures.append(f"{path.name}: pytest modules cannot be executable runners")
    for path in live_tests:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if not path.name.endswith("_test.py"):
            failures.append(f"{relative}: expected *_test.py")
        if not _has_test_candidate(path):
            failures.append(f"{relative}: no pytest test candidate")
        if _has_main_guard(path):
            failures.append(f"{relative}: pytest modules cannot be executable runners")

    actual_live_tests = {path.relative_to(PACKAGE_ROOT).as_posix() for path in live_tests}
    assert len(top_level_tests) >= 7, "the terminal-free suite unexpectedly disappeared"
    assert actual_live_tests == EXPECTED_LIVE_TESTS
    assert not failures, failures


def test_development_runner_layout() -> None:
    """Source-adjacent runners use one no-init execution contract."""
    python_modules = sorted(PACKAGE_ROOT.rglob("*.py"))
    run_local_modules = [
        path
        for path in python_modules
        if "run_local" in path.relative_to(PACKAGE_ROOT).parts
    ]
    runners = [path for path in run_local_modules if path.name.endswith("_run.py")]
    failures = []
    for path in runners:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        definitions = _definitions(path)
        if not {"Locals", "run_local"} <= definitions:
            failures.append(f"{relative}: expected Locals plus run_local")
        if LEGACY_DISPATCHERS & definitions:
            failures.append(f"{relative}: retains legacy dispatcher names")
        if not _main_calls_run_local_directly(path):
            failures.append(f"{relative}: main guard must contain only run_local(local=Locals.*)")
        if _has_test_candidate(path):
            failures.append(f"{relative}: contains pytest tests")

    actual_runners = {path.relative_to(PACKAGE_ROOT).as_posix() for path in runners}
    support_modules = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in run_local_modules
        if not path.name.endswith("_run.py")
    )
    assert actual_runners == EXPECTED_RUNNERS
    assert not failures, failures
    assert not support_modules, f"unexpected run_local support modules: {support_modules}"


def test_production_modules_do_not_own_development_dispatchers() -> None:
    """Production modules stay independent of source-only development runners."""
    failures = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(PACKAGE_ROOT).parts
        if {"run_local", "tests"}.intersection(relative_parts):
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if LEGACY_DISPATCHERS & _definitions(path):
            failures.append(f"{relative}: owns a legacy development dispatcher")
        if _has_main_guard(path):
            failures.append(f"{relative}: owns an executable development runner")
        if _imports_run_local(path):
            failures.append(f"{relative}: imports development-only run_local code")
    assert not failures, failures


def test_examples_use_the_shared_dispatcher_contract() -> None:
    """Broader examples expose the stack-wide local dispatcher names."""
    failures = []
    examples = sorted(EXAMPLES_ROOT.glob("*.py"))
    for path in examples:
        definitions = _definitions(path)
        if not {"Locals", "run_local"} <= definitions:
            failures.append(f"{path.name}: expected Locals plus run_local")
        if LEGACY_DISPATCHERS & definitions:
            failures.append(f"{path.name}: retains legacy dispatcher names")
        if not _main_calls_run_local_directly(path):
            failures.append(f"{path.name}: main guard must contain only run_local(local=Locals.*)")
    assert len(examples) == 5
    assert not failures, failures


def test_development_runners_are_excluded_from_distributions() -> None:
    """Setuptools and the source manifest exclude all development runners."""
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    package_excludes = {
        '"bbg_fetch.run_local*"',
        '"bbg_fetch.*.run_local*"',
        '"bbg_fetch.*.*.run_local*"',
        '"bbg_fetch.*.*.*.run_local*"',
    }
    manifest_prunes = {
        "prune src/bbg_fetch/run_local",
        "prune src/bbg_fetch/*/run_local",
        "prune src/bbg_fetch/*/*/run_local",
        "prune src/bbg_fetch/*/*/*/run_local",
    }
    assert all(pattern in pyproject for pattern in package_excludes)
    assert all(pattern in manifest for pattern in manifest_prunes)
