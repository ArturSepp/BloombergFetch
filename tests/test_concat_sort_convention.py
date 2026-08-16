"""
every axis=1 ``pd.concat`` in library code states ``sort=`` explicitly.

``pd.concat(objs, axis=1)`` joins the frames on their index, and whether the resulting union is
sorted has been changing under us. pandas 2.2 sorted the union of DatetimeIndexes whatever
``sort=`` said; pandas 3.0 honours an explicit ``sort=False`` and leaves the union in appearance
order; pandas 3.0 still sorts when no ``sort=`` is passed, under a ``Pandas4Warning`` announcing
that pandas 4 will not. A call that says nothing therefore means one thing today and another
after the next major release.

This package is where it would hurt most. Every series here arrives from a separate Bloomberg
request with its own history, so the joins in ``fetch_vol_timeseries_per_ticker`` and
``fetch_field_timeseries_per_ticker`` are exactly the union-of-differing-DatetimeIndexes case.
An unsorted price panel leaving this leaf reaches every package downstream, where the symptom is
a resample that raises deep inside pandas, or a forward fill that carries a later price onto an
earlier date and raises nothing.

So every such call states what it wants: ``sort=True`` where the joined index is dates, which is
what pandas 2.2 did, and ``sort=False`` where it is a label index, which pandas has never sorted.

Only ``axis=1`` is covered: an ``axis=0`` concat joins on the columns, which here are field
names rather than dates.

The check is static - it reads the source with ``ast`` and imports nothing - so it runs in CI
without a Bloomberg terminal, like the rest of ``tests/``.

To confirm it can fail, drop ``sort=True`` from either concat in ``core.py``: the call site is
reported below by file, line and the object being concatenated. That was run before this file
was committed.
"""
# packages
import ast
from pathlib import Path
from typing import List, Optional, Tuple
import pytest

# directories that hold scripts rather than library code
EXCLUDED_PARTS: Tuple[str, ...] = ('examples', 'tests', 'notebooks', '_to_delete')


def _package_root() -> Optional[Path]:
    """return the bbg_fetch package directory, or None when running off an installed wheel"""
    for parent in Path(__file__).resolve().parents:
        candidate = parent.joinpath('src', 'bbg_fetch')
        if candidate.is_dir():
            return candidate
    return None


ROOT = _package_root()


def _is_pd_concat(node: ast.Call) -> bool:
    """True for a ``pd.concat(...)`` call node"""
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == 'concat'
            and isinstance(func.value, ast.Name) and func.value.id == 'pd')


def find_implicit_sort_sites(root: Path) -> List[str]:
    """Return one line per axis=1 pd.concat call in library code that omits sort=."""
    offenders = []
    for path in sorted(root.rglob('*.py')):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_pd_concat(node):
                continue
            keywords = {kw.arg: kw for kw in node.keywords if kw.arg is not None}
            axis = keywords.get('axis')
            if axis is None or not isinstance(axis.value, ast.Constant):
                continue
            if axis.value.value not in (1, 'columns') or 'sort' in keywords:
                continue
            objs = ast.unparse(node.args[0]) if node.args else '<no positional objs>'
            rel = path.relative_to(root.parent).as_posix()
            offenders.append(f"{rel}:{node.lineno}: pd.concat({objs[:60]}, axis=1) omits sort=")
    return offenders


@pytest.mark.skipif(ROOT is None, reason='package source not on disk (installed wheel)')
def test_axis1_concat_states_sort() -> None:
    """a concat that does not say whether it sorts means different things in pandas 3 and 4"""
    offenders = find_implicit_sort_sites(ROOT)
    assert not offenders, (
            "axis=1 pd.concat without an explicit sort=; pass sort=True when the index is dates, "
            "sort=False when it is labels:\n" + '\n'.join(offenders))


if __name__ == '__main__':
    for offender in find_implicit_sort_sites(_package_root()):
        print(offender)
