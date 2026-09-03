"""Package-level tests: the public surface and the layering beneath it."""

import ast
import inspect
from pathlib import Path

import pytest
from ase import Atoms

import reactiontools

_PACKAGE = Path(reactiontools.__file__).resolve().parent

#: Modules that must stay free of matplotlib. They build strings and crunch
#: numbers for callers who may never draw anything, and ``tools_fes`` -- the
#: obvious place to reach for the unit conversions -- imports pyplot at module
#: scope. Keeping the edge out is what lets ``tools_style`` and
#: ``tools_units`` exist as separate modules at all.
#:
#: ``tools_path`` is deliberately not in this set: it reads COLVAR files
#: through ``read_plumed_file``, so it depends on ``tools_fes`` by rights, and
#: it needs mdtraj anyway -- a heavier import than matplotlib.
_PLOT_FREE = {"tools_units", "tools_plumed", "tools_cv", "tools_io"}


def _source_of(name: str) -> Path | None:
    """Path of a sibling module, whether it is a file or a package.

    Parameters
    ----------
    name : str
        Module name within the package, without the ``reactiontools.`` prefix.

    Returns
    -------
    pathlib.Path or None
        Its source file, or None if there is no such module.
    """
    for candidate in (_PACKAGE / f"{name}.py", _PACKAGE / name / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_imports(name: str) -> tuple[set[str], set[str]]:
    """Intra-package and third-party top-level imports of one module.

    Read off the source rather than from a live import, so that a module can
    be checked without the package ``__init__`` -- which imports everything --
    having already pulled its dependencies in.

    Parameters
    ----------
    name : str
        Module name within the package, without the ``reactiontools.`` prefix.

    Returns
    -------
    local : set of str
        Names of sibling modules imported with a relative import.
    external : set of str
        Top-level names of everything else imported.
    """
    source = _source_of(name)
    if source is None:
        return set(), set()
    tree = ast.parse(source.read_text())
    local, external = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module:
                    local.add(node.module.split(".")[0])
            elif node.module:
                external.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                external.add(alias.name.split(".")[0])
    return local, external


def _reachable(name: str, seen: set[str] | None = None) -> tuple[set[str], set[str]]:
    """Every sibling module reachable from *name*, and every external import."""
    seen = set() if seen is None else seen
    if name in seen:
        return seen, set()
    seen.add(name)
    local, external = _module_imports(name)
    for sibling in local:
        seen, more = _reachable(sibling, seen)
        external |= more
    return seen, external


def _package_modules() -> list[str]:
    return sorted(path.stem for path in _PACKAGE.glob("tools_*.py"))


@pytest.mark.parametrize("name", _package_modules())
def test_no_module_imports_itself_in_a_cycle(name: str) -> None:
    """The intra-package import graph must stay acyclic.

    A cycle here does not fail at import time in every order, only in some, so
    it shows up as an ImportError that depends on which module the user
    happened to import first.
    """
    local, _ = _module_imports(name)

    for sibling in local:
        reachable, _ = _reachable(sibling)
        assert name not in reachable, f"{name} <-> {sibling} import cycle"


@pytest.mark.parametrize("name", sorted(_PLOT_FREE))
def test_the_plot_free_modules_stay_plot_free(name: str) -> None:
    if _source_of(name) is None:
        pytest.skip(f"{name} does not exist yet")

    _, external = _reachable(name)

    assert "matplotlib" not in external


def test_version_is_a_string() -> None:
    assert isinstance(reactiontools.__version__, str)
    assert reactiontools.__version__.count(".") == 2


def test_all_names_are_importable() -> None:
    """Everything advertised in __all__ must actually be re-exported."""
    missing = [
        name for name in reactiontools.__all__ if not hasattr(reactiontools, name)
    ]
    assert not missing


def test_all_names_are_unique() -> None:
    """Duplicate exports are easy to introduce in the grouped API list."""
    assert len(reactiontools.__all__) == len(set(reactiontools.__all__))


def test_public_api_is_complete() -> None:
    """The documented API must all be reachable from the top-level package."""
    expected = {
        # tools_reaction
        "ConvergenceError",
        "ConvergenceWarning",
        "NebSummary",
        "summarise_neb",
        "restart_neb",
        "restart_parallel_neb",
        "get_neb_path",
        "get_fmax",
        "stitch_path",
        "resample_path",
        "optimise_geom",
        "optimise_reactant_product",
        "prepare_neb",
        "socket_calculators",
        "prepare_parallel_neb",
        "prepare_threaded_neb",
        "optimise_neb",
        "get_ts_image",
        "optimise_ts",
        "optimise_irc",
        "get_vibrations",
        "quick_guess_path",
        "quick_guess_ts",
        # tools_geometry
        "SeedSummary",
        "SeedWarning",
        "seed_minima_from_ts",
        "seed_product_from_ts",
        "bonded_cluster_indices_no_anchor_hub",
        "get_dimer_bonded_cluster_indices",
        "flip_and_face_bases",
        "optimize_with_fixed_anchors",
        "get_best_flip_and_face_bases",
        # tools_plumed
        "plumed_selection",
        "find_molecules",
        "run_sum_hills",
        # tools_plotting
        "n_plot",
        "ax_plot",
        "show_atoms",
        "plot_images",
        "plot_neb",
        "plot_irc",
        "plot_temperature",
        "plot_total_energy",
        "plot_plumed",
        "plot_plumed_multi",
    }
    assert expected <= set(reactiontools.__all__)


def _constant_is_documented(name: str) -> bool:
    """Whether autodoc would find a description for a module-level constant.

    ``__doc__`` is no use here: a dict or str constant inherits its type's
    docstring, so every one of them would look documented. Sphinx instead
    reads a ``#:`` comment block above the assignment, a trailing ``#:``
    comment, or a string literal directly after it -- which is what this looks
    for, in whichever module defines *name*.

    Parameters
    ----------
    name : str
        Name of the constant, as re-exported from the package root.

    Returns
    -------
    bool
        True if a description would be picked up, or if *name* is not a
        module-level assignment anywhere in the package.
    """
    for source in sorted(_PACKAGE.glob("*.py")) + [_PACKAGE / "opes" / "__init__.py"]:
        lines = source.read_text().splitlines()
        tree = ast.parse("\n".join(lines))
        for i, node in enumerate(tree.body):
            targets = (
                [node.target] if isinstance(node, ast.AnnAssign) else
                getattr(node, "targets", [])
            )
            if not any(getattr(t, "id", None) == name for t in targets):
                continue
            above = lines[node.lineno - 2].strip() if node.lineno > 1 else ""
            trailing = lines[node.lineno - 1]
            following = tree.body[i + 1] if i + 1 < len(tree.body) else None
            return (
                above.startswith("#:")
                or "  #:" in trailing
                or (
                    isinstance(following, ast.Expr)
                    and isinstance(following.value, ast.Constant)
                    and isinstance(following.value.value, str)
                )
            )
    return True


def test_every_public_name_is_documented() -> None:
    """__all__ and the rendered documentation must not drift apart.

    The API reference is generated from these docstrings by Sphinx autodoc, so
    a public name without one is a hole in the published documentation rather
    than merely an undocumented function. This used to check the README's
    hand-written API tables; those were deleted once the docstrings began
    generating the reference, and this is their successor.
    """
    missing = []
    for name in reactiontools.__all__:
        if name == "__version__":
            continue
        obj = getattr(reactiontools, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if not (obj.__doc__ or "").strip():
                missing.append(name)
        elif not _constant_is_documented(name):
            missing.append(name)
    assert not missing


def test_quick_guess_path_returns_the_requested_images(water: Atoms) -> None:
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]

    path = reactiontools.quick_guess_path(water, product, n_images=7)

    assert len(path) == 7


def test_quick_guess_ts_returns_a_midpoint_structure(water: Atoms) -> None:
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]

    ts = reactiontools.quick_guess_ts(water, product, n_images=7)

    assert len(ts) == len(water)
