"""Vendored OPES post-processing scripts (see the script headers for provenance).

These are standalone command-line tools, not importable APIs: each parses its
arguments at module scope, so importing one would try to read ``sys.argv``.
:func:`~reactiontools.tools_plumed.run_opes_fes` runs ``FES_from_State.py`` as
a subprocess, which is the only way they are meant to be used from Python.

Making this directory a package is what ensures the scripts ship in wheels, and
gives :func:`script_path` somewhere to resolve them from.
"""

from pathlib import Path

__all__ = ["script_path"]

_HERE = Path(__file__).resolve().parent


def script_path(name="FES_from_State.py"):
    """Filesystem path of a bundled OPES post-processing script.

    Uses ``__file__`` rather than :mod:`importlib.resources`, which hands back
    a ``Traversable`` that only becomes a real path inside an ``as_file``
    block -- no use here, where the path goes into a command that outlives the
    call. The package is installed from a directory, never a zip, so
    ``__file__`` is always a real location.

    Parameters
    ----------
    name : str, optional
        File name of the script.

    Returns
    -------
    pathlib.Path
        Absolute path to the script.

    Raises
    ------
    FileNotFoundError
        If no such script is bundled, listing the ones that are.
    """
    path = _HERE / name
    if not path.is_file():
        available = sorted(p.name for p in _HERE.glob("*.py") if p.name != "__init__.py")
        raise FileNotFoundError(
            f"No OPES script named {name!r} in {_HERE}. Available: {available}")
    return path
