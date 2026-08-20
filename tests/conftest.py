"""Shared fixtures for the reactiontools test suite.

The tests are self-contained: structures are built with ``ase.build`` and
evaluated with EMT, so no external data files or calculators are needed.
"""

import matplotlib

# Must run before reactiontools imports pyplot, or the tests need a display.
matplotlib.use("Agg")

from collections.abc import Callable, Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from ase.io import write  # noqa: E402


@pytest.fixture(autouse=True)
def _work_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in a scratch directory.

    Several functions write trajectories and figures relative to the working
    directory, so this keeps the repository clean.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close any figures a test leaves behind."""
    yield
    plt.close("all")


@pytest.fixture
def calc() -> EMT:
    """An EMT calculator, cheap enough to run in a test."""
    return EMT()


@pytest.fixture
def water() -> Atoms:
    """A single water molecule."""
    return molecule("H2O")


def _make_chain(n_images: int, dz: float = 0.5) -> list[Atoms]:
    """Build a path of single atoms displaced by ``dz`` along z.

    One atom per image makes the path maths exact: the distance between
    consecutive images is exactly ``dz``, so ``get_neb_path`` should return
    ``[0, dz, 2 * dz, ...]``.

    Parameters
    ----------
    n_images : int
        Number of images in the chain.
    dz : float, optional
        Displacement between consecutive images.

    Returns
    -------
    list of ase.Atoms
        The chain.
    """
    return [Atoms("H", positions=[[0.0, 0.0, i * dz]]) for i in range(n_images)]


@pytest.fixture
def make_chain() -> Callable[..., list[Atoms]]:
    """Factory for chains of single atoms, for tests needing several."""
    return _make_chain


@pytest.fixture
def chain() -> list[Atoms]:
    """A five-image chain spaced 0.5 Å apart."""
    return _make_chain(5)


@pytest.fixture
def md_trajectory(tmp_path: Path) -> Path:
    """Write a short MD-like trajectory and return its path.

    Frames carry both momenta and EMT energies, so the trajectory supports
    ``get_temperature`` and ``get_total_energy`` when read back.
    """
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(5):
        atoms = molecule("H2O")
        masses = atoms.get_masses()[:, None]
        atoms.set_momenta(rng.normal(scale=0.01, size=(len(atoms), 3)) * masses)
        atoms.calc = EMT()
        atoms.get_potential_energy()
        frames.append(atoms)

    path = tmp_path / "md.traj"
    write(path, frames)
    return path


# Donor oxygen, the proton it shares, and the acceptor oxygen, as indexed by
# the `pt_atoms` fixture below.
PT_DONOR, PT_HYDROGEN, PT_ACCEPTOR = 0, 1, 2


@pytest.fixture
def pt_atoms() -> Atoms:
    """A hydrogen-bonded triad with a carbon backbone, as ASE atoms.

    Malonaldehyde-like without being malonaldehyde: the proton sits on the
    donor oxygen and points at the acceptor, which is all the path and
    collective-variable tests need. Built rather than committed, so there is
    no data file to keep in step with the tests.
    """
    return Atoms(
        "OHOCCC",
        positions=[
            [0.00, 0.00, 0.00],  # donor O
            [0.98, 0.00, 0.00],  # the shared proton
            [2.65, 0.00, 0.00],  # acceptor O
            [-0.65, 1.18, 0.00],  # backbone
            [0.10, 2.40, 0.00],
            [1.55, 2.35, 0.00],
        ],
    )


@pytest.fixture
def pt_pdb(pt_atoms: Atoms, tmp_path: Path) -> Path:
    """The `pt_atoms` triad written as a PDB, and its path.

    ASE writes ``ATOM`` records, so anything reading this back wants
    ``atom_line="ATOM"`` rather than the ``HETATM`` OpenMM produces.
    """
    path = tmp_path / "index_atoms.pdb"
    write(path, pt_atoms, format="proteindatabank")
    return path


@pytest.fixture
def fes_file(tmp_path: Path) -> Path:
    """Write a minimal PLUMED ``fes.dat`` and return its path.

    Values are in eV, as ``plumed sum_hills`` writes them, so the readers
    should scale them by 1000 to reach meV.
    """
    path = tmp_path / "fes.dat"
    cv = np.linspace(-1.0, 1.0, 21)
    fes = cv**2  # a simple parabola, minimum of 0 eV at cv = 0
    np.savetxt(path, np.column_stack([cv, fes]))
    return path
