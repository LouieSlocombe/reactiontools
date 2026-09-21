"""Tests for :mod:`reactiontools.tools_sella`.

The upstream Sella suite, brought across with the module it exercises and
consolidated into one file: its ``test_utils`` helpers and ``conftest``
fixtures are inlined at the top, and each upstream test module follows under a
banner of its own.

Two upstream test modules are not here. ``tests/utilities/test_math.py`` tested
the Cython that this package does not build; what survives of it is
:func:`test_modified_gram_schmidt` below, run against the NumPy
reimplementation, plus a check that it still matches the compiled original
whenever a copy of upstream sella happens to be importable.
``tests/test_utils/test_poly_factory.py`` is kept, as the helper it covers is
what the eigensolver and linear-operator tests are built on.
"""

import importlib.util
import math
import os
import subprocess
import sys
import warnings
from collections.abc import Callable, Sequence
from itertools import permutations, product
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.units import kB
from numpy.testing import assert_allclose
from scipy.linalg import polar
from scipy.stats import ortho_group

import reactiontools
from reactiontools import tools_sella
from reactiontools.tools_sella import (
    IRC,
    PES,
    ApproximateHessian,
    Bond,
    CellCartesianPES,
    CellInternalPES,
    Constraints,
    DuplicateConstraintError,
    InternalPES,
    Internals,
    NumericalHessian,
    Sella,
    SparseInternalHessians,
    exact,
    full_3x3_to_voigt_6_stress,
    modified_gram_schmidt,
    rayleigh_ritz,
    update_H,
    voigt_6_to_full_3x3_stress,
)

# ===========================================================================
# Helpers, inlined from the upstream tests/test_utils package
# ===========================================================================

def get_matrix(n, m, pd=False, symm=False, rng=None):
    """Generates a random n-by-m matrix"""
    if rng is None:
        rng = np.random.RandomState(1)
    A = rng.normal(size=(n, m))
    if symm:
        assert n == m
        A = 0.5 * (A + A.T)
    if pd:
        assert n == m
        lams, vecs = np.linalg.eigh(A)
        A = vecs @ (np.abs(lams)[:, np.newaxis] * vecs.T)
    return A


def poly_factory(dim, order, rng=None):
    """Generates a random multi-dimensional polynomial function."""
    if rng is None:
        rng = np.random.RandomState(1)

    coeffs = []
    for i in range(order + 1):
        tmp = rng.normal(size=(dim,) * i)
        coeff = np.zeros_like(tmp)
        for n, permute in enumerate(permutations(range(i))):
            coeff += np.transpose(tmp, permute)
        coeffs.append(coeff / ((n + 1) * math.factorial(i)))

    def poly(x):
        res = 0
        grad = np.zeros_like(x)
        hess = np.zeros((dim, dim))
        for i, coeff in enumerate(coeffs):
            lastlast = None
            last = None
            for j in range(i):
                lastlast = last
                last = coeff
                coeff = coeff @ x
            if last is not None:
                grad += i * last
            if lastlast is not None:
                hess += i * (i - 1) * lastlast
            res += coeff
        return res, grad, hess

    return poly

# ===========================================================================
# Fixtures, from the upstream tests/conftest.py
# ===========================================================================

@pytest.fixture
def cu_fcc():
    """Factory for EMT-backed FCC copper.

    Returns a fresh Atoms object on every call so a single test can build
    several independent copies (e.g. to compare two PES configurations).
    """
    def _make(a: float = 3.6, cubic: bool = False) -> Atoms:
        atoms = bulk('Cu', 'fcc', a=a, cubic=cubic)
        atoms.calc = EMT()
        return atoms
    return _make


@pytest.fixture
def two_water_crystal():
    """Factory for two well-separated water molecules in a periodic cell.

    ``cell`` accepts a scalar (cubic cell of that edge length) or a full 3x3
    array, which is what the triclinic and heavily-sheared gradient tests need.
    """
    def _make(cell: Union[float, Sequence] = 7.0) -> Atoms:
        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]
        atoms = water1 + water2
        if np.ndim(cell) == 0:
            cell = [cell, cell, cell]
        atoms.set_cell(cell)
        atoms.pbc = True
        atoms.calc = LennardJones()
        return atoms
    return _make


@pytest.fixture
def methane_box():
    """Factory for methane in a periodic box.

    CH4 has well-defined bonds and angles that behave reliably with internal
    coordinates, which is why the cell-derivative tests use it.
    """
    def _make(vacuum: float = 3.0) -> Atoms:
        mol = molecule('CH4')
        mol.center(vacuum=vacuum)
        mol.pbc = True
        return mol
    return _make


@pytest.fixture
def water_box():
    """Factory for a single water molecule in a periodic box."""
    def _make(vacuum: float = 3.0) -> Atoms:
        mol = molecule('H2O')
        mol.center(vacuum=vacuum)
        mol.pbc = True
        return mol
    return _make


@pytest.fixture
def fd_pes_gradient():
    """Factory for central-difference gradients of a PES with respect to ``x``.

    Perturbs one component of ``pes.get_x()`` at a time, restoring the
    reference state before each probe so errors cannot accumulate across
    components. The PES is left at its original ``x`` on return.

    Parameters
    ----------
    pes
        Any PES exposing ``get_x``/``set_x``/``eval`` and ``dim``.
    indices
        Components to differentiate. Defaults to every component.
    delta
        Step size, either a scalar or a callable mapping component index to a
        step size (internal and cell coordinates want different steps).
    """
    def _fd(
        pes,
        indices: Sequence[int] = None,
        delta: Union[float, Callable[[int], float]] = 1e-6,
    ) -> np.ndarray:
        if indices is None:
            indices = range(pes.dim)
        indices = list(indices)
        step = delta if callable(delta) else (lambda i: delta)

        x0 = pes.get_x()
        g_numeric = np.zeros(len(indices))

        for n, i in enumerate(indices):
            di = step(i)

            pes.set_x(x0)
            x_plus = x0.copy()
            x_plus[i] += di
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            pes.set_x(x0)
            x_minus = x0.copy()
            x_minus[i] -= di
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_numeric[n] = (e_plus - e_minus) / (2 * di)

        pes.set_x(x0)
        return g_numeric
    return _fd


@pytest.fixture
def fd_cell_gradient():
    """Factory for central-difference derivatives w.r.t. the 3x3 cell matrix.

    ``func`` is evaluated on ``atoms`` after each cell perturbation, so this
    works for any scalar-valued internal coordinate (``coord.calc``). Atoms are
    not rescaled with the cell, matching how the analytical cell gradients in
    ``sella.internal`` are defined. The original cell is restored on return.
    """
    def _fd(
        atoms: Atoms,
        func: Callable[[Atoms], float],
        delta: float = 1e-5,
    ) -> np.ndarray:
        cell0 = atoms.get_cell().array.copy()
        grad_numeric = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                cell_plus = cell0.copy()
                cell_plus[i, j] += delta
                atoms.set_cell(cell_plus, scale_atoms=False)
                val_plus = func(atoms)

                cell_minus = cell0.copy()
                cell_minus[i, j] -= delta
                atoms.set_cell(cell_minus, scale_atoms=False)
                val_minus = func(atoms)

                grad_numeric[i, j] = (val_plus - val_minus) / (2 * delta)

        atoms.set_cell(cell0, scale_atoms=False)
        return grad_numeric
    return _fd

# ===========================================================================
# Test helpers
# ===========================================================================

@pytest.mark.parametrize("dim,order", [(1, 1), (2, 2), (10, 5)])
def test_poly_factory(dim, order, eta=1e-6, atol=1e-4):
    rng = np.random.RandomState(1)

    tol = dict(atol=atol, rtol=eta**2)

    my_poly = poly_factory(dim, order)
    x0 = rng.normal(size=dim)
    f0, g0, h0 = my_poly(x0)

    g_numer = np.zeros_like(g0)
    h_numer = np.zeros_like(h0)
    for i in range(dim):
        x = x0.copy()
        x[i] += eta
        fplus, gplus, _ = my_poly(x)
        x[i] = x0[i] - eta
        fminus, gminus, _ = my_poly(x)
        g_numer[i] = (fplus - fminus) / (2 * eta)
        h_numer[i] = (gplus - gminus) / (2 * eta)

    np.testing.assert_allclose(g0, g_numer, **tol)
    np.testing.assert_allclose(h0, h_numer, **tol)

# ===========================================================================
# Hessian update schemes
# ===========================================================================

@pytest.mark.parametrize("dim,subdim,method,symm, pd",
                         [(10, 1, 'TS-BFGS', 2, False),
                          (10, 2, 'TS-BFGS', 0, False),
                          (10, 2, 'TS-BFGS', 1, False),
                          (10, 2, 'TS-BFGS', 2, False),
                          (10, 2, 'BFGS', 2, False),
                          (10, 2, 'PSB', 2, False),
                          (10, 2, 'DFP', 2, False),
                          (10, 2, 'SR1', 2, False),
                          (10, 2, 'Greenstadt', 2, False),
                          (10, 2, 'BFGS_auto', 2, False),
                          (10, 2, 'BFGS_auto', 2, True),
                          ])
def test_update_H(dim, subdim, method, symm, pd):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-6, rtol=1e-6)

    B = get_matrix(dim, dim, pd, True, rng=rng)
    H = get_matrix(dim, dim, pd, True, rng=rng)

    S = get_matrix(dim, subdim, rng=rng)
    Y = H @ S

    B1 = update_H(None, S, Y, method=method, symm=symm)
    np.testing.assert_allclose(B1 @ S, Y, **tol)

    B2 = update_H(B, S, Y, method=method, symm=symm)
    np.testing.assert_allclose(B2 @ S, Y, **tol)

    if subdim == 1:
        B3 = update_H(B, S.ravel(), Y.ravel(), method=method, symm=symm)
        np.testing.assert_allclose(B2, B3, **tol)

        B4 = update_H(B, S.ravel() / 1e12, Y.ravel() / 1e12, method=method,
                      symm=symm)
        np.testing.assert_allclose(B, B4, atol=0, rtol=0)

# ===========================================================================
# Linear operators
# ===========================================================================

@pytest.mark.parametrize("dim,subdim,order,threepoint",
                         [(3, None, 1, False),
                          (3, None, 1, True),
                          (5, 3, 2, True),
                          (10, None, 4, True),
                          (10, 6, 4, False)])
def test_NumericalHessian(dim, subdim, order, threepoint, eta=1e-6, atol=1e-4):
    rng = np.random.RandomState(2)
    tol = dict(rtol=atol, atol=eta**2)

    x = rng.normal(size=dim)

    poly1 = poly_factory(dim, order, rng)
    _, g1, h1 = poly1(x)

    poly2 = poly_factory(dim, order, rng)
    _, g2, h2 = poly2(x)

    if subdim is None:
        U = None
        subdim = dim
        g1proj = g1
        xproj = x
    else:
        U = ortho_group.rvs(dim, random_state=rng)[:, :subdim]
        h1 = U.T @ h1 @ U
        h2 = U.T @ h2 @ U
        g1proj = U.T @ g1
        xproj = U.T @ x

    Hkwargs = dict(x0=x, eta=eta, threepoint=threepoint, Uproj=U)

    H1 = NumericalHessian(lambda x: poly1(x)[:2], g0=g1, **Hkwargs)

    # M1: some random matrix
    M1 = rng.normal(size=(subdim, subdim))

    H2 = H1 + NumericalHessian(lambda x: poly2(x)[:2], g0=g2, **Hkwargs) + M1
    H3 = h1 + h2 + M1

    # Make first column orthogonal to g1
    M1[:, 0] = xproj - g1proj * (xproj @ g1proj) / (g1proj @ g1proj)

    # Make second column orthogonal to g1 and x
    M1[:, 1] -= M1[:, 0] * (M1[:, 1] @ M1[:, 0]) / (M1[:, 0] @ M1[:, 0])
    M1[:, 1] -= g1proj * (M1[:, 1] @ g1proj) / (g1proj @ g1proj)

    np.testing.assert_allclose(H2.T.dot(M1), H3.T @ M1, **tol)


class TestApproximateHessian:
    """Test ApproximateHessian operations.

    Moved here from test_core_functionality.py: sella/linalg.py is ~580 lines
    and this file previously covered only NumericalHessian.
    """

    def test_hessian_arithmetic(self):
        """Test that ApproximateHessian supports addition with arrays."""
        dim = 5
        ncart = 5
        rng = np.random.RandomState(42)

        # Create initialized Hessian
        H1 = ApproximateHessian(dim, ncart, update_method='BFGS')
        # Initialize with a step
        s = rng.normal(size=dim)
        y = rng.normal(size=dim)
        s /= np.linalg.norm(s)
        y /= np.linalg.norm(y)
        H1.update(s, y)
        assert H1.initialized

        # Add to an array
        M = rng.normal(size=(dim, dim))
        M = 0.5 * (M + M.T)

        result = H1 + M
        expected = H1.B + M
        # Result is an ApproximateHessian, compare underlying matrices
        np.testing.assert_allclose(result.B, expected, atol=1e-10)

    def test_hessian_addition_with_uninitialized(self):
        """An uninitialized operand is absorbing: the sum is uninitialized.

        The original form of this test asserted only ``result is not None``,
        which holds for any return value. The real contract is that adding an
        uninitialized Hessian yields an uninitialized Hessian (B is None),
        regardless of operand order.
        """
        dim = 5
        ncart = 5
        rng = np.random.RandomState(42)

        H1 = ApproximateHessian(dim, ncart, update_method='BFGS')
        H2 = ApproximateHessian(dim, ncart, update_method='BFGS')

        # Initialize H1 only
        s = rng.normal(size=dim)
        y = rng.normal(size=dim)
        H1.update(s, y)
        assert H1.initialized
        assert not H2.initialized

        for result in (H1 + H2, H2 + H1):
            assert isinstance(result, ApproximateHessian)
            assert not result.initialized
            assert result.B is None

    def test_eigendecomposition(self):
        """Test eigenvalue decomposition of ApproximateHessian."""
        dim = 6
        ncart = 6
        rng = np.random.RandomState(42)

        H = ApproximateHessian(dim, ncart, update_method='BFGS')

        # Initialize with multiple updates
        for _ in range(3):
            s = rng.normal(size=dim)
            y = rng.normal(size=dim)
            H.update(s, y)

        # Access eigenvalues
        evals = H.evals
        evecs = H.evecs
        assert evals is not None
        assert evecs is not None
        assert len(evals) == dim

        # Verify eigendecomposition
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        np.testing.assert_allclose(H.B, reconstructed, atol=1e-10)


class TestSparseInternalHessians:
    """Test SparseInternalHessians functionality."""

    def test_numpy_array_conversion(self):
        """Test that SparseInternalHessians can be converted to numpy array."""
        # Create a simple molecule
        atoms = molecule('H2O')
        internal = Internals(atoms)
        internal.find_all_bonds()
        internal.find_all_angles()

        # Get the Hessian
        hess = internal.hessian()
        assert isinstance(hess, SparseInternalHessians)

        # Convert to numpy array
        arr = np.asarray(hess)
        assert isinstance(arr, np.ndarray)

        # Check shape consistency
        n = len(internal.calc())
        assert arr.shape == (n, 3 * len(atoms), 3 * len(atoms))

# ===========================================================================
# Eigensolvers
# ===========================================================================

@pytest.mark.parametrize("dim,order,eta,threepoint",
                         [(10, 4, 1e-6, True),
                          (10, 4, 1e-6, False)])
def test_exact(dim, order, eta, threepoint):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-4, rtol=eta**2)

    poly = poly_factory(dim, order, rng=rng)
    x = rng.normal(size=dim)

    _, g, h = poly(x)

    H = NumericalHessian(lambda x: poly(x)[:2], g0=g, x0=x,
                         eta=eta, threepoint=threepoint)

    l1, V1, AV1 = exact(h)
    l2, V2, AV2 = exact(H)

    np.testing.assert_allclose(l1, l2, **tol)
    np.testing.assert_allclose(np.abs(V1.T @ V2), np.eye(dim), **tol)
    np.testing.assert_allclose(h @ V1, AV1, **tol)
    np.testing.assert_allclose(h @ V2, AV2, **tol)

    P = h + get_matrix(dim, dim, rng=rng) * 1e-3
    l3, V3, AV3 = exact(H, P=P)

    np.testing.assert_allclose(l1, l3, **tol)
    np.testing.assert_allclose(np.abs(V1.T @ V2), np.eye(dim), **tol)


@pytest.mark.parametrize("dim,order,eta,threepoint,gamma,method,maxiter",
                         [(10, 4, 1e-6, False, 0., 'jd0', None),
                          (10, 4, 1e-6, False, 1e-32, 'jd0', 3),
                          (10, 4, 1e-6, True, 1e-1, 'jd0', None),
                          (10, 4, 1e-6, False, 1e-1, 'jd0', None),
                          (10, 4, 1e-6, False, 1e-1, 'lanczos', None),
                          (10, 4, 1e-6, False, 1e-1, 'gd', None),
                          (10, 4, 1e-6, False, 1e-1, 'jd0_alt', None),
                          (10, 4, 1e-6, False, 1e-1, 'mjd0_alt', None),
                          (10, 4, 1e-6, False, 1e-1, 'mjd0', None),
                          ])
def test_rayleigh_ritz(dim, order, eta, threepoint, gamma, method, maxiter):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-4, rtol=eta**2)

    poly = poly_factory(dim, order, rng=rng)
    x = rng.normal(size=dim)

    _, g, h = poly(x)
    H = NumericalHessian(lambda x: poly(x)[:2], g0=g, x0=x,
                         eta=eta, threepoint=threepoint)

    l1, V1, AV1 = rayleigh_ritz(H, gamma, np.eye(dim), method=method,
                                maxiter=maxiter)
    np.testing.assert_allclose(l1, np.linalg.eigh(V1.T @ AV1)[0], **tol)

    v0 = rng.normal(size=dim)
    rayleigh_ritz(H, gamma, np.eye(dim), method=method, v0=v0,
                  maxiter=maxiter, vref=np.linalg.eigh(h)[1][:, 0])

# ===========================================================================
# Internal coordinates
# ===========================================================================

@pytest.mark.parametrize("name", ['CH4', 'C6H6', 'C2H6'])
def test_get_internal(name: str) -> None:
    atoms = molecule(name)
    internal = Internals(atoms)
    internal.find_all_bonds()
    internal.find_all_angles()
    internal.find_all_dihedrals()
    jac = internal.jacobian()
    hess = internal.hessian()

    x0 = atoms.positions.ravel().copy()
    x = x0.copy()
    dx = 1e-4

    jac_numer = np.zeros_like(jac)
    hess_numer = np.zeros_like(hess)
    for i in range(len(x)):
        x[i] += dx
        atoms.positions = x.reshape((-1, 3))
        res_plus = internal.calc()
        jac_plus = internal.jacobian()
        x[i] = x0[i] - dx
        atoms.positions = x.reshape((-1, 3))
        res_minus = internal.calc()
        jac_minus = internal.jacobian()
        x[i] = x0[i]
        atoms.positions = x.reshape((-1, 3))
        jac_numer[:, i] = (internal.wrap(res_plus - res_minus)) / (2 * dx)
        hess_numer[:, i, :] = (jac_plus - jac_minus) / (2 * dx)
    np.testing.assert_allclose(jac, jac_numer, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(hess, hess_numer, rtol=1e-7, atol=1e-7)


class TestTRICs:
    """Tests for Translation-Rotation Internal Coordinates (TRICs)."""

    def test_tric_single_atom_fragment(self):
        """Test TRICs with a single-atom fragment (should not raise assertion).

        This tests the bug fix for the line ordering issue in find_all_bonds()
        where single atoms would incorrectly get rotation ICs added.
        """
        # Bi(NO3)3 cluster from the bug report - Bi is a single atom, NO3 are fragments
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )
        # Use scale=1.0 to ensure fragments are detected (not bonded via 1.25 scale)
        ints = Internals(atoms, allow_fragments=True)
        # This should not raise an assertion error even though Bi is a single atom
        ints.find_all_bonds(scale=1.0)
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # Should have translations (including for the single Bi atom)
        assert len(ints.internals['translations']) > 0

        # Rotations should only be for multi-atom fragments (NO3 groups)
        # Bi should NOT have rotation ICs
        for rot in ints.internals['rotations']:
            assert len(rot.indices) >= 2, "Rotation IC added to single atom!"

    def test_tric_scale_parameter(self):
        """Test that scale parameter affects bond detection."""
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )

        # With small scale, should have fragments (TRICs added)
        ints_small = Internals(atoms, allow_fragments=True)
        ints_small.find_all_bonds(scale=1.0)
        n_trans_small = len(ints_small.internals['translations'])
        n_rot_small = len(ints_small.internals['rotations'])

        # With large scale, might connect everything (no TRICs)
        ints_large = Internals(atoms, allow_fragments=True)
        ints_large.find_all_bonds(scale=1.5)
        n_trans_large = len(ints_large.internals['translations'])
        n_rot_large = len(ints_large.internals['rotations'])

        # Smaller scale should result in more fragments (more TRICs)
        assert n_trans_small >= n_trans_large
        assert n_rot_small >= n_rot_large

    def test_tric_two_separate_molecules(self):
        """Test TRICs with two well-separated molecules."""
        # Two water molecules far apart - use explicit element list for clarity
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],     # O (first molecule)
                [0.96, 0.0, 0.0],    # H
                [0.0, 0.96, 0.0],    # H
                [10.0, 0.0, 0.0],    # O (second molecule, far away)
                [10.96, 0.0, 0.0],   # H
                [10.0, 0.96, 0.0],   # H
            ]
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        # Should have 2 fragments, so 2 translation sets (6 coords) and 2 rotation sets (6 coords)
        assert len(ints.internals['translations']) == 6  # 3 per fragment × 2 fragments
        assert len(ints.internals['rotations']) == 6     # 3 per fragment × 2 fragments

    def test_validate_basis_with_trics(self):
        """Test that validate_basis correctly calculates DOF with TRICs."""
        # Two water molecules far apart - use explicit element list for clarity
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],     # O (first molecule)
                [0.96, 0.0, 0.0],    # H
                [0.0, 0.96, 0.0],    # H
                [10.0, 0.0, 0.0],    # O (second molecule, far away)
                [10.96, 0.0, 0.0],   # H
                [10.0, 0.96, 0.0],   # H
            ]
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        # With TRICs, expect 3N = 18 DOF (translations+rotations span full space)
        # validate_basis should not warn about a deficient basis for TRICs.
        # Filter to warnings raised by sella itself -- asserting on the total
        # count makes this fail on unrelated warnings from JAX, ASE or numpy.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ints.validate_basis()

        basis_warnings = [
            w for w in caught
            if w.filename == tools_sella.__file__
        ]
        assert not basis_warnings, (
            f"validate_basis warned: {[str(w.message) for w in basis_warnings]}"
        )

    def test_tric_optimization_convergence(self):
        """Optimization with TRICs runs without an ODE failure.

        Smoke test: the assertion is that no exception is raised. It guards
        the fix for the ODE convergence issue with the ill-conditioned
        Jacobians that arise from TRICs, which surfaced as a RuntimeError.
        """
        # Bi(NO3)3 cluster - a real-world TRIC test case
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )
        atoms.calc = LennardJones()

        # Use TRICs with small scale to ensure fragments are detected
        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds(scale=1.0)
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # This should not raise RuntimeError about ODE convergence
        opt = Sella(atoms, internal=ints)
        # Just run a few steps to verify ODE works
        opt.run(fmax=1.0, steps=5)

# ===========================================================================
# PES wrappers
# ===========================================================================

def _build_pes(pes_class, atoms, cons, trajectory):
    """Construct either PES flavour from the same inputs.

    The two classes take constraints differently: PES accepts a Constraints
    object directly, while InternalPES takes an Internals object that carries
    the constraints on its ``.cons`` attribute.
    """
    if pes_class is InternalPES:
        return InternalPES(
            atoms, Internals(atoms, cons=cons), trajectory=trajectory
        )
    return pes_class(atoms, constraints=cons, trajectory=trajectory)


@pytest.mark.parametrize("pes_class", [PES, InternalPES],
                         ids=["PES", "InternalPES"])
@pytest.mark.parametrize("name,use_traj,fixed_bonds",
                         [("CH4", True, None),
                          ("CH4", False, ((0, 1),)),
                          ("C6H6", False, None),
                          ])
def test_PES(pes_class, name, use_traj, fixed_bonds, tmp_path):
    # C6H6 on InternalPES used to raise from the geometry-update ODE, because
    # a badly truncated Jacobian pseudo-inverse made the geodesic violently
    # stiff. See test_binv_is_truncated_at_the_rank_of_the_jacobian.
    tol = dict(atol=1e-6, rtol=1e-6)

    atoms = molecule(name)

    # EMT is *not* appropriate for molecules like this, but this is one
    # of the few calculators that is guaranteed to be available to all
    # users, and we don't need to use a physical PES to test this.
    atoms.calc = EMT()

    cons = None
    if fixed_bonds is not None:
        cons = Constraints(atoms)
        for indices in fixed_bonds:
            cons.fix_bond(indices)

    # Trajectories go to tmp_path so a test run leaves no files behind.
    traj = str(tmp_path / f"{name}.traj") if use_traj else None

    pes = _build_pes(pes_class, atoms, cons, traj)

    pes.kick(0., diag=True, gamma=0.1)

    for i in range(2):
        pes.kick(-pes.get_g() * 0.01)

    assert pes.H is not None
    # converged() returns (converged, fmax, cmax) -- index it, or the assert
    # is just testing that a 3-tuple is truthy. cmax is loosened alongside
    # fmax because kick() steps along the raw gradient without the
    # constraint correction that Sella's stepper applies, so a constrained
    # PES driven this way sits slightly off the constraint manifold.
    assert not pes.converged(0.)[0]
    assert pes.converged(1e100, cmax=1e100)[0]
    A = pes.get_Ufree().T @ pes.get_Ucons()
    np.testing.assert_allclose(A, 0, **tol)

    pes.kick(-pes.get_g() * 0.001, diag=True, gamma=0.1)


def test_constraint_is_enforced_through_sella():
    """A fixed bond stays fixed when the PES is driven by Sella.

    The parametrized test above only checks that the constrained and free
    subspaces are orthogonal. This checks the end-to-end contract: the
    constraint correction in the stepper actually holds the bond.
    """
    atoms = molecule('CH4')
    atoms.calc = EMT()

    cons = Constraints(atoms)
    cons.fix_bond((0, 1))
    r0 = atoms.get_distance(0, 1)

    opt = Sella(atoms, order=0, constraints=cons, logfile=None)
    opt.run(fmax=1e-3, steps=30)

    assert opt.converged()
    assert atoms.get_distance(0, 1) == pytest.approx(r0, abs=1e-6)


def test_binv_is_truncated_at_the_rank_of_the_jacobian():
    """_get_Binv drops the null space whichever cache path it takes.

    Internal coordinates do not change under rigid-body motion, so the
    Jacobian always carries six vanishing singular values and its inverse
    exists only on the row space. _get_jacobian_qr detects that and builds a
    truncated inverse from its own SVD factors; _get_Binv used to discard it
    and recompute with np.linalg.pinv's default cutoff, which is relative to
    machine precision and so kept the null space. That inverse had entries of
    order 1e12, and since _set_x_ode freezes one Binv for the whole
    integration, the geodesic went stiff enough to stall on a 1e-4 step.
    """
    atoms = molecule('C6H6')
    atoms.calc = EMT()
    pes = InternalPES(atoms, Internals(atoms, cons=None), trajectory=None)

    # Both caches are keyed on the geometry, so moving an atom leaves them
    # cold, as every fresh step does. Order matters from there: _set_x_ode
    # asks for Binv first, which is what the bug needed. _get_Binv missed the
    # cold cache, _get_jacobian_qr filled it with the truncated inverse on the
    # way past, and _get_Binv returned its own recomputation regardless --
    # reaching for the good value only on a later call, once the geometry had
    # stopped moving.
    pes.atoms.positions[0, 0] += 1e-3

    B = pes.int.jacobian()
    svals = np.linalg.svd(B, compute_uv=False)
    kept = svals > 1e-6
    # Twelve atoms, so 36 Cartesian degrees of freedom, six of them rigid.
    assert B.shape[1] == 36
    assert kept.sum() == 30

    Binv = pes._get_Binv()

    assert_allclose(Binv, np.linalg.pinv(B, rcond=1e-6), atol=1e-8)
    # ||Binv||_2 is 1/s_min over the retained singular values, and no entry
    # of a matrix exceeds its spectral norm. Keeping the null space put this
    # at ~1e12 against a bound of ~1.6.
    assert np.abs(Binv).max() <= 1.0 / svals[kept].min()


# ===========================================================================
# Cell optimisation
# ===========================================================================

# System-building factories (cu_fcc, two_water_crystal, methane_box,
# water_box) and the finite-difference helpers (fd_pes_gradient,
# fd_cell_gradient) come from tests/conftest.py.

# Non-orthogonal cells used to exercise the rotation correction, which is
# identity for a cubic cell and so would otherwise go untested.
_TRICLINIC_CELL = np.array([
    [7.0, 0.5, 0.3],
    [0.0, 6.8, 0.4],
    [0.0, 0.0, 7.2],
])

# Larger shear, where the rotation component of the polar decomposition
# deviates substantially from identity.
_SHEARED_CELL = np.array([
    [7.0, 1.5, 0.8],
    [0.0, 6.5, 1.2],
    [0.0, 0.0, 7.5],
])


class TestCellDerivatives:
    """Test cell derivative functions for internal coordinates."""

    def test_bond_cell_derivative_numerical_molecular(
        self, methane_box, fd_cell_gradient
    ):
        """Verify bond cell derivative against numerical finite difference.

        Uses a molecular system with explicit bonds that cross the periodic
        boundary when the cell is compressed.
        """
        atoms = methane_box()

        # Create internals and find bonds (C-H bonds)
        internals = Internals(atoms)
        internals.find_all_bonds()

        assert len(internals.internals['bonds']) > 0, "No bonds found"

        # Test the first bond
        bond = internals.internals['bonds'][0]

        grad_analytic = bond.calc_cell_gradient(atoms)
        grad_numeric = fd_cell_gradient(atoms, bond.calc)

        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_bond_cell_derivative_with_periodic_image(self, fd_cell_gradient):
        """Test bond cell derivative for bond crossing periodic boundary.

        Create a diatomic that spans the periodic boundary to ensure
        ncvec contribution to cell derivative is tested.
        """
        # Create H2 spanning the periodic boundary
        atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
        atoms.set_cell([3.0, 3.0, 3.0])
        atoms.pbc = True

        # Create a bond that crosses the boundary (using ncvec)
        # Bond between atom 0 and atom 1 via periodic image
        bond = Bond(
            indices=np.array([0, 1], dtype=np.int32),
            ncvecs=np.array([[0, 0, -1]], dtype=np.int32)  # Wrap in -z direction
        )

        grad_analytic = bond.calc_cell_gradient(atoms)
        grad_numeric = fd_cell_gradient(atoms, bond.calc)

        # The bond crosses the boundary, so cell derivatives should be non-zero
        assert not np.allclose(grad_analytic, 0), \
            "Cell gradient should be non-zero for PBC bond"
        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_angle_cell_derivative_numerical(self, water_box, fd_cell_gradient):
        """Verify angle cell derivative against numerical finite difference."""
        # Use water - has a well-defined H-O-H angle
        atoms = water_box()

        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()

        assert internals.internals['angles'], "No angles found in water"

        # Get an angle (H-O-H)
        angle = internals.internals['angles'][0]

        grad_analytic = angle.calc_cell_gradient(atoms)
        grad_numeric = fd_cell_gradient(atoms, angle.calc)

        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_cell_jacobian_shape(self, methane_box):
        """Test that cell_jacobian returns correct shape."""
        atoms = methane_box()  # CH4 with bonds and angles

        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()

        J_cell = internals.cell_jacobian()

        # Should have shape (n_active_coords, 9)
        n_active = len(internals.calc())
        assert J_cell.shape == (n_active, 9)


class TestCellInternalPES:
    """Test CellInternalPES class."""

    def test_cell_internal_pes_initialization(self, cu_fcc):
        """Test that CellInternalPES initializes correctly."""
        atoms = cu_fcc()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        # Check dimensions include cell DOF
        assert pes.n_cell_dof == 9  # Full 3x3 cell
        assert pes.dim == pes.n_internal + 9

    def test_cell_internal_pes_get_x(self, cu_fcc):
        """Test get_x returns combined internal + cell vector."""
        atoms = cu_fcc()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        x = pes.get_x()

        # Length should be n_internal + n_cell_dof
        assert len(x) == pes.dim

        # First n_internal elements are internal coords
        q = internals.calc()
        assert_allclose(x[:pes.n_internal], q, rtol=1e-10)

        # Cell params at identity should be approximately zero
        # (log of identity matrix is zero)
        assert_allclose(x[pes.n_internal:], 0, atol=1e-10)

    def test_cell_internal_pes_cell_mask(self, cu_fcc):
        """Test cell_mask parameter."""
        atoms = cu_fcc()

        # Only allow diagonal elements (hydrostatic strain)
        cell_mask = np.eye(3, dtype=bool)

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals, cell_mask=cell_mask)

        assert pes.n_cell_dof == 3  # Only 3 diagonal elements
        assert pes.dim == pes.n_internal + 3

    def test_cell_internal_pes_eval(self, cu_fcc):
        """Test eval returns correct gradient shape."""
        atoms = cu_fcc()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        f, g = pes.eval()

        assert isinstance(f, float)
        assert len(g) == pes.dim


class TestCellCartesianPES:
    """Test CellCartesianPES class."""

    def test_cell_cartesian_pes_initialization(self, cu_fcc):
        """Test that CellCartesianPES initializes correctly."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        # Check dimensions include cell DOF
        assert pes.n_cell_dof == 9  # Full 3x3 cell
        assert pes.n_cart == 3 * len(atoms)
        assert pes.dim == pes.n_cart + 9

    def test_cell_cartesian_pes_get_x(self, cu_fcc):
        """Test get_x returns combined Cartesian + cell vector."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        x = pes.get_x()

        # Length should be n_cart + n_cell_dof
        assert len(x) == pes.dim

        # First n_cart elements are Cartesian positions
        positions_flat = atoms.get_positions().ravel()
        assert_allclose(x[:pes.n_cart], positions_flat, rtol=1e-10)

        # Cell params at identity should be approximately zero
        # (log of identity matrix is zero)
        assert_allclose(x[pes.n_cart:], 0, atol=1e-10)

    def test_cell_cartesian_pes_cell_mask(self, cu_fcc):
        """Test cell_mask parameter."""
        atoms = cu_fcc()

        # Only allow diagonal elements (hydrostatic strain)
        cell_mask = np.eye(3, dtype=bool)

        pes = CellCartesianPES(atoms, cell_mask=cell_mask)

        assert pes.n_cell_dof == 3  # Only 3 diagonal elements
        assert pes.dim == pes.n_cart + 3

    def test_cell_cartesian_pes_eval(self, cu_fcc):
        """Test eval returns correct gradient shape."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        f, g = pes.eval()

        assert isinstance(f, float)
        assert len(g) == pes.dim

    def test_cell_cartesian_pes_save_restore(self, cu_fcc):
        """Test save and restore functionality."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        # Save initial state
        pes.save()
        cell0 = atoms.get_cell().array.copy()
        pos0 = atoms.get_positions().copy()

        # Modify positions and cell
        atoms.positions += 0.1
        new_cell = atoms.get_cell().array * 1.05
        atoms.set_cell(new_cell, scale_atoms=False)

        # Verify modifications took effect
        assert not np.allclose(atoms.get_positions(), pos0)
        assert not np.allclose(atoms.get_cell().array, cell0)

        # Restore and verify
        pes.restore()
        assert_allclose(atoms.get_positions(), pos0)
        assert_allclose(atoms.get_cell().array, cell0)

    def test_cell_cartesian_pes_set_x(self, cu_fcc):
        """Test set_x updates positions and cell."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        # Get initial x
        x0 = pes.get_x()

        # Create target with small perturbations
        x_target = x0.copy()
        x_target[:pes.n_cart] += 0.01  # Small position change
        x_target[pes.n_cart:] += 0.001  # Small cell change

        # Set new x
        pes.set_x(x_target)

        # Get x and verify it's close to target
        x_new = pes.get_x()
        # Cell parameters should match
        assert_allclose(x_new[pes.n_cart:], x_target[pes.n_cart:], rtol=1e-6)
        # Positions should match
        assert_allclose(x_new[:pes.n_cart], x_target[:pes.n_cart], rtol=1e-6)

    def test_cell_cartesian_vs_internal_gradient_shape(self, cu_fcc):
        """Compare CellCartesianPES and CellInternalPES gradient shapes."""
        atoms = cu_fcc()

        # CellCartesianPES
        pes_cart = CellCartesianPES(atoms.copy())
        pes_cart.atoms.calc = EMT()
        _, g_cart = pes_cart.eval()

        # CellInternalPES
        internals = Internals(atoms)
        pes_int = CellInternalPES(atoms, internals)
        _, g_int = pes_int.eval()

        # Both should have 9 cell DOF (full 3x3 cell)
        assert pes_cart.n_cell_dof == 9
        assert pes_int.n_cell_dof == 9

        # Cell parts of gradient should have same length
        assert len(g_cart[pes_cart.n_cart:]) == len(g_int[pes_int.n_internal:])

    def test_cell_cartesian_pes_pressure(self, cu_fcc):
        """Test scalar pressure contribution."""
        atoms = cu_fcc()

        # Without pressure
        pes_no_p = CellCartesianPES(atoms.copy())
        pes_no_p.atoms.calc = EMT()
        f_no_p, _ = pes_no_p.eval()

        # With pressure
        pressure = 0.1  # eV/Å³
        atoms2 = atoms.copy()
        atoms2.calc = EMT()
        pes_p = CellCartesianPES(atoms2, scalar_pressure=pressure)
        f_p, _ = pes_p.eval()

        # Energy with pressure should be higher (positive pressure)
        volume = atoms.get_volume()
        expected_diff = pressure * volume
        assert_allclose(f_p - f_no_p, expected_diff, rtol=1e-10)


class TestCellCartesianGradient:
    """Test cell gradient calculations in CellCartesianPES."""

    def test_cell_gradient_numerical(self, cu_fcc, fd_pes_gradient):
        """Test cell gradient matches numerical finite difference for bulk Cu."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_cart:]  # Cell part of gradient

        # Numerical gradient via finite difference on cell parameters
        g_cell_numeric = fd_pes_gradient(pes, range(pes.n_cart, pes.dim), delta=1e-6)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_cartesian_gradient_numerical(self, cu_fcc, fd_pes_gradient):
        """Test Cartesian gradient matches numerical finite difference."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms)

        # Get analytical gradient
        _, g = pes.eval()
        g_cart = g[:pes.n_cart]  # Cartesian part of gradient

        # Numerical gradient via finite difference
        g_cart_numeric = fd_pes_gradient(pes, range(pes.n_cart), delta=1e-6)

        assert_allclose(g_cart, g_cart_numeric, atol=1e-4, rtol=1e-3)


class TestSellaWithCellOptimization:
    """Integration tests for Sella with cell optimization."""

    def test_sella_optimize_cell_parameter(self, cu_fcc):
        """Test that optimize_cell parameter is properly handled."""
        atoms = cu_fcc()

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True)

        assert opt.optimize_cell is True
        assert isinstance(opt.pes, CellInternalPES)

    def test_sella_cell_optimization_validation(self, cu_fcc):
        """Test validation of cell optimization parameters."""
        atoms = cu_fcc()

        # Should fail with order != 0
        with pytest.raises(ValueError, match="order=0"):
            Sella(atoms, internal=True, order=1, optimize_cell=True)

        # internal=False with optimize_cell=True should use CellCartesianPES
        atoms2 = atoms.copy()
        atoms2.calc = EMT()
        opt = Sella(atoms2, internal=False, order=0, optimize_cell=True)
        assert isinstance(opt.pes, CellCartesianPES)

        # Should fail without PBC
        atoms_nopbc = atoms.copy()
        atoms_nopbc.pbc = False
        atoms_nopbc.calc = EMT()
        with pytest.raises(ValueError, match="periodic"):
            Sella(atoms_nopbc, internal=True, order=0, optimize_cell=True)

    def test_sella_cell_optimization_single_step(self, cu_fcc):
        """Test that cell optimization can take a single step."""
        # Use strained FCC copper
        atoms = cu_fcc(a=3.8)  # Slightly expanded

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True)

        # Record initial state
        cell0 = atoms.get_cell().array.copy()
        e0 = atoms.get_potential_energy()

        # Take one step
        opt.step()

        e1 = atoms.get_potential_energy()
        cell1 = atoms.get_cell().array

        # A cell optimization step must actually move the cell...
        assert not np.allclose(cell0, cell1), "cell did not change"
        # ...toward equilibrium: a=3.8 is expanded, so it should contract.
        assert np.linalg.det(cell1) < np.linalg.det(cell0), \
            "expanded cell should shrink toward equilibrium"
        # ...and not uphill in energy.
        assert e1 <= e0 + 1e-8, f"energy increased: {e0} -> {e1}"

    @pytest.mark.slow
    def test_sella_cell_optimization_convergence(self, cu_fcc):
        """Test that cell optimization converges for a simple system."""
        # Create strained FCC copper - use single primitive cell
        # (larger supercells have issues with angle finding in FCC)
        # Note: bulk() creates a primitive cell, so a=3.8 gives cell param ~2.69 Å
        atoms = cu_fcc(a=3.8)  # Slightly expanded

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            logfile=None,
        )

        # Run optimization. Assert convergence rather than guarding the
        # checks below behind `if converged:` -- that form lets a run that
        # never converges pass silently.
        assert opt.run(fmax=0.05, steps=50), "cell optimization did not converge"

        # Check that cell relaxed toward equilibrium
        # EMT equilibrium for FCC Cu primitive cell is about a = 2.54 Å
        a = atoms.get_cell().cellpar()[0]
        assert 2.4 < a < 2.7

    def test_gradient_converged_delegates_to_converged(self, cu_fcc):
        """Test that gradient_converged routes through converged().

        ASE 3.28 changed irun() to call gradient_converged() instead of
        converged(), bypassing Sella's stress check. This test verifies
        that Sella's gradient_converged override delegates to converged(),
        so the stress convergence check is always reached.
        """
        atoms = cu_fcc(a=3.8)  # Strained → nonzero stress

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True,
                    logfile=None)

        # Force evaluation so pes has cached gradient
        opt.pes.eval()

        # converged() should check stress; with a strained cell and tight
        # tolerance, it should report not converged
        opt.fmax = 1e-10
        opt.smax = 1e-10
        assert not opt.converged()

        # gradient_converged must agree (it delegates to converged)
        assert not opt.gradient_converged()

        # Now with loose tolerances, both should agree on convergence
        opt.fmax = 100.0
        opt.smax = 100.0
        assert opt.converged()
        assert opt.gradient_converged()


class TestVoigtConversion:
    """Test Voigt stress conversion functions."""

    def test_voigt_roundtrip(self):
        """Test conversion roundtrip."""
        # Seeded: an unseeded np.random.randn makes this non-reproducible,
        # so a failure could not be replayed.
        voigt = np.random.RandomState(0).randn(6)

        # Convert to 3x3 and back
        stress_3x3 = voigt_6_to_full_3x3_stress(voigt)
        voigt_back = full_3x3_to_voigt_6_stress(stress_3x3)

        assert_allclose(voigt, voigt_back)

    def test_voigt_symmetry(self):
        """Test that converted tensor is symmetric."""
        voigt = np.array([1, 2, 3, 4, 5, 6])
        stress_3x3 = voigt_6_to_full_3x3_stress(voigt)

        assert_allclose(stress_3x3, stress_3x3.T)


class TestStressTensor:
    """Test how the cell gradient responds to strain.

    These previously asserted only on ``atoms.get_stress()``, which exercises
    ASE's EMT calculator and no Sella code at all. They now assert the Sella
    contract that consumes the stress: the cell block of the PES gradient.
    """

    # EMT equilibrium for the FCC Cu primitive cell.
    A_EQUILIBRIUM = 3.59

    def _cell_gradient(self, atoms):
        pes = CellCartesianPES(atoms)
        _, g = pes.eval()
        return g[pes.n_cart:]

    def test_cell_gradient_vanishes_at_equilibrium(self, cu_fcc):
        """At the equilibrium lattice constant the cell is not driven."""
        g_cell = self._cell_gradient(cu_fcc(a=self.A_EQUILIBRIUM))

        assert np.all(np.isfinite(g_cell))
        assert np.linalg.norm(g_cell) < 1e-2

    def test_cell_gradient_reverses_between_compression_and_expansion(
        self, cu_fcc
    ):
        """Compressed and expanded cells are driven in opposite directions."""
        g_compressed = self._cell_gradient(cu_fcc(a=3.5))
        g_expanded = self._cell_gradient(cu_fcc(a=3.8))

        assert np.all(np.isfinite(g_compressed))
        assert np.all(np.isfinite(g_expanded))

        # Hydrostatic component: negative when compressed (push outward),
        # positive when expanded (pull inward).
        p_compressed = g_compressed.reshape(3, 3).trace()
        p_expanded = g_expanded.reshape(3, 3).trace()

        assert p_compressed < 0 < p_expanded

        # Both are far from equilibrium, so both must be substantial.
        g_eq = self._cell_gradient(cu_fcc(a=self.A_EQUILIBRIUM))
        assert np.linalg.norm(g_compressed) > 100 * np.linalg.norm(g_eq)
        assert np.linalg.norm(g_expanded) > 100 * np.linalg.norm(g_eq)

    def test_cell_gradient_finite_for_molecular_box(self, methane_box):
        """A molecular system in a periodic box yields a finite cell gradient."""
        atoms = methane_box(vacuum=4.0)
        atoms.calc = EMT()

        g_cell = self._cell_gradient(atoms)

        assert len(g_cell) == 9
        assert np.all(np.isfinite(g_cell))


class TestCellGradient:
    """Test cell gradient calculations in CellInternalPES."""

    def test_cell_gradient_numerical_inorganic(self, cu_fcc, fd_pes_gradient):
        """Test cell gradient matches numerical finite difference for bulk Cu."""
        atoms = cu_fcc()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]  # Cell part of gradient

        # Numerical gradient via finite difference on cell parameters
        g_cell_numeric = fd_pes_gradient(pes, range(pes.n_internal, pes.dim), delta=1e-6)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_cell_gradient_shape_molecular(self, two_water_crystal):
        """Test cell gradient has correct shape for molecular system."""
        mol = two_water_crystal(7.0)

        internals = Internals(mol, allow_fragments=True)
        pes = CellInternalPES(mol, internals, auto_find_internals=True)

        _, g = pes.eval()

        # Gradient should have n_internal + n_cell_dof components
        assert len(g) == pes.n_internal + pes.n_cell_dof

        # All components should be finite
        assert np.all(np.isfinite(g))


class TestMolecularCrystal:
    """Test cell optimization for molecular crystal systems.

    Molecular crystals have multiple separate molecules in a periodic cell,
    requiring TRICs (Translation-Rotation Internal Coordinates) for proper
    handling of each molecular fragment.
    """

    def test_molecular_crystal_with_trics(self, two_water_crystal):
        """Test that molecular crystal optimization works with allow_fragments=True."""
        # A simple molecular crystal: two water molecules in a box
        atoms = two_water_crystal()

        # Create optimizer with allow_fragments=True for TRICs
        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            allow_fragments=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellInternalPES)

        # Verify the internal coords have TRICs (translations and rotations)
        internals = opt.pes.int
        assert internals.ntrans > 0, "Should have translation coordinates for fragments"
        assert internals.nrotations > 0, "Should have rotation coordinates for fragments"

        # Take a few steps to verify no errors
        for _ in range(3):
            opt.step()

    def test_molecular_crystal_cartesian(self):
        """Test molecular crystal optimization with Cartesian coordinates."""
        # Create two methane molecules
        ch4_1 = molecule('CH4')
        ch4_2 = molecule('CH4')

        ch4_1.positions += [1.5, 1.5, 1.5]
        ch4_2.positions += [5.0, 5.0, 5.0]

        atoms = ch4_1 + ch4_2
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # Use Cartesian coordinates (internal=False) with cell optimization
        opt = Sella(
            atoms,
            internal=False,
            order=0,
            optimize_cell=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellCartesianPES)

        # Take a few steps
        for _ in range(3):
            opt.step()

    def test_molecular_crystal_trics_dof_count(self):
        """Test that TRICs add correct DOF for molecular fragments."""
        # Create two separate H2 molecules
        atoms = Atoms(
            'H4',
            positions=[
                [0.0, 0.0, 0.0],
                [0.74, 0.0, 0.0],
                [4.0, 4.0, 4.0],
                [4.74, 4.0, 4.0],
            ]
        )
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # With allow_fragments=True, should get TRICs for each fragment
        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()

        # Should have 2 bonds (one per H2)
        assert internals.nbonds == 2

        # Should have translations for the 2 fragments
        assert internals.ntrans > 0

        # Should have rotations for fragments that can rotate
        # (H2 is linear, so rotation DOF may be limited)
        assert internals.nrotations >= 0


class TestTRICsCellDerivatives:
    """Test that TRICs have correct cell derivatives.

    For molecular crystals, translation and rotation coordinates should
    have zero cell derivatives since they describe internal molecular
    motion that doesn't depend on the cell.
    """

    def test_translation_cell_derivative_zero(self):
        """Test that translation coordinates have zero cell derivatives."""
        # Create a simple diatomic in a periodic cell
        atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atoms.set_cell([5.0, 5.0, 5.0])
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()

        # Get the translation coordinates
        translations = internals.internals.get('translations', [])

        for trans in translations:
            grad_cell = trans.calc_cell_gradient(atoms)
            # Translation center of mass should not depend on cell
            assert_allclose(grad_cell, 0, atol=1e-10)

    def test_rotation_cell_derivative_zero(self):
        """Test that rotation coordinates have zero cell derivatives."""
        # Create water molecule (non-linear, has rotations)
        atoms = molecule('H2O')
        atoms.center(vacuum=3.0)
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()
        internals.find_all_angles()

        # Get rotation coordinates
        rotations = internals.internals.get('rotations', [])

        for rot in rotations:
            grad_cell = rot.calc_cell_gradient(atoms)
            # Rotation orientation should not depend on cell
            assert_allclose(grad_cell, 0, atol=1e-10)

    def test_bond_cell_derivative_intramolecular(self):
        """Test intramolecular bond has zero cell derivative if not crossing PBC."""
        # Create H2 well within the cell (not crossing boundary)
        atoms = Atoms('H2', positions=[[2.0, 2.5, 2.5], [2.74, 2.5, 2.5]])
        atoms.set_cell([5.0, 5.0, 5.0])
        atoms.pbc = True

        internals = Internals(atoms)
        internals.find_all_bonds()

        bond = internals.internals['bonds'][0]

        # For intramolecular bond not crossing PBC, cell derivative should be zero
        # (the ncvec should be zero)
        grad_cell = bond.calc_cell_gradient(atoms)

        # Check that gradient is zero (bond doesn't cross boundary)
        assert_allclose(grad_cell, 0, atol=1e-10)

    def test_cell_jacobian_trics_rows_zero(self):
        """Test that TRICs rows in cell_jacobian are zero."""
        atoms = molecule('H2O')
        atoms.center(vacuum=3.0)
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()
        internals.find_all_angles()

        J_cell = internals.cell_jacobian()

        # The rows corresponding to translations and rotations should be zero
        # First ntrans rows are translations
        n_trans = internals.ntrans
        n_rot = internals.nrotations

        if n_trans > 0:
            trans_rows = J_cell[:n_trans, :]
            assert_allclose(trans_rows, 0, atol=1e-10)

        # Last nrotations rows are rotations (after other coords)
        if n_rot > 0:
            # Rotations come after: trans, bonds, angles, dihedrals, other
            rot_start = n_trans + internals.nbonds + internals.nangles + internals.ndihedrals + internals.nother
            rot_rows = J_cell[rot_start:rot_start + n_rot, :]
            assert_allclose(rot_rows, 0, atol=1e-10)


class TestCellConstraints:
    """Test cell optimization with various constraints."""

    def test_hydrostatic_only_dof_count(self, cu_fcc):
        """Test that hydrostatic constraint gives correct number of DOF."""
        atoms = cu_fcc(a=3.8)

        # Only allow diagonal elements
        cell_mask = np.eye(3, dtype=bool)

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=cell_mask,
            logfile=None,
        )

        # Check that only 3 cell DOF (diagonal elements)
        assert opt.pes.n_cell_dof == 3

    def test_isotropic_scaling(self, cu_fcc):
        """Test cell optimization with isotropic scaling only (1 DOF)."""
        atoms = cu_fcc(a=3.8)

        # Only allow uniform scaling (first diagonal element)
        cell_mask = np.zeros((3, 3), dtype=bool)
        cell_mask[0, 0] = True

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=cell_mask,
            logfile=None,
        )

        assert opt.pes.n_cell_dof == 1

    def test_no_shear_mask(self, cu_fcc):
        """Test that shear components can be masked out."""
        atoms = cu_fcc()

        # Allow all 9 components (full cell optimization)
        full_mask = np.ones((3, 3), dtype=bool)
        opt_full = Sella(
            atoms.copy(),
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=full_mask,
            logfile=None,
        )
        atoms.calc = EMT()  # Reset

        # Allow only diagonal (no shear)
        diag_mask = np.eye(3, dtype=bool)
        opt_diag = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=diag_mask,
            logfile=None,
        )

        assert opt_full.pes.n_cell_dof == 9
        assert opt_diag.pes.n_cell_dof == 3


class TestRefineInitialHessian:
    """Test the refine_initial_hessian option for cell optimization."""

    # NOTE: a test named test_refine_hessian_produces_nonzero_coupling used to
    # live here. It asserted only a shape, because on the bulk Cu primitive
    # cell n_internal == 0, so the internal/cell coupling block it named is
    # (0, 9) -- there was nothing to check. On a system that does have
    # internal coordinates the coupling block comes out exactly zero both
    # with and without refinement, so the behaviour the name claimed does not
    # occur. Its only real assertion is covered by
    # test_refine_hessian_cartesian_pes below.

    def test_refine_hessian_vs_default_different(self, cu_fcc):
        """Test that refined Hessian differs from default."""
        # Without refinement
        atoms1 = cu_fcc()
        pes_default = CellInternalPES(
            atoms1,
            Internals(atoms1),
            refine_initial_hessian=False,
        )

        # With refinement
        atoms2 = cu_fcc()
        pes_refined = CellInternalPES(
            atoms2,
            Internals(atoms2),
            refine_initial_hessian=True,
        )

        H_default = pes_default.H.B
        H_refined = pes_refined.H.B

        n_int = pes_default.n_internal

        # The cell-cell blocks should be different
        H_cell_default = H_default[n_int:, n_int:]
        H_cell_refined = H_refined[n_int:, n_int:]

        # Refined is computed from actual curvature, not the default guess
        assert not np.allclose(H_cell_default, H_cell_refined)

    def test_refine_hessian_with_internal_coords(self, two_water_crystal):
        """Refinement on a system that actually has internal coordinates.

        bulk Cu has n_internal == 0, so every other test in this class only
        exercises the cell block. A molecular crystal has both, which checks
        that refinement keeps the full Hessian well formed.
        """
        atoms = two_water_crystal()
        pes = CellInternalPES(
            atoms,
            Internals(atoms, allow_fragments=True),
            refine_initial_hessian=True,
        )

        n_int = pes.n_internal
        assert n_int > 0, "expected internal coordinates for a molecular crystal"

        H = pes.H.B
        assert H.shape == (pes.dim, pes.dim)
        assert np.all(np.isfinite(H))
        assert_allclose(H, H.T, atol=1e-10)

        # Refinement must have moved the cell block off the default guess.
        atoms_default = two_water_crystal()
        pes_default = CellInternalPES(
            atoms_default,
            Internals(atoms_default, allow_fragments=True),
            refine_initial_hessian=False,
        )
        assert not np.allclose(
            H[n_int:, n_int:], pes_default.H.B[n_int:, n_int:]
        )

    def test_refine_hessian_cartesian_pes(self, cu_fcc):
        """Test refine_initial_hessian with CellCartesianPES."""
        atoms = cu_fcc()

        pes = CellCartesianPES(
            atoms,
            refine_initial_hessian=True,
        )

        H = pes.H.B
        n_cart = pes.n_cart

        # Hessian should have correct shape
        assert H.shape == (pes.dim, pes.dim)

        # Cell-cell block should exist
        H_cell = H[n_cart:, n_cart:]
        assert H_cell.shape == (pes.n_cell_dof, pes.n_cell_dof)

    def test_refine_hessian_via_sella_api(self, cu_fcc):
        """Test refine_initial_hessian through Sella API."""
        atoms = cu_fcc()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            refine_initial_hessian=True,
            logfile=None,
        )

        # Should be CellInternalPES
        assert isinstance(opt.pes, CellInternalPES)

        # Hessian should be properly dimensioned
        H = opt.pes.H.B
        assert H.shape == (opt.pes.dim, opt.pes.dim)

    def test_refine_hessian_force_call_count(self, cu_fcc):
        """Test that refinement makes expected number of force calls."""
        atoms = cu_fcc()

        # Only allow diagonal cell DOF for fewer calls
        cell_mask = np.eye(3, dtype=bool)  # 3 DOF

        pes = CellInternalPES(
            atoms,
            Internals(atoms),
            cell_mask=cell_mask,
            refine_initial_hessian=True,
        )

        # Should have made 2 * n_cell_dof = 6 evaluations during init
        # (2 per cell DOF for central difference)
        assert pes.neval == 6


class TestRigidFragments:
    """Test rigid fragment mode for molecular crystal cell optimization."""

    def test_rigid_fragments_auto_detected(self, two_water_crystal):
        """Test that rigid_fragments is auto-detected from allow_fragments."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True
        assert hasattr(pes, 'fragment_groups')
        assert len(pes.fragment_groups) == 2  # Two water molecules

    def test_rigid_fragments_not_detected_without_fragments(self, cu_fcc):
        """Test that rigid_fragments is False when no translations exist."""
        atoms = cu_fcc()
        internals = Internals(atoms)

        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is False

    def test_rigid_fragments_explicit_override(self, two_water_crystal):
        """Test that rigid_fragments can be explicitly set."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        # Explicitly disable
        pes = CellInternalPES(atoms, internals, rigid_fragments=False)
        assert pes.rigid_fragments is False

    def test_fragment_groups_correct_atoms(self, two_water_crystal):
        """Test that fragment groups contain the correct atom indices."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)

        # Each group should have 3 atoms (water has O, H, H)
        for group in pes.fragment_groups:
            assert len(group) == 3

        # All atoms should be covered
        all_atoms = np.sort(np.concatenate(pes.fragment_groups))
        assert_allclose(all_atoms, np.arange(6))

    def test_compute_delta_r(self, two_water_crystal):
        """Test that _compute_delta_r gives positions relative to fragment CoM."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)
        delta_r = pes._compute_delta_r()

        # For each fragment, the mean of delta_r should be zero
        for group in pes.fragment_groups:
            assert_allclose(delta_r[group].mean(axis=0), 0, atol=1e-12)

    def test_rigid_fragment_cell_gradient_numerical(self, two_water_crystal, fd_pes_gradient):
        """Test rigid fragment cell gradient matches numerical finite difference.

        This is the key correctness test: the analytical cell gradient with
        rigid fragment mode should match the energy change when we actually
        move fragment CoMs to maintain fractional positions.
        """
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]

        # Numerical gradient via finite difference on cell parameters
        g_cell_numeric = fd_pes_gradient(pes, range(pes.n_internal, pes.dim), delta=1e-6)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_rigid_fragment_cell_gradient_nonorthogonal(self, two_water_crystal, fd_pes_gradient):
        """Test rigid fragment gradient with non-orthogonal cell."""
        atoms = two_water_crystal(_TRICLINIC_CELL)

        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]

        # Numerical gradient
        g_cell_numeric = fd_pes_gradient(pes, range(pes.n_internal, pes.dim), delta=1e-6)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_intramolecular_geometry_preserved_after_cell_step(self, two_water_crystal):
        """Test that intramolecular geometry is preserved after a cell-only step."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)

        # Record initial bond lengths and angles
        q0 = internals.calc()
        x0 = pes.get_x()

        # Apply a cell-only displacement (change cell params, keep internal targets)
        x_target = x0.copy()
        x_target[pes.n_internal:] += 0.01  # Small uniform cell strain

        pes.set_x(x_target)

        # Get new internal coordinates
        q1 = internals.calc()

        # Internal coords (bonds, angles) should be close to original
        # The internal coord solver handles the small residual
        n_trans = internals.ntrans
        # Skip translations (those change with cell), check bonds/angles
        assert_allclose(q0[n_trans:], q1[n_trans:], atol=1e-3)

    def test_monoatomic_rigid_fragments_trivial(self, cu_fcc):
        """For monoatomic crystals, Δr=0 so rigid_fragments correction vanishes."""
        atoms = cu_fcc()

        # Create internals with allow_fragments - each atom is its own "fragment"
        internals_frag = Internals(atoms, allow_fragments=True)
        has_translations = bool(internals_frag.internals.get('translations', []))

        if not has_translations:
            # No fragments detected for monoatomic - this is expected
            # Just verify rigid_fragments defaults to False
            pes = CellInternalPES(atoms, internals_frag)
            assert pes.rigid_fragments is False
            return

        # If fragments are detected, verify Δr=0 behavior
        pes_rigid = CellInternalPES(atoms, internals_frag, rigid_fragments=True)
        delta_r = pes_rigid._compute_delta_r()
        assert_allclose(delta_r, 0, atol=1e-12)

    def test_rigid_fragments_sella_integration(self, two_water_crystal):
        """Test rigid fragments through the Sella API."""
        atoms = two_water_crystal()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            allow_fragments=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellInternalPES)
        assert opt.pes.rigid_fragments is True

        # Take a few steps to verify no errors
        for _ in range(3):
            opt.step()

    def test_rotation_applied_on_shear(self, two_water_crystal):
        """Test that fragment atoms rotate under shear, not just translate."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)

        x0 = pes.get_x()
        pos0 = atoms.get_positions().copy()

        # Apply a pure shear cell displacement (off-diagonal log-deformation)
        x_shear = x0.copy()
        # Find an off-diagonal cell DOF (shear)
        cell_mask = pes.cell_mask
        offdiag_indices = []
        for idx, (i, j) in enumerate(zip(*np.where(cell_mask))):
            if i != j:
                offdiag_indices.append(idx)
        if not offdiag_indices:
            pytest.skip("No off-diagonal cell DOF available")

        shear_idx = offdiag_indices[0]
        x_shear[pes.n_internal + shear_idx] += 0.05

        pes.set_x(x_shear)
        pos_after = atoms.get_positions().copy()

        # For each fragment, check that the relative geometry changed orientation
        # (rotation applied) but bond lengths are preserved
        for group in pes.fragment_groups:
            dr_before = pos0[group] - pos0[group].mean(axis=0)
            dr_after = pos_after[group] - pos_after[group].mean(axis=0)

            # Bond lengths should be nearly preserved
            dists_before = np.linalg.norm(dr_before, axis=1)
            dists_after = np.linalg.norm(dr_after, axis=1)
            assert_allclose(dists_before, dists_after, atol=1e-3)

            # But the orientation should have changed (dr_after != dr_before)
            # The rotation under shear should produce a nonzero angular change
            if len(group) > 1:
                diff = dr_after - dr_before
                assert np.max(np.abs(diff)) > 1e-6, (
                    "Fragment atoms should rotate under shear deformation"
                )

    def test_gradient_numerical_large_shear(self, two_water_crystal, fd_pes_gradient):
        """Test gradient correctness with a heavily sheared cell.

        This stress-tests the rotation correction at large deformation
        where the rotation component R deviates significantly from identity.
        """
        atoms = two_water_crystal(_SHEARED_CELL)

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]

        # Numerical gradient via finite difference
        g_cell_numeric = fd_pes_gradient(pes, range(pes.n_internal, pes.dim), delta=1e-6)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_gradient_after_cell_step(self, two_water_crystal, fd_pes_gradient):
        """Test gradient correctness after the cell has already been deformed.

        After a cell step, F != I and the rotation correction is nonzero.
        Verify gradient still matches numerical FD at the deformed geometry.
        """
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)

        # First, apply a cell deformation to move away from F = I
        x0 = pes.get_x()
        x_deformed = x0.copy()
        # Apply a mix of diagonal and off-diagonal strains
        n_cell = pes.n_cell_dof
        for i in range(n_cell):
            x_deformed[pes.n_internal + i] += 0.02 * ((-1)**i)
        pes.set_x(x_deformed)

        # Now verify gradient at this deformed state
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]

        g_cell_numeric = fd_pes_gradient(pes, range(pes.n_internal, pes.dim), delta=1e-6)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    @staticmethod
    def _full_gradient_fd(fd_pes_gradient, pes, delta_int=1e-5, delta_cell=1e-6):
        """Analytical and numerical gradient over ALL DOFs.

        Internal and cell coordinates need different step sizes, so the step
        is chosen per component.
        """
        _, g_analytical = pes.eval()
        g_numeric = fd_pes_gradient(
            pes,
            delta=lambda i: delta_cell if i >= pes.n_internal else delta_int,
        )
        return g_analytical, g_numeric

    def test_full_gradient_numerical_rigid_fragments(self, two_water_crystal, fd_pes_gradient):
        """Verify analytical gradient matches FD for ALL DOFs (internal + cell)."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        g_analytical, g_numeric = self._full_gradient_fd(fd_pes_gradient, pes)
        assert_allclose(g_analytical, g_numeric, atol=1e-4, rtol=1e-3)

    def test_full_gradient_numerical_triclinic(self, two_water_crystal, fd_pes_gradient):
        """Verify full gradient with non-orthogonal cell (rotation correction active)."""
        atoms = two_water_crystal(_TRICLINIC_CELL)

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        g_analytical, g_numeric = self._full_gradient_fd(fd_pes_gradient, pes)
        assert_allclose(g_analytical, g_numeric, atol=1e-4, rtol=1e-3)

    def test_full_gradient_numerical_after_deformation(self, two_water_crystal, fd_pes_gradient):
        """Verify full gradient after cell deformation (F != I)."""
        atoms = two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)

        x0 = pes.get_x()
        x_deformed = x0.copy()
        for i in range(pes.n_cell_dof):
            x_deformed[pes.n_internal + i] += 0.02 * ((-1)**i)
        pes.set_x(x_deformed)

        g_analytical, g_numeric = self._full_gradient_fd(fd_pes_gradient, pes)
        assert_allclose(g_analytical, g_numeric, atol=1e-4, rtol=1e-3)

    def test_rotation_correction_nonzero_under_shear(self, two_water_crystal):
        """Verify that the rotation correction to the gradient is nonzero
        when the cell is sheared (so R != I in polar decomposition)."""
        atoms = two_water_crystal(7.0)

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)

        # Apply shear to move away from R = I
        x0 = pes.get_x()
        x_sheared = x0.copy()
        cell_mask = pes.cell_mask
        offdiag_indices = []
        for idx, (i, j) in enumerate(zip(*np.where(cell_mask))):
            if i != j:
                offdiag_indices.append(idx)
        if not offdiag_indices:
            pytest.skip("No off-diagonal cell DOF")

        x_sheared[pes.n_internal + offdiag_indices[0]] += 0.05
        pes.set_x(x_sheared)

        # At this point F should have a nontrivial rotation component
        F = pes._get_deformation_gradient()
        R, U = polar(F)
        assert not np.allclose(R, np.eye(3), atol=1e-6), (
            "R should differ from identity after shear"
        )


class TestNiggliReduction:
    """Tests for periodic Niggli reduction during cell optimization."""

    def test_niggli_triggers_on_skewed_cell_cartesian(self, cu_fcc):
        """CellCartesianPES.maybe_niggli_reduce fires when angles are extreme."""
        atoms = cu_fcc()

        pes = CellCartesianPES(atoms, eta=1e-4)

        # Shear the cell to create a skewed angle
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 3.0  # Large off-diagonal shear
        atoms.set_cell(cell, scale_atoms=True)
        pes.orig_cell = atoms.get_cell().array.copy()

        # Shear further so cell relative to orig_cell is identity but
        # the cell itself is skewed
        angles = atoms.get_cell().angles()
        assert min(angles) < 60.0 or max(angles) > 120.0, (
            f"Test setup: cell should be skewed, got angles {angles}"
        )

        reduced = pes.maybe_niggli_reduce()
        assert reduced, "Niggli reduction should have been triggered"

        # After reduction, angles should be closer to 90
        new_angles = atoms.get_cell().angles()
        new_max_dev = max(abs(a - 90.0) for a in new_angles)
        assert new_max_dev <= 30.0 + 1e-10, (
            f"After Niggli reduction, angles should be near 90, got {new_angles}"
        )

        # orig_cell should be updated
        assert_allclose(pes.orig_cell, atoms.get_cell().array)

    def test_niggli_triggers_on_skewed_cell_internal(self, methane_box):
        """CellInternalPES.maybe_niggli_reduce fires when angles are extreme."""
        atoms = methane_box()
        atoms.calc = EMT()

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals=internals, eta=1e-4)

        # Shear the cell
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 6.0
        atoms.set_cell(cell, scale_atoms=False)
        pes.orig_cell = atoms.get_cell().array.copy()

        angles = atoms.get_cell().angles()
        assert min(angles) < 60.0 or max(angles) > 120.0

        reduced = pes.maybe_niggli_reduce()
        assert reduced

        new_angles = atoms.get_cell().angles()
        new_max_dev = max(abs(a - 90.0) for a in new_angles)
        assert new_max_dev < 30.0, f"Angles after reduction: {new_angles}"

    def test_niggli_does_not_trigger_on_good_cell(self, cu_fcc):
        """No reduction when cell angles are fine."""
        # Cubic cell — angles are all 60 or 90 depending on conventional vs
        # primitive. Use the conventional cell to get 90 degree angles.
        atoms2 = cu_fcc(cubic=True)
        pes2 = CellCartesianPES(atoms2, eta=1e-4)

        reduced = pes2.maybe_niggli_reduce()
        assert not reduced, "Should not reduce a cubic cell"

    def test_niggli_transforms_hessian_cell_block(self, cu_fcc):
        """After Niggli reduction, cell block of Hessian is transformed."""
        atoms = cu_fcc(cubic=True)

        pes = CellCartesianPES(atoms, eta=1e-4)
        n = pes.n_cart

        # Set Hessian to identity for cell block to track the transformation
        H = pes.H.B.copy()
        H[n:, n:] = np.eye(pes.n_cell_dof)
        H[:n, n:] = 0.0
        H[n:, :n] = 0.0
        pes.set_H(H, initialized=True)

        # Shear the cell to trigger reduction
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 5.0
        atoms.set_cell(cell, scale_atoms=True)
        pes.orig_cell = atoms.get_cell().array.copy()

        pes.maybe_niggli_reduce()

        H_new = pes.H.B
        # Cell block should be transformed (T^T @ I @ T = T^T @ T), symmetric
        H_cell = H_new[n:, n:]
        assert_allclose(H_cell, H_cell.T, atol=1e-10)
        # Should not be identity (transformation is non-trivial for skewed cell)
        assert not np.allclose(H_cell, np.eye(pes.n_cell_dof), atol=0.1)
        # Cartesian block should be preserved
        assert_allclose(H_new[:n, :n], H[:n, :n])

    def test_niggli_in_sella_step(self, cu_fcc):
        """Niggli reduction integrates with Sella.step() without crashing.

        Smoke test: the assertion is that no exception is raised while
        stepping a cell skewed badly enough to trigger reduction mid-run.
        """
        atoms = cu_fcc(cubic=True)

        opt = Sella(atoms, order=0, optimize_cell=True, niggli=True,
                    logfile=None)

        # Shear the cell to extreme angles
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 5.0
        atoms.set_cell(cell, scale_atoms=True)
        opt.pes.orig_cell = atoms.get_cell().array.copy()

        # Run a few steps — should not crash even if Niggli fires
        for _ in range(3):
            opt.step()

# ===========================================================================
# IRC
# ===========================================================================

# A first-order saddle of the 6-atom Lennard-Jones cluster (one imaginary mode).
_LJ6_TS = [[-0.819098, -0.456198, -0.221436], [0.233188, -0.375765, -0.567537],
           [-0.259168, 0.501054, -0.063452], [-0.504031, -0.163298, 0.804587],
           [0.800850, 0.577070, -0.410659], [0.548259, -0.082862, 0.458498]]


def _ts():
    atoms = Atoms('Ar6', positions=_LJ6_TS)
    atoms.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)
    return atoms


def test_irc_takes_first_step_from_converged_ts():
    """IRC must apply the initial displacement even when the input TS already
    has |F| < fmax.

    Regression for the ASE >= 3.28 optimizer loop, which checks
    ``gradient_converged()`` rather than ``converged()``. IRC only overrode
    ``converged()`` (where the ``first``-step guard lives), so an IRC started
    from a converged TS "converged" at 0 steps and returned the TS unchanged.
    """
    x0 = _ts().get_positions()
    ends = {}
    for direction in ('forward', 'reverse'):
        atoms = _ts()
        irc = IRC(atoms, dx=0.1, keep_going=True, logfile=None)
        irc.run(fmax=0.05, steps=100, direction=direction)
        assert irc.nsteps > 0, f"{direction} IRC took no steps"
        assert np.abs(atoms.get_positions() - x0).max() > 1e-3, \
            f"{direction} IRC did not leave the TS"
        ends[direction] = atoms.get_positions()

    # forward and reverse must descend to opposite sides of the TS
    assert np.abs(ends['forward'] - ends['reverse']).max() > 1e-2

# ===========================================================================
# Integration: linear molecules
# ===========================================================================

def test_n2_cartesian():
    """N2 optimization in Cartesian coordinates."""
    atoms = molecule('N2')
    atoms.calc = EMT()
    opt = Sella(atoms, order=0, logfile=None)
    opt.run(fmax=0.01, steps=100)

    assert opt.converged()
    # The failure mode being guarded against is NaN, not just non-convergence.
    assert np.all(np.isfinite(atoms.get_positions()))
    assert np.all(np.isfinite(atoms.get_forces()))


def test_n2_internal():
    """N2 optimization with internal coordinates (TRICs)."""
    atoms = molecule('N2')
    atoms.calc = EMT()
    opt = Sella(atoms, order=0, internal=True, logfile=None)
    opt.run(fmax=0.01, steps=100)

    assert opt.converged()
    assert np.all(np.isfinite(atoms.get_positions()))
    assert np.all(np.isfinite(atoms.get_forces()))

# ===========================================================================
# Integration: Morse cluster
# ===========================================================================

@pytest.mark.parametrize(
    "internal,order",
    [
        (False, 0),
        (False, 1),
        (True, 0),
        (True, 1),
    ],
)
def test_morse_cluster(internal, order, trajectory=None):
    rng = np.random.RandomState(4)

    nat = 4
    atoms = Atoms(['Xe'] * nat, rng.normal(size=(nat, 3), scale=3.0))
    # parameters from DOI: 10.1515/zna-1987-0505
    atoms.calc = MorsePotential(alpha=226.9 * kB, r0=4.73, rho0=4.73*1.099)

    cons = Constraints(atoms)
    cons.fix_translation()
    cons.fix_rotation()

    opt = Sella(
        atoms,
        order=order,
        internal=internal,
        trajectory=trajectory,
        gamma=1e-3,
        constraints=cons,
    )
    opt.run(fmax=1e-3)

    Ufree = opt.pes.get_Ufree()
    np.testing.assert_allclose(opt.pes.get_g() @ Ufree, 0, atol=5e-3)
    opt.pes.diag(gamma=1e-16)
    H = opt.pes.get_HL().project(Ufree)
    assert np.sum(H.evals < 0) == order, H.evals

# ===========================================================================
# Integration: TIP3P water cluster
# ===========================================================================

water = molecule('H2O')
water.set_distance(0, 1, rOH)
water.set_distance(0, 2, rOH)
water.set_angle(1, 0, 2, angleHOH)
a = 3.106162559099496
rng = np.random.RandomState(0)

atoms_ref = Atoms()
for offsets in product(*((0, 1),) * 3):
    atoms = water.copy()
    for axis in ['x', 'y', 'z']:
        atoms.rotate(rng.random() * 360, axis)
    atoms.positions += a * np.asarray(offsets)
    atoms_ref += atoms


# internal=True builds a coordinate basis of only 30 coordinates for this
# 72-DOF cluster (sella warns "30 coords found! Expected 72"), and all 30 of
# them are the rigid-water constraints. Ufree therefore comes out (30, 0) --
# an empty free subspace -- so the optimizer has no direction to move in and
# reports convergence immediately. The Cartesian path correctly yields
# 72 - 30 = 42 free DOF. Until the internal basis covers the full space these
# two cases cannot pass; order=0 previously *appeared* to pass only because
# every assertion is vacuously true on an empty spectrum.
_EMPTY_FREE_SUBSPACE = pytest.mark.xfail(
    reason="internal coords span only the 30 constraints; Ufree is (30, 0)",
    strict=True,
)


@pytest.mark.parametrize("internal,order",
                         [pytest.param(True, 0, marks=_EMPTY_FREE_SUBSPACE),
                          (False, 0),
                          pytest.param(True, 1, marks=_EMPTY_FREE_SUBSPACE),
                          (False, 1),
                          ])
def test_water_dimer(internal, order, tmp_path):
    rng = np.random.RandomState(1)

    atoms = atoms_ref.copy()
    atoms.calc = TIP3P()
    atoms.rattle(0.01, rng=rng)

    nwater = len(atoms) // 3
    cons = Constraints(atoms)
    for i in range(nwater):
        cons.fix_bond((3 * i, 3 * i + 1), target=rOH)
        cons.fix_bond((3 * i, 3 * i + 2), target=rOH)
        cons.fix_angle((3 * i + 1, 3 * i, 3 * i + 2), target=angleHOH)

    # Remove net translation and rotation
    try:
        cons.fix_translation()
    except DuplicateConstraintError:
        pass
    try:
        cons.fix_rotation()
    except DuplicateConstraintError:
        pass

    sella_kwargs = dict(
        order=order,
        trajectory=str(tmp_path / 'test.traj'),
        eta=1e-6,
        delta0=1e-2,
    )
    if internal:
        sella_kwargs['internal'] = Internals(
            atoms, cons=cons, allow_fragments=True
        )
    else:
        sella_kwargs['constraints'] = cons
    opt = Sella(atoms, **sella_kwargs)

    opt.delta = 0.05
    opt.run(fmax=1e-3)

    atoms.rattle()
    opt.run(fmax=1e-3)

    Ufree = opt.pes.get_Ufree()
    # Without this guard every assertion below is vacuously true when the
    # free subspace is empty, and the test passes while checking nothing.
    assert Ufree.shape[1] > 0, f"empty free subspace: Ufree {Ufree.shape}"

    g = opt.pes.get_g() @ Ufree
    np.testing.assert_allclose(g, 0, atol=1e-3)
    opt.pes.diag(gamma=1e-16)
    H = opt.pes.get_HL().project(Ufree)
    assert np.sum(H.evals < 0) == order, H.evals

# ===========================================================================
# Modified Gram-Schmidt
# ===========================================================================
# What is left of the upstream tests/utilities/test_math.py. The rest of that
# file covered Cython this package does not build: pseudo_inverse, which
# nothing in the library ever called, and the cdef helpers behind it.

@pytest.mark.parametrize(
    "n,mx,my,eps1,eps2,maxiter",
    [
        (3, 2, 1, 1e-15, 1e-6, 100),
        (100, 50, 25, 1e-15, 1e-6, 100),
    ],
)
def test_modified_gram_schmidt(n, mx, my, eps1, eps2, maxiter):
    rng = np.random.RandomState(2)

    tol = dict(atol=1e-6, rtol=1e-6)
    mgskw = dict(eps1=eps1, eps2=eps2, maxiter=maxiter)

    X = get_matrix(n, mx, rng=rng)

    Xout1 = modified_gram_schmidt(X, **mgskw)
    _, nxout1 = Xout1.shape

    np.testing.assert_allclose(Xout1.T @ Xout1, np.eye(nxout1), **tol)
    np.testing.assert_allclose(
        np.linalg.det(X.T @ X), np.linalg.det(X.T @ Xout1) ** 2, **tol
    )

    Y = get_matrix(n, my, rng=rng)
    Xout2 = modified_gram_schmidt(X, Y, **mgskw)
    _, nxout2 = Xout2.shape

    np.testing.assert_allclose(Xout2.T @ Xout2, np.eye(nxout2), **tol)
    np.testing.assert_allclose(Xout2.T @ Y, np.zeros((nxout2, my)), **tol)

    X[:, 1] = X[:, 0]

    Xout3 = modified_gram_schmidt(X, **mgskw)
    _, nxout3 = Xout3.shape
    assert nxout3 == nxout1 - 1

    np.testing.assert_allclose(Xout2.T @ Xout2, np.eye(nxout2), **tol)


def test_modified_gram_schmidt_leaves_its_input_alone():
    """Upstream took copies internally; the NumPy version has to as well."""
    X = np.random.RandomState(7).normal(size=(20, 5))
    X0 = X.copy()
    modified_gram_schmidt(X)
    np.testing.assert_array_equal(X, X0)


def test_modified_gram_schmidt_passes_an_empty_basis_straight_back():
    empty = np.zeros((7, 0))
    assert modified_gram_schmidt(empty) is empty


def test_modified_gram_schmidt_rejects_mismatched_shapes():
    X = np.random.RandomState(8).normal(size=(10, 3))
    Y = np.random.RandomState(9).normal(size=(9, 2))
    with pytest.raises(ValueError, match="10 rows but Y has 9"):
        modified_gram_schmidt(X, Y)


def test_modified_gram_schmidt_gives_up_rather_than_spinning():
    """One sweep is never enough to converge a column, so this must raise."""
    X = np.random.RandomState(10).normal(size=(10, 2))
    with pytest.raises(RuntimeError, match="MGS failed"):
        modified_gram_schmidt(X, maxiter=1)


@pytest.mark.skipif(
    importlib.util.find_spec("sella") is None,
    reason="needs an upstream sella to compare against",
)
@pytest.mark.parametrize(
    "n,mx,my,degenerate",
    [
        (3, 2, 1, False),
        (100, 50, 25, False),
        (10, 4, 2, True),
        (50, 50, None, False),
        (200, 10, 5, True),
        (12, 8, 3, False),
    ],
)
def test_modified_gram_schmidt_matches_the_cython_it_replaced(n, mx, my, degenerate):
    """The reimplementation has to agree with the compiled original.

    Skipped unless upstream sella is importable, which it will not be on a
    plain install of this package -- the point of the migration was to stop
    needing it. It runs where a developer still has a checkout to hand.
    """
    from sella.utilities.math import modified_gram_schmidt as cython_mgs

    rng = np.random.RandomState(11)
    X = get_matrix(n, mx, rng=rng)
    Y = get_matrix(n, my, rng=rng) if my else None
    if degenerate:
        X[:, -1] = X[:, 0]

    expected = cython_mgs(X.copy(), None if Y is None else Y.copy())
    actual = modified_gram_schmidt(X.copy(), None if Y is None else Y.copy())

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


# ===========================================================================
# Import-time environment
# ===========================================================================
# This module is imported by reactiontools/__init__.py, so anything it does at
# import time it does to every user of the package.


def test_an_unwritable_cache_directory_does_not_break_the_import(
    tmp_path: Path,
) -> None:
    """Regression: an unwritable home made ``import reactiontools`` fail.

    The JAX compilation cache only saves tracing time, but the ``os.makedirs``
    that created it was unguarded, so a read-only home -- a compute node, a
    container without a writable HOME -- raised PermissionError out of the
    package __init__ and took every workflow down with it.

    Run in a subprocess because the module under test is already imported.
    """
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    env = {
        **os.environ,
        "HOME": str(home),
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
    }
    env.pop("JAX_COMPILATION_CACHE_DIR", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import reactiontools;"
            "from reactiontools import tools_sella;"
            "import os;"
            "print(tools_sella._JAX_CACHE_DIR);"
            "print('JAX_COMPILATION_CACHE_DIR' in os.environ)",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(reactiontools.__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    cache_dir, var_set = result.stdout.split()[-2:]
    # The cache is off rather than pointed at a directory that is not there,
    # and the variable this module would have set is not left behind.
    assert cache_dir == "None"
    assert var_set == "False"
