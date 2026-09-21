"""Geodesic interpolation: a smooth first guess at a reaction path.

The band :func:`reactiontools.tools_reaction.prepare_neb` starts from, and what
:func:`reactiontools.tools_reaction.quick_guess_path` and
:func:`reactiontools.tools_reaction.quick_guess_ts` hand back. Given two end
states -- or more, when an intermediate is already known --
:func:`geodesic_interpolate` returns a path of images that is short in a metric
of scaled inter-atomic distances rather than in Cartesian space, which is what
keeps atoms from passing through each other on the way across.

It runs in two stages. :func:`redistribute` builds a raw path with the right
number of images, bisecting the largest gap until there are enough of them and
dropping the least costly image while there are too many. :class:`Geodesic`
then smooths that path by minimising its length in the internal metric, holding
the two end points fixed. Up to :data:`SWEEP_ABOVE_N_ATOMS` atoms the whole
path is optimised at once; past that the images are smoothed one at a time,
sweeping back and forth, as SciPy's least-squares optimisers slow down badly
with problem size.

The coordinates are redundant internals: every atom pair within a cut-off, plus
every pair within three bonds of each other however far apart they are, each
distance passed through a scaling function that decays with distance --
:func:`morse_scaler` by default. The optimisation is carried out in Cartesians
all the same, because an arbitrary set of redundant internal coordinate values
need not correspond to any real geometry.

This module is a leaf of the package: it reaches ``ase``, ``numpy`` and
``scipy``, and nothing else here beyond
:func:`reactiontools.tools_io.write_xyz_frame`. Its alignment helpers --
:func:`align_geom`, :func:`align_path` and :func:`align_path_to` -- exist for
that reason. They are the bare Kabsch fit the interpolation needs between
adjacent images; :func:`reactiontools.tools_geometry.align_atom_sets` is the
fuller one, with weights, subset selection and validation, for superposing two
structures you already have.

The interpolation itself is not periodic. The internal coordinates are plain
inter-atomic distances with no minimum image convention, so a bond that crosses
a cell boundary is not handled. A periodic path is nonetheless moved back onto
the frame of reference it arrived in, since a cell only says where the atoms
are in the frame it was given with.

Derived from `geodesic-interpolate
<https://github.com/virtualzx-nad/geodesic-interpolate>`_ by Xiaolei Zhu, MIT
licensed; the copyright notice travels with this package in ``LICENSE``. It
came by way of the fork at
https://github.com/LouieSlocombe/geodesic_interpolate,
commit ``1e37b32aab77bcab0273e806b5a0a35df81160dd`` of 20 August 2026, which is
the tree to diff this file against and the one to re-sync from. Cite
``zhu2019geodesic`` when you use it -- see ``CITATIONS.bib``.
"""

import os
from collections.abc import Callable, Iterable
from itertools import pairwise
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from scipy.optimize import least_squares
from scipy.sparse import bmat, coo_matrix, csr_matrix, identity, triu, vstack
from scipy.spatial import KDTree

from .tools_io import write_xyz_frame

__all__ = [
    "SWEEP_ABOVE_N_ATOMS",
    "Geodesic",
    "align_geom",
    "align_path",
    "align_path_to",
    "compute_rij",
    "compute_wij",
    "elu_scaler",
    "from_ase_atoms",
    "geodesic_interpolate",
    "get_bond_list",
    "morse_scaler",
    "read_xyz",
    "redistribute",
    "to_ase_atoms",
    "write_xyz",
]

#: System size, in atoms, above which :func:`geodesic_interpolate` sweeps the
#: images one at a time rather than optimising the whole path at once. SciPy's
#: least-squares optimisers slow down faster than linearly in the number of
#: degrees of freedom, so past this the sweep is the cheaper of the two.
SWEEP_ABOVE_N_ATOMS = 35


def _kabsch_rotation(ref_centered: np.ndarray, geom_centered: np.ndarray) -> np.ndarray:
    """Find the rotation that best takes one centred geometry onto another.

    The SVD of the covariance matrix of the two geometries gives the optimal
    rotation. Both arguments must already have their centres at the origin.

    Parameters
    ----------
    ref_centered : numpy.ndarray
        The centred reference geometry to rotate towards.
    geom_centered : numpy.ndarray
        The centred geometry to be rotated.

    Returns
    -------
    numpy.ndarray
        The rotation matrix, to be applied on the right of a set of centred
        coordinates.
    """
    cov = np.dot(geom_centered.T, ref_centered)
    v, _, w = np.linalg.svd(cov)

    # A negative determinant means the SVD produced a reflection rather than a
    # rotation. Flipping the least significant axis gives a proper rotation.
    if np.linalg.det(v) * np.linalg.det(w) < 0.0:
        v[:, -1] *= -1

    return np.dot(v, w)


def align_geom(
    ref_geom: np.ndarray,
    geom: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find the rigid-body motion that maximally overlaps two geometries.

    Implemented with the Kabsch algorithm: both geometries are centred, the SVD
    of their covariance matrix gives the optimal rotation, and the result is
    moved back onto the centre of the reference.

    Every atom counts equally and all of them are used. See
    :func:`reactiontools.tools_geometry.align_atom_sets` for the weighted,
    subset-aware superposition of two structures.

    Parameters
    ----------
    ref_geom : numpy.ndarray
        The reference geometry to rotate towards, of shape ``(n_atoms, 3)``.
    geom : numpy.ndarray
        The geometry to be rotated and shifted, of the same shape.

    Returns
    -------
    rmsd : float
        Root-mean-squared difference between the rotated geometry and the
        reference.
    aligned_geom : numpy.ndarray
        The rotated geometry that maximally overlaps the reference.
    """
    center = np.mean(ref_geom, axis=0)
    ref_geom_centered = ref_geom - center
    geom_centered = geom - np.mean(geom, axis=0)

    rotation_matrix = _kabsch_rotation(ref_geom_centered, geom_centered)
    aligned_geom = np.dot(geom_centered, rotation_matrix) + center
    rmsd = np.sqrt(np.mean((aligned_geom - ref_geom) ** 2))

    return rmsd, aligned_geom


def align_path(path: np.ndarray | list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Rotate and translate the images of a path to minimise movement along it.

    The first image is shifted so its geometric centre sits at the origin, and
    each following image is aligned against the one before it, which leaves
    every image centred on the origin as well.

    Parameters
    ----------
    path : array_like
        Sequence of geometries, of shape ``(n_images, n_atoms, 3)``.

    Returns
    -------
    max_rmsd : float
        Largest RMSD between any two adjacent images after alignment.
    path : numpy.ndarray
        The aligned path. This is a copy, so the input is left untouched.
    """
    path = np.array(path, dtype=float)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0.0
    for g, next_g in pairwise(path):
        rmsd, aligned_geom = align_geom(g, next_g)
        next_g[:] = aligned_geom
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_path_to(ref_geom: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Move a whole path rigidly so its first image sits on a reference.

    :func:`align_path` leaves the path centred on the origin and rotated onto
    its own first image, which is a frame of the optimiser's choosing rather
    than the one the caller handed in. That is immaterial for an isolated
    molecule, but a unit cell only describes where the atoms are in the frame
    it came with, so a periodic path has to be put back before its cell means
    anything.

    One rotation and translation, taken from the first image, is applied to
    every image, so the shape of the path and its length are untouched.

    Parameters
    ----------
    ref_geom : numpy.ndarray
        Geometry to put the first image back onto, of shape ``(n_atoms, 3)``.
    path : numpy.ndarray
        The path to move, of shape ``(n_images, n_atoms, 3)``.

    Returns
    -------
    numpy.ndarray
        The path in the reference geometry's frame. This is a copy.
    """
    path = np.asarray(path, dtype=float)
    center = np.mean(ref_geom, axis=0)
    origin = np.mean(path[0], axis=0)

    rotation_matrix = _kabsch_rotation(ref_geom - center, path[0] - origin)
    return np.dot(path - origin, rotation_matrix) + center


def _pairs_within_three_bonds(
    tree: KDTree,
    n_atoms: int,
    bond_threshold: float,
) -> list[tuple[int, int]]:
    """List the atom pairs separated by three or fewer bonds in one geometry.

    Every such pair is a bonded pair with an optional extra bond tacked on at
    either end, so the answer is the sparsity pattern of ``(A + I) A (A + I)``
    for the bond adjacency ``A``. Phrasing it as two sparse matrix products
    keeps the cost down on large molecules, where walking the neighbour lists
    in Python does not scale.

    Parameters
    ----------
    tree : scipy.spatial.KDTree
        KD-tree of the geometry to work from.
    n_atoms : int
        Number of atoms in the geometry.
    bond_threshold : float
        Distance below which two atoms count as bonded.

    Returns
    -------
    list of tuple of int
        The ``(i, j)`` pairs, with ``i < j``.
    """
    bonded = tree.query_pairs(bond_threshold, output_type="ndarray")
    if len(bonded) == 0:
        return []
    # The adjacency matrix needs both directions of every bond
    rows = np.concatenate([bonded[:, 0], bonded[:, 1]])
    cols = np.concatenate([bonded[:, 1], bonded[:, 0]])
    adjacency = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_atoms, n_atoms)
    ).tocsr()
    with_self = adjacency + identity(n_atoms, format="csr")
    reach = triu(with_self @ adjacency @ with_self, k=1).tocoo()
    return list(zip(reach.row.tolist(), reach.col.tolist()))


def get_bond_list(
    geom: np.ndarray | list[np.ndarray],
    atoms: list[str] | None = None,
    threshold: float = 4.0,
    min_neighbors: int = 4,
    snapshots: int = 30,
    bond_threshold: float = 1.8,
    enforce: Iterable[tuple[int, int]] = (),
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Get the list of atom pairs that define the internal coordinate system.

    Samples images from the path and collects every pair of atoms that comes
    within ``threshold`` of each other in any of them. Pairs linked by three or
    fewer bonds are always included, as are the pairs given in ``enforce``, and
    further pairs are added for any atom left with fewer than ``min_neighbors``
    neighbours.

    Parameters
    ----------
    geom : array_like
        A single geometry or a path of them. Anything that is not already
        ``(n_images, n_atoms, 3)`` is promoted to that shape.
    atoms : list of str, optional
        Element symbols, used to look up covalent radii. When omitted, every
        pair is given a nominal equilibrium distance of 2.0 angstrom instead.
    threshold : float, optional
        Distance cut-off for including a pair in the coordinate system.
    min_neighbors : int, optional
        Minimum number of pairs each atom must take part in. Atoms below this
        get their nearest neighbours added.
    snapshots : int, optional
        Maximum number of images to sample. Keeps the cost down when the path
        is long and the atoms numerous.
    bond_threshold : float, optional
        Distance below which two atoms count as bonded, used to work out which
        pairs are within three bonds of each other.
    enforce : iterable of tuple of int, optional
        Pairs to include regardless of how far apart the atoms are.
    rng : numpy.random.Generator, optional
        Random source for choosing which images to sample. Defaults to a fresh
        unseeded generator. Pass a seeded one to keep the choice reproducible
        without disturbing anyone else's random state.

    Returns
    -------
    rij_list : list of tuple of int
        The ``(i, j)`` atom pairs making up the coordinates.
    re : numpy.ndarray
        Equilibrium distance for each pair, taken as the sum of the two
        covalent radii.
    """
    # Type casting and value checks on the input parameters
    geom = np.asarray(geom, dtype=float)
    if len(geom.shape) < 3:
        # A single geometry, or a flattened one, is promoted to 3d
        geom = geom.reshape(1, -1, 3)
    n_atoms = geom.shape[1]
    min_neighbors = min(min_neighbors, n_atoms - 1)
    if rng is None:
        rng = np.random.default_rng()

    # Always look at both end points, plus a random selection of the images
    # between them, so a long path costs no more to analyse than a short one
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(
            rng.choice(np.arange(1, len(geom) - 1), snapshots - 2, replace=False)
        )
    # Build the neighbour list for each sampled image and merge them together
    rij_set = set(enforce)
    for image in images:
        tree = KDTree(geom[image])
        rij_set.update(
            map(tuple, tree.query_pairs(threshold, output_type="ndarray").tolist())
        )
        # Anything within three bonds is included whatever the distance
        rij_set.update(_pairs_within_three_bonds(tree, n_atoms, bond_threshold))
    rij_list = sorted(rij_set)
    # Count how many pairs each atom appears in, so `min_neighbors` can be checked
    pairs = np.asarray(rij_list, dtype=int).reshape(-1, 2)
    count = np.bincount(pairs.ravel(), minlength=n_atoms)
    # Top up any under-connected atom with its nearest neighbours in the final
    # geometry. Atoms only ever gain neighbours here, so querying the whole
    # under-connected set in one go is safe; the count is still re-checked in
    # turn, because an atom may have been brought up to `min_neighbors` by an
    # earlier atom's additions.
    under_connected = np.flatnonzero(count < min_neighbors)
    if len(under_connected):
        tree = KDTree(geom[-1])
        _, neighbors = tree.query(geom[-1, under_connected], k=min_neighbors + 1)
        neighbors = np.asarray(neighbors).reshape(len(under_connected), -1)
        for idx, nearest in zip(under_connected.tolist(), neighbors.tolist()):
            if count[idx] >= min_neighbors:
                continue
            for i in nearest:
                if i == idx:
                    continue
                pair = (i, idx) if i < idx else (idx, i)
                if pair in rij_set:
                    continue
                else:
                    rij_set.add(pair)
                    rij_list.append(pair)
                    count[i] += 1
                    count[idx] += 1
        pairs = np.asarray(rij_list, dtype=int).reshape(-1, 2)
    if atoms is None:
        re = np.full(len(rij_list), 2.0)
    else:
        atom_numbers = [atomic_numbers[atom.capitalize()] for atom in atoms]
        radius = np.array([covalent_radii[num] for num in atom_numbers])
        re = radius[pairs[:, 0]] + radius[pairs[:, 1]]
    return rij_list, re


# Index bookkeeping derived from a pair list. Building it costs about as much
# as one evaluation, and the same pair list is used for every image over the
# whole run, so the last few are kept around. A strong reference to the list is
# held alongside its `id`, which is what makes the identity check sound: the
# list cannot be collected and have its address handed to something else while
# the entry lives.
_PAIR_INDEX_CACHE: dict[
    tuple[int, int, int],
    tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
] = {}
_PAIR_INDEX_CACHE_SIZE = 8


def _pair_index(
    rij_list: list[tuple[int, int]],
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build, or look up, the index arrays describing a pair list.

    Parameters
    ----------
    rij_list : list of tuple of int
        Indices of the atom pairs.
    n_atoms : int
        Number of atoms, which sets the width of the B matrix.

    Returns
    -------
    pairs : numpy.ndarray
        The pair list as an ``(n_pairs, 2)`` integer array.
    indptr, indices : numpy.ndarray
        CSR skeleton of the B matrix. Every distance depends on six Cartesian
        components, three for each of its atoms.
    sign : numpy.ndarray
        ``+1`` where the pair is stored low index first, ``-1`` otherwise, so
        the gradients can be written in the column order CSR requires.
    """
    key = (id(rij_list), len(rij_list), n_atoms)
    cached = _PAIR_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is rij_list:
        return cached[1:]

    pairs = np.asarray(rij_list, dtype=int).reshape(-1, 2)
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    indices = np.empty((len(pairs), 6), dtype=np.int32)
    indices[:, 0:3] = 3 * lo[:, None] + np.arange(3)
    indices[:, 3:6] = 3 * hi[:, None] + np.arange(3)
    indptr = np.arange(0, 6 * len(pairs) + 1, 6, dtype=np.int32)
    sign = np.where(pairs[:, 0] < pairs[:, 1], 1.0, -1.0)

    if len(_PAIR_INDEX_CACHE) >= _PAIR_INDEX_CACHE_SIZE:
        _PAIR_INDEX_CACHE.clear()
    entry = (rij_list, pairs, indptr, indices.ravel(), sign)
    _PAIR_INDEX_CACHE[key] = entry
    return entry[1:]


def compute_rij(
    geom: np.ndarray,
    rij_list: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate inter-atomic distances and their Cartesian derivatives.

    Parameters
    ----------
    geom : numpy.ndarray
        Cartesian geometry of all the atoms, of shape ``(n_atoms, 3)``.
    rij_list : list of tuple of int
        Indices of the atom pairs to evaluate.

    Returns
    -------
    rij : numpy.ndarray
        The distance for each pair.
    b_mat : numpy.ndarray
        Wilson B matrix, of shape ``(n_pairs, n_atoms, 3)``, holding the
        Cartesian gradient of every distance.
    """
    pairs = _pair_index(rij_list, len(geom))[0]
    d_vec = geom[pairs[:, 0]] - geom[pairs[:, 1]]
    rij = np.linalg.norm(d_vec, axis=1)
    # A distance only depends on its own two atoms, and moving one of them is
    # the exact opposite of moving the other
    grad = d_vec / rij[:, None]
    b_mat = np.zeros((len(pairs), len(geom), 3))
    rows = np.arange(len(pairs))
    b_mat[rows, pairs[:, 0]] = grad
    b_mat[rows, pairs[:, 1]] = -grad
    return rij, b_mat


def compute_wij(
    geom: np.ndarray,
    rij_list: list[tuple[int, int]],
    func: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    sparse: bool = False,
) -> tuple[np.ndarray, np.ndarray | csr_matrix]:
    """Calculate scaled distances and their Cartesian derivatives.

    Same as :func:`compute_rij`, except each distance is passed through a
    scaling function which sets the metric of the internal coordinates.

    Parameters
    ----------
    geom : numpy.ndarray
        Cartesian geometry of all the atoms. Flattened input is accepted and
        reshaped to ``(n_atoms, 3)``.
    rij_list : list of tuple of int
        Indices of the atom pairs to evaluate.
    func : callable
        Scaling function returning both the scaled value and its derivative
        with respect to the raw distance. Must broadcast over arrays.
    sparse : bool, optional
        Return the B matrix as a sparse matrix rather than a dense array. Only
        six of its entries per row are ever non-zero, so this is what the
        optimisers are given: it saves building the dense array, and lets SciPy
        solve the least-squares steps iteratively instead of by dense
        factorisation.

    Returns
    -------
    wij : numpy.ndarray
        The scaled distance for each pair.
    b_mat : numpy.ndarray or scipy.sparse.csr_matrix
        Cartesian gradients of the scaled distances, of shape
        ``(n_pairs, 3 * n_atoms)``, with the atom and component axes flattened
        together so ``scipy.optimize`` can use it directly. A dense array by
        default, or a sparse matrix when ``sparse`` is set.
    """
    geom = np.asarray(geom, dtype=float).reshape(-1, 3)
    if not sparse:
        rij, b_mat = compute_rij(geom, rij_list)
        wij, d_wdr = func(rij)
        # Chain rule: scale each pair's gradient by dw/dr for that pair
        b_mat *= d_wdr[:, None, None]
        return wij, b_mat.reshape(len(rij_list), -1)

    pairs, indptr, indices, sign = _pair_index(rij_list, len(geom))
    d_vec = geom[pairs[:, 0]] - geom[pairs[:, 1]]
    rij = np.linalg.norm(d_vec, axis=1)
    wij, d_wdr = func(rij)
    # Normalise, then apply the chain rule, in that order so the result matches
    # the dense branch bit for bit. `sign` puts the gradient of the
    # lower-numbered atom first, which is the column order CSR wants.
    grad = (d_vec / rij[:, None]) * (d_wdr * sign)[:, None]
    data = np.concatenate([grad, -grad], axis=1).ravel()
    b_mat = csr_matrix((data, indices, indptr), shape=(len(pairs), geom.size))
    return wij, b_mat


def morse_scaler(
    re: float | np.ndarray = 1.5,
    alpha: float = 1.7,
    beta: float = 0.01,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a scaling function based on a Morse potential.

    This is the metric geodesic interpolation uses by default. A small
    ``beta / r`` tail is added so the coordinate keeps responding to atoms that
    are far apart, where the exponential has already decayed away.

    Parameters
    ----------
    re : float or numpy.ndarray, optional
        Equilibrium distance. Usually the per-pair array returned by
        :func:`get_bond_list` rather than a single number.
    alpha : float, optional
        Decay constant of the exponential. Larger values are more localised,
        which tracks a sharp energy landscape better, while smaller values have
        longer range and give smoother paths from few images.
    beta : float, optional
        Weight of the long-range tail.

    Returns
    -------
    callable
        A function taking an array of distances to the scaled distances and
        their derivatives with respect to the raw distance, in the form
        :func:`compute_wij` expects.
    """

    def scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale an array of distances, and differentiate the scaling."""
        ratio = x / re
        val1 = np.exp(alpha * (1.0 - ratio))
        val2 = beta / ratio
        d_val = (-alpha * val1 / re) - (val2 / x)
        return val1 + val2, d_val

    return scaler


def elu_scaler(
    re: float | np.ndarray = 2.0,
    alpha: float = 2.0,
    beta: float = 0.01,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a scaling function with an exponential tail and a linear core.

    Shaped like an ELU: beyond ``re`` the scaled distance decays exponentially,
    as in :func:`morse_scaler`, while below ``re`` it continues linearly along
    the tangent at ``re`` instead of blowing up. A ``beta * re / r`` tail is
    added for the same reason as in the Morse case.

    Parameters
    ----------
    re : float or numpy.ndarray, optional
        Distance at which the behaviour switches from linear to exponential.
    alpha : float, optional
        Decay constant of the exponential, which also sets the linear slope.
    beta : float, optional
        Weight of the long-range tail.

    Returns
    -------
    callable
        A function taking an array of distances to the scaled distances and
        their derivatives with respect to the raw distance, in the form
        :func:`compute_wij` expects.
    """

    def scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale an array of distances, and differentiate the scaling."""
        tail = alpha * (1.0 - x / re)
        decay = np.exp(tail)
        outer = x > re
        val1 = np.where(outer, decay, tail + 1.0)
        d_val = np.where(outer, -alpha / re * decay, -alpha / re)
        val2 = beta * re / x
        return val1 + val2, d_val - val2 / x

    return scaler


class Geodesic:
    """Optimiser that finds a geodesic in redundant internal coordinates.

    The heart of it is the path length in the internal metric: the length of
    each segment is measured through its midpoint, so the target function is a
    sum of squared displacements and can be handed straight to a least-squares
    optimiser.

    :meth:`smooth` optimises every image at once and :meth:`sweep` takes them
    one at a time; both leave the result in :attr:`path`. The two end points
    are held fixed either way.

    Attributes
    ----------
    path : numpy.ndarray
        The current geometries, of shape ``(n_images, n_atoms, 3)``. Updated in
        place as the optimisation proceeds.
    n_images, n_atoms : int
        Shape of the path, as images and atoms.
    rij_list : list of tuple of int
        The atom pairs making up the internal coordinates.
    n_rij : int
        How many of them there are.
    re : numpy.ndarray
        Equilibrium distance for each of those pairs.
    scaler : callable
        The scaling function setting the metric.
    friction : float
        Default weight of the friction term regularising the step size.
    length : float
        Path length in the internal metric, over the section last measured.
        Only set once the displacements have been computed.
    optimality : float
        Infinity norm of the gradient of the length, which is what convergence
        is judged on. Only set once the target function has been evaluated.
    """

    length: float
    optimality: float

    def __init__(
        self,
        atoms: list[str],
        path: np.ndarray | list[np.ndarray],
        scaler: float | Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = 1.7,
        threshold: float = 3.0,
        min_neighbors: int = 4,
        friction: float = 1e-3,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise the interpolater.

        Parameters
        ----------
        atoms : list of str
            Atom symbols, used to look up covalent radii.
        path : array_like
            Initial geometries of the path, of shape
            ``(n_images, n_atoms, 3)``.
        scaler : float or callable, optional
            Either the alpha parameter for :func:`morse_scaler`, or an explicit
            scaling function. Small values have long range and make it easier
            to get smooth paths out of few images; larger values usually give
            better energetics because they represent the sharp energy landscape
            better.
        threshold : float, optional
            Distance cut-off for building the inter-nuclear distance
            coordinates. Atoms linked by three or fewer bonds are added too,
            whatever their distance.
        min_neighbors : int, optional
            Minimum number of neighbours each atom must have in the atom pair
            list.
        friction : float, optional
            Weight of the friction term in the target function, which keeps the
            optimiser from taking steps large enough to blow the path up.
        rng : numpy.random.Generator, optional
            Random source used when sampling images to build the coordinates.
            Defaults to a fresh unseeded generator.

        Raises
        ------
        ValueError
            If the path is not three dimensional.
        """
        path = np.asarray(path, dtype=float)
        if path.ndim != 3:
            raise ValueError("The path to be interpolated must have 3 dimensions")
        _, self.path = align_path(path)
        self.n_images, self.n_atoms, _ = self.path.shape
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(
            self.path,
            atoms,
            threshold=threshold,
            min_neighbors=min_neighbors,
            rng=rng,
        )
        if isinstance(scaler, (int, float, np.number)):
            self.scaler = morse_scaler(re=self.re, alpha=scaler)
        else:
            self.scaler = scaler
        self.n_rij = len(self.rij_list)
        self.friction = friction
        # Internal storage for the midpoints, internal coordinates and B
        # matrices. `None` marks a value as unknown, so it is only ever
        # computed when it is needed.
        self.w: list[np.ndarray | None] = [None] * self.n_images
        self.dw_dr: list[csr_matrix | None] = [None] * self.n_images
        self.w_mid: list[np.ndarray | None] = [None] * (self.n_images - 1)
        self.dw_dr_mid: list[csr_matrix | None] = [None] * (self.n_images - 1)
        self.displacements: np.ndarray | None = None
        self.grad: csr_matrix | None = None
        self.segment: tuple[int, int] | None = None

    def update_intc(self) -> None:
        """Fill in any internal coordinates currently marked unknown.

        Missing entries are flagged with ``None`` in the internal storage; this
        finds them, works out the midpoint geometries where needed, and
        evaluates the coordinates and their gradients. Everything already known
        is left alone, so nothing is evaluated twice.

        The derivatives are kept sparse, since each scaled distance only
        depends on the six Cartesian components of its own two atoms.
        """
        for i, geom in enumerate(self.path):
            if self.w[i] is None:
                self.w[i], self.dw_dr[i] = compute_wij(
                    geom, self.rij_list, self.scaler, sparse=True
                )
        for i, (geom_l, geom_r) in enumerate(zip(self.path, self.path[1:])):
            if self.w_mid[i] is None:
                geom_mid = (geom_l + geom_r) / 2
                self.w_mid[i], self.dw_dr_mid[i] = compute_wij(
                    geom_mid, self.rij_list, self.scaler, sparse=True
                )

    def update_geometry(self, geom: np.ndarray, start: int, end: int) -> bool:
        """Move a segment of the path, invalidating what depended on it.

        The internal coordinates, derivatives and midpoints of the affected
        images are reset to unknown so :meth:`update_intc` recomputes them.
        Note that moving images ``start:end`` also invalidates the midpoint
        just before ``start``, if there is one.

        Parameters
        ----------
        geom : numpy.ndarray
            New Cartesian coordinates for the segment, flattened or otherwise.
        start, end : int
            The section of the path being replaced.

        Returns
        -------
        bool
            True if the geometry actually changed, False if ``geom`` was
            already the current geometry and nothing needed invalidating.
        """
        geom = geom.reshape(self.path[start:end].shape)
        if np.array_equal(geom, self.path[start:end]):
            return False
        self.path[start:end] = geom
        self.w[start:end] = [None] * (end - start)
        # Clamped, because a negative slice bound would wrap round to the end
        # of the list and insert entries instead of overwriting them
        first_mid = max(start - 1, 0)
        self.w_mid[first_mid:end] = [None] * (
            min(end, self.n_images - 1) - first_mid
        )
        return True

    def compute_displacements(
        self,
        start: int = 1,
        end: int = -1,
        dx: np.ndarray | None = None,
        friction: float = 1e-3,
    ) -> None:
        """Compute the displacements along a section of the path, and its length.

        Each segment is split at its midpoint and measured in two halves, which
        is what makes the length a sum of squares. The friction term is
        appended to the same vector so the least-squares optimiser sees it as
        extra residuals.

        Sets :attr:`length` and :attr:`displacements`.

        Parameters
        ----------
        start, end : int, optional
            Section of the path to measure. A negative ``end`` counts back from
            the last image. ``start`` must be at least 1: the end points of the
            path are fixed, and each segment is measured against the image
            before it.
        dx : numpy.ndarray, optional
            Displacement of the segment from its reference geometry. When
            given, it enters the target function scaled by ``friction``.
        friction : float, optional
            Weight of the friction term.

        Raises
        ------
        ValueError
            If the section does not lie between the two fixed end points.
        """
        if end < 0:
            end += self.n_images
        if not 1 <= start < end <= self.n_images - 1:
            raise ValueError(
                f"Section ({start}, {end}) must lie between the fixed end "
                f"points of a {self.n_images} image path"
            )
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [
            wm - wl
            for wl, wm in zip(self.w[start - 1 : end], self.w_mid[start - 1 : end])
        ]
        vecs_r = [
            wr - wm
            for wr, wm in zip(self.w[start : end + 1], self.w_mid[start - 1 : end])
        ]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(
            np.linalg.norm(vecs_r, axis=1)
        )
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from the initial geometry
        self.displacements = np.concatenate(vecs_l + vecs_r + [trans])

    def compute_disp_grad(
        self,
        start: int,
        end: int,
        friction: float = 1e-3,
    ) -> None:
        """Differentiate the displacement vectors with respect to Cartesians.

        Moving one image changes the two half-segments on either side of it
        directly, and the two midpoints it shares with its neighbours at half
        the rate, which is where the factors of a half come from. The friction
        residuals contribute a diagonal block at the bottom of the matrix.

        An image only appears in the two half-segments it touches, so the
        matrix is block-bidiagonal, and each block is itself sparse because a
        distance only depends on its own two atoms. It is assembled sparse and
        left that way: for a few dozen atoms fewer than one entry in a hundred
        is non-zero, and handing a dense array to
        :func:`scipy.optimize.least_squares` makes it factorise the whole thing
        on every iteration.

        Sets :attr:`grad`.

        Parameters
        ----------
        start, end : int
            Section of the path being differentiated.
        friction : float, optional
            Weight of the friction term, matching
            :meth:`compute_displacements`.
        """
        # Derivatives of the displacement vectors w.r.t. the image Cartesians
        n_seg = end - start
        n_rows = n_seg + 1  # One more image than segments, as each joins two
        n_dof = 3 * self.n_atoms
        blocks_l = [[None] * n_seg for _ in range(n_rows)]
        blocks_r = [[None] * n_seg for _ in range(n_rows)]
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dw_dr_mid[image - 1] / 2
            dmid2 = self.dw_dr_mid[image] / 2
            blocks_l[i + 1][i] = dmid2 - self.dw_dr[image]
            blocks_l[i][i] = dmid1
            blocks_r[i + 1][i] = -dmid2
            blocks_r[i][i] = self.dw_dr[image] - dmid1
        # The friction residuals are one scaled identity, laid out block by
        # block so it lines up with the image columns above it
        friction_block = identity(n_dof, format="csr") * friction
        blocks_f = [
            [friction_block if k == i else None for i in range(n_seg)]
            for k in range(n_seg)
        ]
        self.grad = bmat(blocks_l + blocks_r + blocks_f, format="csr")

    def compute_target_func(
        self,
        geom: np.ndarray | None = None,
        start: int = 1,
        end: int = -1,
        x0: np.ndarray | None = None,
        friction: float = 1e-3,
    ) -> None:
        """Compute the vectorised target function the minimisation uses.

        Sets :attr:`displacements`, :attr:`grad` and :attr:`optimality`.

        Parameters
        ----------
        geom : numpy.ndarray, optional
            Geometry to evaluate at. If it matches the geometry already stored
            for this segment, the whole evaluation is skipped.
        start, end : int, optional
            Section of the path being optimised.
        x0 : numpy.ndarray, optional
            Reference geometry the friction term pulls back towards. Defaults
            to no pull at all.
        friction : float, optional
            Weight of the friction term.
        """
        if end < 0:
            end += self.n_images
        if (
            geom is not None
            and self.segment == (start, end)
            and not self.update_geometry(geom, start, end)
        ):
            return

        self.segment = (start, end)
        if x0 is None:
            dx = np.zeros(self.path[start:end].size)
        else:
            dx = self.path[start:end].ravel() - x0.ravel()
        self.compute_displacements(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.abs(self.grad.T @ self.displacements).max()

    def target_func(self, geom: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Residuals for the optimiser.

        Wraps :meth:`compute_target_func`, which skips the work if the geometry
        has not moved since the last call.

        Parameters
        ----------
        geom : numpy.ndarray
            Geometry of the segment, flattened as SciPy hands it over.
        **kwargs
            Passed straight through to :meth:`compute_target_func`.

        Returns
        -------
        numpy.ndarray
            The displacement vectors of the segment, with the friction
            residuals appended.
        """
        self.compute_target_func(geom, **kwargs)
        return self.displacements

    def target_deriv(self, geom: np.ndarray, **kwargs: Any) -> csr_matrix:
        """Jacobian for the optimiser.

        Wraps :meth:`compute_target_func`, which skips the work if the geometry
        has not moved since the last call. Paired with :meth:`target_func`,
        this means each geometry is only ever evaluated once even though SciPy
        asks for value and Jacobian separately.

        Parameters
        ----------
        geom : numpy.ndarray
            Geometry of the segment, flattened as SciPy hands it over.
        **kwargs
            Passed straight through to :meth:`compute_target_func`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivatives of those displacement vectors, held sparse.
        """
        self.compute_target_func(geom, **kwargs)
        return self.grad

    def smooth(
        self,
        tol: float = 1e-3,
        max_iter: int = 50,
        start: int = 1,
        end: int = -1,
        friction: float | None = None,
        xref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Minimise the path length as one function of all the coordinates.

        In principle this is very efficient, but it can get costly for large
        systems with many images, in which case :meth:`sweep` is the better
        option.

        Parameters
        ----------
        tol : float, optional
            Convergence tolerance on the optimality, i.e. the uniform gradient
            of the target function.
        max_iter : int, optional
            Ceiling on the number of function evaluations, passed to SciPy as
            ``max_nfev``.
        start, end : int, optional
            Section of the path to optimise.
        friction : float, optional
            Weight of the friction term. Defaults to the value given to the
            constructor.
        xref : numpy.ndarray, optional
            Reference geometry for the friction term. Defaults to the starting
            geometry of the segment.

        Returns
        -------
        numpy.ndarray
            The optimised path. This is also stored in :attr:`path`.
        """
        x0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref = x0
        self.displacements = self.grad = self.segment = None
        if friction is None:
            friction = self.friction
        # Keyword arguments that will be sent to the target function
        kwargs = dict(start=start, end=end, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        if self.optimality > tol:
            # `soft_l1` keeps a single badly placed image from dominating the fit
            result = least_squares(
                self.target_func,
                x0,
                self.target_deriv,
                ftol=tol,
                gtol=tol,
                max_nfev=max_iter,
                kwargs=kwargs,
                loss="soft_l1",
            )
            self.update_geometry(result["x"], start, end)
        _, self.path = align_path(self.path)
        return self.path

    def sweep(
        self,
        tol: float = 1e-3,
        max_iter: int = 50,
        micro_iter: int = 20,
        start: int = 1,
        end: int = -1,
    ) -> np.ndarray:
        """Minimise the path length one image at a time, sweeping back and forth.

        Less efficient per iteration than :meth:`smooth`, but it scales far
        more kindly with system size given how slow SciPy's optimisers get on
        large problems. It also allows finer control, and makes it cheap to
        skip images that are already close to optimal.

        Parameters
        ----------
        tol : float, optional
            Convergence tolerance on the optimality, i.e. the uniform gradient
            of the target function.
        max_iter : int, optional
            Maximum number of sweeps through the path.
        micro_iter : int, optional
            Number of micro-iterations spent optimising each image.
        start, end : int, optional
            Section of the path to optimise.

        Returns
        -------
        numpy.ndarray
            The optimised path. This is also stored in :attr:`path`.
        """
        if end < 0:
            end = self.n_images + end
        images = list(range(start, end))
        # The micro-iteration tolerance is tightened as convergence improves
        curr_tol = tol * 10
        for iteration in range(max_iter):
            max_dl = 0.0
            for i in images:  # Use self.smooth() to optimise individual images
                # Each image is pulled back towards the midpoint of its
                # neighbours, with heavy friction on the first sweep to keep
                # the initial guess from being thrown around
                xmid = (self.path[i - 1] + self.path[i + 1]) * 0.5
                self.smooth(
                    curr_tol,
                    max_iter=min(micro_iter, iteration + 6),
                    start=i,
                    end=i + 1,
                    friction=self.friction if iteration else 0.1,
                    xref=xmid,
                )
                max_dl = max(max_dl, self.optimality)
            if max_dl < tol:  # Check for convergence
                break
            curr_tol = max(tol * 0.5, max_dl * 0.2)
            images.reverse()  # Alternate the sweeping direction
        _, self.path = align_path(self.path)
        return self.path


def _mid_point(
    atoms: list[str],
    geom1: np.ndarray,
    geom2: np.ndarray,
    tol: float = 1e-2,
    nudge: float = 0.01,
    threshold: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Find the geometry whose internals sit closest to the average of two.

    A least-squares minimisation against the average of the two end points, run
    twice, starting from just beside either end point. Do not use the Cartesian
    average as the guess: things will blow up. The two runs are then compared
    by local geodesic length and the shorter one wins.

    The point produced here need not join smoothly onto either end point; it
    only has to be a good enough starting guess for the smoothing that follows.

    Random nudges are added to the starting geometry, so repeated runs need not
    converge to the same answer -- for larger systems they essentially never
    will. Running several times and keeping the best result is therefore
    worthwhile.

    Parameters
    ----------
    atoms : list of str
        Atom symbols, used to look up covalent radii.
    geom1, geom2 : numpy.ndarray
        Cartesian geometries of the two end points.
    tol : float, optional
        Convergence tolerance for the least-squares minimisation.
    nudge : float, optional
        Size of the random nudge added to the starting geometry. Helps to turn
        up different solutions, and to break symmetry when the optimal path
        does.
    threshold : float, optional
        Distance cut-off for including an atom pair in the coordinates.
    rng : numpy.random.Generator, optional
        Random source for the nudge and the image sampling. Defaults to a fresh
        unseeded generator.

    Returns
    -------
    numpy.ndarray
        The optimised mid-point, bisecting the two end points in internal
        coordinates.
    """
    geom1, geom2 = np.array(geom1, dtype=float), np.array(geom2, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    add_pair: set[tuple[int, int]] = set()
    geom_list: list[np.ndarray] = [geom1, geom2]

    # The outer loop makes sure the coordinate system is large enough. The
    # interpolated point can bring atom pairs into contact that are far apart
    # at both end points, which would let them collide unnoticed. Including
    # every pair would blow up for large molecules, so the compromise is to
    # start from a screened list, add any pair that comes into contact, and
    # redo the minimisation until the coordinate system and the interpolated
    # geometry agree.
    while True:
        rij_list, re = get_bond_list(
            geom_list, threshold=threshold + 1.0, enforce=add_pair, rng=rng
        )
        scaler = morse_scaler(alpha=0.7, re=re)
        w = (
            compute_wij(geom1, rij_list, scaler, sparse=True)[0]
            + compute_wij(geom2, rij_list, scaler, sparse=True)[0]
        ) / 2
        d_min: float = np.inf
        x_min: np.ndarray | None = None
        friction: float = 0.1 / np.sqrt(geom1.shape[0])
        # The friction residuals never change, so build them once rather than
        # on every call
        friction_block = identity(geom1.size, format="csr") * friction
        # SciPy asks for the residuals and the Jacobian in separate calls but
        # at the same geometry, so holding on to the last evaluation halves the
        # work
        last_eval: list[np.ndarray | tuple[np.ndarray, csr_matrix] | None] = [
            None,
            None,
        ]

        def wij_at(x: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
            """Scaled distances and B matrix at ``x``, reusing the last result."""
            if last_eval[0] is None or not np.array_equal(last_eval[0], x):
                last_eval[0] = np.array(x)
                last_eval[1] = compute_wij(x, rij_list, scaler, sparse=True)
            return last_eval[1]

        # The inner loop minimises from either end point in turn as the guess
        for coef in [0.02, 0.98]:
            x0: np.ndarray = (geom1 * coef + geom2 * (1 - coef)).ravel() + (
                nudge * rng.random(geom1.size)
            )
            # Residuals are the difference from the target internals, plus a
            # friction term holding the geometry near where it started
            result = least_squares(
                lambda x: np.concatenate([wij_at(x)[0] - w, (x - x0) * friction]),
                x0,
                lambda x: vstack([wij_at(x)[1], friction_block], format="csr"),
                ftol=tol,
                gtol=tol,
            )
            x_mid: np.ndarray = result["x"].reshape(-1, 3)
            # Rebuild the pair list including the new point, and check for any
            # fresh contacts
            new_rij, _ = get_bond_list(
                [*geom_list, x_mid], threshold=threshold, min_neighbors=0, rng=rng
            )
            extras = set(new_rij) - set(rij_list)

            if extras:
                # Widen the coordinate system and start the minimisation over
                geom_list.append(x_mid)
                add_pair.update(extras)
                break

            # Score this candidate by locally smoothing the three-image path
            # that runs through it
            smoother = Geodesic(
                atoms,
                [geom1, x_mid, geom2],
                scaler=0.7,
                threshold=threshold,
                friction=1,
                rng=rng,
            )
            smoother.compute_displacements()
            width = max(
                np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2]
            )
            dist = width + smoother.length
            if dist < d_min:
                d_min, x_min = dist, smoother.path[1]
        else:
            # Both starting guesses finished without new atom pairs, so the
            # coordinate system held up and the minimisation is done
            break

    return x_min


def redistribute(
    atoms: list[str],
    geoms: list[np.ndarray],
    n_images: int,
    tol: float = 1e-2,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Add or remove images so the path has the requested number of them.

    If there are too few, new points are added by bisecting the largest RMSD
    gap. If there are too many, images are dropped one at a time, each time
    choosing the one whose removal leaves the shortest merged segment.

    Parameters
    ----------
    atoms : list of str
        Atom symbols, used to look up covalent radii.
    geoms : list of numpy.ndarray
        Geometries of the original path.
    n_images : int
        The desired number of images.
    tol : float, optional
        Convergence tolerance for the bisection.
    rng : numpy.random.Generator, optional
        Random source for the bisection, which is stochastic. Defaults to a
        fresh unseeded generator.

    Returns
    -------
    list of numpy.ndarray
        An aligned path with the correct number of images.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)

    # Add bisection points if there are too few images
    while len(geoms) < n_images:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i: int = int(np.argmax(dists))
        insertion: np.ndarray = _mid_point(
            atoms, geoms[max_i], geoms[max_i + 1], tol, rng=rng
        )
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])

    # Remove points if there are too many images
    while len(geoms) > n_images:
        # Each distance here spans two segments, so the smallest one marks the
        # image that can be dropped with the least disruption
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i: int = int(np.argmin(dists))
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])

    return geoms


def from_ase_atoms(atoms: list[Atoms]) -> tuple[list[str], list[np.ndarray]]:
    """Split a list of ASE Atoms objects into symbols and coordinates.

    Every frame must hold the same atoms in the same order, since the symbols
    are taken from the first one and the interpolation matches the frames up
    atom by atom. Two end states built separately are the usual way to get
    that wrong, so it is checked here rather than assumed: the coordinates
    alone carry no record of which atom is which, and a path interpolated
    between mismatched frames is silently a path to the wrong structure.

    Parameters
    ----------
    atoms : list of ase.Atoms
        Frames of the path.

    Returns
    -------
    atom_names : list of str
        Element symbols of all the atoms.
    coords : list of numpy.ndarray
        Cartesian coordinates for every frame.

    Raises
    ------
    ValueError
        If the frames do not all describe the same atoms in the same order.
    """
    atom_names = atoms[0].get_chemical_symbols()
    coords: list[np.ndarray] = []
    for i, atom in enumerate(atoms):
        symbols = atom.get_chemical_symbols()
        if symbols != atom_names:
            first = next(
                (j for j, (a, b) in enumerate(zip(atom_names, symbols)) if a != b),
                min(len(symbols), len(atom_names)),
            )
            if len(symbols) != len(atom_names):
                detail = f"{len(symbols)} atoms, against {len(atom_names)}"
            elif sorted(symbols) == sorted(atom_names):
                detail = (
                    f"the same atoms in a different order: atom {first} is "
                    f"{symbols[first]}, not {atom_names[first]}"
                )
            else:
                detail = (
                    f"a different composition: atom {first} is "
                    f"{symbols[first]}, not {atom_names[first]}"
                )
            raise ValueError(
                f"Frame {i} has {detail}. Every frame of a path must hold the "
                f"same atoms in the same order as the first."
            )
        coords.append(np.array(atom.get_positions()))
    return atom_names, coords


def to_ase_atoms(
    atoms: list[str],
    coords: np.ndarray | list[np.ndarray],
    template: Atoms | None = None,
) -> list[Atoms]:
    """Rebuild a list of ASE Atoms objects from symbols and coordinates.

    Parameters
    ----------
    atoms : list of str
        Element symbols of all the atoms.
    coords : array_like
        Cartesian coordinates, of shape ``(n_images, n_atoms, 3)``. A single
        frame of shape ``(n_atoms, 3)`` is accepted too.
    template : ase.Atoms, optional
        Atoms object to take everything other than the positions from, so the
        unit cell, boundary conditions, constraints, tags and the rest survive
        a round trip through this module. Each frame is a copy of the template
        with its positions replaced; as with :meth:`ase.Atoms.copy`, the
        calculator is not carried over, since its results belong to the
        geometry it was attached to. Any constraints come along for later use
        but are not applied to the coordinates, which are always exactly the
        ones given. Without a template the frames are built from the symbols
        alone and carry no cell.

    Returns
    -------
    list of ase.Atoms
        One Atoms object per frame.

    Raises
    ------
    ValueError
        If ``template`` describes a different set of atoms from ``atoms``.
    """
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]  # Add a new axis for a single frame
    if template is None:
        return [Atoms(symbols=atoms, positions=frame) for frame in coords]

    symbols = [atom.capitalize() for atom in atoms]
    if template.get_chemical_symbols() != symbols:
        raise ValueError(
            f"Template describes {template.get_chemical_formula()}, which is "
            f"not the same system as the {len(symbols)} atoms given"
        )
    images = []
    for frame in coords:
        image = template.copy()
        # Not the default, which would let a constraint such as `FixAtoms`
        # quietly drag atoms back to where the template had them and corrupt
        # the interpolated path
        image.set_positions(frame, apply_constraint=False)
        images.append(image)
    return images


def read_xyz(filename: str | os.PathLike) -> tuple[list[str], list[np.ndarray]]:
    """Read an XYZ file and return the atom names and coordinates.

    Parameters
    ----------
    filename : str or os.PathLike
        Name of the XYZ data file. It may hold any number of frames, and blank
        lines between them are ignored.

    Returns
    -------
    atom_names : list of str
        Element symbols of all the atoms, read from the last frame.
    coords : list of numpy.ndarray
        Cartesian coordinates for every frame.

    Raises
    ------
    ValueError
        If the file is empty or does not parse as XYZ.
    """
    coords: list[np.ndarray] = []
    with open(filename, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():  # Blank line between or after frames
                continue
            try:
                n_atoms = int(line)  # Read the number of atoms
                next(fh)  # Skip over the comment line
                atom_names: list[str] = []
                geom = np.zeros((n_atoms, 3), float)
                for i in range(n_atoms):
                    line = next(fh).split()
                    atom_names.append(line[0])
                    geom[i] = line[1:4]  # NumPy auto-converts str to float
            except (TypeError, ValueError, OSError, IndexError, StopIteration) as err:
                raise ValueError("Incorrect XYZ file format") from err
            coords.append(geom)
    if not coords:
        raise ValueError("File is empty")
    return atom_names, coords


def write_xyz(
    filename: str | os.PathLike,
    atoms: list[str],
    coords: np.ndarray | list[np.ndarray],
) -> None:
    """Write atom names and coordinate data to a multi-frame XYZ file.

    Parameters
    ----------
    filename : str or os.PathLike
        Name of the XYZ data file to write.
    atoms : list of str
        Element symbol for each atom.
    coords : array_like
        Coordinates, of shape ``(n_images, n_atoms, 3)``. A single frame of
        shape ``(n_atoms, 3)`` is accepted too.
    """
    # Not `np.atleast_3d`, which would turn a single frame into
    # `(n_atoms, 3, 1)`
    coords = np.asarray(coords, dtype=float).reshape(-1, len(atoms), 3)
    with open(filename, "w", encoding="utf-8") as fh:
        for i, frame in enumerate(coords):
            write_xyz_frame(fh, atoms, frame, comment=f"Frame {i}")


def geodesic_interpolate(
    atoms: list[Atoms] | str | os.PathLike,
    n_images: int = 17,
    output: str | os.PathLike = "interpolated.xyz",
    tol: float = 2e-3,
    max_iter: int = 50,
    micro_iter: int = 20,
    scaling: float = 1.7,
    friction: float = 1e-2,
    dist_cutoff: float = 3.0,
    seed: int = 42,
) -> list[Atoms] | None:
    """Interpolate a reaction path between two or more geometries.

    Runs the two stages in turn: :func:`redistribute` builds a raw path with
    the requested number of images, then :class:`Geodesic` smooths it into a
    geodesic under the internal coordinate metric.

    Which smoothing is used depends on the size of the system. Up to
    :data:`SWEEP_ABOVE_N_ATOMS` atoms the whole path is optimised at once;
    beyond that the images are smoothed one at a time, sweeping back and forth
    along the path, because SciPy's optimisers slow down badly as the problem
    grows.

    Input and output mirror each other. Given ASE Atoms objects the
    interpolated path comes back as Atoms objects; given a filename it is
    written to ``output`` instead.

    Given Atoms objects, everything the interpolation does not itself touch is
    taken from the first frame and carried onto every image: the unit cell, the
    boundary conditions, constraints, tags and so on. A periodic path is also
    moved back onto the frame of reference of the input, since the optimisation
    otherwise leaves it centred and rotated into a frame of its own, which
    would put the atoms in the wrong place relative to the cell. Note that the
    interpolation itself is not periodic: the internal coordinates are plain
    inter-atomic distances with no minimum image convention, so a bond that
    crosses a cell boundary is not handled.

    Parameters
    ----------
    atoms : list of ase.Atoms, or str or os.PathLike
        Either a list of ASE Atoms objects, or the name of an XYZ file holding
        the end points. Only the first and last geometries need be meaningful,
        but intermediate ones are used if present.
    n_images : int, optional
        Number of images in the interpolated path.
    output : str or os.PathLike, optional
        XYZ file to write to. Only used when ``atoms`` is a filename.
    tol : float, optional
        Convergence tolerance for the smoothing, judged on the uniform gradient
        of the path length. The raw path is built at five times this, being
        only a starting guess for the smoothing.
    max_iter : int, optional
        Ceiling on the smoothing, counting function evaluations when the whole
        path is optimised at once and sweeps when sweeping. It is a ceiling
        rather than a target, as the optimisation stops as soon as it meets
        ``tol``: the bundled test systems all converge somewhere between 10 and
        35 at the default seed, though the count moves around with the seed.
        Setting it too low silently truncates the path part-way through the
        descent.
    micro_iter : int, optional
        Micro-iterations spent on each image. Only used when sweeping, so it
        has no effect on systems of :data:`SWEEP_ABOVE_N_ATOMS` atoms or fewer.
    scaling : float, optional
        Alpha parameter of the Morse scaler setting the coordinate metric.
    friction : float, optional
        Weight of the friction term regularising the optimisation step size.
    dist_cutoff : float, optional
        Distance cut-off for building the internal coordinates. Atoms within
        three bonds of each other are included whatever their distance.
    seed : int, optional
        Seed for the random nudges and image sampling, so runs reproduce. The
        bisection is stochastic, and without a fixed seed larger systems will
        not give the same path twice. The seed goes to a
        :class:`numpy.random.Generator` used only here, so the caller's own
        random state is left alone.

    Returns
    -------
    list of ase.Atoms or None
        The interpolated path when ``atoms`` was a list of Atoms objects,
        otherwise None, with the path written to ``output``.

    Raises
    ------
    TypeError
        If ``atoms`` is neither a list of ASE Atoms nor a filename.
    ValueError
        If fewer than two geometries are supplied.
    """
    rng = np.random.default_rng(seed)
    template = None
    if isinstance(atoms, (str, os.PathLike)):
        symbols, geometries = read_xyz(atoms)
    elif isinstance(atoms, list):
        symbols, geometries = from_ase_atoms(atoms)
        # Everything the interpolation does not touch is carried over from the
        # first frame, which is also where the symbols come from
        template = atoms[0]
    else:
        raise TypeError("Input must be an ASE Atoms object or a filename.")

    if len(geometries) < 2:
        raise ValueError("Need at least two initial geometries.")

    # A looser tolerance is enough for the raw path, which only has to be a
    # decent starting guess for the smoothing that follows
    raw = redistribute(symbols, geometries, n_images, tol=tol * 5, rng=rng)
    smoother = Geodesic(
        symbols, raw, scaling, threshold=dist_cutoff, friction=friction, rng=rng
    )

    # Optimising the whole path at once is faster, but SciPy's optimisers slow
    # down badly as the system grows, so past this size sweep one image at a
    # time instead
    if len(symbols) > SWEEP_ABOVE_N_ATOMS:
        smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
    else:
        smoother.smooth(tol=tol, max_iter=max_iter)

    if template is None:
        write_xyz(output, symbols, smoother.path)
        return None

    # The optimisation leaves the path in its own centred and rotated frame.
    # That is of no consequence for an isolated molecule, but a cell describes
    # where the atoms are only in the frame it was given in, so a periodic path
    # is moved back onto its input
    path = smoother.path
    if template.cell.rank > 0 or template.pbc.any():
        path = align_path_to(template.get_positions(), path)
    return to_ase_atoms(symbols, path, template=template)
