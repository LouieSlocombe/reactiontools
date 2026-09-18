"""Coordinate utilities used by the interpolation program.

Geodesic interpolation works in a redundant internal coordinate system built from
scaled inter-atomic distances.  This module provides the pieces needed to set that
system up: alignment of geometries and paths (Kabsch), selection of the atom pairs
that make up the coordinates, evaluation of those coordinates together with their
Cartesian derivatives (the Wilson B matrix), and the scaling functions that define
the metric.
"""
from collections.abc import Callable, Iterable
from itertools import pairwise

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from scipy.sparse import coo_matrix, csr_matrix, identity, triu
from scipy.spatial import KDTree


def align_path(path: np.ndarray | list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Rotate and translate the images of a path to minimise RMSD movement along it.

    The first image is shifted so its geometric centre sits at the origin, and each
    following image is aligned against the one before it, which leaves every image
    centred on the origin as well.

    Args:
        path: Sequence of geometries, of shape ``(n_images, n_atoms, 3)``.

    Returns:
        max_rmsd: Largest RMSD between any two adjacent images after alignment.
        path: The aligned path.  This is a copy, so the input is left untouched.
    """
    path = np.array(path, dtype=float)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0.0
    for g, next_g in pairwise(path):
        rmsd, aligned_geom = align_geom(g, next_g)
        next_g[:] = aligned_geom
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def _kabsch_rotation(ref_centered: np.ndarray, geom_centered: np.ndarray) -> np.ndarray:
    """Find the rotation that best takes one centred geometry onto another.

    The SVD of the covariance matrix of the two geometries gives the optimal rotation.
    Both arguments must already have their centres at the origin.

    Args:
        ref_centered: The centred reference geometry to rotate towards.
        geom_centered: The centred geometry to be rotated.

    Returns:
        The rotation matrix, to be applied on the right of a set of centred coordinates.
    """
    cov = np.dot(geom_centered.T, ref_centered)
    v, _, w = np.linalg.svd(cov)

    # A negative determinant means the SVD produced a reflection rather than a
    # rotation.  Flipping the least significant axis gives a proper rotation.
    if np.linalg.det(v) * np.linalg.det(w) < 0.0:
        v[:, -1] *= -1

    return np.dot(v, w)


def align_geom(ref_geom: np.ndarray, geom: np.ndarray) -> tuple[float, np.ndarray]:
    """Find the rigid-body motion that maximally overlaps one geometry with another.

    Implemented with the Kabsch algorithm: both geometries are centred, the SVD of
    their covariance matrix gives the optimal rotation, and the result is moved back
    onto the centre of the reference.

    Args:
        ref_geom: The reference geometry to rotate towards.
        geom: The geometry to be rotated and shifted.

    Returns:
        rmsd: Root-mean-squared difference between the rotated geometry and the
            reference.
        aligned_geom: The rotated geometry that maximally overlaps the reference.
    """
    center = np.mean(ref_geom, axis=0)
    ref_geom_centered = ref_geom - center
    geom_centered = geom - np.mean(geom, axis=0)

    rotation_matrix = _kabsch_rotation(ref_geom_centered, geom_centered)
    aligned_geom = np.dot(geom_centered, rotation_matrix) + center
    rmsd = np.sqrt(np.mean((aligned_geom - ref_geom) ** 2))

    return rmsd, aligned_geom


def align_path_to(ref_geom: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Move a whole path rigidly so that its first image sits on a reference geometry.

    `align_path` leaves the path centred on the origin and rotated onto its own first
    image, which is a frame of the optimiser's choosing rather than the one the caller
    handed in.  That is immaterial for an isolated molecule, but a unit cell only
    describes where the atoms are in the frame it came with, so a periodic path has to be
    put back before its cell means anything.

    One rotation and translation, taken from the first image, is applied to every image,
    so the shape of the path and its length are untouched.

    Args:
        ref_geom: Geometry to put the first image back onto, of shape ``(n_atoms, 3)``.
        path: The path to move, of shape ``(n_images, n_atoms, 3)``.

    Returns:
        The path in the reference geometry's frame.  This is a copy.
    """
    path = np.asarray(path, dtype=float)
    center = np.mean(ref_geom, axis=0)
    origin = np.mean(path[0], axis=0)

    rotation_matrix = _kabsch_rotation(ref_geom - center, path[0] - origin)
    return np.dot(path - origin, rotation_matrix) + center


def _pairs_within_three_bonds(tree: KDTree, n_atoms: int, bond_threshold: float) -> list[tuple[int, int]]:
    """List the atom pairs separated by three or fewer bonds in one geometry.

    Every such pair is a bonded pair with an optional extra bond tacked on at either
    end, so the answer is the sparsity pattern of ``(A + I) A (A + I)`` for the bond
    adjacency ``A``.  Phrasing it as two sparse matrix products keeps the cost down on
    large molecules, where walking the neighbour lists in Python does not scale.

    Args:
        tree: KD-tree of the geometry to work from.
        n_atoms: Number of atoms in the geometry.
        bond_threshold: Distance below which two atoms count as bonded.

    Returns:
        The ``(i, j)`` pairs, with ``i < j``.
    """
    bonded = tree.query_pairs(bond_threshold, output_type='ndarray')
    if len(bonded) == 0:
        return []
    # The adjacency matrix needs both directions of every bond
    rows = np.concatenate([bonded[:, 0], bonded[:, 1]])
    cols = np.concatenate([bonded[:, 1], bonded[:, 0]])
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_atoms, n_atoms)).tocsr()
    with_self = adjacency + identity(n_atoms, format='csr')
    reach = triu(with_self @ adjacency @ with_self, k=1).tocoo()
    return list(zip(reach.row.tolist(), reach.col.tolist()))


def get_bond_list(geom: np.ndarray | list[np.ndarray],
                  atoms: list[str] | None = None,
                  threshold: float = 4.0,
                  min_neighbors: int = 4,
                  snapshots: int = 30,
                  bond_threshold: float = 1.8,
                  enforce: Iterable[tuple[int, int]] = (),
                  rng: np.random.Generator | None = None) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Get the list of atom pairs that define the internal coordinate system.

    Samples images from the path and collects every pair of atoms that comes within
    ``threshold`` of each other in any of them.  Pairs linked by three or fewer bonds
    are always included, as are the pairs given in ``enforce``, and further pairs are
    added for any atom left with fewer than ``min_neighbors`` neighbours.

    Args:
        geom: A single geometry or a path of them.  Anything that is not already
            ``(n_images, n_atoms, 3)`` is promoted to that shape.
        atoms: Element symbols, used to look up covalent radii.  When omitted, every
            pair is given a nominal equilibrium distance of 2.0 Angstrom instead.
        threshold: Distance cut-off for including a pair in the coordinate system.
        min_neighbors: Minimum number of pairs each atom must take part in.  Atoms
            below this get their nearest neighbours added.
        snapshots: Maximum number of images to sample.  Keeps the cost down when the
            path is long and the atoms numerous.
        bond_threshold: Distance below which two atoms count as bonded, used to work
            out which pairs are within three bonds of each other.
        enforce: Pairs to include regardless of how far apart the atoms are.
        rng: Random source for choosing which images to sample.  Defaults to a fresh
            unseeded `numpy.random.Generator`.  Pass a seeded one to keep the choice
            reproducible without disturbing anyone else's random state.

    Returns:
        rij_list: List of the ``(i, j)`` atom pairs making up the coordinates.
        re: Equilibrium distance for each pair, taken as the sum of the two covalent
            radii.
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

    # Always look at both end points, plus a random selection of the images between
    # them, so that a long path costs no more to analyse than a short one
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(rng.choice(np.arange(1, len(geom) - 1), snapshots - 2, replace=False))
    # Build the neighbour list for each sampled image and merge them together
    rij_set = set(enforce)
    for image in images:
        tree = KDTree(geom[image])
        rij_set.update(map(tuple, tree.query_pairs(threshold, output_type='ndarray').tolist()))
        # Anything within three bonds of each other is included whatever the distance
        rij_set.update(_pairs_within_three_bonds(tree, n_atoms, bond_threshold))
    rij_list = sorted(rij_set)
    # Count how many pairs each atom appears in, so `min_neighbors` can be checked
    pairs = np.asarray(rij_list, dtype=int).reshape(-1, 2)
    count = np.bincount(pairs.ravel(), minlength=n_atoms)
    # Top up any under-connected atom with its nearest neighbours in the final geometry.
    # Atoms only ever gain neighbours here, so querying the whole under-connected set in
    # one go is safe; the count is still re-checked in turn, because an atom may have
    # been brought up to `min_neighbors` by an earlier atom's additions.
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


# Index bookkeeping derived from a pair list.  Building it costs about as much as one
# evaluation, and the same pair list is used for every image over the whole run, so the
# last few are kept around.  A strong reference to the list is held alongside its `id`,
# which is what makes the identity check sound: the list cannot be collected and have
# its address handed to something else while the entry lives.
_PAIR_INDEX_CACHE: dict[tuple[int, int, int],
                        tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_PAIR_INDEX_CACHE_SIZE = 8


def _pair_index(rij_list: list[tuple[int, int]],
                n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build, or look up, the index arrays describing a pair list.

    Args:
        rij_list: Indices of the atom pairs.
        n_atoms: Number of atoms, which sets the width of the B matrix.

    Returns:
        pairs: The pair list as an ``(n_pairs, 2)`` integer array.
        indptr, indices: CSR skeleton of the B matrix.  Every distance depends on six
            Cartesian components, three for each of its atoms.
        sign: ``+1`` where the pair is stored low index first, ``-1`` otherwise, so the
            gradients can be written in the column order CSR requires.
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


def compute_rij(geom: np.ndarray,
                rij_list: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a list of inter-atomic distances and their Cartesian derivatives.

    Args:
        geom: Cartesian geometry of all the atoms, shape ``(n_atoms, 3)``.
        rij_list: Indices of the atom pairs to evaluate.

    Returns:
        rij: The distance for each pair.
        b_mat: Wilson B matrix, of shape ``(n_pairs, n_atoms, 3)``, holding the
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


def compute_wij(geom: np.ndarray,
                rij_list: list[tuple[int, int]],
                func: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                sparse: bool = False) -> tuple[np.ndarray, np.ndarray | csr_matrix]:
    """Calculate a list of scaled distances and their Cartesian derivatives.

    Same as `compute_rij`, except each distance is passed through a scaling function
    which sets the metric of the internal coordinates.

    Args:
        geom: Cartesian geometry of all the atoms.  Flattened input is accepted and
            reshaped to ``(n_atoms, 3)``.
        rij_list: Indices of the atom pairs to evaluate.
        func: Scaling function returning both the scaled value and its derivative
            with respect to the raw distance.  Must broadcast over arrays.
        sparse: Return the B matrix as a sparse matrix rather than a dense array.  Only
            six of its entries per row are ever non-zero, so this is what the optimisers
            are given: it saves building the dense array, and lets scipy solve the
            least-squares steps iteratively instead of by dense factorisation.

    Returns:
        wij: The scaled distance for each pair.
        b_mat: Cartesian gradients of the scaled distances, of shape
            ``(n_pairs, 3 * n_atoms)``, with the atom and component axes flattened
            together so scipy.optimize can use it directly.  A dense array by default,
            or a `scipy.sparse.csr_matrix` when ``sparse`` is set.
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
    # Normalise, then apply the chain rule, in that order so the result matches the
    # dense branch bit for bit.  `sign` puts the gradient of the lower-numbered atom
    # first, which is the column order CSR wants.
    grad = (d_vec / rij[:, None]) * (d_wdr * sign)[:, None]
    data = np.concatenate([grad, -grad], axis=1).ravel()
    b_mat = csr_matrix((data, indices, indptr), shape=(len(pairs), geom.size))
    return wij, b_mat


def morse_scaler(re: float | np.ndarray = 1.5, alpha: float = 1.7,
                 beta: float = 0.01) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a scaling function based on a Morse potential.

    This is the metric geodesic interpolation uses by default.  A small ``beta / r``
    tail is added so that the coordinate keeps responding to atoms that are far apart,
    where the exponential has already decayed away.

    Args:
        re: Equilibrium distance.  Usually the per-pair array returned by
            `get_bond_list` rather than a single number.
        alpha: Decay constant of the exponential.  Larger values are more localised,
            which tracks a sharp energy landscape better, while smaller values have
            longer range and give smoother paths from few images.
        beta: Weight of the long-range tail.

    Returns:
        A function taking an array of distances to the scaled distances and their
        derivatives with respect to the raw distance, in the form `compute_wij` expects.
    """

    def scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale an array of distances, and differentiate the scaling."""
        ratio = x / re
        val1 = np.exp(alpha * (1.0 - ratio))
        val2 = beta / ratio
        d_val = (-alpha * val1 / re) - (val2 / x)
        return val1 + val2, d_val

    return scaler


def elu_scaler(re: float | np.ndarray = 2.0, alpha: float = 2.0,
               beta: float = 0.01) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a scaling function with an exponential tail and a linear core.

    Shaped like an ELU: beyond ``re`` the scaled distance decays exponentially, as in
    `morse_scaler`, while below ``re`` it continues linearly along the tangent at
    ``re`` instead of blowing up.  A ``beta * re / r`` tail is added for the same
    reason as in the Morse case.

    Args:
        re: Distance at which the behaviour switches from linear to exponential.
        alpha: Decay constant of the exponential, which also sets the linear slope.
        beta: Weight of the long-range tail.

    Returns:
        A function taking an array of distances to the scaled distances and their
        derivatives with respect to the raw distance, in the form `compute_wij` expects.
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
