"""Geometry manipulation for building reaction end states.

A NEB needs a product as well as a reactant, and for anything bigger than a
single bond rearrangement the product is the awkward one to draw by hand. The
functions here build one instead: :func:`get_dimer_bonded_cluster_indices`
works out which atoms form each half of a stacked dimer, and
:func:`get_best_flip_and_face_bases` swaps those halves over to give the
flipped structure, ready to pass to
:func:`~reactiontools.tools_reaction.prepare_neb`.

:func:`swap_bonding_configuration` does the same job for proton transfers,
moving one or more hydrogens across their hydrogen bonds. Where those two build
a product out of what the reaction is known to do,
:func:`seed_product_from_ts` builds one out of a transition state instead,
stepping past the saddle along the geodesic that reaches it -- no calculator,
and no need to have guessed the mechanism first. For structures that
already describe the same atoms, :func:`align_atom_sets` superposes one on the
other with the optimal rigid Kabsch transform, and :func:`atom_set_rmsd`
measures what remains.
"""

import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import permutations
from numbers import Integral
from typing import Any, TextIO

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS

from .tools_reaction import _check_converged, get_neb_path, quick_guess_path

#: Cosine below which the direction a path arrives at the saddle in counts as
#: unrelated to the direction the reaction is going in overall -- 0.5 is 60
#: degrees off. Interpolating between two structures that barely differ gives a
#: local direction made of numerical wander rather than chemistry, and this is
#: what tells the two apart: a real reaction comes in above 0.9, while two
#: copies of the same structure score about zero.
_SEED_MIN_ALIGNMENT = 0.5


def _alignment_positions(
    points: Sequence[Sequence[float]] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Return a validated ``(n, 3)`` floating-point coordinate array.

    Parameters
    ----------
    points : array_like
        Cartesian coordinates of shape ``(n, 3)``.
    name : str
        Name of the caller's argument, quoted back in the error messages.

    Returns
    -------
    numpy.ndarray
        The coordinates as a float array of shape ``(n, 3)``.

    Raises
    ------
    TypeError
        If *points* cannot be converted to a numeric array.
    ValueError
        If the shape is not ``(n, 3)``, no positions are supplied, or a
        coordinate is non-finite.
    """
    try:
        positions = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array-like collection of positions.") from exc

    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (n, 3); got {positions.shape}.")
    if len(positions) == 0:
        raise ValueError(f"{name} must contain at least one position.")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    return positions


def _alignment_weights(
    weights: Sequence[float] | np.ndarray | None,
    count: int,
) -> np.ndarray:
    """Return validated weights for a rigid fit or RMSD calculation.

    Parameters
    ----------
    weights : array_like or None
        One non-negative weight per correspondence, at least one of them
        positive. ``None`` weights every correspondence equally.
    count : int
        Number of correspondences the weights have to cover.

    Returns
    -------
    numpy.ndarray
        The weights as a float array of shape ``(count,)``.

    Raises
    ------
    TypeError
        If *weights* cannot be converted to a numeric array.
    ValueError
        If the length does not match *count*, a value is non-finite or
        negative, or every weight is zero.
    """
    if weights is None:
        return np.ones(count, dtype=float)

    try:
        weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("weights must be an array-like collection of numbers.") from exc

    if weights.shape != (count,):
        raise ValueError(f"weights must have shape ({count},); got {weights.shape}.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values.")
    if np.any(weights < 0.0):
        raise ValueError("weights must not contain negative values.")
    if not np.any(weights > 0.0):
        raise ValueError("at least one weight must be greater than zero.")
    return weights


def kabsch_transform(
    mobile_positions: Sequence[Sequence[float]] | np.ndarray,
    reference_positions: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best proper rigid transform from one point set to another.

    The points correspond by row. The returned transform uses NumPy's row
    vector convention, so it is applied as
    ``mobile_positions @ rotation + translation``. Reflections are excluded:
    the rotation is always right-handed, even when a reflected fit would have
    a lower residual.

    Parameters
    ----------
    mobile_positions, reference_positions : array_like
        Corresponding Cartesian coordinates, each with shape ``(n, 3)``.
    weights : array_like, optional
        One non-negative weight per correspondence. At least one must be
        positive. By default every point has equal weight.

    Returns
    -------
    tuple of numpy.ndarray
        ``(rotation, translation)`` with shapes ``(3, 3)`` and ``(3,)``.

    Raises
    ------
    TypeError
        If positions or weights cannot be converted to numeric arrays.
    ValueError
        If the coordinate shapes differ, no positions are supplied, a value
        is non-finite, or the weights are invalid.
    """
    mobile = _alignment_positions(mobile_positions, "mobile_positions")
    reference = _alignment_positions(reference_positions, "reference_positions")
    if mobile.shape != reference.shape:
        raise ValueError(
            "mobile_positions and reference_positions must have the same shape; "
            f"got {mobile.shape} and {reference.shape}."
        )

    fit_weights = _alignment_weights(weights, len(mobile))
    mobile_centre = np.average(mobile, axis=0, weights=fit_weights)
    reference_centre = np.average(reference, axis=0, weights=fit_weights)
    mobile_centred = mobile - mobile_centre
    reference_centred = reference - reference_centre

    covariance = (mobile_centred * fit_weights[:, None]).T @ reference_centred
    left, _singular_values, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose

    # Kabsch's unconstrained orthogonal fit may be a reflection. Negating one
    # singular vector chooses the best proper rotation instead.
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose

    translation = reference_centre - mobile_centre @ rotation
    return rotation, translation


def _atom_indices(
    atoms: Atoms,
    indices: int | Iterable[int] | None,
    name: str,
) -> np.ndarray:
    """Validate an alignment selection and return it as an integer array.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the selection indexes into.
    indices : int or iterable of int or None
        Atoms to select. ``None`` selects every atom in *atoms*.
    name : str
        Name of the caller's argument, quoted back in the error messages.

    Returns
    -------
    numpy.ndarray
        The selection as an integer array, in the order given.

    Raises
    ------
    TypeError
        If *indices* is neither an integer nor an iterable of integers.
    ValueError
        If the selection is empty or repeats an index.
    IndexError
        If an index lies outside *atoms*.
    """
    if indices is None:
        if len(atoms) == 0:
            raise ValueError(f"{name} must not be empty.")
        return np.arange(len(atoms), dtype=int)

    if isinstance(indices, Integral) and not isinstance(indices, bool):
        raw_indices = [indices]
    else:
        try:
            raw_indices = list(indices)
        except TypeError as exc:
            raise TypeError(f"{name} must contain integer atom indices.") from exc

    if not raw_indices:
        raise ValueError(f"{name} must not be empty.")
    if any(
        not isinstance(index, Integral) or isinstance(index, bool)
        for index in raw_indices
    ):
        raise TypeError(f"{name} must contain only integer atom indices.")

    selected = np.asarray(raw_indices, dtype=int)
    if len(np.unique(selected)) != len(selected):
        raise ValueError(f"{name} must not contain repeated atom indices.")
    if np.any(selected < 0) or np.any(selected >= len(atoms)):
        raise IndexError(f"{name} contains an index out of range for {len(atoms)} atoms.")
    return selected


def _atom_alignment_data(
    mobile: Atoms,
    reference: Atoms,
    mobile_indices: int | Iterable[int] | None,
    reference_indices: int | Iterable[int] | None,
    weights: str | Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve atom selections and weights shared by alignment operations.

    Parameters
    ----------
    mobile, reference : ase.Atoms
        Structure to move and structure to fit it onto.
    mobile_indices, reference_indices : int or iterable of int or None
        Corresponding selections used for the fit, matched by position within
        each. ``None`` selects every atom in that structure.
    weights : {None, "masses"} or array_like
        Fit weights, interpreted as in :func:`align_atom_sets`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mobile_positions, reference_positions, weights)``, the positions
        of shape ``(n, 3)`` and the weights of shape ``(n,)``.

    Raises
    ------
    TypeError
        If either structure is not an :class:`ase.Atoms`, or a selection or
        the weights have the wrong type.
    ValueError
        If the selections differ in length, *weights* is a string other than
        ``"masses"``, or a selection or weight is otherwise invalid.
    IndexError
        If a selection contains an atom index outside its structure.
    """
    if not isinstance(mobile, Atoms) or not isinstance(reference, Atoms):
        raise TypeError("mobile and reference must both be ase.Atoms objects.")

    mobile_selection = _atom_indices(mobile, mobile_indices, "mobile_indices")
    reference_selection = _atom_indices(
        reference, reference_indices, "reference_indices"
    )
    if len(mobile_selection) != len(reference_selection):
        raise ValueError(
            "mobile_indices and reference_indices must select the same number "
            f"of atoms; got {len(mobile_selection)} and {len(reference_selection)}."
        )

    if isinstance(weights, str):
        if weights != "masses":
            raise ValueError("weights must be None, 'masses', or a numeric sequence.")
        weights = mobile.get_masses()[mobile_selection]

    fit_weights = _alignment_weights(weights, len(mobile_selection))
    return (
        _alignment_positions(
            mobile.positions[mobile_selection], "selected mobile positions"
        ),
        _alignment_positions(
            reference.positions[reference_selection], "selected reference positions"
        ),
        fit_weights,
    )


def align_atom_sets(
    mobile: Atoms,
    reference: Atoms,
    mobile_indices: int | Iterable[int] | None = None,
    reference_indices: int | Iterable[int] | None = None,
    weights: str | Sequence[float] | np.ndarray | None = None,
) -> Atoms:
    """Rigidly superpose one atom set on a corresponding reference set.

    The selected atoms determine the least-squares Kabsch fit, while the
    resulting rotation and translation are applied to every atom in
    ``mobile``. This keeps the mobile structure internally rigid and makes it
    possible to fit on a stable substructure while carrying spectators or a
    flexible region along with it. Neither input is modified.

    Correspondence is positional: the first mobile selection index is matched
    to the first reference selection index, and so on. Atom identities are not
    automatically matched or reordered.

    Parameters
    ----------
    mobile, reference : ase.Atoms
        Structure to move and structure to fit it onto.
    mobile_indices, reference_indices : int or iterable of int, optional
        Corresponding selections used for the fit. Each defaults to every atom
        in its structure and the selections must have equal length.
    weights : {None, "masses"} or array_like, optional
        Fit weights. ``None`` gives every correspondence equal influence;
        ``"masses"`` uses the selected mobile atoms' masses; a numeric
        sequence supplies one non-negative weight per correspondence.

    Returns
    -------
    ase.Atoms
        A copy of ``mobile`` with every position transformed into the
        reference frame.

    Raises
    ------
    TypeError
        If either structure is not an :class:`ase.Atoms`, or selections or
        weights have the wrong type.
    ValueError
        If the selections differ in length, are empty or repeated, or weights
        are invalid.
    IndexError
        If a selection contains an atom index outside its structure.
    """
    mobile_fit, reference_fit, fit_weights = _atom_alignment_data(
        mobile, reference, mobile_indices, reference_indices, weights
    )
    rotation, translation = kabsch_transform(
        mobile_fit, reference_fit, weights=fit_weights
    )

    aligned = mobile.copy()
    # Alignment changes the coordinate frame, so positional constraints must
    # not suppress part of the rigid transform. The constraints themselves
    # remain attached to the returned copy.
    aligned.set_positions(
        mobile.positions @ rotation + translation, apply_constraint=False
    )
    return aligned


def atom_set_rmsd(
    mobile: Atoms,
    reference: Atoms,
    mobile_indices: int | Iterable[int] | None = None,
    reference_indices: int | Iterable[int] | None = None,
    weights: str | Sequence[float] | np.ndarray | None = None,
    align: bool = False,
) -> float:
    """Calculate the RMSD between corresponding atoms, optionally after fitting.

    Parameters are the same as :func:`align_atom_sets`. When ``align`` is
    ``True``, the optimal rigid transform is applied to the selected mobile
    coordinates before the RMSD is measured; the input structures still are
    not modified.

    Parameters
    ----------
    mobile, reference : ase.Atoms
        Structures whose corresponding positions are compared.
    mobile_indices, reference_indices : int or iterable of int, optional
        Corresponding selections to compare. Each defaults to every atom.
    weights : {None, "masses"} or array_like, optional
        RMSD and fit weights, interpreted as in :func:`align_atom_sets`.
    align : bool, optional
        Remove the best rigid rotation and translation before measuring.
        Defaults to ``False`` so the function can also report displacement in
        the current coordinate frame.

    Returns
    -------
    float
        Root-mean-square Cartesian distance in Å.
    """
    mobile_fit, reference_fit, fit_weights = _atom_alignment_data(
        mobile, reference, mobile_indices, reference_indices, weights
    )
    if align:
        rotation, translation = kabsch_transform(
            mobile_fit, reference_fit, weights=fit_weights
        )
        mobile_fit = mobile_fit @ rotation + translation

    squared_distances = np.sum((mobile_fit - reference_fit) ** 2, axis=1)
    return float(np.sqrt(np.average(squared_distances, weights=fit_weights)))


def bonded_cluster_indices_no_anchor_hub(
    atoms: Atoms, anchor: int, mult: float = 1.0, multi_h: float = 1.3
) -> list[int]:
    """Collect the atoms bonded to an anchor without routing through it.

    A plain flood fill over the bonded graph would leak through the anchor
    into whatever else it touches, which for a stacked dimer means swallowing
    both halves at once. Here the anchor is marked visited before the walk
    starts, so the search reaches its immediate neighbours and then spreads
    outwards from them only — the anchor is a starting point, never a
    thoroughfare.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose bonded graph is walked.
    anchor : int
        Index of the anchor atom, included in the returned cluster.
    mult : float, optional
        Scale applied to ASE's natural covalent cutoffs. Raise it to make
        bonding more permissive.
    multi_h : float, optional
        Separate cutoff scale for hydrogen, applied to its covalent radius.
        Hydrogen's natural cutoff is tight enough to miss hydrogen bonds,
        which are exactly the contacts that matter across a base pair.

    Returns
    -------
    list of int
        Sorted indices of the anchor and everything bonded to it, in ascending
        order.

    Raises
    ------
    IndexError
        If ``anchor`` is out of range.
    """
    n = len(atoms)
    if not (0 <= anchor < n):
        raise IndexError(f"Anchor index {anchor} out of range for {n} atoms.")

    cutoffs = natural_cutoffs(atoms, mult=mult)
    for i, atom in enumerate(atoms):
        if atom.symbol == "H":
            cutoffs[i] = covalent_radii[atom.number] * multi_h

    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    first_neighbors, _ = nl.get_neighbors(anchor)
    # NOTE: depends on atom ordering, dropping whichever atom sits two indices
    # before the anchor. Carried over from the original implementation.
    first_neighbors = [i for i in first_neighbors if i != anchor - 2]

    # The anchor counts as visited before the walk starts, so the search
    # spreads outwards from its neighbours and never back through it.
    visited = {anchor} | set(first_neighbors)
    stack = list(first_neighbors)

    while stack:
        i = stack.pop()
        nbrs, _ = nl.get_neighbors(i)
        for j in nbrs:
            if j == anchor:
                continue
            if j not in visited:
                visited.add(j)
                stack.append(j)

    return sorted(visited)


def get_dimer_bonded_cluster_indices(
    atoms: Atoms,
    anchors: list[int],
    mults: Sequence[float] | None = None,
    multi_h: float = 1.3,
) -> list[int]:
    """Collect the atoms of both halves of a dimer, as one index list.

    Runs :func:`bonded_cluster_indices_no_anchor_hub` from each anchor and
    merges the results, so an atom bridging the two halves is counted once.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose bonded graph is walked.
    anchors : list of int
        Exactly two anchor indices, one per half.
    mults : sequence of float, optional
        Per-anchor cutoff scales. Defaults to ``[1.0, 1.0]``.
    multi_h : float, optional
        Hydrogen cutoff scale, shared by both walks.

    Returns
    -------
    list of int
        Sorted union of the two clusters.

    Raises
    ------
    ValueError
        If ``anchors`` or ``mults`` does not hold exactly two values.
    """
    if mults is None:
        mults = [1.0, 1.0]

    if len(anchors) != 2:
        raise ValueError("Anchors list must contain exactly two indices.")

    if len(mults) != 2:
        raise ValueError("Mults list must contain exactly two values.")

    base_a = bonded_cluster_indices_no_anchor_hub(
        atoms, anchors[0], mult=mults[0], multi_h=multi_h
    )
    base_b = bonded_cluster_indices_no_anchor_hub(
        atoms, anchors[1], mult=mults[1], multi_h=multi_h
    )

    return sorted(set(base_a + base_b))


def _pca_frame(
    positions: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a right-handed frame to a set of points by PCA.

    For a roughly planar group the two largest-variance axes span the plane
    and the smallest-variance axis is its normal, which is what makes this a
    usable local frame for a flat molecular fragment.

    Parameters
    ----------
    positions : array_like
        Points of shape ``(n, 3)``.

    Returns
    -------
    tuple
        ``(origin, R)`` with the centroid and a ``(3, 3)`` rotation whose
        columns are the x, y and z axes, z being the plane normal.
    """
    pts = np.asarray(positions)
    origin = pts.mean(axis=0)
    X = pts - origin
    # PCA by SVD: the principal axes are the rows of Vt, in descending order
    # of variance, so the last of them is the plane normal.
    _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
    x = Vt[0]
    y = Vt[1]
    z = Vt[2]
    # Rebuild y and z from cross products rather than trusting SVD's output to
    # be exactly orthonormal and right-handed.
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    R = np.vstack([x, y, z]).T  # columns are axes
    return origin, R


def _orient_normal_toward(
    R: np.ndarray,
    origin: Sequence[float] | np.ndarray,
    target_point: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Flip a frame's normal so it points at a target.

    SVD fixes each axis only up to a sign, so two fragments fitted
    independently can come back with normals pointing opposite ways. Pinning
    both at the other fragment removes that ambiguity.

    Parameters
    ----------
    R : numpy.ndarray
        ``(3, 3)`` frame whose columns are the axes.
    origin : array_like
        Origin the frame is anchored at.
    target_point : array_like
        Point the normal should point towards.

    Returns
    -------
    numpy.ndarray
        The frame, with y and z negated together if z pointed away, which
        flips the normal while keeping the frame right-handed.
    """
    z = R[:, 2]
    d = np.asarray(target_point) - np.asarray(origin)
    if np.dot(z, d) < 0.0:
        R = np.column_stack((R[:, 0], -R[:, 1], -R[:, 2]))
    return R


def _rigid_transform(
    points: Sequence[Sequence[float]] | np.ndarray,
    anchor_pos: Sequence[float] | np.ndarray,
    R_target: np.ndarray,
    new_anchor_pos: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Rotate points about one anchor and translate them onto another.

    Parameters
    ----------
    points : array_like
        Points of shape ``(n, 3)``.
    anchor_pos : array_like
        Point the rotation is taken about.
    R_target : numpy.ndarray
        ``(3, 3)`` rotation to apply.
    new_anchor_pos : array_like
        Where the anchor ends up.

    Returns
    -------
    numpy.ndarray
        Transformed points, same shape as the input.
    """
    P = np.asarray(points) - anchor_pos
    P_rot = P @ R_target.T
    return P_rot + new_anchor_pos


def flip_and_face_bases(
    atoms: Atoms,
    baseA_idxs: list,
    baseB_idxs: list,
    anchors: list,
    rot_matrix: list | None = None,
) -> Atoms:
    """Swap two fragments over, each landing on the other's anchor.

    Fits a local frame to each fragment, maps one frame onto the other through
    a reflection, and moves each fragment onto the opposite anchor. The
    reflection is what makes the two end up facing each other rather than
    merely translated, so the result is a plausible flipped end state rather
    than a structure with the fragments back to back.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to rearrange. Not modified; a copy is returned.
    baseA_idxs : list of int
        Indices of the first fragment.
    baseB_idxs : list of int
        Indices of the second fragment.
    anchors : list of int
        Two anchor indices, one per fragment, in the same order as the
        fragments. Fragment A lands on anchor B and vice versa.
    rot_matrix : list of float, optional
        Diagonal of the reflection applied between the two frames, as three
        signs. Defaults to ``[-1.0, 1.0, -1.0]``. Which one is right depends on
        the geometry; :func:`get_best_flip_and_face_bases` searches them.

    Returns
    -------
    ase.Atoms
        Copy of ``atoms`` with the two fragments swapped. Atoms outside both
        fragments keep their positions.
    """
    anchorA_idx = anchors[0]
    anchorB_idx = anchors[1]
    atoms = atoms.copy()

    pos = atoms.get_positions()
    baseA = np.array(baseA_idxs, dtype=int)
    baseB = np.array(baseB_idxs, dtype=int)

    anchorA = pos[anchorA_idx].copy()
    anchorB = pos[anchorB_idx].copy()

    originA, RA = _pca_frame(pos[baseA])
    originB, RB = _pca_frame(pos[baseB])

    # Point both normals at the other fragment, so that "facing" means the
    # same thing for each of them.
    RB = _orient_normal_toward(RB, originB, originA)
    RA = _orient_normal_toward(RA, originA, originB)

    # The reflection flips the normal while keeping x, so the fragments end up
    # facing each other rather than back to back.
    if rot_matrix is None:
        rot_matrix = [-1.0, 1.0, -1.0]
    M = np.diag(rot_matrix)

    # Mapping A's frame onto B's: R_target_A @ RA == RB @ M, and RA is
    # orthonormal, so R_target_A = RB @ M @ RA.T. B onto A is the mirror image.
    R_target_A = RB @ M @ RA.T
    R_target_B = RA @ M @ RB.T

    newA = _rigid_transform(pos[baseA], anchorA, R_target_A, anchorB)
    newB = _rigid_transform(pos[baseB], anchorB, R_target_B, anchorA)

    new_pos = pos.copy()
    new_pos[baseA] = newA
    new_pos[baseB] = newB
    atoms.set_positions(new_pos)
    return atoms


def optimize_with_fixed_anchors(
    atoms: Atoms,
    baseA_idxs: list,
    baseB_idxs: list,
    anchor_indices: list,
    calc: Any,
    fmax: float = 0.05,
    steps: int = 1000,
    raise_on_unconverged: bool = False,
    optimiser: Callable[..., Any] = BFGS,
    logfile: str | TextIO | None = "-",
) -> Atoms:
    """Relax the two fragments while holding their anchors still.

    A structure straight out of :func:`flip_and_face_bases` is rigid-body
    exact but strained, because the fragments were moved as blocks. This
    relieves the strain without letting the fragments drift back, which is
    what pinning the anchors buys.

    Only the fragment atoms are optimised; anything outside both fragments is
    left exactly where it was.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to relax. Not modified; a copy is returned.
    baseA_idxs : list of int
        Indices of the first fragment.
    baseB_idxs : list of int
        Indices of the second fragment.
    anchor_indices : list of int
        Indices held fixed during the relaxation.
    calc : ase.calculators.Calculator
        Calculator used for the relaxation.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps. A flipped structure needing more
        than the default has usually landed somewhere unphysical, which the
        :class:`~reactiontools.ConvergenceWarning` then says out loud instead
        of grinding on against ASE's effectively unlimited default.
    raise_on_unconverged : bool, optional
        Raise :exc:`~reactiontools.ConvergenceError` instead of warning when
        the relaxation hits ``steps`` without reaching ``fmax``.
    optimiser : callable, optional
        ASE optimiser class to relax with, or anything callable as
        ``optimiser(atoms, logfile=...)``. Defaults to
        :class:`~ase.optimize.BFGS`. See
        :func:`~reactiontools.tools_reaction.optimise_geom`.
    logfile : str, file object or None, optional
        Where the optimiser writes its per-step table. ``'-'``, the default,
        is stdout; a filename writes there instead; ``None`` silences it.

    Returns
    -------
    ase.Atoms
        Copy of ``atoms`` with the relaxed fragment positions written back,
        and ``info["converged"]`` recording whether the relaxation reached
        ``fmax``.

    Raises
    ------
    ConvergenceError
        If the relaxation did not converge and ``raise_on_unconverged`` is
        True.
    """
    atoms_opt = atoms.copy()
    constraint = FixAtoms(indices=anchor_indices)
    atoms_opt.set_constraint(constraint)

    selection = list(baseA_idxs) + list(baseB_idxs)
    atoms_opt = atoms_opt[selection]

    atoms_opt.calc = calc
    converged = optimiser(atoms_opt, logfile=logfile).run(fmax=fmax, steps=steps)

    # Write the relaxed coordinates back through the full position array:
    # indexing an Atoms object returns a new object, so assigning to
    # atoms_out[selection] would update a copy and discard the optimisation.
    atoms_out = atoms.copy()
    positions = atoms_out.get_positions()
    positions[selection] = atoms_opt.get_positions()
    atoms_out.set_positions(positions)
    atoms_out.info["converged"] = _check_converged(
        converged, "Fixed-anchor relaxation", fmax, steps, raise_on_unconverged
    )

    return atoms_out


def get_best_flip_and_face_bases(
    atoms: Atoms,
    baseA_idxs: list,
    baseB_idxs: list,
    anchors: list,
    optimise_after: bool = True,
    calc: Any = None,
    raise_on_unconverged: bool = False,
    optimiser: Callable[..., Any] = BFGS,
    logfile: str | TextIO | None = "-",
) -> Atoms:
    """Search the reflection signs for the tightest flipped structure.

    :func:`flip_and_face_bases` takes a reflection whose correct signs depend
    on how the fragments happen to sit, and a wrong choice throws them apart.
    This tries every sign combination that is a genuine reflection and keeps
    whichever leaves the two fragment centres of mass closest together.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to rearrange. Not modified.
    baseA_idxs : list of int
        Indices of the first fragment.
    baseB_idxs : list of int
        Indices of the second fragment.
    anchors : list of int
        Two anchor indices, one per fragment.
    optimise_after : bool, optional
        Relax the result with :func:`optimize_with_fixed_anchors`.
    calc : ase.calculators.Calculator, optional
        Calculator for that relaxation. Required when ``optimise_after`` is
        ``True``.
    raise_on_unconverged : bool, optional
        Passed to :func:`optimize_with_fixed_anchors`; ignored when
        ``optimise_after`` is ``False``, since nothing is then relaxed.
    optimiser, logfile
        Passed to :func:`optimize_with_fixed_anchors` for the single
        relaxation at the end; the reflection search itself moves the
        fragments rigidly and never touches the calculator. Ignored when
        ``optimise_after`` is ``False``.

    Returns
    -------
    ase.Atoms
        Flipped structure, relaxed when ``optimise_after`` is ``True``, in
        which case ``info["converged"]`` records whether that relaxation
        reached ``fmax``.

    Raises
    ------
    ValueError
        If ``optimise_after`` is ``True`` and no calculator is given.
    ConvergenceError
        If the relaxation did not converge and ``raise_on_unconverged`` is
        True.
    """
    if optimise_after and calc is None:
        raise ValueError(
            "optimise_after=True needs a calculator; pass calc=, "
            "or set optimise_after=False to skip the relaxation."
        )

    # Sorted so the search order -- and which matrix wins a tie in the COM
    # distance below -- does not depend on set iteration order.
    rot_matrix_permutations = sorted(
        set(permutations((-1.0, 1.0, 1.0))) | set(permutations((-1.0, -1.0, 1.0)))
    )
    print(f"All permutations of rot_matrix: {rot_matrix_permutations}", flush=True)

    best_rot_matrix = None
    best_dist_after = float("inf")
    for rot_matrix in rot_matrix_permutations:
        rot_matrix = list(rot_matrix)
        print(f"Trying rot_matrix: {rot_matrix}", flush=True)
        swapped = flip_and_face_bases(
            atoms,
            baseA_idxs=baseA_idxs,
            baseB_idxs=baseB_idxs,
            anchors=anchors,
            rot_matrix=rot_matrix,
        )

        com_a_before = atoms[baseA_idxs].get_center_of_mass()
        com_b_before = atoms[baseB_idxs].get_center_of_mass()
        com_a_after = swapped[baseA_idxs].get_center_of_mass()
        com_b_after = swapped[baseB_idxs].get_center_of_mass()

        dist_before = np.linalg.norm(com_a_before - com_b_before)
        dist_after = np.linalg.norm(com_a_after - com_b_after)

        print(f"dist_before COM: {dist_before}", flush=True)
        print(f"dist_after COM:  {dist_after}", flush=True)
        print(flush=True)

        if dist_after < best_dist_after:
            best_dist_after = dist_after
            best_rot_matrix = rot_matrix

    print("Best rot_matrix:", best_rot_matrix)

    swapped = flip_and_face_bases(
        atoms,
        baseA_idxs=baseA_idxs,
        baseB_idxs=baseB_idxs,
        anchors=anchors,
        rot_matrix=best_rot_matrix,
    )
    if optimise_after:
        swapped = optimize_with_fixed_anchors(
            swapped,
            baseA_idxs=baseA_idxs,
            baseB_idxs=baseB_idxs,
            anchor_indices=anchors,
            calc=calc,
            raise_on_unconverged=raise_on_unconverged,
            optimiser=optimiser,
            logfile=logfile,
        )

    return swapped


def swap_bonding_configuration(
    atoms: Atoms,
    donor_index: int | Iterable[int],
    hydrogen_index: int | Iterable[int],
    acceptor_index: int | Iterable[int],
) -> Atoms:
    """Swap one or more donor-H...acceptor bonds to donor...H-acceptor.

    Builds the product end state of one or more proton transfers. Each hydrogen
    is moved to the acceptor side of its hydrogen bond, keeping the bond length
    it had to its donor, so the result is a sensible starting geometry for
    :func:`~reactiontools.tools_reaction.optimise_geom` and then a band.

    A scalar donor or acceptor index is shared by all the hydrogens. Otherwise,
    donor and acceptor iterables must contain one index per hydrogen.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure holding the hydrogen bond. Not modified; a copy is returned.
    donor_index : int or iterable of int
        Index of a shared donor atom, or one donor index per hydrogen.
    hydrogen_index : int or iterable of int
        Index or indices of the hydrogens being moved.
    acceptor_index : int or iterable of int
        Index of a shared acceptor atom, or one acceptor index per hydrogen.

    Returns
    -------
    ase.Atoms
        Copy of ``atoms`` with each hydrogen on its acceptor side.

    Raises
    ------
    TypeError
        If an index argument is neither an integer nor an iterable of integers.
    ValueError
        If no hydrogen is supplied, donor or acceptor counts do not match the
        hydrogen count, a hydrogen is repeated, a transfer reuses an atom in
        more than one role, or a donor and acceptor occupy the same position.
    IndexError
        If an atom index is out of range.
    """
    def as_indices(value: int | Iterable[int], name: str) -> list[int]:
        """Normalise one index argument to a list of plain integers.

        Parameters
        ----------
        value : int or iterable of int
            A single shared index, or one index per hydrogen.
        name : str
            Name of the caller's argument, quoted back in the error messages.

        Returns
        -------
        list of int
            The indices, in the order given.

        Raises
        ------
        TypeError
            If *value* is neither an integer nor an iterable of integers.
        ValueError
            If an empty iterable is supplied.
        """
        if isinstance(value, Integral) and not isinstance(value, bool):
            return [int(value)]

        try:
            indices = list(value)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an integer or iterable of integers."
            ) from exc

        if not indices:
            raise ValueError(f"{name} must not be empty.")
        if any(
            not isinstance(index, Integral) or isinstance(index, bool)
            for index in indices
        ):
            raise TypeError(f"{name} must contain only integers.")
        return [int(index) for index in indices]

    donors = as_indices(donor_index, "donor_index")
    hydrogens = as_indices(hydrogen_index, "hydrogen_index")
    acceptors = as_indices(acceptor_index, "acceptor_index")
    transfer_count = len(hydrogens)

    def one_per_hydrogen(indices: list[int], name: str) -> list[int]:
        """Broadcast a shared index over the hydrogens, or check the count.

        Parameters
        ----------
        indices : list of int
            Indices as ``as_indices`` returned them.
        name : str
            Name of the caller's argument, quoted back in the error message.

        Returns
        -------
        list of int
            One index per hydrogen, the single shared index repeated when
            only one was given.

        Raises
        ------
        ValueError
            If more than one index was given and the count does not match the
            number of hydrogens.
        """
        if len(indices) == 1:
            return indices * transfer_count
        if len(indices) != transfer_count:
            raise ValueError(
                f"{name} must contain one index or one index per hydrogen "
                f"({transfer_count}); got {len(indices)}."
            )
        return indices

    donors = one_per_hydrogen(donors, "donor_index")
    acceptors = one_per_hydrogen(acceptors, "acceptor_index")

    if len(set(hydrogens)) != transfer_count:
        raise ValueError("hydrogen_index must not contain repeated indices.")

    atom_count = len(atoms)
    for name, indices in (
        ("donor_index", donors),
        ("hydrogen_index", hydrogens),
        ("acceptor_index", acceptors),
    ):
        for index in indices:
            if not 0 <= index < atom_count:
                raise IndexError(
                    f"{name} contains index {index}, which is out of range for "
                    f"{atom_count} atoms."
                )

    for donor, hydrogen, acceptor in zip(donors, hydrogens, acceptors):
        if len({donor, hydrogen, acceptor}) != 3:
            raise ValueError(
                "Each transfer needs distinct donor, hydrogen, and acceptor indices."
            )
        if atoms[hydrogen].symbol != "H":
            raise ValueError(
                f"hydrogen_index contains index {hydrogen}, whose element is "
                f"{atoms[hydrogen].symbol}, not H."
            )

    donor_positions = atoms.positions[donors]
    hydrogen_positions = atoms.positions[hydrogens]
    acceptor_positions = atoms.positions[acceptors]
    directions = acceptor_positions - donor_positions
    donor_acceptor_distances = np.linalg.norm(directions, axis=1)
    if np.any(donor_acceptor_distances == 0):
        raise ValueError("Donor and acceptor positions must be different.")

    directions /= donor_acceptor_distances[:, np.newaxis]
    donor_hydrogen_distances = np.linalg.norm(
        hydrogen_positions - donor_positions, axis=1
    )
    new_hydrogen_positions = (
        acceptor_positions - directions * donor_hydrogen_distances[:, np.newaxis]
    )

    swapped = atoms.copy()
    swapped.positions[hydrogens] = new_hydrogen_positions
    return swapped


class SeedWarning(UserWarning):
    """A seeded end state did not come out looking like a new structure.

    Warned rather than raised by :func:`seed_product_from_ts`, because a seed
    that stopped short of where it was aimed is still a starting point, and
    because whether it landed in the right basin is settled by relaxing it
    rather than by measuring it. Turn it into an error with
    ``warnings.simplefilter("error", SeedWarning)`` in a batch script, where a
    seed that is really a second copy of the reactant would otherwise be
    carried into everything downstream.
    """


def _seed_clash(
    atoms: Atoms,
    radii_sum: np.ndarray,
    clash_scale: float,
) -> tuple[int, int, float, float] | None:
    """Find the worst-compressed contact in a structure, if there is one.

    The geometric stand-in for an energy: with no calculator to say a step has
    gone somewhere unphysical, what says it instead is two atoms driven closer
    than any bond between them would ever sit.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to check.
    radii_sum : numpy.ndarray
        ``(n_atoms, n_atoms)`` sums of covalent radii.
    clash_scale : float
        Fraction of a pair's covalent radii below which their separation
        counts as a clash.

    Returns
    -------
    tuple or None
        ``(i, j, distance, limit)`` for the worst offending pair, or None when
        nothing is compressed past ``clash_scale``.
    """
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    scaled = distances / radii_sum
    # Every atom sits at zero distance from itself, which would otherwise be
    # the worst contact in every structure.
    np.fill_diagonal(scaled, np.inf)
    i, j = np.unravel_index(np.argmin(scaled), scaled.shape)
    if scaled[i, j] >= clash_scale:
        return None
    return int(i), int(j), float(distances[i, j]), float(radii_sum[i, j])


def seed_product_from_ts(
    reactant: Atoms,
    ts: Atoms,
    n_images: int = 25,
    push: float = 1.0,
    n_steps: int = 10,
    tangent_images: int = 2,
    weights: str | Sequence[float] | np.ndarray | None = "masses",
    clash_scale: float | None = 0.7,
    return_path: bool = False,
) -> Atoms | tuple[Atoms, list[Atoms]]:
    """Seed the far end state by stepping past a transition state.

    Geodesically interpolates from ``reactant`` to ``ts``, reads the direction
    the path is travelling in as it arrives at the saddle, and keeps stepping
    that way past it. What comes out is the structure on the other side: a
    product guess built from a reactant and a transition state alone.

    This is the cheap counterpart to
    :func:`~reactiontools.tools_reaction.optimise_irc`, which answers the same
    question properly by following the true reaction coordinate downhill, at
    the cost of a converged saddle, a calculator and hundreds of gradients.
    Nothing here is evaluated: the seed is unrelaxed, and the way to turn it
    into an end state worth using is to relax it with
    :func:`~reactiontools.tools_reaction.optimise_geom`. Running
    :func:`~reactiontools.tools_reaction.prepare_neb` between the reactant and
    the relaxed seed, and checking the band comes back over a barrier near the
    transition state it started from, is what confirms the seed landed in the
    basin it was aimed at.

    Nothing about this is specific to products. The reactant is only the end
    state you already have, so passing a product as ``reactant`` seeds the
    reactant instead, by stepping off the other side of the same saddle.

    Parameters
    ----------
    reactant : ase.Atoms
        End state to start from. Not modified.
    ts : ase.Atoms
        Transition state, or a guess at one -- it is used for its geometry and
        nothing else, so it need not be converged. Not modified. Must describe
        the same atoms, in the same order, as ``reactant``.
    n_images : int, optional
        Number of images in the geodesic interpolation from ``reactant`` to
        ``ts``. More images resolve the arriving direction more finely.
    push : float, optional
        How far past the transition state to step, as a multiple of the
        geodesic length from ``reactant`` to ``ts``. The default of 1.0 steps
        as far beyond the saddle as the reactant sits before it, which for a
        near-symmetric reaction -- a proton transfer, say -- puts the seed
        roughly where the reactant would be reflected through the saddle.
        Raise it for a reaction whose product lies further out, lower it for a
        saddle that sits late along the path.
    n_steps : int, optional
        Number of equal increments the push is taken in. Nothing but the clash
        check looks between them, so this sets both how finely a push that has
        to stop early can stop and how well that check sees: increments long
        enough to carry an atom clean through another one step over the clash
        without noticing it.
    tangent_images : int, optional
        Number of trailing images the arrival direction is measured over. The
        default of 2 is the difference between the last two images; a larger
        chord is less sensitive to the geodesic wobbling near its endpoint,
        at the cost of averaging in curvature from further back.
    weights : {"masses", None} or array_like, optional
        How the two images are superposed before their difference is taken,
        interpreted as in :func:`align_atom_sets`.
        The default weights by mass, which is the frame a reaction coordinate
        is conventionally defined in and which keeps the heavy atoms still
        while a hydrogen does the travelling. Weighting every atom equally
        instead spreads a proton's motion back over the atoms it left, and
        extrapolating that pulls them apart. Pass None for equal weights, or
        an array to fix the frame on part of the structure.
    clash_scale : float or None, optional
        Stop stepping before any two atoms come closer than this fraction of
        the sum of their covalent radii, and warn :class:`SeedWarning` saying
        how far it got. The default of 0.7 sits well inside a normal bond, so
        a bond forming across the saddle does not trip it, while a step
        driving atoms through each other does. None steps the whole way
        regardless.
    return_path : bool, optional
        Also return every structure the seed was built from.

    Returns
    -------
    ase.Atoms or tuple
        The seeded end state, carrying ``info["seeded"]``, whether both the
        checks below passed; ``info["seed_push"]``,
        ``info["seed_rmsd_reactant"]`` and ``info["seed_rmsd_ts"]``, how far it
        actually travelled, in Å; and ``info["seed_alignment"]``, how much the
        direction it was stepped along had to do with the direction the path
        travelled overall, as a cosine. With ``return_path`` it is instead
        ``(seed, path)``, where ``path`` is the geodesic from the reactant to
        the transition state followed by every extrapolated image, ending on
        the seed -- a band to plot, or to hand to
        :func:`~reactiontools.tools_reaction.restart_neb`. It is
        ``n_images + n_steps`` long unless the push stopped early. The whole
        band is put in the frame of ``ts``, so its image ``n_images - 1`` is
        exactly the transition state as given, while its first image is the
        reactant up to the rigid drift the interpolation accumulated along the
        way.

    Raises
    ------
    ValueError
        If ``reactant`` and ``ts`` do not describe the same atoms, if any of
        the arguments are out of range, or if the two structures are so alike
        that the path between them gives no direction to extrapolate along.

    Warns
    -----
    SeedWarning
        If the push had to stop early to avoid a clash; if the seed did not end
        up further from the reactant than the transition state already was,
        which means the step went nowhere useful; or if the direction the path
        arrived in bore little relation to the direction it travelled overall,
        which means there was no reaction coordinate there to read.

    Notes
    -----
    Geodesic interpolation measures plain Cartesian distances, with no
    minimum-image convention and no knowledge of the cell, and hands its path
    back rigidly rotated. For a periodic system that rotation is meaningless,
    since the cell does not follow it -- which is why the band is put back in
    the frame of ``ts`` before anything is measured off it, and why the seed
    comes out where the cell expects it. What that does not rescue is a
    reaction whose atoms cross a cell boundary: the interpolation never sees
    the periodic image, so there is nothing sensible to extrapolate and this is
    the wrong tool for it.

    Examples
    --------
    >>> seed = seed_product_from_ts(reactant, ts)        # doctest: +SKIP
    >>> product = optimise_geom(seed, calc)              # doctest: +SKIP
    >>> summarise_neb(optimise_neb(prepare_neb(reactant, product, calc)))
    ...                                                  # doctest: +SKIP
    """
    if len(reactant) != len(ts):
        raise ValueError(
            f"reactant and ts must describe the same atoms, got "
            f"{len(reactant)} and {len(ts)}"
        )
    if reactant.get_chemical_symbols() != ts.get_chemical_symbols():
        raise ValueError(
            "reactant and ts must have the same chemical symbols in the same "
            "order, so that the path between them connects each atom to "
            "itself. Reorder one of them to match the other."
        )
    if n_images < 3:
        raise ValueError(
            f"Geodesic interpolation needs an interior image to move, so "
            f"n_images must be at least 3, got {n_images}"
        )
    if not 2 <= tangent_images <= n_images:
        raise ValueError(
            f"tangent_images must be between 2 and n_images={n_images} to "
            f"measure a direction across the path, got {tangent_images}"
        )
    if push <= 0:
        raise ValueError(
            f"push must be positive to step past the transition state, got "
            f"{push}. Swap reactant and ts to step the other way."
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if atom_set_rmsd(reactant, ts, align=True) < 1e-6:
        raise ValueError(
            "reactant and ts are the same structure, so there is no reaction "
            "coordinate to read a direction off and nothing to step past."
        )

    path = quick_guess_path(reactant, ts, n_images=n_images)

    # Geodesic interpolation hands the path back in a frame of its own, so the
    # last image is the transition state rotated and shifted away from where
    # the caller put it. Taking the transform from that image and applying it
    # to the whole path moves the band into the caller's frame in one piece,
    # leaving the seed a plain displacement of the transition state as given,
    # overlayable on it. The fit is exact rather than a compromise, because
    # the image being fitted is that same structure and not merely a similar
    # one -- which is why it is anchored on the transition state and not on
    # the reactant at the other end.
    rotation, translation = kabsch_transform(path[-1].positions, ts.positions)
    for image in path:
        # apply_constraint=False: this is a change of frame, not a step, and a
        # constraint would hold the fixed atoms behind in the old one.
        image.set_positions(
            image.positions @ rotation + translation, apply_constraint=False
        )

    # Superpose the trailing image on the last one before differencing them:
    # the difference of two frames that differ by a rotation is mostly that
    # rotation, and the direction wanted here is the internal motion alone.
    behind = align_atom_sets(path[-tangent_images], path[-1], weights=weights)
    direction = path[-1].positions - behind.positions

    # Frobenius norm over the whole 3N vector, the same measure get_neb_path
    # sums, so that push is a multiple of a length on the same scale.
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError(
            "The last two images of the path came back identical, so there is "
            "no direction to extrapolate along. Raise n_images, or lower "
            "tangent_images to measure across a shorter stretch of the path."
        )
    direction = direction / norm

    # How much the direction the path arrives in has to do with where the path
    # came from. Interpolation between two structures that differ only a
    # little returns one whose local direction is mostly numerical wander, and
    # extrapolating that steps somewhere arbitrary; this is the number that
    # shows it, and it is worth reporting even when it is healthy.
    start = align_atom_sets(path[0], path[-1], weights=weights)
    overall = path[-1].positions - start.positions
    alignment = float(np.sum(direction * overall) / np.linalg.norm(overall))

    step = push * get_neb_path(path)[-1] / n_steps

    if clash_scale is not None:
        radii = covalent_radii[ts.get_atomic_numbers()]
        radii_sum = radii[:, None] + radii[None, :]

    extrapolated = []
    stopped_at = None
    for i in range(1, n_steps + 1):
        image = ts.copy()
        # Whether the transition state converged says nothing about a
        # structure extrapolated away from it.
        image.info.pop("converged", None)
        # apply_constraint=True, and set_positions rather than assigning to
        # .positions, which writes the array straight through: this is a step,
        # not a change of frame, so an atom the caller fixed has to stay where
        # it was fixed. It also means the geometry checked for clashes below
        # is the one that would be kept.
        image.set_positions(
            path[-1].positions + i * step * direction, apply_constraint=True
        )

        if clash_scale is not None:
            clash = _seed_clash(image, radii_sum, clash_scale)
            if clash is not None:
                stopped_at = (i, clash)
                break

        extrapolated.append(image)

    if stopped_at is not None:
        i, (first, second, distance, limit) = stopped_at
        warnings.warn(
            f"Seeding stopped after {i - 1} of {n_steps} steps "
            f"({(i - 1) * step:.3f} Å of the {n_steps * step:.3f} Å push): the "
            f"next step brought {ts.symbols[first]}{first} and "
            f"{ts.symbols[second]}{second} within {distance:.3f} Å, under "
            f"clash_scale={clash_scale} of their {limit:.3f} Å covalent radii. "
            f"Lower push, or pass clash_scale=None to step anyway.",
            SeedWarning,
            stacklevel=2,
        )

    # Every step clashed, so there is nothing to hand back but the transition
    # state itself, in the frame the path put it in.
    if not extrapolated:
        seed = ts.copy()
        seed.info.pop("converged", None)
        seed.set_positions(path[-1].positions, apply_constraint=False)
    else:
        seed = extrapolated[-1]

    # Plain Cartesian RMSDs, whatever ``weights`` says: a mass-weighted one
    # barely registers a hydrogen crossing, which is exactly the motion these
    # numbers are here to report on.
    rmsd_reactant = atom_set_rmsd(seed, reactant, align=True)
    rmsd_ts = atom_set_rmsd(seed, ts, align=True)
    ts_from_reactant = atom_set_rmsd(ts, reactant, align=True)

    if rmsd_reactant <= ts_from_reactant:
        warnings.warn(
            f"The seed is {rmsd_reactant:.3f} Å from the reactant, no further "
            f"than the transition state already was ({ts_from_reactant:.3f} "
            f"Å), so stepping past the saddle went nowhere useful. Raise push, "
            f"or check that ts really lies between the two end states.",
            SeedWarning,
            stacklevel=2,
        )
    if alignment < _SEED_MIN_ALIGNMENT:
        warnings.warn(
            f"The path arrives at the transition state {alignment:.2f} aligned "
            f"with the direction it travelled overall, so the direction the "
            f"seed was stepped along has little to do with the reaction. "
            f"reactant and ts are probably too alike to interpolate between "
            f"usefully.",
            SeedWarning,
            stacklevel=2,
        )

    seed.info["seeded"] = bool(
        rmsd_reactant > ts_from_reactant and alignment >= _SEED_MIN_ALIGNMENT
    )
    seed.info["seed_push"] = float(len(extrapolated) * step)
    seed.info["seed_alignment"] = alignment
    seed.info["seed_rmsd_reactant"] = rmsd_reactant
    seed.info["seed_rmsd_ts"] = rmsd_ts

    if return_path:
        return seed, list(path) + extrapolated
    return seed
