"""Geometry manipulation for building reaction end states.

A NEB needs a product as well as a reactant, and for anything bigger than a
single bond rearrangement the product is the awkward one to draw by hand. The
functions here build one instead: :func:`get_dimer_bonded_cluster_indices`
works out which atoms form each half of a stacked dimer, and
:func:`get_best_flip_and_face_bases` swaps those halves over to give the
flipped structure, ready to pass to
:func:`~reactiontools.tools_reaction.prepare_neb`.

:func:`swap_bonding_configuration` does the same job for the much smaller case
of a proton transfer, moving one hydrogen across a hydrogen bond.
"""

from itertools import permutations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS

from .tools_reaction import _check_converged


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
    atoms: Atoms, anchors: list[int], mults=None, multi_h: float = 1.3
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


def _pca_frame(positions):
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


def _orient_normal_toward(R, origin, target_point):
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


def _rigid_transform(points, anchor_pos, R_target, new_anchor_pos):
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
    calc,
    fmax: float = 0.05,
    steps: int = 1000,
    raise_on_unconverged: bool = False,
    optimiser=BFGS,
    logfile="-",
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
    calc=None,
    raise_on_unconverged: bool = False,
    optimiser=BFGS,
    logfile="-",
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


def swap_bonding_configuration(atoms, donor_index, hydrogen_index, acceptor_index):
    """Swap an O-H...O hydrogen bond over to O...H-O.

    Builds the product end state of a proton transfer: the hydrogen is moved
    to the acceptor side of the hydrogen bond, keeping the same bond length it
    had to the donor, so the result is a sensible starting geometry for
    :func:`~reactiontools.tools_reaction.optimise_geom` and then a band.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure holding the hydrogen bond. Not modified; a copy is returned.
    donor_index : int
        Index of the donor oxygen.
    hydrogen_index : int
        Index of the hydrogen being moved.
    acceptor_index : int
        Index of the acceptor oxygen.

    Returns
    -------
    ase.Atoms
        Copy of ``atoms`` with the hydrogen on the acceptor side.
    """
    atoms = atoms.copy()
    donor_pos = atoms.positions[donor_index]
    hydrogen_pos = atoms.positions[hydrogen_index]
    acceptor_pos = atoms.positions[acceptor_index]

    direction = acceptor_pos - donor_pos
    direction /= np.linalg.norm(direction)
    new_hydrogen_pos = acceptor_pos - direction * np.linalg.norm(
        hydrogen_pos - donor_pos
    )

    atoms.positions[hydrogen_index] = new_hydrogen_pos

    return atoms
