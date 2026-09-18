"""Top-level driver tying the interpolation and smoothing stages together.

`geodesic_interpolate` is the one function most callers need.  It takes the end points,
builds a raw path with `interpolation.redistribute`, smooths it with `geodesic.Geodesic`,
and hands the result back in whichever form the end points came in.
"""
import os

import numpy as np
from ase import Atoms

from .coord_utils import align_path_to
from .fileio import from_ase_atoms, read_xyz, to_ase_atoms, write_xyz
from .geodesic import Geodesic
from .interpolation import redistribute


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

    Runs the two stages in turn: `redistribute` builds a raw path with the requested
    number of images, then `Geodesic` smooths it into a geodesic under the internal
    coordinate metric.

    Which smoothing is used depends on the size of the system.  Up to 35 atoms the whole
    path is optimised at once; beyond that the images are smoothed one at a time,
    sweeping back and forth along the path, because scipy's optimisers slow down badly
    as the problem grows.

    Input and output mirror each other.  Given ASE Atoms objects the interpolated path
    comes back as Atoms objects; given a filename it is written to `output` instead.

    Given Atoms objects, everything the interpolation does not itself touch is taken from
    the first frame and carried onto every image: the unit cell, the boundary conditions,
    constraints, tags and so on.  A periodic path is also moved back onto the frame of
    reference of the input, since the optimisation otherwise leaves it centred and rotated
    into a frame of its own, which would put the atoms in the wrong place relative to the
    cell.  Note that the interpolation itself is not periodic: the internal coordinates
    are plain inter-atomic distances with no minimum image convention, so a bond that
    crosses a cell boundary is not handled.

    Args:
        atoms: Either a list of ASE Atoms objects, or the name of an XYZ file holding
            the end points.  Only the first and last geometries need be meaningful,
            but intermediate ones are used if present.
        n_images: Number of images in the interpolated path.
        output: XYZ file to write to.  Only used when `atoms` is a filename.
        tol: Convergence tolerance for the smoothing, judged on the uniform gradient of
            the path length.  The raw path is built at five times this, being only a
            starting guess for the smoothing.
        max_iter: Ceiling on the smoothing, counting function evaluations when the whole
            path is optimised at once and sweeps when sweeping.  It is a ceiling rather
            than a target, as the optimisation stops as soon as it meets `tol`: the
            bundled test systems all converge somewhere between 10 and 35 at the default
            seed, though the count moves around with the seed.  Setting it too low
            silently truncates the path part-way through the descent.
        micro_iter: Micro-iterations spent on each image.  Only used when sweeping, so
            it has no effect on systems of 35 atoms or fewer.
        scaling: Alpha parameter of the Morse scaler setting the coordinate metric.
        friction: Weight of the friction term regularising the optimisation step size.
        dist_cutoff: Distance cut-off for building the internal coordinates.  Atoms
            within three bonds of each other are included whatever their distance.
        seed: Seed for the random nudges and image sampling, so runs reproduce.  The
            bisection is stochastic, and without a fixed seed larger systems will not
            give the same path twice.  The seed goes to a `numpy.random.Generator` used
            only here, so the caller's own random state is left alone.

    Returns:
        The interpolated path as a list of ASE Atoms objects when `atoms` was a list of
        them, otherwise None, with the path written to `output`.

    Raises:
        TypeError: If `atoms` is neither a list of ASE Atoms nor a filename.
        ValueError: If fewer than two geometries are supplied.
    """
    rng = np.random.default_rng(seed)
    template = None
    if isinstance(atoms, (str, os.PathLike)):
        symbols, geometries = read_xyz(atoms)
    elif isinstance(atoms, list):
        symbols, geometries = from_ase_atoms(atoms)
        # Everything the interpolation does not touch is carried over from the first
        # frame, which is also where the symbols come from
        template = atoms[0]
    else:
        raise TypeError("Input must be an ASE Atoms object or a filename.")

    if len(geometries) < 2:
        raise ValueError("Need at least two initial geometries.")

    # A looser tolerance is enough for the raw path, which only has to be a decent
    # starting guess for the smoothing that follows
    raw = redistribute(symbols, geometries, n_images, tol=tol * 5, rng=rng)
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction, rng=rng)

    # Optimising the whole path at once is faster, but scipy's optimisers slow down
    # badly as the system grows, so past this size sweep one image at a time instead
    sweep = len(symbols) > 35

    if sweep:
        smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
    else:
        smoother.smooth(tol=tol, max_iter=max_iter)

    if template is None:
        write_xyz(output, symbols, smoother.path)
        return None

    # The optimisation leaves the path in its own centred and rotated frame.  That is of
    # no consequence for an isolated molecule, but a cell describes where the atoms are
    # only in the frame it was given in, so a periodic path is moved back onto its input
    path = smoother.path
    if template.cell.rank > 0 or template.pbc.any():
        path = align_path_to(template.get_positions(), path)
    return to_ase_atoms(symbols, path, template=template)
