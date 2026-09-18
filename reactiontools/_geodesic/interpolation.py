"""Raw path construction, the first of the two interpolation stages.

Adds or drops images until the count matches what was asked for, bisecting the largest
gap each time and using geodesic length to choose between candidate midpoints.  The
result is only a starting guess: the geodesic smoothing in `geodesic` is what turns it
into the final path.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, identity, vstack

from .coord_utils import align_geom, align_path, compute_wij, get_bond_list, morse_scaler
from .geodesic import Geodesic


def _mid_point(atoms: list[str],
               geom1: np.ndarray,
               geom2: np.ndarray,
               tol: float = 1e-2,
               nudge: float = 0.01,
               threshold: float = 4.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Find the geometry whose internal coordinates sit closest to the average of two others.

    A least-squares minimisation against the average of the two end points, run twice,
    starting from just beside either end point.  DON'T USE THE CARTESIAN AVERAGE AS THE
    GUESS, THINGS WILL BLOW UP.  The two runs are then compared by local geodesic length
    and the shorter one wins.

    The point produced here need not join smoothly onto either end point; it only has to
    be a good enough starting guess for the smoothing that follows.

    Random nudges are added to the starting geometry, so repeated runs need not converge
    to the same answer — for larger systems they essentially never will.  Running several
    times and keeping the best result is therefore worthwhile.

    Args:
        atoms: Atom symbols, used to look up covalent radii.
        geom1, geom2: Cartesian geometries of the two end points.
        tol: Convergence tolerance for the least-squares minimisation.
        nudge: Size of the random nudge added to the starting geometry.  Helps to turn
            up different solutions, and to break symmetry when the optimal path does.
        threshold: Distance cut-off for including an atom pair in the coordinates.
        rng: Random source for the nudge and the image sampling.  Defaults to a fresh
            unseeded `numpy.random.Generator`.

    Returns:
        The optimised mid-point, bisecting the two end points in internal coordinates.
    """
    geom1, geom2 = np.array(geom1, dtype=float), np.array(geom2, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    add_pair: set[tuple[int, int]] = set()
    geom_list: list[np.ndarray] = [geom1, geom2]

    # The outer loop makes sure the coordinate system is large enough.  The interpolated
    # point can bring atom pairs into contact that are far apart at both end points,
    # which would let them collide unnoticed.  Including every pair would blow up for
    # large molecules, so the compromise is to start from a screened list, add any pair
    # that comes into contact, and redo the minimisation until the coordinate system and
    # the interpolated geometry agree.
    while True:
        rij_list, re = get_bond_list(geom_list, threshold=threshold + 1.0, enforce=add_pair, rng=rng)
        scaler = morse_scaler(alpha=0.7, re=re)
        w = (compute_wij(geom1, rij_list, scaler, sparse=True)[0]
             + compute_wij(geom2, rij_list, scaler, sparse=True)[0]) / 2
        d_min: float = np.inf
        x_min: np.ndarray | None = None
        friction: float = 0.1 / np.sqrt(geom1.shape[0])
        # The friction residuals never change, so build them once rather than per call
        friction_block = identity(geom1.size, format='csr') * friction
        # scipy asks for the residuals and the Jacobian in separate calls but at the same
        # geometry, so holding on to the last evaluation halves the work
        last_eval: list[np.ndarray | tuple[np.ndarray, csr_matrix] | None] = [None, None]

        def wij_at(x: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
            """Scaled distances and B matrix at `x`, reusing the last result if it fits."""
            if last_eval[0] is None or not np.array_equal(last_eval[0], x):
                last_eval[0] = np.array(x)
                last_eval[1] = compute_wij(x, rij_list, scaler, sparse=True)
            return last_eval[1]

        # The inner loop minimises from either end point in turn as the starting guess
        for coef in [0.02, 0.98]:
            x0: np.ndarray = (geom1 * coef + geom2 * (1 - coef)).ravel() + nudge * rng.random(geom1.size)
            # Residuals are the difference from the target internals, plus a friction
            # term holding the geometry near where it started
            result = least_squares(
                lambda x: np.concatenate([wij_at(x)[0] - w, (x - x0) * friction]),
                x0,
                lambda x: vstack([wij_at(x)[1], friction_block], format='csr'),
                ftol=tol,
                gtol=tol,
            )
            x_mid: np.ndarray = result["x"].reshape(-1, 3)
            # Rebuild the pair list including the new point and check for fresh contacts
            new_rij, _ = get_bond_list([*geom_list, x_mid], threshold=threshold, min_neighbors=0, rng=rng)
            extras = set(new_rij) - set(rij_list)

            if extras:
                # Widen the coordinate system and start the minimisation over
                geom_list.append(x_mid)
                add_pair.update(extras)
                break

            # Score this candidate by locally smoothing the three-image path through it
            smoother = Geodesic(atoms,
                                [geom1, x_mid, geom2],
                                scaler=0.7,
                                threshold=threshold,
                                friction=1,
                                rng=rng)
            smoother.compute_displacements()
            width = max(np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2])
            dist = width + smoother.length
            if dist < d_min:
                d_min, x_min = dist, smoother.path[1]
        else:
            # Both starting guesses finished without new atom pairs, so the coordinate
            # system held up and the minimisation is done
            break

    return x_min


def redistribute(atoms: list[str], geoms: list[np.ndarray], n_images: int, tol: float = 1e-2,
                 rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Add or remove images so the path has the requested number of them.

    If there are too few, new points are added by bisecting the largest RMSD gap.  If
    there are too many, images are dropped one at a time, each time choosing the one
    whose removal leaves the shortest merged segment.

    Args:
        atoms: Atom symbols, used to look up covalent radii.
        geoms: Geometries of the original path.
        n_images: The desired number of images.
        tol: Convergence tolerance for the bisection.
        rng: Random source for the bisection, which is stochastic.  Defaults to a fresh
            unseeded `numpy.random.Generator`.

    Returns:
        An aligned path with the correct number of images.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)

    # Add bisection points if there are too few images
    while len(geoms) < n_images:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i: int = int(np.argmax(dists))
        insertion: np.ndarray = _mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol, rng=rng)
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])

    # Remove points if there are too many images
    while len(geoms) > n_images:
        # Each distance here spans two segments, so the smallest one marks the image
        # that can be dropped with the least disruption
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i: int = int(np.argmin(dists))
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])

    return geoms
