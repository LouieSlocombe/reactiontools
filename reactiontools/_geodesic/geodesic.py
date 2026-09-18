"""Geodesic smoothing.

Minimises the length of a reaction path measured with a redundant internal coordinate
metric, but optimises the Cartesian coordinates directly.  Working in Cartesians avoids
the feasibility problems that come with redundant internals, where an arbitrary set of
internal coordinate values need not correspond to any real geometry.
"""
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import bmat, csr_matrix, identity

from .coord_utils import align_path, compute_wij, get_bond_list, morse_scaler


class Geodesic:
    """Optimiser that finds a geodesic in redundant internal coordinates.

    The heart of it is the path length in the internal metric: the length of each
    segment is measured through its midpoint, so that the target function is a sum of
    squared displacements and can be handed straight to a least-squares optimiser.

    `smooth` optimises every image at once and `sweep` takes them one at a time; both
    leave the result in `path`.  The two end points are held fixed either way.

    Attributes:
        path: The current geometries, of shape ``(n_images, n_atoms, 3)``.  Updated in
            place as the optimisation proceeds.
        rij_list: The atom pairs making up the internal coordinates.
        re: Equilibrium distance for each of those pairs.
        length: Path length in the internal metric, over the section last measured.
            Only set once the displacements have been computed.
        optimality: Infinity norm of the gradient of the length, which is what
            convergence is judged on.  Only set once the target function has been
            evaluated.
    """

    length: float
    optimality: float

    def __init__(self,
                 atoms: list[str],
                 path: np.ndarray | list[np.ndarray],
                 scaler: float | Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = 1.7,
                 threshold: float = 3.0,
                 min_neighbors: int = 4,
                 friction: float = 1e-3,
                 rng: np.random.Generator | None = None) -> None:
        """Initialise the interpolater.

        Args:
            atoms: Atom symbols, used to look up covalent radii.
            path: Initial geometries of the path, of shape ``(n_images, n_atoms, 3)``.
            scaler: Either the alpha parameter for the Morse scaler, or an explicit
                scaling function.  Small values have long range and make it easier to
                get smooth paths out of few images; larger values usually give better
                energetics because they represent the sharp energy landscape better.
            threshold: Distance cut-off for building the inter-nuclear distance
                coordinates.  Atoms linked by three or fewer bonds are added too,
                whatever their distance.
            min_neighbors: Minimum number of neighbours each atom must have in the
                atom pair list.
            friction: Weight of the friction term in the target function, which keeps
                the optimiser from taking steps large enough to blow the path up.
            rng: Random source used when sampling images to build the coordinates.
                Defaults to a fresh unseeded `numpy.random.Generator`.

        Raises:
            ValueError: If the path is not three dimensional.
        """
        path = np.asarray(path, dtype=float)
        if path.ndim != 3:
            raise ValueError('The path to be interpolated must have 3 dimensions')
        _, self.path = align_path(path)
        self.n_images, self.n_atoms, _ = self.path.shape
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(self.path, atoms, threshold=threshold,
                                               min_neighbors=min_neighbors, rng=rng)
        if isinstance(scaler, (int, float, np.number)):
            self.scaler = morse_scaler(re=self.re, alpha=scaler)
        else:
            self.scaler = scaler
        self.n_rij = len(self.rij_list)
        self.friction = friction
        # Initalize interal storages for mid points, internal coordinates and B matrices.
        # `None` marks a value as unknown, so it is only ever computed when needed.
        self.w: list[np.ndarray | None] = [None] * self.n_images
        self.dw_dR: list[csr_matrix | None] = [None] * self.n_images
        self.w_mid: list[np.ndarray | None] = [None] * (self.n_images - 1)
        self.dwdR_mid: list[csr_matrix | None] = [None] * (self.n_images - 1)
        self.displacements: np.ndarray | None = None
        self.grad: csr_matrix | None = None
        self.segment: tuple[int, int] | None = None

    def update_intc(self) -> None:
        """Fill in any internal coordinates and derivatives currently marked unknown.

        Missing entries are flagged with `None` in the internal storage; this finds
        them, works out the midpoint geometries where needed, and evaluates the
        coordinates and their gradients.  Everything already known is left alone, so
        nothing is evaluated twice.

        The derivatives are kept sparse, since each scaled distance only depends on the
        six Cartesian components of its own two atoms.
        """
        for i, X in enumerate(self.path):
            if self.w[i] is None:
                self.w[i], self.dw_dR[i] = compute_wij(X, self.rij_list, self.scaler, sparse=True)
        for i, (X0, X1) in enumerate(zip(self.path, self.path[1:])):
            if self.w_mid[i] is None:
                Xm = (X0 + X1) / 2
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(Xm, self.rij_list, self.scaler, sparse=True)

    def update_geometry(self, X: np.ndarray, start: int, end: int) -> bool:
        """Move a segment of the path, invalidating everything that depended on it.

        The internal coordinates, derivatives and midpoints of the affected images are
        reset to unknown so that `update_intc` recomputes them.  Note that moving
        images ``start:end`` also invalidates the midpoint just before ``start``, if
        there is one.

        Args:
            X: New Cartesian coordinates for the segment, flattened or otherwise.
            start, end: The section of the path being replaced.

        Returns:
            True if the geometry actually changed, False if `X` was already the
            current geometry and nothing needed invalidating.
        """
        X = X.reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        self.path[start:end] = X
        self.w[start:end] = [None] * (end - start)
        # Clamped, because a negative slice bound would wrap round to the end of the
        # list and insert entries instead of overwriting them
        first_mid = max(start - 1, 0)
        self.w_mid[first_mid:end] = [None] * (min(end, self.n_images - 1) - first_mid)
        return True

    def compute_displacements(self, start: int = 1, end: int = -1,
                              dx: np.ndarray | None = None, friction: float = 1e-3) -> None:
        """Compute the displacement vectors along a section of the path, and its length.

        Each segment is split at its midpoint and measured in two halves, which is what
        makes the length a sum of squares.  The friction term is appended to the same
        vector so the least-squares optimiser sees it as extra residuals.

        Args:
            start, end: Section of the path to measure.  A negative ``end`` counts back
                from the last image.  ``start`` must be at least 1: the end points of
                the path are fixed, and each segment is measured against the image
                before it.
            dx: Displacement of the segment from its reference geometry.  When given,
                it enters the target function scaled by ``friction``.
            friction: Weight of the friction term.

        Sets `self.length` and `self.displacements`.

        Raises:
            ValueError: If the section does not lie between the two fixed end points.
        """
        if end < 0:
            end += self.n_images
        if not 1 <= start < end <= self.n_images - 1:
            raise ValueError(f'Section ({start}, {end}) must lie between the fixed end '
                             f'points of a {self.n_images} image path')
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [wm - wl for wl, wm in zip(self.w[start - 1:end], self.w_mid[start - 1:end])]
        vecs_r = [wr - wm for wr, wm in zip(self.w[start:end + 1], self.w_mid[start - 1:end])]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(np.linalg.norm(vecs_r, axis=1))
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry.  friction term
        self.displacements = np.concatenate(vecs_l + vecs_r + [trans])

    def compute_disp_grad(self, start: int, end: int, friction: float = 1e-3) -> None:
        """Compute derivatives of the displacement vectors with respect to Cartesians.

        Moving one image changes the two half-segments on either side of it directly,
        and the two midpoints it shares with its neighbours at half the rate, which is
        where the factors of a half come from.  The friction residuals contribute a
        diagonal block at the bottom of the matrix.

        An image only appears in the two half-segments it touches, so the matrix is
        block-bidiagonal, and each block is itself sparse because a distance only
        depends on its own two atoms.  It is assembled sparse and left that way: for a
        few dozen atoms fewer than one entry in a hundred is non-zero, and handing a
        dense array to `scipy.optimize.least_squares` makes it factorise the whole
        thing on every iteration.

        Args:
            start, end: Section of the path being differentiated.
            friction: Weight of the friction term, matching `compute_displacements`.

        Sets `self.grad`.
        """
        # Calculate derivatives of displacement vectors with respect to image Cartesians
        n_seg = end - start
        n_rows = n_seg + 1  # One more image than segments, as each segment joins two
        n_dof = 3 * self.n_atoms
        blocks_l = [[None] * n_seg for _ in range(n_rows)]
        blocks_r = [[None] * n_seg for _ in range(n_rows)]
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dwdR_mid[image - 1] / 2
            dmid2 = self.dwdR_mid[image] / 2
            blocks_l[i + 1][i] = dmid2 - self.dw_dR[image]
            blocks_l[i][i] = dmid1
            blocks_r[i + 1][i] = -dmid2
            blocks_r[i][i] = self.dw_dR[image] - dmid1
        # The friction residuals are one scaled identity, laid out block by block so it
        # lines up with the image columns above it
        friction_block = identity(n_dof, format='csr') * friction
        blocks_f = [[friction_block if k == i else None for i in range(n_seg)] for k in range(n_seg)]
        self.grad = bmat(blocks_l + blocks_r + blocks_f, format='csr')

    def compute_target_func(self, X: np.ndarray | None = None, start: int = 1, end: int = -1,
                            x0: np.ndarray | None = None, friction: float = 1e-3) -> None:
        """Compute the vectorised target function used for least-squares minimisation.

        Args:
            X: Geometry to evaluate at.  If it matches the geometry already stored for
                this segment, the whole evaluation is skipped.
            start, end: Section of the path being optimised.
            x0: Reference geometry the friction term pulls back towards.  Defaults to
                no pull at all.
            friction: Weight of the friction term.

        Sets `self.displacements`, `self.grad` and `self.optimality`.
        """
        if end < 0:
            end += self.n_images
        if X is not None and self.segment == (start, end) and not self.update_geometry(X, start, end):
            return

        self.segment = (start, end)
        dx = np.zeros(self.path[start:end].size) if x0 is None else self.path[start:end].ravel() - x0.ravel()
        self.compute_displacements(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.abs(self.grad.T @ self.displacements).max()

    def target_func(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Residuals for the optimiser.

        Wraps `compute_target_func`, which skips the work if the geometry has not moved
        since the last call.

        Args:
            X: Geometry of the segment, flattened as scipy hands it over.
            **kwargs: Passed straight through to `compute_target_func`.

        Returns:
            The displacement vectors of the segment, with the friction residuals
            appended.
        """
        self.compute_target_func(X, **kwargs)
        return self.displacements

    def target_deriv(self, X: np.ndarray, **kwargs: Any) -> csr_matrix:
        """Jacobian for the optimiser.

        Wraps `compute_target_func`, which skips the work if the geometry has not moved
        since the last call.  Paired with `target_func`, this means each geometry is
        only ever evaluated once even though scipy asks for value and Jacobian
        separately.

        Args:
            X: Geometry of the segment, flattened as scipy hands it over.
            **kwargs: Passed straight through to `compute_target_func`.

        Returns:
            The derivatives of those displacement vectors, held sparse.
        """
        self.compute_target_func(X, **kwargs)
        return self.grad

    def smooth(self,
               tol: float = 1e-3,
               max_iter: int = 50,
               start: int = 1,
               end: int = -1,
               friction: float | None = None,
               xref: np.ndarray | None = None) -> np.ndarray:
        """Minimise the path length as a single function of all the image coordinates.

        In principle this is very efficient, but it can get costly for large systems
        with many images, in which case `sweep` is the better option.

        Args:
            tol: Convergence tolerance on the optimality, i.e. the uniform gradient of
                the target function.
            max_iter: Ceiling on the number of function evaluations, passed to scipy as
                ``max_nfev``.
            start, end: Section of the path to optimise.
            friction: Weight of the friction term.  Defaults to the value given to the
                constructor.
            xref: Reference geometry for the friction term.  Defaults to the starting
                geometry of the segment.

        Returns:
            The optimised path.  This is also stored in `self.path`.
        """

        X0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref = X0
        self.displacements = self.grad = self.segment = None
        if friction is None:
            friction = self.friction
        # Configure the keyword arguments that will be sent to the target function.
        kwargs = dict(start=start, end=end, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        if self.optimality > tol:
            # `soft_l1` keeps a single badly placed image from dominating the fit
            result = least_squares(self.target_func, X0, self.target_deriv, ftol=tol, gtol=tol,
                                   max_nfev=max_iter, kwargs=kwargs, loss='soft_l1')
            self.update_geometry(result['x'], start, end)
        _, self.path = align_path(self.path)
        return self.path

    def sweep(self, tol: float = 1e-3, max_iter: int = 50, micro_iter: int = 20,
              start: int = 1, end: int = -1) -> np.ndarray:
        """Minimise the path length one image at a time, sweeping back and forth.

        Less efficient per iteration than `smooth`, but it scales far more kindly with
        system size given how slow scipy's optimisers get on large problems.  It also
        allows finer control, and makes it cheap to skip images that are already close
        to optimal.

        Args:
            tol: Convergence tolerance on the optimality, i.e. the uniform gradient of
                the target function.
            max_iter: Maximum number of sweeps through the path.
            micro_iter: Number of micro-iterations spent optimising each image.
            start, end: Section of the path to optimise.

        Returns:
            The optimised path.  This is also stored in `self.path`.
        """
        if end < 0:
            end = self.n_images + end
        images = list(range(start, end))
        # Microiteration convergence tolerances are adjusted on the fly based on level of convergence.
        curr_tol = tol * 10
        for iteration in range(max_iter):
            max_dL = 0
            for i in images:  # Use self.smooth() to optimise individual images
                # Each image is pulled back towards the midpoint of its neighbours,
                # with heavy friction on the first sweep to keep the initial guess
                # from being thrown around
                xmid = (self.path[i - 1] + self.path[i + 1]) * 0.5
                self.smooth(curr_tol, max_iter=min(micro_iter, iteration + 6),
                            start=i, end=i + 1,
                            friction=self.friction if iteration else 0.1,
                            xref=xmid)
                max_dL = max(max_dL, self.optimality)
            if max_dL < tol:  # Check for convergence.
                break
            curr_tol = max(tol * 0.5, max_dL * 0.2)  # Adjust micro-iteration threshold
            images.reverse()  # Alternate sweeping direction.
        _, self.path = align_path(self.path)
        return self.path
