"""Reaction paths: build a band, relax it, refine the top of it.

The workflow the package is built around, in the order it runs.
:func:`optimise_reactant_product` relaxes the two end states,
:func:`prepare_neb` interpolates a band between them, :func:`optimise_neb`
relaxes that, :func:`summarise_neb` reduces it to the barriers it was run for
and :func:`get_ts_image` picks off the highest image. From there
:func:`optimise_ts` refines that image onto a true saddle point,
:func:`optimise_irc` follows the reaction coordinate down either side of it to
show which states it actually connects, and :func:`get_vibrations` confirms it
is a first-order saddle rather than something else that stopped moving.

:func:`restart_neb` picks a band back up instead of interpolating a fresh one,
which is how a run that stopped early is continued, or a converged one
tightened. The parallel variants, :func:`prepare_parallel_neb` and
:func:`restart_parallel_neb`, do the same jobs with one socket calculator per
image, so a band whose images are expensive evaluates them concurrently rather
than one after another.

Every ``optimise_*`` function records whether it reached its force criterion in
``info["converged"]`` on the structures it returns, and warns
:class:`ConvergenceWarning` when it did not; pass ``raise_on_unconverged=True``
for a :class:`ConvergenceError` instead.

Sella is installed with the package and imported on demand by the
saddle-point searches, :func:`optimise_ts` and :func:`optimise_irc`.
"""

import copy
import os
import warnings
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import geodesic_interpolate as gi
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.socketio import SocketIOCalculator
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS
from ase.parallel import world
from ase.vibrations import Vibrations
from scipy.interpolate import CubicSpline

_SELLA_HINT = (
    "{name} needs sella, which is not installed. "
    "Reinstall reactiontools with its dependencies."
)


class ConvergenceWarning(UserWarning):
    """An optimisation stopped before reaching its force criterion.

    Warned rather than raised because a partly relaxed structure is often
    still worth looking at, and because a band that ran out of steps is a
    perfectly good starting point for the next run. Pass
    ``raise_on_unconverged=True`` to any of the ``optimise_*`` functions for a
    :exc:`ConvergenceError` instead, or turn every one of them into an error
    at once with ``warnings.simplefilter("error", ConvergenceWarning)``.
    """


class ConvergenceError(RuntimeError):
    """An optimisation stopped before reaching its force criterion.

    Raised by the ``optimise_*`` functions in place of
    :class:`ConvergenceWarning` when they are called with
    ``raise_on_unconverged=True``.
    """


def _check_converged(converged, what, fmax, steps, raise_on_unconverged):
    """Report an optimisation that ran out of steps.

    ASE's optimisers return whether they converged and otherwise say nothing,
    so a run that hits its step limit hands back a structure that looks like
    any other. That is the worst way to find out, because the next thing to
    notice is usually the vibrational analysis, several expensive steps later.

    Parameters
    ----------
    converged : bool
        What the optimiser's ``run`` returned.
    what : str
        Name of the run, quoted in the message so that the two halves of an
        IRC, or the two endpoints of a band, can be told apart. Also what
        keeps the warnings distinct, since the warnings module shows a
        repeated message from one call site only once.
    fmax : float
        Force criterion the run was asked for, in eV/Å.
    steps : int
        Step limit it hit.
    raise_on_unconverged : bool
        Raise :exc:`ConvergenceError` instead of warning.

    Returns
    -------
    bool
        ``converged`` unchanged, for the caller to record on the structures it
        returns.

    Raises
    ------
    ConvergenceError
        If the run did not converge and ``raise_on_unconverged`` is True.
    """
    if converged:
        return True

    message = (
        f"{what} hit its {steps}-step limit without reaching "
        f"fmax={fmax} eV/Å, so the result is not a converged "
        f"stationary point. Raise steps, loosen fmax, or start from a "
        f"better guess."
    )
    if raise_on_unconverged:
        raise ConvergenceError(message)
    # stacklevel=3: past this helper and past the optimise_* function that
    # called it, onto the caller's own line.
    warnings.warn(message, ConvergenceWarning, stacklevel=3)
    return False


def _import_sella(name):
    """Import sella on demand, with an install hint when it is missing.

    Importing Sella only for saddle-point searches keeps package startup
    lightweight.

    Parameters
    ----------
    name : str
        Name of the calling function, quoted in the error message.

    Returns
    -------
    tuple
        The ``(Sella, IRC)`` classes.

    Raises
    ------
    ImportError
        If sella is not installed.
    """
    try:
        from sella import IRC, Sella
    except ImportError as exc:
        raise ImportError(_SELLA_HINT.format(name=name)) from exc
    return Sella, IRC


def get_neb_path(images):
    """Compute the cumulative reaction-path distance for NEB images.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Images along a path.

    Returns
    -------
    numpy.ndarray
        Cumulative distance coordinate starting at zero.
    """
    positions = [atoms.positions for atoms in images]
    path = [0] + [
        np.linalg.norm(positions[i + 1] - positions[i])
        for i in range(len(positions) - 1)
    ]
    return np.cumsum(path)


def get_fmax(atoms):
    """Return the largest force acting on any single atom.

    This is the quantity ASE's optimisers converge against, so it is the one
    to print when checking how far a structure is from a stationary point.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure carrying a calculator that can supply forces.

    Returns
    -------
    float
        Maximum per-atom force magnitude in eV/Å.
    """
    return np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())


def stitch_path(path1, path2, f_reverse_path=False):
    """Join two reaction paths into a single IRC-like sequence.

    Parameters
    ----------
    path1 : sequence of ase.Atoms
        First path, typically the reactant side.
    path2 : sequence of ase.Atoms
        Second path, typically the product side.
    f_reverse_path : bool, optional
        Reverse the stitched path before returning it.

    Returns
    -------
    list
        Concatenated path.
    """
    irc = list(path1)[::-1] + list(path2)[1:]
    if f_reverse_path:
        irc = irc[::-1]
    return irc


def resample_path(path, n_resample):
    """Resample a path to a fixed number of images using cubic splines.

    Parameters
    ----------
    path : sequence of ase.Atoms
        Path to resample.
    n_resample : int
        Number of images in the resampled path.

    Returns
    -------
    list of ase.Atoms
        Resampled path with the first and last images preserved.
    """
    path_distance = get_neb_path(path)
    path_interp = np.linspace(0, path_distance[-1], n_resample)
    positions = np.array([image.positions for image in path])
    positions_interp = CubicSpline(path_distance, positions)(path_interp)
    irc_resampled = [path[0]]
    for ii in range(1, n_resample - 1):
        atoms = path[0].copy()
        atoms.positions = positions_interp[ii, :, :]
        irc_resampled.append(atoms)
    irc_resampled.append(path[-1])
    return irc_resampled


def optimise_geom(
    atoms,
    calc,
    fmax=0.01,
    steps=1000,
    opti_traj="opti.traj",
    use_socket=False,
    socket_port=None,
    socket_unixsocket=None,
    socket_log=None,
    raise_on_unconverged=False,
    optimiser=BFGS,
    logfile="-",
    keep_traj=False,
    _what="Geometry optimisation",
):
    """Relax a structure and return the final image.

    Whether the optimiser actually converged is recorded in
    ``info["converged"]`` on the returned structure, and a run that hits
    ``steps`` first warns :class:`ConvergenceWarning`.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to optimise.
    calc : ase.calculators.Calculator
        Calculator attached during the optimisation.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    opti_traj : str, optional
        Trajectory filename used to store the optimisation. Deleted on the way
        out unless ``keep_traj`` says otherwise.
    use_socket : bool, optional
        Drive ``calc`` through an
        ``ase.calculators.socketio.SocketIOCalculator`` instead of calling
        it directly, so the external code launches once and stays running
        for every BFGS step instead of restarting on each one. Needs a
        calculator ASE can launch as an i-PI client (e.g. ``Espresso``,
        ``Aims``, ``Siesta``); a calculator without that support, such as
        EMT, will fail.
    socket_port : int, optional
        Port for the socket server, used when ``use_socket`` is True.
        Defaults to ASE's own default (31415) when neither this nor
        ``socket_unixsocket`` is given.
    socket_unixsocket : str, optional
        Name of a Unix socket to use instead of ``socket_port``.
    socket_log : file object, optional
        Logfile for the socket communication, for debugging.
    raise_on_unconverged : bool, optional
        Raise :exc:`ConvergenceError` instead of warning when the run hits
        ``steps`` without reaching ``fmax``. Worth turning on in a batch
        script, where a silently unrelaxed structure would otherwise be
        carried into everything downstream.
    optimiser : callable, optional
        ASE optimiser class to relax with, or anything callable as
        ``optimiser(atoms, trajectory=..., logfile=...)`` returning an object
        with a ``run(fmax, steps)``. Defaults to
        :class:`~ase.optimize.BFGS`. Pass a class for a plain swap
        (``optimiser=FIRE``) or a ``functools.partial`` to preset an
        optimiser's own arguments.
    logfile : str, file object or None, optional
        Where the optimiser writes its per-step table. ``'-'``, the default,
        is stdout; a filename writes there instead; ``None`` silences it.
    keep_traj : bool, optional
        Keep ``opti_traj`` instead of deleting it. Off by default because a
        successful relaxation only needs its final structure, and on is what
        to reach for when one misbehaves and the path it took is the evidence.

    Returns
    -------
    ase.Atoms
        Relaxed structure, with ``info["converged"]`` recording whether the
        force criterion was met.

    Raises
    ------
    ConvergenceError
        If the run did not converge and ``raise_on_unconverged`` is True. The
        trajectory survives that only if ``keep_traj`` is True.
    """
    atoms = atoms.copy()
    if use_socket:
        context = SocketIOCalculator(
            calc, port=socket_port, unixsocket=socket_unixsocket, log=socket_log
        )
    else:
        context = nullcontext(calc)
    with context as live_calc:
        atoms.calc = live_calc
        converged = optimiser(atoms, trajectory=opti_traj, logfile=logfile).run(
            fmax=fmax, steps=steps
        )
    atoms = read(opti_traj, index=-1)
    if not keep_traj:
        Path(opti_traj).unlink()
    atoms.calc = calc
    atoms.info["converged"] = _check_converged(
        converged, _what, fmax, steps, raise_on_unconverged
    )
    return atoms


def optimise_reactant_product(
    reactant,
    product,
    calc,
    fmax=0.01,
    steps=1000,
    reactant_opti="reactant_opti.traj",
    product_opti="product_opti.traj",
    use_socket=False,
    socket_port=None,
    socket_unixsocket=None,
    socket_log=None,
    raise_on_unconverged=False,
    optimiser=BFGS,
    logfile="-",
    keep_traj=False,
):
    """Optimise reactant and product structures independently.

    Each endpoint carries its own ``info["converged"]``, and the two are
    reported separately, so an endpoint that failed to relax can be told from
    one that did.

    Parameters
    ----------
    reactant : ase.Atoms
        Reactant structure.
    product : ase.Atoms
        Product structure.
    calc : ase.calculators.Calculator
        Calculator used for both optimisations.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    reactant_opti : str, optional
        Temporary trajectory filename for the reactant optimisation.
    product_opti : str, optional
        Temporary trajectory filename for the product optimisation.
    use_socket, socket_port, socket_unixsocket, socket_log
        See :func:`optimise_geom`. Passed through to both optimisations,
        which run one after the other and so can safely reuse the same
        port or Unix socket.
    raise_on_unconverged : bool, optional
        Raise :exc:`ConvergenceError` on the first endpoint that fails to
        reach ``fmax`` within ``steps``, instead of warning. See
        :func:`optimise_geom`.
    optimiser, logfile, keep_traj
        See :func:`optimise_geom`. Both endpoints use the same optimiser and
        log, and keep their own trajectory under the name given for it.

    Returns
    -------
    tuple of ase.Atoms
        Optimised reactant and product structures, each with
        ``info["converged"]`` recording whether it reached ``fmax``.

    Raises
    ------
    ConvergenceError
        If either endpoint did not converge and ``raise_on_unconverged`` is
        True.
    """
    print("Optimising reactant...", flush=True)
    reactant = optimise_geom(
        reactant,
        calc,
        fmax=fmax,
        steps=steps,
        opti_traj=reactant_opti,
        use_socket=use_socket,
        socket_port=socket_port,
        socket_unixsocket=socket_unixsocket,
        socket_log=socket_log,
        raise_on_unconverged=raise_on_unconverged,
        optimiser=optimiser,
        logfile=logfile,
        keep_traj=keep_traj,
        _what="Reactant optimisation",
    )

    print("Optimising product...", flush=True)
    product = optimise_geom(
        product,
        calc,
        fmax=fmax,
        steps=steps,
        opti_traj=product_opti,
        use_socket=use_socket,
        socket_port=socket_port,
        socket_unixsocket=socket_unixsocket,
        socket_log=socket_log,
        raise_on_unconverged=raise_on_unconverged,
        optimiser=optimiser,
        logfile=logfile,
        keep_traj=keep_traj,
        _what="Product optimisation",
    )
    return reactant, product


def _build_band(
    reactant,
    product,
    n_images,
    climb,
    rm_ro_trans,
    geo_int,
    k,
    parallel=False,
    world=None,
):
    """Build an interpolated NEB whose images carry no calculators yet.

    Shared by :func:`prepare_neb` and :func:`prepare_parallel_neb`, which
    differ only in how they attach calculators afterwards.

    Parameters
    ----------
    reactant, product : ase.Atoms
        End states. They become the first and last images as given, so pass
        copies if the caller's objects must not be touched.
    n_images : int
        Total number of images, including endpoints.
    climb : bool
        Enable the climbing-image NEB variant.
    rm_ro_trans : bool
        Remove rigid-body rotation and translation during interpolation.
    geo_int : bool
        Use geodesic interpolation instead of linear plus IDPP.
    k : float
        Spring constant passed to ASE's NEB.
    parallel : bool, optional
        Distribute the images over threads or MPI ranks. See
        :func:`prepare_parallel_neb`.
    world : object, optional
        MPI communicator used to distribute images. ``None`` leaves ASE to
        use its own default, ``ase.parallel.world``.

    Returns
    -------
    ase.mep.NEB
        Interpolated band with ``image.calc`` still ``None`` throughout.
    """
    neb_images = [reactant] + [reactant.copy() for _ in range(n_images - 2)] + [product]

    if geo_int:
        neb_images = gi.geodesic_interpolate(neb_images, n_images=n_images)

    neb = NEB(
        neb_images,
        climb=climb,
        remove_rotation_and_translation=rm_ro_trans,
        k=k,
        method="improvedtangent",
        parallel=parallel,
        world=world,
    )
    if not geo_int:
        neb.interpolate()
        neb.interpolate("idpp")
    return neb


def _validate_band(images):
    """Check a sequence of images can be relaxed as a band, and listify it.

    ASE builds a two-image NEB happily and then fails with an ``IndexError``
    from inside the tangent calculation, because there is no interior image to
    take a tangent between. Catching it here says what is actually wrong.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Band to check.

    Returns
    -------
    list of ase.Atoms
        The same images as a list, so a one-shot iterator can be indexed.

    Raises
    ------
    ValueError
        If there are fewer than three images.
    """
    images = list(images)
    if len(images) < 3:
        raise ValueError(
            f"A band needs at least 3 images to leave an interior image to "
            f"relax, got {len(images)}"
        )
    return images


def _band_from_images(
    images, n_images, climb, rm_ro_trans, k, parallel=False, world=None
):
    """Build a NEB from an existing band, interpolating nothing.

    The counterpart to :func:`_build_band`: the images are already where they
    belong, so the only work is optionally re-spacing them and handing them to
    ASE. Shared by :func:`restart_neb` and :func:`restart_parallel_neb`, which
    differ only in how they attach calculators afterwards.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Band to continue from. Copied, so the caller keeps the images it
        passed in — they are what to fall back on if the restart goes worse
        than the run it continues.
    n_images : int or None
        Resample the band to this many images with :func:`resample_path`.
        ``None`` keeps it as it is.
    climb, rm_ro_trans, k, parallel, world
        As for :func:`_build_band`.

    Returns
    -------
    ase.mep.NEB
        Band with ``image.calc`` still ``None`` throughout.

    Raises
    ------
    ValueError
        If there are fewer than three images, before or after resampling.
    """
    images = _validate_band(images)

    if n_images is not None:
        images = _validate_band(resample_path(images, n_images))

    band = [image.copy() for image in images]
    for image in band:
        # Whether the run that produced these images converged says nothing
        # about the band they are about to become, and a stale True is worse
        # than no answer.
        image.info.pop("converged", None)

    return NEB(
        band,
        climb=climb,
        remove_rotation_and_translation=rm_ro_trans,
        k=k,
        method="improvedtangent",
        parallel=parallel,
        world=world,
    )


def _attach_calculators(neb, calc):
    """Give every image its own copy of ``calc`` and evaluate the band once.

    Parameters
    ----------
    neb : ase.mep.NEB
        Band whose images carry no calculators yet.
    calc : ase.calculators.Calculator
        Calculator to copy onto each image.

    Returns
    -------
    ase.mep.NEB
        The same band, primed.
    """
    # deepcopy, not copy: shallow copies of a calculator that has already run
    # share its internal arrays, so the images overwrite each other's forces.
    for image in neb.images:
        image.calc = copy.deepcopy(calc)

    # Evaluate once the geometries are final, and through the NEB itself
    # rather than image by image: that's what makes parallel=True actually
    # run the images concurrently instead of always evaluating them in turn.
    neb.get_forces()
    return neb


def prepare_neb(
    reactant,
    product,
    calc,
    n_images=5,
    climb=True,
    rm_ro_trans=True,
    geo_int=True,
    k=2.0,
    parallel=False,
    world=None,
):
    """Build an ASE NEB object from reactant and product end states.

    Parameters
    ----------
    reactant : ase.Atoms
        Initial state.
    product : ase.Atoms
        Final state.
    calc : ase.calculators.Calculator
        Calculator copied onto each NEB image.
    n_images : int, optional
        Total number of images, including endpoints.
    climb : bool, optional
        Enable the climbing-image NEB variant.
    rm_ro_trans : bool, optional
        Remove rigid-body rotation and translation during interpolation.
    geo_int : bool, optional
        Use geodesic interpolation before NEB construction.
    k : float, optional
        Spring constant passed to ASE's NEB.
    parallel : bool, optional
        Evaluate the interior images' energies and forces concurrently
        instead of one at a time. Without an MPI launcher this runs each
        image's calculator in its own thread, which only speeds things up
        if ``calc`` releases the GIL while it runs (e.g. one that shells out
        to an external code); under ``mpirun`` ASE instead distributes the
        images across MPI ranks. Applies to every force evaluation on the
        returned band, including the ones ``optimise_neb`` runs.
    world : object, optional
        MPI communicator used to distribute images when ``parallel`` is True
        and the process is launched under MPI. Defaults to
        ``ase.parallel.world``.

    Returns
    -------
    ase.mep.NEB
        Configured NEB object.
    """
    neb = _build_band(
        reactant,
        product,
        n_images,
        climb,
        rm_ro_trans,
        geo_int,
        k,
        parallel=parallel,
        world=world,
    )
    return _attach_calculators(neb, calc)


def restart_neb(
    images,
    calc,
    n_images=None,
    climb=True,
    rm_ro_trans=True,
    k=2.0,
    parallel=False,
    world=None,
):
    """Build a NEB from a band that has already been relaxed once.

    :func:`prepare_neb` interpolates a fresh band between two endpoints, which
    throws away everything a previous run learned. This takes the band itself,
    so an optimisation that ran out of steps can be continued, a converged one
    tightened, or the whole thing re-run against a better calculator, each
    starting from the path already found rather than from a straight line.

    The usual source is whatever :func:`optimise_neb` returned. From disk it
    is the last ``n_images`` entries of the trajectory, since the optimiser
    writes the whole band on every step::

        images = read("ts.traj", index="-7:")

    A band records its geometries and nothing else, so the settings of the run
    that produced it have to be given again here. ``rm_ro_trans`` is the one
    that bites: it defaults to ``True``, as in :func:`prepare_neb`, and
    leaving it there for the periodic or constrained system that was built
    with ``rm_ro_trans=False`` stops the continued band converging just as
    surely as it would have stopped the first one.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Band to continue from, endpoints included. Copied, so the caller keeps
        the originals. Any calculators they carry are dropped in favour of
        ``calc``.
    calc : ase.calculators.Calculator
        Calculator copied onto each image, as in :func:`prepare_neb`. Needed
        even when continuing with the same one as before, because the images
        come back from a trajectory holding only stored results.
    n_images : int or None, optional
        Resample the band to this many images with :func:`resample_path`
        before relaxing it, for a path too coarse to resolve the barrier.
        ``None`` keeps the images as they are. Note that resampling spaces the
        images evenly along the path, so passing the count the band already
        has is not a no-op — it re-spaces a band whose images have bunched up.
    climb : bool, optional
        Enable the climbing-image NEB variant. Turning this on for a second
        pass, having left it off for the first, is the standard way to run a
        band that is expensive to converge.
    rm_ro_trans : bool, optional
        Remove rigid-body rotation and translation. Give this the same value
        the original run had; see above.
    k : float, optional
        Spring constant passed to ASE's NEB.
    parallel : bool, optional
        Evaluate the interior images concurrently. See :func:`prepare_neb`.
    world : object, optional
        MPI communicator used to distribute images. Defaults to
        ``ase.parallel.world``.

    Returns
    -------
    ase.mep.NEB
        Configured band, ready for :func:`optimise_neb`.

    Raises
    ------
    ValueError
        If fewer than three images are given, or asked for.

    Examples
    --------
    Continue an unconverged band, with a tighter criterion and climbing on::

        images = optimise_neb(neb, fmax=0.1, steps=100)
        if not images[0].info["converged"]:
            neb = restart_neb(images, calc, climb=True, rm_ro_trans=False)
            images = optimise_neb(neb, fmax=0.05, steps=500)
    """
    neb = _band_from_images(
        images, n_images, climb, rm_ro_trans, k, parallel=parallel, world=world
    )
    return _attach_calculators(neb, calc)


def optimise_neb(
    neb,
    fmax=0.01,
    steps=1000,
    ts_traj="ts.traj",
    raise_on_unconverged=False,
    optimiser=BFGS,
    logfile="-",
):
    """Optimise an NEB band and return the final trajectory images.

    A band that runs out of steps warns :class:`ConvergenceWarning` rather
    than failing, because the images it reached are still the best starting
    point for the next attempt — and are still on disk in ``ts_traj``.

    Parameters
    ----------
    neb : ase.mep.NEB
        NEB object to optimise.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    ts_traj : str, optional
        Output trajectory filename.
    raise_on_unconverged : bool, optional
        Raise :exc:`ConvergenceError` instead of warning when the band hits
        ``steps`` without reaching ``fmax``. The top of an unconverged band
        is not a transition state, so this is worth turning on wherever
        :func:`get_ts_image` feeds a barrier straight into a result.
    optimiser : callable, optional
        ASE optimiser class to relax the band with; see
        :func:`optimise_geom`. Defaults to :class:`~ase.optimize.BFGS`.
        :class:`~ase.optimize.FIRE` is worth trying for a band that BFGS
        cannot settle, being less easily thrown by the spring forces.
    logfile : str, file object or None, optional
        Where the optimiser writes its per-step table. ``'-'``, the default,
        is stdout; a filename writes there instead; ``None`` silences it.

    Returns
    -------
    list of ase.Atoms
        Final NEB images read back from ``ts_traj``. Every image carries the
        band's ``info["converged"]``, which is a property of the band as a
        whole rather than of any one image.

    Raises
    ------
    ConvergenceError
        If the band did not converge and ``raise_on_unconverged`` is True.
    """
    n_images = len(neb.images)
    converged = optimiser(neb, trajectory=ts_traj, logfile=logfile).run(
        fmax=fmax, steps=steps
    )
    images = read(ts_traj, index=f"-{n_images}:")
    converged = _check_converged(
        converged, "NEB optimisation", fmax, steps, raise_on_unconverged
    )
    for image in images:
        image.info["converged"] = converged
    return images


class _FixedEnergy(Calculator):
    """Report a pre-computed energy wherever the atoms happen to sit.

    The NEB tangent needs the endpoint energies on every force call, but the
    band never relaxes its endpoints, so one evaluation is enough. Holding
    that value in a ``SinglePointCalculator`` does not work: with
    ``remove_rotation_and_translation`` the band rigidly re-aligns the final
    image on every call, and a ``SinglePointCalculator`` refuses to hand back
    an energy once the atoms have moved. Rigid-body motion leaves the energy
    unchanged, so returning the stored value stays correct.

    Forces are deliberately not implemented — the band only ever asks the
    endpoints for their energy, and writing zeros to the trajectory would
    claim a converged minimum that was never verified.
    """

    implemented_properties = ["energy", "free_energy"]

    def __init__(self, energy):
        super().__init__()
        self.energy = energy

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes or [])
        self.results = {"energy": self.energy, "free_energy": self.energy}


def _cached_energy(atoms):
    """Return an energy ``atoms`` already holds, without running anything.

    Deliberately reads the stored result rather than calling
    ``get_potential_energy``, which would evaluate the calculator when the
    result is missing or stale. Both are bad here: the calculator on an
    endpoint is typically the socket that priced it, and by the time the band
    is built that socket may have been closed, or — because
    :func:`optimise_reactant_product` sends both endpoints through one
    calculator — be holding the *other* endpoint's result.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure that may or may not carry a calculator holding an energy.

    Returns
    -------
    float or None
        The stored energy, or ``None`` if there is none, or it belongs to a
        different geometry. Callers price the endpoint themselves in that case.
    """
    calc = atoms.calc
    if calc is None or "energy" not in getattr(calc, "results", {}):
        return None
    if calc.check_state(atoms):  # non-empty: the atoms moved since
        return None
    return calc.results["energy"]


@contextmanager
def socket_calculators(
    n_calculators,
    make_calc=None,
    make_launcher=None,
    unixsocket=None,
    port=None,
    timeout=None,
    log=None,
):
    """Open a pool of socket calculators, one per image.

    Each :class:`~ase.calculators.socketio.SocketIOCalculator` gets its own
    socket, so the external codes behind them run as independent processes and
    can compute at the same time. The calculators are closed — and their
    clients shut down — when the block exits, including when it exits by
    exception.

    Parameters
    ----------
    n_calculators : int
        How many calculators to open.
    make_calc : callable or None, optional
        Called as ``make_calc(index)`` and must return a fresh ASE calculator
        for that image, which ASE then launches as a socket client. The index
        is passed so each client can be given its own working directory:
        file-based codes write their input and output relative to
        ``calc.directory``, and clients sharing a directory overwrite each
        other. Pass ``None`` to run only the servers and launch the clients
        yourself, for example from a batch script.
    make_launcher : callable or None, optional
        Called as ``make_launcher(index)`` and must return a client launcher,
        such as :class:`~ase.calculators.socketio.PySocketIOClient`, which
        drives a Python calculator in a separate process. Mutually exclusive
        with ``make_calc``.
    unixsocket : str or None, optional
        Prefix for UNIX socket names; image ``i`` uses ``f"{unixsocket}-{i}"``,
        which ASE places at ``/tmp/ipi_{unixsocket}-{i}``. Defaults to a
        prefix containing this process's PID, so concurrent jobs on one node
        do not collide. Mutually exclusive with ``port``.
    port : int or None, optional
        Base TCP port; image ``i`` listens on ``port + i``. Mutually exclusive
        with ``unixsocket``.
    timeout : float or None, optional
        Socket timeout in seconds, unlimited by default. Worth setting for
        long jobs, so a client that dies without closing its socket raises
        instead of hanging the run forever.
    log : str or None, optional
        Filename prefix for the socket communication logs; image ``i`` writes
        ``f"{log}-{i}.log"``. Off by default, but the first thing to turn on
        when a run stalls with no output.

    Yields
    ------
    list of ase.calculators.socketio.SocketIOCalculator
        One calculator per image, in index order.

    Raises
    ------
    ValueError
        If ``n_calculators`` is not positive, or if both members of a
        mutually exclusive pair are given.
    """
    if n_calculators < 1:
        raise ValueError(f"n_calculators must be positive, got {n_calculators}")
    if make_calc is not None and make_launcher is not None:
        raise ValueError("Specify only one of make_calc and make_launcher")
    if unixsocket is not None and port is not None:
        raise ValueError("Specify only one of unixsocket and port")
    if unixsocket is None and port is None:
        unixsocket = f"reactiontools-{os.getpid()}"

    # ExitStack, not a loop of with-blocks: if the fourth socket of eight
    # fails to bind, the three already listening still have to be closed.
    with ExitStack() as stack:
        calculators = [
            stack.enter_context(
                SocketIOCalculator(
                    calc=None if make_calc is None else make_calc(i),
                    launch_client=None if make_launcher is None else make_launcher(i),
                    unixsocket=None if unixsocket is None else f"{unixsocket}-{i}",
                    port=None if port is None else port + i,
                    timeout=timeout,
                    log=None if log is None else f"{log}-{i}.log",
                )
            )
            for i in range(n_calculators)
        ]
        yield calculators


def _require_single_rank(name):
    """Refuse to run under more than one MPI rank.

    The parallelism here is threads and sockets, so the ranks belong to the
    clients, not the driver. Under ``mpirun`` ASE would distribute the images
    over MPI ranks instead and every rank would try to bind the same sockets,
    so this raises rather than hanging.

    Parameters
    ----------
    name : str
        Name of the calling function, quoted in the error message.

    Raises
    ------
    RuntimeError
        If more than one MPI rank is running.
    """
    if world.size != 1:
        raise RuntimeError(
            f"{name} parallelises over sockets within one process, but "
            f"{world.size} MPI ranks are running. Run it without mpirun and "
            f"give the clients the ranks instead."
        )


@contextmanager
def _parallel_band(neb, energies, make_calc, socket_kwargs):
    """Put a built band onto a pool of socket calculators.

    Only the interior images need sockets. The endpoints are pinned to a fixed
    energy, reusing one they already carry if there is one and otherwise
    pricing them through the first socket. Without pinning, ``rm_ro_trans``
    would have the band recompute the final endpoint on every step, for an
    energy that rigid-body motion cannot change.

    Parameters
    ----------
    neb : ase.mep.NEB
        Band whose images carry no calculators yet.
    energies : sequence of (float or None)
        Known energies of the first and last image, ``None`` where unknown.
    make_calc : callable or None
        Builds the calculator for each interior image; see
        :func:`socket_calculators`.
    socket_kwargs : dict
        Passed to :func:`socket_calculators`.

    Yields
    ------
    ase.mep.NEB
        The band, wired up. The sockets close when the block exits.
    """
    with socket_calculators(len(neb.images) - 2, make_calc, **socket_kwargs) as calcs:
        for endpoint, energy in zip((neb.images[0], neb.images[-1]), energies):
            if energy is None:
                endpoint.calc = calcs[0]
                energy = endpoint.get_potential_energy()
            endpoint.calc = _FixedEnergy(energy)

        for image, calc in zip(neb.images[1:-1], calcs):
            image.calc = calc

        yield neb


@contextmanager
def prepare_parallel_neb(
    reactant,
    product,
    make_calc,
    n_images=5,
    climb=True,
    rm_ro_trans=True,
    geo_int=True,
    k=2.0,
    **socket_kwargs,
):
    """Build a NEB that evaluates its images concurrently over sockets.

    The serial :func:`prepare_neb` walks the band one image at a time, so a
    seven-image band costs five sequential energy evaluations per step. Here
    each interior image gets its own socket calculator, and ASE's parallel
    NEB runs one thread per interior image. Every thread blocks in
    ``socket.recv`` while its external code works, which releases the GIL, so
    the calculations genuinely overlap and a step costs about as much as its
    slowest image.

    Only the interior images need sockets. The endpoints are evaluated once
    and pinned, reusing the energy their calculator already holds if there is
    one — normally the one left behind by
    :func:`optimise_reactant_product` — and otherwise evaluating them through
    the first socket. Without pinning, ``rm_ro_trans`` would have the band
    recompute the final endpoint on every step, for an energy that rigid-body
    motion cannot change.

    Because the parallelism is threads and sockets, this must run as a single
    process. Launching it under ``mpirun`` makes ASE distribute the images
    over MPI ranks instead, and every rank would then try to bind the same
    sockets.

    Parameters
    ----------
    reactant : ase.Atoms
        Initial state. Not modified; the band is built from a copy.
    product : ase.Atoms
        Final state. Not modified.
    make_calc : callable or None
        Called as ``make_calc(index)`` to build the calculator for interior
        image ``index``, counting from zero. Give each one its own working
        directory. Pass ``None`` to launch the clients yourself.
    n_images : int, optional
        Total number of images, including endpoints. Must be at least three.
    climb : bool, optional
        Enable the climbing-image NEB variant.
    rm_ro_trans : bool, optional
        Remove rigid-body rotation and translation during interpolation.
    geo_int : bool, optional
        Use geodesic interpolation before NEB construction.
    k : float, optional
        Spring constant passed to ASE's NEB.
    **socket_kwargs
        Passed to :func:`socket_calculators`: ``make_launcher``,
        ``unixsocket``, ``port``, ``timeout`` and ``log``.

    Yields
    ------
    ase.mep.NEB
        Configured band, ready for :func:`optimise_neb`. The sockets stay
        open for the lifetime of the block and close on the way out, so the
        band must be optimised inside it.

    Raises
    ------
    ValueError
        If ``n_images`` is less than three.
    RuntimeError
        If more than one MPI rank is running.
    ImportError
        If ``geo_int`` is ``True`` and geodesic_interpolate is not installed.

    Examples
    --------
    Each image runs its own Quantum Espresso client, in its own directory::

        from ase.calculators.espresso import Espresso

        def make_calc(index):
            return Espresso(directory=f"image-{index}", pseudopotentials=...)

        with prepare_parallel_neb(reactant, product, make_calc,
                                  n_images=7, timeout=600) as neb:
            images = optimise_neb(neb, fmax=0.05)

        # The sockets are shut by now, but the band carries its energies.
        ts = get_ts_image(images)
    """
    if n_images - 2 < 1:
        raise ValueError(
            f"n_images must be at least 3 to leave an interior image to "
            f"relax, got {n_images}"
        )
    _require_single_rank("prepare_parallel_neb")

    # Read the endpoint energies before copying: Atoms.copy() drops the
    # calculator, and with it the energy the endpoint optimisation left.
    energies = [_cached_energy(reactant), _cached_energy(product)]

    neb = _build_band(
        reactant.copy(),
        product.copy(),
        n_images,
        climb,
        rm_ro_trans,
        geo_int,
        k,
        parallel=True,
    )

    with _parallel_band(neb, energies, make_calc, socket_kwargs) as band:
        yield band


@contextmanager
def restart_parallel_neb(
    images,
    make_calc,
    n_images=None,
    climb=True,
    rm_ro_trans=True,
    k=2.0,
    **socket_kwargs,
):
    """Continue an existing band over sockets, evaluating its images at once.

    :func:`restart_neb` for the case where the images are expensive enough to
    want one client each — the same relationship
    :func:`prepare_parallel_neb` has to :func:`prepare_neb`, and all three of
    that function's caveats apply here too: run it as a single process, only
    the interior images get sockets, and set a ``timeout``.

    A band read back from a trajectory carries its endpoint energies in the
    single-point calculators ASE stores with it, so those are reused and the
    endpoints cost nothing to pin — the case
    :func:`prepare_parallel_neb` has to fall back to a socket for whenever the
    endpoints were not relaxed in the same session.

    As for :func:`restart_neb`, the settings of the run being continued have
    to be given again, ``rm_ro_trans`` above all.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Band to continue from, endpoints included. Copied, so the caller keeps
        the originals.
    make_calc : callable or None
        Called as ``make_calc(index)`` to build the calculator for interior
        image ``index``, counting from zero. Give each one its own working
        directory. Pass ``None`` to launch the clients yourself.
    n_images : int or None, optional
        Resample the band to this many images before relaxing it. See
        :func:`restart_neb`.
    climb : bool, optional
        Enable the climbing-image NEB variant.
    rm_ro_trans : bool, optional
        Remove rigid-body rotation and translation. Give this the same value
        the original run had; see above.
    k : float, optional
        Spring constant passed to ASE's NEB.
    **socket_kwargs
        Passed to :func:`socket_calculators`: ``make_launcher``,
        ``unixsocket``, ``port``, ``timeout`` and ``log``.

    Yields
    ------
    ase.mep.NEB
        Configured band, ready for :func:`optimise_neb`. The sockets stay open
        for the lifetime of the block and close on the way out, so the band
        must be optimised inside it.

    Raises
    ------
    ValueError
        If fewer than three images are given, or asked for.
    RuntimeError
        If more than one MPI rank is running.

    Examples
    --------
    Pick a run back up where it stopped, with more steps::

        images = read("ts.traj", index="-7:")

        with restart_parallel_neb(images, make_calc, timeout=600) as neb:
            images = optimise_neb(neb, fmax=0.05, steps=500)
    """
    _require_single_rank("restart_parallel_neb")
    images = _validate_band(images)

    # Read the endpoint energies before copying, as above. resample_path
    # passes the endpoints through untouched, so these stay right either way.
    energies = [_cached_energy(images[0]), _cached_energy(images[-1])]

    neb = _band_from_images(images, n_images, climb, rm_ro_trans, k, parallel=True)

    with _parallel_band(neb, energies, make_calc, socket_kwargs) as band:
        yield band


def _get_energy(image, calc):
    """Return the energy an image already carries, else evaluate with ``calc``.

    Images read back from a trajectory hold their energies, and re-running the
    calculator would recompute the band's most expensive quantity. An image
    without one gets its own deepcopy of ``calc``, so images never share
    calculator state.

    Parameters
    ----------
    image : ase.Atoms
        Image whose energy is wanted.
    calc : ase.calculators.Calculator or None
        Fallback calculator for images that carry no energy.

    Returns
    -------
    float
        Potential energy in eV.
    """
    if image.calc is not None:
        try:
            return image.calc.results["energy"]
        except (AttributeError, KeyError):
            pass
    image.calc = copy.deepcopy(calc)
    return image.get_potential_energy()


@dataclass
class NebSummary:
    """The numbers a relaxed band is run for.

    Attributes
    ----------
    energies : numpy.ndarray
        Potential energy of each image in eV, as the calculator reported it.
        Absolute, not shifted: subtract ``energies[0]`` for a profile
        referenced to the reactant, as the barriers below are.
    ts_index : int
        Index of the highest-energy image, the one :func:`get_ts_image`
        returns.
    barrier : float
        Forward barrier in eV, from the first image up to the highest.
    reverse_barrier : float
        Reverse barrier in eV, from the last image up to the highest.
    reaction_energy : float
        Energy of the last image relative to the first, in eV. Negative for
        an exothermic reaction.
    """

    energies: np.ndarray
    ts_index: int
    barrier: float
    reverse_barrier: float
    reaction_energy: float

    def __post_init__(self):
        """Coerce *energies* to a float array, whatever it was built from."""
        self.energies = np.asarray(self.energies, dtype=float)

    @property
    def is_barrierless(self):
        """bool: True when the highest image is one of the endpoints.

        The band then has no maximum in between: it runs downhill throughout,
        and what :func:`get_ts_image` returns is an endpoint rather than a
        transition state. Worth checking before paying for
        :func:`optimise_ts`, which would otherwise be started from a
        structure that is not a saddle at all.
        """
        return self.ts_index in (0, len(self.energies) - 1)

    @staticmethod
    def _ev(value):
        """Format an energy, without a sign on a value that rounds to zero.

        A thermoneutral reaction comes out a hair either side of zero, and
        "-0.000 eV" reads as a finding rather than as the rounding it is.
        """
        text = f"{value:.3f}"
        return "0.000" if text == "-0.000" else text

    def __str__(self):
        """Report the barriers and the reaction energy, one per line."""
        return (
            f"Barrier:         {self._ev(self.barrier)} eV\n"
            f"Reverse barrier: {self._ev(self.reverse_barrier)} eV\n"
            f"Reaction energy: {self._ev(self.reaction_energy)} eV\n"
            f"TS image:        {self.ts_index} of "
            f"{len(self.energies) - 1}"
        )


def summarise_neb(images, calc=None):
    """Reduce a relaxed band to the numbers it was run for.

    The barrier is measured from the highest image, so it agrees with
    :func:`get_ts_image` and with the profile :func:`plot_neb` draws. It is
    not spline-fitted, unlike ASE's ``NEBTools.get_barrier``, whose default
    interpolates between images and so can report a maximum that sits at no
    image at all.

    A band only resolves a barrier as well as its images allow: the true
    saddle lies between them, so this underestimates. Refining the top image
    with :func:`optimise_ts` is what turns it into a number worth quoting.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Images along the band, reactant first.
    calc : ase.calculators.Calculator or None, optional
        Fallback for images carrying no energy. Not needed for a band from
        :func:`optimise_neb`, whose images already have theirs. Unlike
        :func:`get_ts_image`, a calculator given here does not replace the
        energies the images already hold.

    Returns
    -------
    NebSummary
        Barriers, reaction energy and the position of the top image.

    Raises
    ------
    ValueError
        If ``images`` is empty.

    Examples
    --------
    >>> summary = summarise_neb(images)
    >>> print(summary)                                  # doctest: +SKIP
    Barrier:         0.374 eV
    Reverse barrier: 0.374 eV
    Reaction energy: 0.000 eV
    TS image:        3 of 6
    """
    images = list(images)
    if not images:
        raise ValueError("Cannot summarise an empty band")

    energies = np.array([_get_energy(image, calc) for image in images])
    ts_index = int(np.argmax(energies))
    return NebSummary(
        energies=energies,
        ts_index=ts_index,
        barrier=float(energies[ts_index] - energies[0]),
        reverse_barrier=float(energies[ts_index] - energies[-1]),
        reaction_energy=float(energies[-1] - energies[0]),
    )


def get_ts_image(neb_images, calc=None):
    """Return the highest-energy image along a NEB band.

    Parameters
    ----------
    neb_images : sequence of ase.Atoms
        Images along the band.
    calc : ase.calculators.Calculator or None, optional
        Calculator used to evaluate the potential energies, replacing
        whatever the images carry. Leave it out to use the energies they
        already hold, as images read back by :func:`optimise_neb` do. That is
        the way to pick the TS out of a band from
        :func:`prepare_parallel_neb`, whose sockets are closed by the time the
        band comes back and so cannot be handed to anything.

    Returns
    -------
    ase.Atoms
        Image with the maximum potential energy.
    """
    if calc is not None:
        for image in neb_images:
            image.calc = copy.deepcopy(calc)
    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]


def optimise_ts(
    ts_image,
    calc,
    fmax=0.01,
    steps=1000,
    eta=1e-4,
    gamma=0.1,
    sella_traj="sella.traj",
    raise_on_unconverged=False,
    logfile="-",
    internal=False,
):
    """Refine a transition-state guess to a true saddle point with Sella.

    A NEB band gets close to the saddle but rarely converges tightly onto it,
    so the usual route is :func:`get_ts_image` to pick the top of the band and
    this to polish it. Unlike :func:`optimise_geom`, the trajectory is kept:
    a saddle search is the step most likely to wander off, and the path it
    took is what tells you it did.

    Parameters
    ----------
    ts_image : ase.Atoms
        Transition-state guess. Not modified; a copy is optimised.
    calc : ase.calculators.Calculator
        Calculator attached during the search.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    eta : float, optional
        Finite-difference step for Sella's curvature estimate.
    gamma : float, optional
        Convergence criterion for Sella's iterative diagonalisation.
    sella_traj : str, optional
        Trajectory filename, kept after the run.
    raise_on_unconverged : bool, optional
        Raise :exc:`ConvergenceError` instead of warning when the search hits
        ``steps`` without reaching ``fmax``.
    logfile : str, file object or None, optional
        Where Sella writes its per-step table. ``'-'``, the default, is
        stdout; a filename writes there instead; ``None`` silences it. The
        two energy lines printed before the search are not affected. There is
        no ``optimiser`` argument here: the search is Sella's.
    internal : bool, optional
        Use Sella's internal coordinates instead of Cartesian coordinates.

    Returns
    -------
    ase.Atoms
        Refined transition state, read back from the trajectory, with
        ``info["converged"]`` recording whether the search reached ``fmax``.
        Converging here says the search found *a* stationary point, not that
        it is a first-order saddle — that is what :func:`get_vibrations` is
        for.

    Raises
    ------
    ImportError
        If sella is not installed.
    ConvergenceError
        If the search did not converge and ``raise_on_unconverged`` is True.
    """
    Sella, _IRC = _import_sella("optimise_ts")

    print("Running Sella TS search", flush=True)
    ts_image = ts_image.copy()
    ts_image.calc = calc

    print(f"Initial energy: {ts_image.get_potential_energy():.3} eV", flush=True)
    print(f"Initial max force: {get_fmax(ts_image):.3} eV/A", flush=True)

    sella_ts = Sella(
        ts_image,
        trajectory=sella_traj,
        logfile=logfile,
        eta=eta,
        gamma=gamma,
        internal=internal,
    )
    converged = sella_ts.run(fmax=fmax, steps=steps)

    ts = read(sella_traj, index=-1)
    ts.info["converged"] = _check_converged(
        converged, "Sella TS search", fmax, steps, raise_on_unconverged
    )
    return ts


def optimise_irc(
    ts_image,
    calc,
    fmax=0.01,
    steps=1000,
    dx=0.1,
    eta=1e-4,
    gamma=0.1,
    keep_going=True,
    irc_f_traj="irc_f.traj",
    irc_r_traj="irc_r.traj",
    raise_on_unconverged=False,
    logfile="-",
):
    """Follow the intrinsic reaction coordinate downhill from a saddle point.

    Runs Sella's IRC in both directions, which is what confirms that a saddle
    found by :func:`optimise_ts` actually connects the reactant and product
    you meant rather than some other pair of minima. The two halves come back
    separately and can be joined into one profile with :func:`stitch_path`.

    Parameters
    ----------
    ts_image : ase.Atoms
        Converged transition state. Not modified; each direction runs on its
        own copy.
    calc : ase.calculators.Calculator
        Calculator attached during both runs.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of steps per direction.
    dx : float, optional
        Step length along the reaction coordinate in Å.
    eta : float, optional
        Finite-difference step for Sella's curvature estimate.
    gamma : float, optional
        Convergence criterion for Sella's iterative diagonalisation.
    keep_going : bool, optional
        Carry on past a step that fails to converge instead of stopping.
    irc_f_traj : str, optional
        Trajectory filename for the forward direction.
    irc_r_traj : str, optional
        Trajectory filename for the reverse direction.
    raise_on_unconverged : bool, optional
        Raise :exc:`ConvergenceError` instead of warning when either
        direction hits ``steps`` without reaching ``fmax``. A half that
        stopped early has not reached its minimum, so it does not show which
        state that direction connects to — the whole point of running it.
    logfile : str, file object or None, optional
        Where Sella writes its per-step table. ``'-'``, the default, is
        stdout; a filename writes there instead; ``None`` silences it. Both
        directions share it, so a filename collects the two runs in order.

    Returns
    -------
    tuple of list of ase.Atoms
        ``(forward, reverse)`` paths, each read back from its trajectory and
        starting at the transition state. Every image of a path carries that
        direction's ``info["converged"]``; the two are reported separately,
        since one direction commonly reaches its minimum while the other
        does not.

    Raises
    ------
    ImportError
        If sella is not installed.
    ConvergenceError
        If either direction did not converge and ``raise_on_unconverged`` is
        True. The forward direction is checked first, and only once both have
        run, so a failure there does not cost the reverse run.
    """
    _Sella, IRC = _import_sella("optimise_irc")

    irc_f = ts_image.copy()
    irc_f.calc = calc
    print("Running IRC forward", flush=True)
    sella_irc_f = IRC(
        irc_f,
        trajectory=irc_f_traj,
        logfile=logfile,
        dx=dx,
        eta=eta,
        gamma=gamma,
        keep_going=keep_going,
    )
    converged_f = sella_irc_f.run(fmax=fmax, steps=steps, direction="forward")

    irc_r = ts_image.copy()
    irc_r.calc = calc

    print("Running IRC reverse", flush=True)
    sella_irc_r = IRC(
        irc_r,
        trajectory=irc_r_traj,
        logfile=logfile,
        dx=dx,
        eta=eta,
        gamma=gamma,
        keep_going=keep_going,
    )
    converged_r = sella_irc_r.run(fmax=fmax, steps=steps, direction="reverse")

    forward = read(irc_f_traj, index=":")
    reverse = read(irc_r_traj, index=":")
    for path, ran_to_fmax, what in (
        (forward, converged_f, "Forward IRC"),
        (reverse, converged_r, "Reverse IRC"),
    ):
        converged = _check_converged(
            ran_to_fmax, what, fmax, steps, raise_on_unconverged
        )
        for image in path:
            image.info["converged"] = converged
    return forward, reverse


def get_vibrations(atoms, calc):
    """Compute vibrational frequencies by finite differences.

    Mostly used to characterise a stationary point: a minimum has all-real
    frequencies, while a transition state from :func:`optimise_ts` should show
    exactly one imaginary mode, which ASE reports as a complex number.

    The displacement cache ASE writes is removed before and after the run, so
    a stale cache from an earlier geometry cannot silently be reused.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to displace. Not modified; a copy is used.
    calc : ase.calculators.Calculator
        Calculator used for the displaced evaluations.

    Returns
    -------
    numpy.ndarray
        Frequencies in cm⁻¹, complex where a mode is imaginary.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    vib = Vibrations(atoms)
    vib.clean()
    vib.run()
    vib.summary()
    freqs = vib.get_frequencies()
    vib.clean()
    return freqs


def quick_guess_path(reactant, product, n_images=25):
    """Generate a quick geodesic path guess between two endpoints.

    Parameters
    ----------
    reactant : ase.Atoms
        Starting structure.
    product : ase.Atoms
        Ending structure.
    n_images : int, optional
        Number of images in the interpolated path.

    Returns
    -------
    list of ase.Atoms
        Interpolated path.
    """
    return gi.geodesic_interpolate([reactant, product], n_images=n_images)


def quick_guess_ts(reactant, product, n_images=25):
    """Return the midpoint image from a quick geodesic path guess.

    Parameters
    ----------
    reactant : ase.Atoms
        Starting structure.
    product : ase.Atoms
        Ending structure.
    n_images : int, optional
        Number of images used in the interpolated path.

    Returns
    -------
    ase.Atoms
        Midpoint image of the interpolated path.
    """
    path = gi.geodesic_interpolate([reactant, product], n_images=n_images)
    return path[n_images // 2]
