"""PLUMED support: build the input, drive a biased run, sum the hills.

The three stages of a metadynamics calculation, in the order they happen.
:func:`plumed_selection` and :func:`find_molecules` pick out the atoms a
collective variable is built from, :func:`plumed_metad_input` turns a CV and a
few METAD settings into the input PLUMED reads, :func:`plumed_calculator`
wraps an ASE calculator in that bias so an ordinary ASE molecular-dynamics run
becomes a biased one, and :func:`run_sum_hills` turns the hills it deposited
into a free-energy surface for :mod:`reactiontools.tools_fes` to plot.

:func:`run_opes_fes` is the ``OPES_METAD`` counterpart of
:func:`run_sum_hills`. OPES deposits no hills to add up, writing a running
estimate of the bias to a ``STATE`` file instead, so the surface is read back
out of that by one of the scripts bundled in :mod:`reactiontools.opes`.

Only :func:`plumed_calculator` needs the plumed Python module; only
:func:`run_sum_hills` needs the ``plumed`` executable. The rest is string
handling and works without either.
"""

import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from ase.calculators.plumed import Plumed
from ase.neighborlist import build_neighbor_list
from ase.units import kB
from scipy.sparse.csgraph import connected_components

from .opes import script_path

#: PLUMED reads and writes its own units unless the input says otherwise, and
#: they are not ASE's: without this line a run driven from ASE reports lengths
#: in nm and energies in kJ/mol. Declaring it makes ``COLVAR``, ``HILLS`` and
#: ``fes.dat`` come out in the Å and eV that the rest of this package, and
#: :func:`~reactiontools.plot_plumed` in particular, assume.
PLUMED_ASE_UNITS = "UNITS ENERGY=eV LENGTH=A TIME=fs"

_PLUMED_HINT = (
    "plumed_calculator needs the plumed Python module, which is "
    "not installed. Install it with "
    "`conda install -c conda-forge py-plumed`."
)

_CV_LABEL = re.compile(r"^\s*([A-Za-z_]\w*)\s*:")


def plumed_selection(indices):
    """Format atom indices as a PLUMED ``ATOMS=`` selection string.

    Parameters
    ----------
    indices : iterable of int
        Zero-based atom indices.

    Returns
    -------
    str
        Comma-separated PLUMED selection using one-based indexing and compact
        ranges.

    Raises
    ------
    ValueError
        If no indices are given.
    """
    idx = sorted({int(i) + 1 for i in indices})
    if not idx:
        raise ValueError("empty atom selection")
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def find_molecules(atoms):
    """Return connected atom groups identified as molecules.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure for which the bonded graph should be analysed.

    Returns
    -------
    list of numpy.ndarray
        Atom-index arrays, one per connected component.
    """
    nl = build_neighbor_list(atoms, self_interaction=False, bothways=True)
    n, labels = connected_components(
        nl.get_connectivity_matrix(sparse=True), directed=False
    )
    return [np.where(labels == k)[0] for k in range(n)]


def _cv_labels(cvs):
    """Pull the label off each collective-variable line.

    PLUMED refers to a CV by the label it was defined with, so ``METAD`` and
    ``PRINT`` need those names. Taking them from the lines rather than asking
    for them separately keeps the two from drifting apart.

    Parameters
    ----------
    cvs : sequence of str
        PLUMED action lines, each of the form ``"label: ACTION ..."``.

    Returns
    -------
    list of str
        One label per line, in order.

    Raises
    ------
    ValueError
        If a line carries no label, or two lines share one.
    """
    labels = []
    for line in cvs:
        match = _CV_LABEL.match(line)
        if match is None:
            raise ValueError(
                f"Collective variable {line!r} has no label. PLUMED needs one "
                f"to refer to it by, as in 'd1: DISTANCE ATOMS=1,2'."
            )
        labels.append(match.group(1))

    duplicates = {name for name in labels if labels.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Collective variable labels must be unique, got {sorted(duplicates)} "
            f"more than once."
        )
    return labels


def plumed_metad_input(
    cvs,
    sigma,
    height,
    pace,
    biasfactor=None,
    temperature=None,
    hills="HILLS",
    colvar="COLVAR",
    stride=10,
    units=True,
    metad_extra=None,
    extra=None,
):
    """Build the PLUMED input for a metadynamics run.

    Assembles the lines :func:`plumed_calculator` takes: the units, the
    collective variables as given, a ``METAD`` action biasing all of them, and
    a ``PRINT`` collecting them into a ``COLVAR``. The ``METAD`` file defaults
    to the name :func:`run_sum_hills` looks for, so the two ends of the
    workflow line up without being told to.

    Parameters
    ----------
    cvs : sequence of str
        Collective-variable lines in PLUMED syntax, each labelled, as in
        ``"d1: DISTANCE ATOMS=1,2"``. :func:`plumed_selection` formats the
        atom lists. Passed through untouched, so any PLUMED action can be
        used.
    sigma : float or sequence of float
        Gaussian width per collective variable, in the units of that variable.
        A single value is used for all of them.
    height : float
        Gaussian height in eV, or in kJ/mol if ``units`` is ``False``.
    pace : int
        Number of steps between deposited Gaussians.
    biasfactor : float or None, optional
        Bias factor for well-tempered metadynamics, which must be greater than
        one. ``None``, the default, deposits Gaussians of fixed height
        instead, which never stops filling and so has no converged surface to
        read off.
    temperature : float or None, optional
        Simulation temperature in kelvin, written as ``TEMP``. Required with
        ``biasfactor``, and must match the temperature the dynamics actually
        runs at — nothing checks that for you.
    hills : str, optional
        File the Gaussians are written to.
    colvar : str or None, optional
        File the collective variables are printed to. ``None`` omits the
        ``PRINT``.
    stride : int, optional
        Number of steps between ``PRINT`` lines.
    units : bool, optional
        Prepend :data:`PLUMED_ASE_UNITS`, so the run reads and writes Å and
        eV. Leave it on unless you mean to work in PLUMED's own nm and
        kJ/mol, in which case ``sigma``, ``height`` and the output files are
        all in those units and :func:`~reactiontools.plot_plumed` will read
        them wrongly.
    metad_extra : str or None, optional
        Extra keywords appended to the ``METAD`` line, for the settings not
        given their own argument here — ``GRID_MIN``, ``GRID_MAX`` and
        ``GRID_BIN`` above all, which a long run wants.
    extra : sequence of str or None, optional
        Extra whole lines appended at the end, for walls, further prints, or
        anything else.

    Returns
    -------
    list of str
        The input lines, ready for :func:`plumed_calculator`.

    Raises
    ------
    ValueError
        If no collective variable is given, if a line carries no label, if
        the number of sigmas does not match the number of variables, or if
        ``biasfactor`` is given without a temperature or is not above one.

    Examples
    --------
    Well-tempered metadynamics along one distance::

        lines = plumed_metad_input(
            cvs=["d1: DISTANCE ATOMS=1,2"],
            sigma=0.05, height=0.02, pace=100,
            biasfactor=10, temperature=300)
    """
    cvs = list(cvs)
    if not cvs:
        raise ValueError("Metadynamics needs at least one collective variable")
    labels = _cv_labels(cvs)

    sigmas = [sigma] * len(labels) if np.isscalar(sigma) else list(sigma)
    if len(sigmas) != len(labels):
        raise ValueError(
            f"Got {len(sigmas)} sigmas for {len(labels)} collective "
            f"variables; give one each, or one for all of them."
        )

    if biasfactor is not None:
        if biasfactor <= 1:
            raise ValueError(
                f"biasfactor must be greater than 1, got {biasfactor}. A bias "
                f"factor of 1 is unbiased sampling; leave it out for "
                f"non-well-tempered metadynamics."
            )
        if temperature is None:
            raise ValueError(
                "Well-tempered metadynamics needs temperature, in kelvin, to "
                "scale the deposited height by."
            )

    arg = ",".join(labels)
    lines = [PLUMED_ASE_UNITS] if units else []
    lines.extend(cvs)

    metad = [
        f"METAD ARG={arg}",
        f"SIGMA={','.join(str(s) for s in sigmas)}",
        f"HEIGHT={height}",
        f"PACE={pace}",
        f"FILE={hills}",
    ]
    if biasfactor is not None:
        metad.append(f"BIASFACTOR={biasfactor}")
    if temperature is not None:
        metad.append(f"TEMP={temperature}")
    if metad_extra:
        metad.append(metad_extra)
    lines.append(" ".join(metad))

    if colvar:
        lines.append(f"PRINT ARG={arg} FILE={colvar} STRIDE={stride}")
    if extra:
        lines.extend(extra)
    return lines


@contextmanager
def plumed_calculator(
    atoms, calc, input_lines, timestep, temperature=None, log="", restart=False
):
    """Bias an ASE calculator with PLUMED, for the length of the block.

    Wraps ``calc`` in an :class:`ase.calculators.plumed.Plumed` and hangs it
    on ``atoms``, so an ordinary ASE dynamics run becomes a biased one:
    the integrator asks for forces as usual and PLUMED adds the bias on top.

    A context manager because PLUMED buffers what it writes and only flushes
    on ``finalize``. Run the dynamics inside the block; a run that returns
    without it leaves ``HILLS`` short of the hills it deposited last, and a
    free-energy surface summed from that is quietly wrong rather than
    obviously missing.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to simulate. Its calculator is swapped for the biased one
        for the length of the block and put back on the way out, so the atoms
        come back as they were, holding a calculator that still works.
    calc : ase.calculators.Calculator
        Calculator computing the unbiased forces.
    input_lines : sequence of str
        PLUMED input, one action per line, from :func:`plumed_metad_input` or
        written by hand. Hand-written input wants :data:`PLUMED_ASE_UNITS`
        first.
    timestep : float
        Timestep in ASE time units — the same value the integrator is given,
        for example ``0.5 * ase.units.fs``. PLUMED counts its own steps from
        it, so a mismatch silently misdates every hill.
    temperature : float or None, optional
        Simulation temperature in kelvin, converted to the thermal energy
        PLUMED is told about. Needed by the methods that scale a bias by it,
        well-tempered metadynamics above all; ``None`` leaves ASE's default,
        which those methods must not be run with. Match the ``temperature``
        given to :func:`plumed_metad_input` and to the thermostat.
    log : str, optional
        PLUMED's own log file. Empty, the default, writes to stdout.
    restart : bool, optional
        Continue a previous run, appending to its hills rather than starting
        the bias from nothing.

    Yields
    ------
    ase.calculators.plumed.Plumed
        The biased calculator, already attached to ``atoms``.

    Raises
    ------
    ImportError
        If the plumed Python module is not installed.

    Examples
    --------
    Well-tempered metadynamics along one distance, at 300 K::

        from ase import units
        from ase.md.langevin import Langevin

        lines = plumed_metad_input(cvs=["d1: DISTANCE ATOMS=1,2"],
                                   sigma=0.05, height=0.02, pace=100,
                                   biasfactor=10, temperature=300)

        with plumed_calculator(atoms, calc, lines, timestep=0.5 * units.fs,
                               temperature=300):
            Langevin(atoms, 0.5 * units.fs, temperature_K=300,
                     friction=0.01).run(10000)

        run_sum_hills()
    """
    kT = 1.0 if temperature is None else kB * temperature
    # Read before constructing: an ASE calculator handed `atoms=` hangs itself
    # on them, so by the time Plumed returns this would already be the biased
    # one and the block would restore nothing.
    previous = atoms.calc
    try:
        biased = Plumed(
            calc=calc,
            input=list(input_lines),
            timestep=timestep,
            atoms=atoms,
            kT=kT,
            log=log,
            restart=restart,
        )
    except ImportError as exc:
        raise ImportError(_PLUMED_HINT) from exc

    # `with biased`: its __exit__ is what finalises PLUMED and flushes the
    # files, and it has to run even if the dynamics raises partway through.
    with biased:
        atoms.calc = biased
        try:
            yield biased
        finally:
            atoms.calc = previous


def _grid_bound(value):
    """Format a per-variable grid bound, which PLUMED takes comma-separated.

    Parameters
    ----------
    value : float or sequence of float
        One bound, or one per collective variable.

    Returns
    -------
    str
        The value, or the sequence joined with commas.
    """
    if np.isscalar(value) or isinstance(value, str):
        return str(value)
    return ",".join(str(item) for item in value)


def sum_hills_files(outfile="fes.dat"):
    """List the surfaces a strided :func:`run_sum_hills` wrote, in order.

    ``--stride`` does not number the file it was given: it writes
    ``f"{outfile}{n}.dat"``, so the default ``outfile`` yields
    ``fes.dat0.dat``, ``fes.dat1.dat`` and so on. Two things follow, and this
    exists because both are easy to get wrong. The obvious glob, ``fes*.dat``,
    is right only by accident; and sorting the names puts ``fes.dat10.dat``
    before ``fes.dat2.dat``, which for a convergence series — where the order
    is the entire point — silently scrambles the answer.

    Parameters
    ----------
    outfile : str or path-like, optional
        The ``outfile`` given to :func:`run_sum_hills`.

    Returns
    -------
    list of pathlib.Path
        The surfaces, ordered by the index PLUMED gave them, ready to hand to
        :func:`~reactiontools.plot_fes_1d`. Empty if the run was not strided.

    Examples
    --------
    A convergence series, labelled by the simulated time each surface covers::

        run_sum_hills(stride=100, outfile="fes")
        surfaces = sum_hills_files("fes")
        plot_fes_1d(surfaces,
                    labels=[(i + 1) * 100 for i in range(len(surfaces))],
                    label_template="{:g} hills")
    """
    outfile = Path(outfile)
    pattern = re.compile(rf"^{re.escape(outfile.name)}(\d+)\.dat$")

    numbered = []
    for path in (outfile.parent or Path(".")).iterdir():
        match = pattern.match(path.name)
        if match is not None:
            numbered.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered)]


def run_sum_hills(
    hills="HILLS",
    outfile="fes.dat",
    mintozero=True,
    stride=None,
    nohistory=False,
    grid_min=None,
    grid_max=None,
    grid_bin=None,
    idw=None,
    kt=None,
    negbias=False,
    extra=None,
    verbose=True,
):
    """Run ``plumed sum_hills`` to build a free-energy surface from the hills.

    The paths are resolved by the plumed executable, so this acts on the
    current working directory unless absolute paths are given.

    Parameters
    ----------
    hills : str or path-like, optional
        Hills file written by the ``METAD`` action.
    outfile : str or path-like, optional
        Free-energy surface file to write, as read by
        :func:`~reactiontools.tools_plotting.plot_plumed`. With ``stride`` it
        is a stem rather than a filename; see there.
    mintozero : bool, optional
        Pass ``--mintozero`` so the surface minimum sits at zero.
    stride : int or None, optional
        Write a surface every ``stride`` hills instead of one at the end,
        which is how a convergence series is made: a run is converged when
        the last few surfaces lie on top of each other. The files are named
        ``f"{outfile}{n}.dat"``, so the default ``outfile`` gives the
        unlovely ``fes.dat0.dat`` — pass ``outfile="fes"`` for ``fes0.dat``.
        :func:`sum_hills_files` collects them in the right order either way.
    nohistory : bool, optional
        With ``stride``, make each surface from only the hills in its own
        interval rather than from everything up to it. Useful to watch where
        the bias is being deposited; not what a convergence series wants.
    grid_min, grid_max : float or sequence of float, optional
        Bounds of the output grid, one per collective variable. Worth setting
        for a series, since PLUMED otherwise picks bounds per surface from
        the hills it has so far and the surfaces come back on grids that do
        not line up.
    grid_bin : int or sequence of int, optional
        Number of bins per collective variable.
    idw : str or sequence of str, optional
        Collective variables to keep, by label; the rest are integrated out,
        which needs ``kt``. This is how a two-dimensional surface is
        projected onto one of its variables.
    kt : float or None, optional
        Thermal energy for that integration, in the energy units of the hills
        file — eV for a run built by :func:`plumed_metad_input`, where
        ``ase.units.kB * 300`` is 300 K. Only used with ``idw``.
    negbias : bool, optional
        Print the negative bias rather than the free energy.
    extra : sequence of str, optional
        Further arguments appended to the command line, for the options
        without their own keyword here — ``--spacing``, ``--fmt``,
        ``--histo`` and the rest.
    verbose : bool, optional
        Print the command being run.

    Returns
    -------
    str
        The command line that was run. With ``stride``, the surfaces
        themselves are gathered by :func:`sum_hills_files`.

    Raises
    ------
    ValueError
        If ``kt`` is given without ``idw``, which would silently do nothing.
    subprocess.CalledProcessError
        If plumed exits non-zero.
    """
    if kt is not None and idw is None:
        raise ValueError(
            "kt only applies when idw names the variables to keep, since it "
            "is the temperature the others are integrated out at. Pass idw, "
            "or leave kt out."
        )

    cmd = ["plumed", "sum_hills", "--hills", str(hills), "--outfile", str(outfile)]
    if mintozero:
        cmd.append("--mintozero")
    if stride is not None:
        cmd += ["--stride", str(stride)]
    if nohistory:
        cmd.append("--nohistory")
    if grid_min is not None:
        cmd += ["--min", _grid_bound(grid_min)]
    if grid_max is not None:
        cmd += ["--max", _grid_bound(grid_max)]
    if grid_bin is not None:
        cmd += ["--bin", _grid_bound(grid_bin)]
    if idw is not None:
        cmd += ["--idw", idw if isinstance(idw, str) else ",".join(idw)]
    if kt is not None:
        cmd += ["--kt", str(kt)]
    if negbias:
        cmd.append("--negbias")
    if extra:
        cmd += [str(item) for item in extra]
    cmd_str = " ".join(cmd)

    if verbose:
        print(f"Running: {cmd_str}", flush=True)

    subprocess.run(cmd, check=True)
    return cmd_str


def _opes_fes_command(
    state="STATE",
    outfile="fes.dat",
    grid_min=None,
    grid_max=None,
    grid_bin=None,
    kt=None,
    extra=None,
):
    """Build the command line that reconstructs a FES from an OPES state file.

    ``OPES_METAD`` does not deposit hills for ``plumed sum_hills`` to add up;
    it writes a running estimate of the bias to a ``STATE`` file instead, and
    the surface is read back out of that by the bundled ``FES_from_State.py``.

    Parameters
    ----------
    state : str or path-like, optional
        State file written by the ``OPES_METAD`` action's ``STATE_WFILE``.
    outfile : str or path-like, optional
        Free-energy surface file to write.
    grid_min, grid_max : float or sequence of float, optional
        Bounds of the output grid, one per collective variable. Both must be
        given together, or neither.
    grid_bin : int or sequence of int, optional
        Number of bins per collective variable.
    kt : float or None, optional
        Thermal energy in the energy units of the state file -- kJ/mol for a
        run driven from OpenMM, which is what
        :func:`~reactiontools.tools_units.thermal_energy` returns by default.
    extra : sequence of str, optional
        Further arguments appended to the command line.

    Returns
    -------
    list of str
        The command, as an argument list.

    Raises
    ------
    ValueError
        If only one of *grid_min* and *grid_max* is given.
    """
    if (grid_min is None) != (grid_max is None):
        raise ValueError(
            "Give both grid_min and grid_max or neither; FES_from_State.py "
            "needs the two bounds together to size its grid."
        )

    # sys.executable, not "python3": the scripts need this environment's
    # pandas, and whatever "python3" resolves to on PATH may not have it.
    cmd = [
        sys.executable,
        str(script_path("FES_from_State.py")),
        "--state",
        str(state),
        "--outfile",
        str(outfile),
    ]
    if grid_min is not None:
        cmd += ["--min", _grid_bound(grid_min), "--max", _grid_bound(grid_max)]
    if grid_bin is not None:
        cmd += ["--bin", _grid_bound(grid_bin)]
    if kt is not None:
        cmd += ["--kt", f"{float(kt):.6g}"]
    if extra:
        cmd += [str(item) for item in extra]
    return cmd


def run_opes_fes(
    state="STATE",
    outfile="fes.dat",
    grid_min=None,
    grid_max=None,
    grid_bin=None,
    kt=None,
    extra=None,
    verbose=True,
):
    """Rebuild a free-energy surface from an OPES state file.

    The ``OPES_METAD`` counterpart of :func:`run_sum_hills`, and the other half
    of what the ``f_opes`` switch on the
    :mod:`reactiontools.tools_cv` builders selects: they emit the bias action,
    this reads the surface back out of what it wrote.

    Paths are resolved by the script, so this acts on the current working
    directory unless absolute paths are given.

    Parameters
    ----------
    state : str or path-like, optional
        State file written by the ``OPES_METAD`` action's ``STATE_WFILE``.
    outfile : str or path-like, optional
        Free-energy surface file to write, as read by
        :func:`~reactiontools.as_fes`.
    grid_min, grid_max : float or sequence of float, optional
        Bounds of the output grid, one per collective variable.
    grid_bin : int or sequence of int, optional
        Number of bins per collective variable.
    kt : float or None, optional
        Thermal energy in the energy units of the state file. See
        :func:`~reactiontools.tools_units.thermal_energy`.
    extra : sequence of str, optional
        Further arguments appended to the command line, for the options
        without their own keyword here -- ``--deltaFat``, ``--all_stored``,
        ``--der`` and the rest.
    verbose : bool, optional
        Print the command being run.

    Returns
    -------
    str
        The command line that was run.

    Raises
    ------
    ValueError
        If only one of *grid_min* and *grid_max* is given.
    subprocess.CalledProcessError
        If the script exits non-zero.
    """
    cmd = _opes_fes_command(
        state=state,
        outfile=outfile,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_bin=grid_bin,
        kt=kt,
        extra=extra,
    )
    cmd_str = " ".join(cmd)

    if verbose:
        print(f"Running: {cmd_str}", flush=True)

    subprocess.run(cmd, check=True)
    return cmd_str
