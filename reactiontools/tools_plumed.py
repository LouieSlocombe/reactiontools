"""PLUMED support: build the input, drive a biased run, sum the hills.

The three stages of a metadynamics calculation, in the order they happen.
:func:`plumed_selection` and :func:`find_molecules` pick out the atoms a
collective variable is built from, :func:`plumed_metad_input` turns a CV and a
few METAD settings into the input PLUMED reads, :func:`plumed_calculator`
wraps an ASE calculator in that bias so an ordinary ASE molecular-dynamics run
becomes a biased one, and :func:`run_sum_hills` turns the hills it deposited
into a free-energy surface for :mod:`reactiontools.tools_fes` to plot.

Only :func:`plumed_calculator` needs the plumed Python module; only
:func:`run_sum_hills` needs the ``plumed`` executable. The rest is string
handling and works without either.
"""

import re
import subprocess
from contextlib import contextmanager

import numpy as np
from ase.calculators.plumed import Plumed
from ase.neighborlist import build_neighbor_list
from ase.units import kB
from scipy.sparse.csgraph import connected_components

#: PLUMED reads and writes its own units unless the input says otherwise, and
#: they are not ASE's: without this line a run driven from ASE reports lengths
#: in nm and energies in kJ/mol. Declaring it makes ``COLVAR``, ``HILLS`` and
#: ``fes.dat`` come out in the Å and eV that the rest of this package, and
#: :func:`~reactiontools.plot_plumed` in particular, assume.
PLUMED_ASE_UNITS = "UNITS ENERGY=eV LENGTH=A TIME=fs"

_PLUMED_HINT = ("plumed_calculator needs the plumed Python module, which is "
                "not installed. Install it with "
                "`conda install -c conda-forge py-plumed`.")

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
    n, labels = connected_components(nl.get_connectivity_matrix(sparse=True),
                                     directed=False)
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
                f"to refer to it by, as in 'd1: DISTANCE ATOMS=1,2'.")
        labels.append(match.group(1))

    duplicates = {name for name in labels if labels.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Collective variable labels must be unique, got {sorted(duplicates)} "
            f"more than once.")
    return labels


def plumed_metad_input(cvs,
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
                       extra=None):
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
            f"variables; give one each, or one for all of them.")

    if biasfactor is not None:
        if biasfactor <= 1:
            raise ValueError(
                f"biasfactor must be greater than 1, got {biasfactor}. A bias "
                f"factor of 1 is unbiased sampling; leave it out for "
                f"non-well-tempered metadynamics.")
        if temperature is None:
            raise ValueError(
                "Well-tempered metadynamics needs temperature, in kelvin, to "
                "scale the deposited height by.")

    arg = ",".join(labels)
    lines = [PLUMED_ASE_UNITS] if units else []
    lines.extend(cvs)

    metad = [f"METAD ARG={arg}",
             f"SIGMA={','.join(str(s) for s in sigmas)}",
             f"HEIGHT={height}",
             f"PACE={pace}",
             f"FILE={hills}"]
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
def plumed_calculator(atoms, calc, input_lines, timestep,
                      temperature=None,
                      log="",
                      restart=False):
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
        biased = Plumed(calc=calc,
                        input=list(input_lines),
                        timestep=timestep,
                        atoms=atoms,
                        kT=kT,
                        log=log,
                        restart=restart)
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


def run_sum_hills(hills="HILLS",
                  outfile="fes.dat",
                  mintozero=True,
                  verbose=True):
    """Run ``plumed sum_hills`` to build a free-energy surface from the hills.

    The paths are resolved by the plumed executable, so this acts on the
    current working directory unless absolute paths are given.

    Parameters
    ----------
    hills : str or path-like, optional
        Hills file written by the ``METAD`` action.
    outfile : str or path-like, optional
        Free-energy surface file to write, as read by
        :func:`~reactiontools.tools_plotting.plot_plumed`.
    mintozero : bool, optional
        Pass ``--mintozero`` so the surface minimum sits at zero.
    verbose : bool, optional
        Print the command being run.

    Returns
    -------
    str
        The command line that was run.

    Raises
    ------
    subprocess.CalledProcessError
        If plumed exits non-zero.
    """
    cmd = ["plumed", "sum_hills", "--hills", str(hills), "--outfile", str(outfile)]
    if mintozero:
        cmd.append("--mintozero")
    cmd_str = " ".join(cmd)

    if verbose:
        print(f"Running: {cmd_str}", flush=True)

    subprocess.run(cmd, check=True)
    return cmd_str
