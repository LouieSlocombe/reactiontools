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
:func:`run_opes_reweighting` rebuilds the surface from the ``COLVAR`` samples
and their bias column instead, which is how several independent walkers are
combined into one estimate -- :func:`combine_colvar_files` does the merge.

Only :func:`plumed_calculator` needs the plumed Python module; only
:func:`run_sum_hills` needs the ``plumed`` executable. The rest is string
handling and works without either.
"""

import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
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
    "not installed. Build it, and the PLUMED it binds to, with "
    "`bash build_tools/conda_install.sh`."
)

_CV_LABEL = re.compile(r"^\s*([A-Za-z_]\w*)\s*:")


def plumed_selection(indices: Iterable[int]) -> str:
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


def find_molecules(atoms: Atoms) -> list[np.ndarray]:
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


def _cv_labels(cvs: Sequence[str]) -> list[str]:
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
    cvs: Sequence[str],
    sigma: float | Sequence[float],
    height: float,
    pace: int,
    biasfactor: float | None = None,
    temperature: float | None = None,
    hills: str = "HILLS",
    colvar: str | None = "COLVAR",
    stride: int = 10,
    units: bool = True,
    metad_extra: str | None = None,
    extra: Sequence[str] | None = None,
) -> list[str]:
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
    atoms: Atoms,
    calc: Any,
    input_lines: Sequence[str],
    timestep: float,
    temperature: float | None = None,
    log: str = "",
    restart: bool = False,
) -> Iterator[Plumed]:
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


def _grid_bound(value: float | str | Sequence[float]) -> str:
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


def sum_hills_files(outfile: str | Path = "fes.dat") -> list[Path]:
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
    hills: str | Path = "HILLS",
    outfile: str | Path = "fes.dat",
    mintozero: bool = True,
    stride: int | None = None,
    nohistory: bool = False,
    grid_min: float | Sequence[float] | None = None,
    grid_max: float | Sequence[float] | None = None,
    grid_bin: int | Sequence[int] | None = None,
    idw: str | Sequence[str] | None = None,
    kt: float | None = None,
    negbias: bool = False,
    extra: Sequence[str] | None = None,
    verbose: bool = True,
) -> str:
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


def combine_colvar_files(
    colvar_files: Sequence[str | Path],
    outfile: str | Path = "COLVAR",
    sort_by_time: bool = True,
) -> Path:
    """Merge several ``COLVAR`` files into one, keeping a single header.

    The merge :func:`run_opes_reweighting` wants for independent walkers,
    done the two ways the bundled script's own notes describe. Sorted by
    time (``sort -gs COLVAR.*``), the walkers interleave into one series,
    which is what ``stride`` and ``skiprows`` expect -- a stride then samples
    across walkers, and skipped rows discard every walker's transient
    together. Unsorted (``cat COLVAR.*``), the walkers stay contiguous, so
    ``blocks`` set to the number of walkers gives one block per walker and
    the error bars come from their scatter.

    Parameters
    ----------
    colvar_files : sequence of path-like
        The ``COLVAR`` files, one per walker. Their header blocks -- the
        leading ``#! FIELDS`` line and any ``#! SET`` lines after it -- must
        be identical, since one copy speaks for all of them.
    outfile : str or path-like, optional
        The merged file to write. Must not be one of the inputs.
    sort_by_time : bool, optional
        Stable-sort the rows on their first column. Default True; pass
        False for the contiguous per-walker layout ``blocks`` wants.

    Returns
    -------
    pathlib.Path
        The merged file.

    Raises
    ------
    ValueError
        If no files are given, *outfile* is one of the inputs, a file does
        not open with a ``#! FIELDS`` header, the headers disagree, or a
        row's time is not a number.
    """
    paths = [Path(path) for path in colvar_files]
    if not paths:
        raise ValueError("combine_colvar_files needs at least one COLVAR file")
    target = Path(outfile)
    for path in paths:
        if path.resolve() == target.resolve():
            raise ValueError(
                f"outfile {target} is one of the inputs; it would be "
                "overwritten while being read"
            )

    header: list[str] | None = None
    rows: list[tuple[float, str]] = []
    for path in paths:
        lines = path.read_text().splitlines()
        block = []
        for line in lines:
            if not line.startswith("#"):
                break
            block.append(line)
        if not block or not block[0].startswith("#! FIELDS"):
            raise ValueError(f"{path} does not start with a #! FIELDS header")
        if header is None:
            header = block
        elif block != header:
            raise ValueError(
                f"{path} has a different header block from {paths[0]}; "
                "only walkers printing the same fields can be combined"
            )
        for line in lines[len(block):]:
            # A restarted walker re-prints its header mid-file; one copy at
            # the top speaks for all of them.
            if not line.strip() or line.startswith("#"):
                continue
            if sort_by_time:
                try:
                    time = float(line.split()[0])
                except ValueError as exc:
                    raise ValueError(
                        f"{path} has a non-numeric time in row {line!r}"
                    ) from exc
            else:
                time = 0.0
            rows.append((time, line))

    if sort_by_time:
        rows.sort(key=lambda row: row[0])
    assert header is not None
    content = "\n".join([*header, *(line for _, line in rows)])
    target.write_text(content + "\n")
    return target


def _opes_fes_command(
    state: str | Path = "STATE",
    outfile: str | Path = "fes.dat",
    grid_min: float | str | Sequence[float] | None = None,
    grid_max: float | str | Sequence[float] | None = None,
    grid_bin: int | str | Sequence[int] | None = None,
    kt: float | None = None,
    extra: Sequence[str] | None = None,
) -> list[str]:
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
    state: str | Path = "STATE",
    outfile: str | Path = "fes.dat",
    grid_min: float | Sequence[float] | None = None,
    grid_max: float | Sequence[float] | None = None,
    grid_bin: int | Sequence[int] | None = None,
    kt: float | None = None,
    extra: Sequence[str] | None = None,
    verbose: bool = True,
) -> str:
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


def _opes_reweighting_command(
    sigma: float | str | Sequence[float],
    kt: float,
    colvar: str | Path = "COLVAR",
    outfile: str | Path = "fes.dat",
    cv: str | int | Sequence[str | int] | None = None,
    bias: str | None = None,
    grid_min: float | str | Sequence[float] | None = None,
    grid_max: float | str | Sequence[float] | None = None,
    grid_bin: int | str | Sequence[int] | None = None,
    blocks: int | None = None,
    stride: int | None = None,
    skiprows: int | None = None,
    extra: Sequence[str] | None = None,
) -> list[str]:
    """Build the command line that reweights ``COLVAR`` samples into a FES.

    The bundled ``FES_from_Reweighting.py`` weights every sample by its own
    recorded bias, so *sigma* -- the kernel width of its density estimate --
    and *kt* are required by the script and therefore lead here.

    Parameters
    ----------
    sigma : float or sequence of float
        Kernel bandwidth of the weighted density estimate, one value per
        collective variable. A fraction of the OPES ``SIGMA`` is the usual
        starting point.
    kt : float
        Thermal energy in the energy units of the ``COLVAR`` file -- kJ/mol
        for a run driven from OpenMM, which is what
        :func:`~reactiontools.tools_units.thermal_energy` returns by default.
    colvar : str or path-like, optional
        The ``COLVAR`` file, or the merged one
        :func:`combine_colvar_files` wrote.
    outfile : str or path-like, optional
        Free-energy surface file to write, as read by
        :func:`~reactiontools.as_fes`.
    cv : str, int, or sequence, optional
        Collective variable to bin, by column name or 1-based column
        number; two entries make a two-dimensional surface. Default is the
        script's, column 2 -- the first variable after the time.
    bias : str or None, optional
        Bias column to weight by. The script's default matches any column
        containing ``.bias``, which finds what the
        :mod:`reactiontools.tools_cv` builders print; ``'NO'`` disables
        the weighting entirely.
    grid_min, grid_max : float or sequence of float, optional
        Bounds of the output grid, one per collective variable. Both must
        be given together, or neither.
    grid_bin : int or sequence of int, optional
        Number of bins per collective variable.
    blocks : int or None, optional
        Split the samples into this many blocks and add an uncertainty
        column from their scatter. On a file merged with
        ``sort_by_time=False``, one block per walker makes that the
        cross-walker error.
    stride : int or None, optional
        Also write the surface every *stride* samples, numbered like
        :func:`run_sum_hills`'s convergence series. The script treats this
        and *blocks* as alternatives.
    skiprows : int or None, optional
        Initial rows to discard, the transient before the bias settled.
    extra : sequence of str, optional
        Further arguments appended to the command line -- ``--reverse``,
        ``--nomintozero``, ``--fmt`` and the rest.

    Returns
    -------
    list of str
        The command, as an argument list.

    Raises
    ------
    ValueError
        If only one of *grid_min* and *grid_max* is given, or both *blocks*
        and *stride* are.
    """
    if (grid_min is None) != (grid_max is None):
        raise ValueError(
            "Give both grid_min and grid_max or neither; "
            "FES_from_Reweighting.py needs the two bounds together to size "
            "its grid."
        )
    if blocks is not None and stride is not None:
        raise ValueError(
            "FES_from_Reweighting.py treats --blocks and --stride as "
            "alternatives; pass one or the other."
        )

    # sys.executable, not "python3": the scripts need this environment's
    # pandas, and whatever "python3" resolves to on PATH may not have it.
    cmd = [
        sys.executable,
        str(script_path("FES_from_Reweighting.py")),
        "--colvar",
        str(colvar),
        "--outfile",
        str(outfile),
        "--sigma",
        _grid_bound(sigma),
        "--kt",
        f"{float(kt):.6g}",
    ]
    if cv is not None:
        cmd += ["--cv", _grid_bound(cv)]
    if bias is not None:
        cmd += ["--bias", str(bias)]
    if grid_min is not None:
        cmd += ["--min", _grid_bound(grid_min), "--max", _grid_bound(grid_max)]
    if grid_bin is not None:
        cmd += ["--bin", _grid_bound(grid_bin)]
    if blocks is not None:
        cmd += ["--blocks", str(blocks)]
    if stride is not None:
        cmd += ["--stride", str(stride)]
    if skiprows is not None:
        cmd += ["--skiprows", str(skiprows)]
    if extra:
        cmd += [str(item) for item in extra]
    return cmd


def run_opes_reweighting(
    sigma: float | str | Sequence[float],
    kt: float,
    colvar: str | Path = "COLVAR",
    outfile: str | Path = "fes.dat",
    cv: str | int | Sequence[str | int] | None = None,
    bias: str | None = None,
    grid_min: float | Sequence[float] | None = None,
    grid_max: float | Sequence[float] | None = None,
    grid_bin: int | Sequence[int] | None = None,
    blocks: int | None = None,
    stride: int | None = None,
    skiprows: int | None = None,
    extra: Sequence[str] | None = None,
    verbose: bool = True,
) -> str:
    """Rebuild a free-energy surface by reweighting ``COLVAR`` samples.

    The reweighting counterpart of :func:`run_opes_fes`: that reads the
    bias's own running estimate out of the ``STATE`` file, this rebuilds
    the surface from the samples and the bias they were collected under.
    For a single run the two should agree, and their disagreement is a
    convergence check. For several independent walkers this is the one
    that combines them -- each sample carries its own walker's bias, so
    the merged file :func:`combine_colvar_files` writes is reweighted as
    one: sorted by time for a single series (*stride*, *skiprows*),
    concatenated per walker for cross-walker error bars (*blocks*).

    Paths are resolved by the script, so this acts on the current working
    directory unless absolute paths are given.

    Parameters
    ----------
    sigma : float or sequence of float
        Kernel bandwidth of the weighted density estimate, one value per
        collective variable. A fraction of the OPES ``SIGMA`` is the usual
        starting point.
    kt : float
        Thermal energy in the energy units of the ``COLVAR`` file. See
        :func:`~reactiontools.tools_units.thermal_energy`.
    colvar : str or path-like, optional
        The ``COLVAR`` file, or the merged one
        :func:`combine_colvar_files` wrote.
    outfile : str or path-like, optional
        Free-energy surface file to write, as read by
        :func:`~reactiontools.as_fes`.
    cv : str, int, or sequence, optional
        Collective variable to bin, by column name or 1-based column
        number; two entries make a two-dimensional surface.
    bias : str or None, optional
        Bias column to weight by; the script's default matches any column
        containing ``.bias``.
    grid_min, grid_max : float or sequence of float, optional
        Bounds of the output grid, one per collective variable.
    grid_bin : int or sequence of int, optional
        Number of bins per collective variable.
    blocks : int or None, optional
        Number of blocks for the uncertainty column -- one per walker on
        an unsorted merge.
    stride : int or None, optional
        Also write the surface every *stride* samples, as a convergence
        series.
    skiprows : int or None, optional
        Initial rows to discard, the transient before the bias settled.
    extra : sequence of str, optional
        Further arguments appended to the command line.
    verbose : bool, optional
        Print the command being run.

    Returns
    -------
    str
        The command line that was run.

    Raises
    ------
    ValueError
        If only one of *grid_min* and *grid_max* is given, or both
        *blocks* and *stride* are.
    subprocess.CalledProcessError
        If the script exits non-zero.
    """
    cmd = _opes_reweighting_command(
        sigma=sigma,
        kt=kt,
        colvar=colvar,
        outfile=outfile,
        cv=cv,
        bias=bias,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_bin=grid_bin,
        blocks=blocks,
        stride=stride,
        skiprows=skiprows,
        extra=extra,
    )
    cmd_str = " ".join(cmd)

    if verbose:
        print(f"Running: {cmd_str}", flush=True)

    subprocess.run(cmd, check=True)
    return cmd_str
