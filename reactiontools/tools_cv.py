"""Collective variables for proton transfer, and the PLUMED input that biases them.

A proton transfer will not happen on its own inside an MD run: the barrier is
several times kBT and the event is rare enough that unbiased sampling never
sees it. Every builder here writes a PLUMED script that biases the system
along a collective variable so that it does, and returns alongside it the
shell command that turns the resulting bias back into a free-energy surface --
``plumed sum_hills`` for well-tempered metadynamics, or the bundled
:mod:`reactiontools.opes` scripts when ``OPES_METAD`` is used instead. Hand the
script to whatever drives PLUMED (``openmmnqe.run_openmm_prod`` for an OpenMM
run, :func:`~reactiontools.tools_plumed.plumed_calculator` for an ASE one) and
the surface it produces to :func:`~reactiontools.plot_plumed_fes`.

The builders fall into two families:

* **Coordination difference** -- :func:`plumed_input_1pt`,
  :func:`plumed_input_2pt_1d` and :func:`plumed_input_2pt_2d` bias one or two
  proton transfers directly, the CV running from +1 at the donor to -1 at the
  acceptor.
* **Path** -- :func:`plumed_input_neb_path` biases progress along a reference
  path instead, which is the option that copes when the reaction is not well
  described by any one geometric coordinate. Build the reference with
  :mod:`reactiontools.tools_path`.

:func:`plumed_input_steered` stands apart: it drags a CV rather than biasing
it, and exists to generate the reference path the last family needs.

Building your own
-----------------
A CV that encodes one particular system belongs with that study rather than
here -- the wobble base-pair builders that used to sit alongside these moved out
to the study that uses them for exactly that reason. The plumbing under these
builders is worth sharing though, so it is public:
:func:`plumed_bias_and_fes` returns the bias line together with the command
that reads the surface back, which is the pairing that goes wrong when the two
are written separately; :func:`plumed_one_based`, :func:`plumed_units_header`,
:func:`plumed_temperature_pair` and :func:`plumed_angle_radians` handle the
conversions every script needs. Write the CV lines yourself, hand the label you
biased to :func:`plumed_bias_and_fes`, and the result is a script indistinguishable
from the ones here.

Units
-----
These scripts are written for an external PLUMED, so they carry no ``UNITS``
line by default and are therefore in PLUMED's own nanometres and kJ/mol --
which is what it uses when driven from OpenMM. Pass ``units="ase"`` for
angstrom and eV instead, which prepends
:data:`~reactiontools.tools_plumed.PLUMED_ASE_UNITS` and is what a run driven
through :func:`~reactiontools.tools_plumed.plumed_calculator` needs.

Note that this is the opposite default to
:func:`~reactiontools.tools_plumed.plumed_metad_input`, which writes the
``UNITS`` line unless told otherwise because it only ever feeds ASE. The two
are not interchangeable: a script moved between them without changing ``units``
is wrong by a factor of ten in every length.

Lengths that come from the geometry are converted for you. Lengths you supply
yourself -- ``wall`` in :func:`plumed_input_neb_path` -- are in whichever unit
you asked for, and nothing checks that.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from .tools_plumed import PLUMED_ASE_UNITS, _opes_fes_command
from .tools_units import as_kelvin, thermal_energy

__all__ = [
    "as_positions",
    "plumed_angle_radians",
    "plumed_bias_and_fes",
    "plumed_input_1pt",
    "plumed_input_2pt_1d",
    "plumed_input_2pt_2d",
    "plumed_input_neb_path",
    "plumed_input_steered",
    "plumed_input_steered_pt",
    "plumed_one_based",
    "plumed_temperature_pair",
    "plumed_units_header",
    "switching_value",
]

#: Script length unit per angstrom, keyed by the ``units`` argument. PLUMED
#: works in nanometres unless a UNITS line says otherwise; ASE hands it
#: angstrom and PLUMED_ASE_UNITS declares that.
_LENGTH_SCALE = {"plumed": 0.1, "ase": 1.0}

#: Energy unit each choice implies, for the ``--kt`` of the reconstruction.
_ENERGY_UNIT = {"plumed": "kJ/mol", "ase": "eV"}

#: Default Gaussian height, in the energy unit each choice implies. These are
#: the same deposit: 15 kJ/mol is 0.155 eV. A kJ/mol default carried unchanged
#: into an eV script would be a Gaussian 96.5 times too tall.
_DEFAULT_HEIGHT = {"plumed": 15.0, "ase": 0.155}


def as_positions(source: Any) -> np.ndarray:
    """Coordinates as a plain ``(n_atoms, 3)`` array of angstrom.

    Accepts whatever holds a geometry: an array of coordinates, an
    :class:`ase.Atoms`, an ``openmm.app.Modeller`` or anything else whose
    ``positions`` carry OpenMM units, or a path to a structure file ASE can
    read. OpenMM works in nanometres and ASE in angstrom; the conversion
    happens here so that no builder has to know which world it was called
    from, and so that nothing in this package imports OpenMM.

    Parameters
    ----------
    source : ase.Atoms or openmm.app.Modeller or array_like or str or path-like
        The geometry, or somewhere to read one from.

    Returns
    -------
    numpy.ndarray
        Coordinates in angstrom, of shape ``(n_atoms, 3)``.

    Raises
    ------
    ValueError
        If the coordinates are not ``(n_atoms, 3)``.
    """
    if isinstance(source, (str, Path)):
        source = read(source)

    positions = getattr(source, "positions", source)

    # Only reachable when OpenMM built this object, so a caller who never
    # touches OpenMM never reaches the import and never needs it installed.
    if type(positions).__module__.split(".")[0] == "openmm":
        from openmm import unit as openmm_unit

        positions = positions.value_in_unit(openmm_unit.angstrom)

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected coordinates of shape (n_atoms, 3), got {positions.shape}."
        )
    return positions


def _check_units(units: str) -> float:
    """Validate the ``units`` argument and return its length scale.

    Parameters
    ----------
    units : str
        ``'plumed'`` or ``'ase'``.

    Returns
    -------
    float
        Script length units per angstrom.

    Raises
    ------
    ValueError
        If *units* is neither.
    """
    if units not in _LENGTH_SCALE:
        raise ValueError(
            f"Unknown units {units!r}. Use 'plumed' for the nanometres and "
            f"kJ/mol an external PLUMED works in, or 'ase' for the angstrom "
            f"and eV a run driven through plumed_calculator needs."
        )
    return _LENGTH_SCALE[units]


def _geometry(source: Any, units: str) -> np.ndarray:
    """Coordinates in the script's own length unit.

    Parameters
    ----------
    source : ase.Atoms or openmm.app.Modeller or array_like or str
        The geometry.
    units : str
        ``'plumed'`` or ``'ase'``.

    Returns
    -------
    numpy.ndarray
        Coordinates of shape ``(n_atoms, 3)``, scaled out of angstrom.
    """
    return as_positions(source) * _check_units(units)


def plumed_units_header(units: str) -> str:
    """The UNITS line the script needs, if any.

    Parameters
    ----------
    units : str
        ``'plumed'`` or ``'ase'``.

    Returns
    -------
    str
        The line plus a newline, or an empty string for PLUMED's own units.
    """
    _check_units(units)
    return f"{PLUMED_ASE_UNITS}\n" if units == "ase" else ""


def _default_height(height: float | None, units: str) -> float:
    """The Gaussian height to deposit, defaulted in the script's energy unit.

    Parameters
    ----------
    height : float or None
        The height asked for, or None to take the default for *units*.
    units : str
        ``'plumed'`` or ``'ase'``.

    Returns
    -------
    float
        *height* if it was given, else 15 kJ/mol expressed in the right unit.
    """
    _check_units(units)
    return _DEFAULT_HEIGHT[units] if height is None else height


def plumed_one_based(indices: Iterable[int]) -> list[int]:
    """Convert 0-based atom indices to PLUMED's 1-based convention.

    Order is preserved, which is what a CV needs: the builders here read their
    indices positionally, as donor, hydrogen, acceptor. This is not
    interchangeable with
    :func:`~reactiontools.tools_plumed.plumed_selection`, which sorts and
    dedupes its indices to collapse them into ranges -- right for the atom
    *group* of a ``COORDINATION``, wrong for anything positional.

    Parameters
    ----------
    indices : iterable of int
        0-based indices, as OpenMM and ASE number them.

    Returns
    -------
    list of int
        1-based indices, as PLUMED numbers them, in the order given.
    """
    return [int(index) + 1 for index in indices]


def _distance(positions: np.ndarray, i: int, j: int) -> float:
    """Distance between two atoms, in whatever unit *positions* is in.

    Parameters
    ----------
    positions : numpy.ndarray
        Coordinates of shape ``(n_atoms, 3)``.
    i, j : int
        0-based atom indices.

    Returns
    -------
    float
        The distance.
    """
    return float(np.linalg.norm(positions[i] - positions[j]))


def _pt_distances(
    positions: np.ndarray,
    idx: Sequence[int],
) -> tuple[float, float, float]:
    """The three distances that size a proton-transfer CV.

    Parameters
    ----------
    positions : numpy.ndarray
        Coordinates of shape ``(n_atoms, 3)``.
    idx : sequence of int
        Three 0-based indices, ordered donor, hydrogen, acceptor.

    Returns
    -------
    r_dh, r_ah, r_da : float
        Donor-hydrogen, acceptor-hydrogen and donor-acceptor distances.
    """
    return (
        _distance(positions, idx[0], idx[1]),
        _distance(positions, idx[2], idx[1]),
        _distance(positions, idx[0], idx[2]),
    )


def _size_r0(r_dh: float, r_ah: float, r_0: float) -> float:
    """Size the switching function from the shorter of the two bond distances.

    Parameters
    ----------
    r_dh, r_ah : float
        Donor-hydrogen and acceptor-hydrogen distances.
    r_0 : float
        Multiplier on the shorter of them.

    Returns
    -------
    float
        ``R_0``, rounded to two decimals as the scripts write it.
    """
    return float(np.round(min(r_dh, r_ah) * r_0, decimals=2))


def _wall_value(distance: float, wall: float) -> float:
    """Place a wall at a multiple of the distance a pair currently sits at.

    Parameters
    ----------
    distance : float
        The present separation.
    wall : float
        Multiplier on it.

    Returns
    -------
    float
        Where the wall goes, rounded to two decimals.
    """
    return float(np.round(distance * wall, decimals=2))


def plumed_angle_radians(angle_lim: float) -> float:
    """Convert a wall angle from degrees to the radians PLUMED wants.

    Parameters
    ----------
    angle_lim : float
        Angle in degrees.

    Returns
    -------
    float
        The angle in radians, rounded to two decimals as the scripts write it.
    """
    return float(np.round(np.deg2rad(angle_lim), decimals=2))


def switching_value(r: float, r_0: float, nn: int = 6, mm: int | None = None) -> float:
    """Evaluate PLUMED's default rational switching function.

    This is the same function ``COORDINATION`` applies to every pair distance,
    ``s(r) = (1 - (r/r_0)^nn) / (1 - (r/r_0)^mm)``, so it can be used to work
    out what a coordination-based CV is worth for a given geometry without
    running PLUMED -- which is how :func:`plumed_input_steered_pt` decides
    where a pull should start and stop.

    Parameters
    ----------
    r : float
        Distance between the two atoms, in the same units as *r_0*.
    r_0 : float
        The ``R_0`` parameter of the switching function.
    nn : int, optional
        Numerator exponent. Default is 6, as in PLUMED.
    mm : int or None, optional
        Denominator exponent. If None, ``2 * nn`` is used, again as in PLUMED.

    Returns
    -------
    float
        The value of the switching function, between 0 and 1.
    """
    mm = 2 * nn if mm is None else mm
    x = (r / r_0) ** nn
    y = (r / r_0) ** mm
    if np.isclose(y, 1.0):
        # r == r_0 makes both halves vanish; the limit there is nn / mm
        return nn / mm
    return (1.0 - x) / (1.0 - y)


def plumed_temperature_pair(temperature: Any, units: str) -> tuple[float, float]:
    """Temperature in kelvin and kBT in the script's energy unit.

    Parameters
    ----------
    temperature : float or openmm.unit.Quantity
        Simulation temperature.
    units : str
        ``'plumed'`` or ``'ase'``.

    Returns
    -------
    kelvin : float
        For the ``TEMP`` of the bias action.
    kt : float
        For the ``--kt`` of the reconstruction command.
    """
    _check_units(units)
    kelvin_value = as_kelvin(temperature)
    return kelvin_value, thermal_energy(kelvin_value, _ENERGY_UNIT[units])


def plumed_bias_and_fes(
    f_opes: bool,
    arg: str,
    pace: float | str,
    height: float | str,
    sigma: float | str,
    bias: float | str,
    temperature: float,
    kt: float,
    grid_bin: int | str,
    grid_min: float | str | None = None,
    grid_max: float | str | None = None,
    label: str = "metad:      ",
) -> tuple[str, str]:
    """Build the metadynamics bias line and the matching FES-reconstruction command.

    Every builder in this module goes through here, so that the bias it emits
    and the command that reads the surface back cannot disagree about the grid
    or the temperature -- which is exactly what happened while three of them
    hand-rolled their own.

    Parameters
    ----------
    f_opes : bool
        If True, bias with ``OPES_METAD`` (reconstructed with the bundled OPES
        ``FES_from_State.py``); otherwise use well-tempered ``METAD`` with
        ``plumed sum_hills``.
    arg : str
        The PLUMED ``ARG=`` value the bias acts on.
    pace, height, sigma, bias : int or float or str
        Values interpolated into the bias line. For a multi-dimensional bias
        pass pre-joined strings, e.g. ``f'{sigma},{sigma}'``.
    temperature : float
        Temperature in kelvin.
    kt : float
        kBT in the script's energy unit, for the ``--kt`` of the FES command.
    grid_bin : int or str
        Grid bin count, pre-joined for a multi-dimensional bias.
    grid_min, grid_max : float or str or None, optional
        Grid bounds. When None the grid is left out of both commands, and
        PLUMED sizes it itself. Default is None.
    label : str, optional
        Prefix of the bias line, being the label plus enough spaces to keep
        the script's columns aligned. Default is ``'metad:      '``.

    Returns
    -------
    metad_line : str
        The PLUMED bias line.
    fes_command : str
        Shell command reconstructing the free-energy surface from it.
    """
    if f_opes:
        metad_line = (
            f"{label}OPES_METAD ARG={arg} PACE={pace} BARRIER={height} "
            f"SIGMA={sigma} TEMP={temperature} "
            f"STATE_WFILE=STATE STATE_WSTRIDE={pace}"
        )
        fes_command = " ".join(
            _opes_fes_command(
                state="STATE",
                outfile="fes.dat",
                grid_min=grid_min,
                grid_max=grid_max,
                grid_bin=grid_bin,
                kt=kt,
            )
        )
    else:
        metad_grid = (
            f" GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}"
            if grid_min is not None
            else ""
        )
        metad_line = (
            f"{label}METAD ARG={arg} PACE={pace} HEIGHT={height} "
            f"SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature} "
            f"FILE=HILLS{metad_grid}"
        )
        fes_grid = f" --min {grid_min} --max {grid_max}" if grid_min is not None else ""
        fes_command = (
            f"plumed sum_hills --hills HILLS --outfile fes.dat"
            f"{fes_grid} --bin {grid_bin} --kt {kt:.6g}"
        )
    return metad_line, fes_command


def _pt_cv_block(
    idx: Sequence[int],
    r_0: float,
    wall: float,
    angle_lim: float,
    kappa: float,
    suffix: str = "",
    pad: int = 12,
) -> tuple[str, str]:
    """Build the PLUMED lines defining one proton-transfer CV and its walls.

    The coordination difference that every builder in the first two families
    is made of: how strongly the hydrogen is bound to the donor, minus how
    strongly it is bound to the acceptor, so the CV runs from +1 to -1 as the
    proton crosses. The distance wall stops the two heavy atoms drifting apart
    while it does, and the angle wall stops the hydrogen bond bending out of
    line, either of which would let the CV change without a transfer happening.

    Parameters
    ----------
    idx : sequence of int
        Three 1-based atom indices, ordered donor, hydrogen, acceptor.
    r_0 : float
        ``R_0`` of the coordination switching function.
    wall : float
        Where the donor-acceptor upper wall goes.
    angle_lim : float
        Where the donor-hydrogen-acceptor lower wall goes, in radians.
    kappa : float
        Spring constant of both walls.
    suffix : str, optional
        Appended to every label, to tell two transfers apart. Empty, the
        default, gives the single-transfer names.
    pad : int, optional
        Column the actions are aligned to. Default is 12.

    Returns
    -------
    cv_lines, wall_lines : str
        The CV definition and the walls, as two blocks of text.
    """
    # The single- and double-transfer builders grew different label
    # conventions; both are in use in saved scripts, so they are preserved
    # rather than unified.
    if suffix:
        c_d, c_a, cv = f"c_d{suffix}", f"c_a{suffix}", f"cv_diff{suffix}"
        dist, u_wall = f"dist_da_{suffix}", f"u_wall_{suffix}"
        ang, a_wall = f"ang_{suffix}", f"w_{suffix}"
    else:
        c_d, c_a, cv = "c_d", "c_a", "pt_cv"
        dist, u_wall = "dist_da", "dist_wall"
        ang, a_wall = "ang_1", "ang_wall"

    def line(label: str, body: str) -> str:
        """Pad a label out to *pad* columns and put the action after it."""
        return f"{label + ':':<{pad}}{body}"

    cv_lines = "\n".join(
        [
            line(c_d, f"COORDINATION GROUPA={idx[0]} GROUPB={idx[1]} R_0={r_0}"),
            line(c_a, f"COORDINATION GROUPA={idx[2]} GROUPB={idx[1]} R_0={r_0}"),
            line(cv, f"COMBINE ARG={c_d},{c_a} COEFFICIENTS=1,-1 PERIODIC=NO"),
        ]
    )
    wall_lines = "\n".join(
        [
            line(dist, f"DISTANCE ATOMS={idx[2]},{idx[0]}"),
            line(u_wall, f"UPPER_WALLS ARG={dist} AT={wall} KAPPA={kappa}"),
            line(ang, f"ANGLE ATOMS={idx[2]},{idx[1]},{idx[0]}"),
            line(a_wall, f"LOWER_WALLS ARG={ang} AT={angle_lim} KAPPA={kappa}"),
        ]
    )
    return cv_lines, wall_lines


def plumed_input_steered(
    cv_block: str,
    cv_start: float,
    cv_stop: float,
    steps: int,
    cv_name: str = "cv",
    kappa: float = 2000.0,
    stride: int = 100,
    steps_equil: int = 0,
    steps_relax: int = 0,
    colvar_file: str = "COLVAR_SMD",
    extra_lines: str | None = None,
    units: str = "plumed",
) -> tuple[str, int]:
    """Build a PLUMED input that drags a collective variable (steered MD).

    The centre of a harmonic restraint is moved linearly from *cv_start* to
    *cv_stop* over *steps* MD steps, optionally after holding at the starting
    value for *steps_equil* steps and before holding at the final value for
    *steps_relax* steps. The trajectory this produces is a first guess at the
    reaction path, which
    :func:`~reactiontools.tools_path.path_from_steered_md` turns into a
    reference for ``PATHMSD``.

    Parameters
    ----------
    cv_block : str
        PLUMED lines defining the CV to steer, ending with an action labelled
        *cv_name*.
    cv_start : float
        Value of the CV the restraint starts at, normally the value of the
        reactant.
    cv_stop : float
        Value of the CV the restraint finishes at, normally the value of the
        product.
    steps : int
        Number of MD steps spent pulling from *cv_start* to *cv_stop*. Pulling
        slowly costs more time but leaves a path that is closer to the free
        energy valley.
    cv_name : str, optional
        Label of the CV defined in *cv_block*. Must be a plain label rather
        than a component such as ``path.sss``, because it is used to name the
        restraint's output components. Default is ``'cv'``.
    kappa : float, optional
        Spring constant of the moving restraint in kJ/mol per CV unit squared.
        Default is 2000.0. Too soft and the system lags behind the restraint,
        too stiff and the pulling heats it.
    stride : int, optional
        How often the CV is written to *colvar_file*, in steps. Default is 100.
        Match this to the reporter interval of the MD run so that every
        trajectory frame has a CV value.
    steps_equil : int, optional
        Steps held at *cv_start* before pulling starts. Default is 0.
    steps_relax : int, optional
        Steps held at *cv_stop* after pulling finishes. Default is 0.
    colvar_file : str, optional
        File the CV, the restraint centre and the work are written to.
        Default is ``'COLVAR_SMD'``.
    extra_lines : str or None, optional
        Further PLUMED lines (walls, restraints, extra prints) inserted after
        the CV definition. Default is None.
    units : str, optional
        ``'plumed'`` (the default) for nanometres and kJ/mol, or ``'ase'`` for
        angstrom and eV. See the module docstring.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    n_steps : int
        Total number of MD steps the pulling schedule covers, i.e.
        ``steps_equil + steps + steps_relax``. Pass this to the MD run so the
        simulation does not stop mid-pull.
    """
    # Each milestone is a (step, restraint centre) pair; PLUMED interpolates
    # the centre linearly in between them.
    milestones = [(0, cv_start)]
    step = 0
    if steps_equil > 0:
        step += steps_equil
        milestones.append((step, cv_start))
    step += steps
    milestones.append((step, cv_stop))
    if steps_relax > 0:
        step += steps_relax
        milestones.append((step, cv_stop))

    schedule = " ".join(
        f"STEP{i}={at_step} AT{i}={at:.4f} KAPPA{i}={kappa}"
        for i, (at_step, at) in enumerate(milestones)
    )

    plumed_input = f"""
{plumed_units_header(units)}# Collective variable
{cv_block.strip()}
{extra_lines.strip() if extra_lines else ""}
# Steered MD: pull the CV from {cv_start:.4f} to {cv_stop:.4f}
smd:        MOVINGRESTRAINT ARG={cv_name} {schedule}
PRINT       ARG={cv_name},smd.{cv_name}_cntr,smd.work STRIDE={stride} FILE={colvar_file}
        """
    return plumed_input, step


def plumed_input_steered_pt(
    geometry: Any,
    idx: Sequence[int],
    steps: int,
    r_0: float = 1.1,
    wall: float = 1.5,
    angle_lim: float = 130.0,
    kappa: float = 2000.0,
    stride: int = 100,
    cv_start: float | None = None,
    cv_stop: float | None = None,
    steps_equil: int = 0,
    steps_relax: int = 0,
    colvar_file: str = "COLVAR_SMD",
    wall_kappa: float = 500.0,
    units: str = "plumed",
) -> tuple[str, int]:
    """Build a steered MD input that pulls a proton across a hydrogen bond.

    The collective variable is the one :func:`plumed_input_1pt` biases, the
    difference between the donor-hydrogen and acceptor-hydrogen coordination
    numbers, and the same walls keep the donor and acceptor from drifting
    apart while the proton is dragged over. Unless they are given, the start
    and end values of the CV are read off *geometry*: the current value for
    the start, and its mirror image for the end, which is where the proton
    sits once transferred.

    Parameters
    ----------
    geometry : ase.Atoms or openmm.app.Modeller or array_like or str
        The reactant geometry, used to size the switching function and to work
        out where the pull should start. See :func:`as_positions`.
    idx : list of int
        Three 0-based atom indices, ordered donor, hydrogen, acceptor.
    steps : int
        Number of MD steps spent pulling the proton across.
    r_0 : float, optional
        Multiplier on the shorter of the two donor/acceptor-hydrogen distances
        that sets ``R_0`` of the switching function. Default is 1.1.
    wall : float, optional
        Multiplier on the donor-acceptor distance that sets the upper wall
        keeping the hydrogen bond intact. Default is 1.5.
    angle_lim : float, optional
        Lower wall on the donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    kappa : float, optional
        Spring constant of the moving restraint. Default is 2000.0.
    stride : int, optional
        How often the CV is written, in steps. Default is 100.
    cv_start, cv_stop : float or None, optional
        Explicit start and end values for the CV. If None, the start is
        computed from *geometry* and the end is its negative. Default is None.
    steps_equil, steps_relax : int, optional
        Steps held at the start and end values. Default is 0.
    colvar_file : str, optional
        File the CV is written to. Default is ``'COLVAR_SMD'``.
    wall_kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    units : str, optional
        ``'plumed'`` (the default) for nanometres and kJ/mol, or ``'ase'`` for
        angstrom and eV. See the module docstring.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    n_steps : int
        Total number of MD steps the pulling schedule covers.
    """
    positions = _geometry(geometry, units)
    r_dh, r_ah, r_da = _pt_distances(positions, idx)
    r_0 = _size_r0(r_dh, r_ah, r_0)

    if cv_start is None:
        # What the CV is worth right now: bonded to the donor gives +1,
        # bonded to the acceptor gives -1
        cv_start = np.round(
            switching_value(r_dh, r_0) - switching_value(r_ah, r_0), decimals=2
        )
    if cv_stop is None:
        cv_stop = -cv_start

    cv_lines, wall_lines = _pt_cv_block(
        plumed_one_based(idx),
        r_0,
        _wall_value(r_da, wall),
        plumed_angle_radians(angle_lim),
        wall_kappa,
    )

    return plumed_input_steered(
        cv_lines,
        cv_start,
        cv_stop,
        steps,
        cv_name="pt_cv",
        kappa=kappa,
        stride=stride,
        steps_equil=steps_equil,
        steps_relax=steps_relax,
        colvar_file=colvar_file,
        extra_lines=f"\n# Limits\n{wall_lines}\n",
        units=units,
    )


def plumed_input_1pt(
    geometry: Any,
    idx: Sequence[int],
    temperature: Any,
    r_0: float = 1.1,
    wall: float = 1.5,
    angle_lim: float = 130.0,
    pace: int = 500,
    height: float | None = None,
    sigma: float = 0.05,
    bias: float = 20.0,
    grid_min: float | None = -1.1,
    grid_max: float | None = 1.1,
    grid_bin: int = 200,
    kappa: float = 500.0,
    f_opes: bool = False,
    units: str = "plumed",
) -> tuple[str, str]:
    """Build a PLUMED input that biases a single proton transfer with metadynamics.

    The collective variable is the same one :func:`plumed_input_steered_pt`
    drags: the difference between the donor-hydrogen and acceptor-hydrogen
    coordination numbers, running from +1 (bonded to the donor) to -1
    (bonded to the acceptor). Distance and angle walls keep the hydrogen
    bond intact while the proton moves. Use this to bias the transfer with
    metadynamics; use :func:`plumed_input_steered_pt` to drag it instead.

    Parameters
    ----------
    geometry : ase.Atoms or openmm.app.Modeller or array_like or str
        The reactant geometry, used to size the switching function and the
        walls. See :func:`as_positions`.
    idx : list of int
        Three 0-based atom indices, ordered donor, hydrogen, acceptor.
    temperature : float or openmm.unit.Quantity
        Simulation temperature in kelvin, used for the BIASFACTOR/OPES scaling
        and to report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Multiplier on the shorter of the two donor/acceptor-hydrogen
        distances that sets ``R_0`` of the coordination switching function.
        Default is 1.1.
    wall : float, optional
        Multiplier on the donor-acceptor distance that sets the upper wall
        keeping the hydrogen bond intact. Default is 1.5.
    angle_lim : float, optional
        Lower wall on the donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in the script's energy unit. None, the default,
        deposits 15 kJ/mol -- written as 15.0 for ``units='plumed'`` and as
        its equivalent 0.155 for ``units='ase'``.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid. Default is -1.1 and 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.
    kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.
    units : str, optional
        ``'plumed'`` (the default) for nanometres and kJ/mol, or ``'ase'`` for
        angstrom and eV. See the module docstring.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    fes_command : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*: ``plumed sum_hills``, or the bundled
        OPES ``FES_from_State.py`` when *f_opes* is True.
    """
    positions = _geometry(geometry, units)
    r_dh, r_ah, r_da = _pt_distances(positions, idx)

    cv_lines, wall_lines = _pt_cv_block(
        plumed_one_based(idx),
        _size_r0(r_dh, r_ah, r_0),
        _wall_value(r_da, wall),
        plumed_angle_radians(angle_lim),
        kappa,
    )

    height = _default_height(height, units)
    kelvin, kt = plumed_temperature_pair(temperature, units)
    metad_line, fes_command = plumed_bias_and_fes(
        f_opes,
        "pt_cv",
        pace,
        height,
        sigma,
        bias,
        kelvin,
        kt,
        grid_bin,
        grid_min=grid_min,
        grid_max=grid_max,
    )

    plumed_input = f"""
{plumed_units_header(units)}# Proton transfer
{cv_lines}

# Limits
{wall_lines}

# Metadynamics
{metad_line}
PRINT       ARG=c_d,c_a,pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, fes_command


def _two_pt_body(
    geometry: Any,
    idx1: Sequence[int],
    idx2: Sequence[int],
    r_0: float,
    wall: float,
    angle_lim: float,
    kappa: float,
    units: str,
) -> str:
    """The CV and wall blocks shared by the two double-transfer builders.

    Parameters
    ----------
    geometry : ase.Atoms or openmm.app.Modeller or array_like or str
        The reactant geometry.
    idx1, idx2 : sequence of int
        Three 0-based indices each, ordered donor, hydrogen, acceptor.
    r_0, wall : float
        Multipliers sizing the switching functions and the walls.
    angle_lim : float
        Lower wall on each donor-hydrogen-acceptor angle, in degrees.
    kappa : float
        Spring constant of the walls.
    units : str
        ``'plumed'`` or ``'ase'``.

    Returns
    -------
    str
        The two labelled transfer blocks, walls included.
    """
    positions = _geometry(geometry, units)
    r1_dh, r1_ah, r1_da = _pt_distances(positions, idx1)
    r2_dh, r2_ah, r2_da = _pt_distances(positions, idx2)

    # One wall value for both pairs, taken from whichever is further apart, so
    # neither hydrogen bond is squeezed by the other's geometry.
    at = _wall_value(max(r1_da, r2_da), wall)
    angle = plumed_angle_radians(angle_lim)

    blocks = []
    for n, (idx, r_dh, r_ah) in enumerate(
        ((idx1, r1_dh, r1_ah), (idx2, r2_dh, r2_ah)), start=1
    ):
        cv_lines, wall_lines = _pt_cv_block(
            plumed_one_based(idx),
            _size_r0(r_dh, r_ah, r_0),
            at,
            angle,
            kappa,
            suffix=str(n),
        )
        blocks.append(f"# Proton transfer {n}\n{cv_lines}\n\n# Limits\n{wall_lines}")
    return "\n\n".join(blocks)


def plumed_input_2pt_1d(
    geometry: Any,
    idx1: Sequence[int],
    idx2: Sequence[int],
    temperature: Any,
    r_0: float = 1.1,
    wall: float = 1.5,
    angle_lim: float = 130.0,
    pace: int = 500,
    height: float | None = None,
    sigma: float = 0.05,
    bias: float = 20.0,
    grid_min: float | None = -1.1,
    grid_max: float | None = 1.1,
    grid_bin: int = 200,
    kappa: float = 500.0,
    f_opes: bool = False,
    units: str = "plumed",
) -> tuple[str, str]:
    """Build a PLUMED input that biases two proton transfers along one coordinate.

    Each transfer gets its own coordination-difference CV, as in
    :func:`plumed_input_1pt`, and the two are averaged into a single
    concerted coordinate. That is the right picture when the two protons are
    expected to move together; use :func:`plumed_input_2pt_2d` when whether
    they do is the question.

    Parameters
    ----------
    geometry : ase.Atoms or openmm.app.Modeller or array_like or str
        The reactant geometry, used to size the switching functions and the
        walls. See :func:`as_positions`.
    idx1, idx2 : list of int
        Three 0-based atom indices each, ordered donor, hydrogen, acceptor,
        for the first and second proton transfer.
    temperature : float or openmm.unit.Quantity
        Simulation temperature in kelvin.
    r_0 : float, optional
        Multiplier on the shorter donor/acceptor-hydrogen distance in each
        pair that sets ``R_0`` of its coordination switching function.
        Default is 1.1.
    wall : float, optional
        Multiplier on the larger of the two donor-acceptor distances that
        sets the upper wall keeping both hydrogen bonds intact. Default is
        1.5.
    angle_lim : float, optional
        Lower wall on each donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in the script's energy unit. None, the default,
        deposits 15 kJ/mol -- written as 15.0 for ``units='plumed'`` and as
        its equivalent 0.155 for ``units='ase'``.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid. Default is -1.1 and 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.
    kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.
    units : str, optional
        ``'plumed'`` (the default) for nanometres and kJ/mol, or ``'ase'`` for
        angstrom and eV. See the module docstring.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    fes_command : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*.
    """
    body = _two_pt_body(geometry, idx1, idx2, r_0, wall, angle_lim, kappa, units)

    height = _default_height(height, units)
    kelvin, kt = plumed_temperature_pair(temperature, units)
    metad_line, fes_command = plumed_bias_and_fes(
        f_opes,
        "pt_cv",
        pace,
        height,
        sigma,
        bias,
        kelvin,
        kt,
        grid_bin,
        grid_min=grid_min,
        grid_max=grid_max,
    )

    plumed_input = f"""
{plumed_units_header(units)}{body}

# Combine the two proton transfers into a single CV
pt_cv:      COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

# Metadynamics
{metad_line}
PRINT       ARG=pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, fes_command


def plumed_input_2pt_2d(
    geometry: Any,
    idx1: Sequence[int],
    idx2: Sequence[int],
    temperature: Any,
    r_0: float = 1.1,
    wall: float = 1.5,
    angle_lim: float = 130.0,
    pace: int = 500,
    height: float | None = None,
    sigma: float = 0.05,
    bias: float = 20.0,
    grid_min: float = -1.1,
    grid_max: float = 1.1,
    grid_bin: int = 200,
    kappa: float = 500.0,
    f_opes: bool = False,
    units: str = "plumed",
) -> tuple[str, str]:
    """Build a PLUMED input that biases two proton transfers on a 2-D surface.

    Each proton transfer gets its own donor-hydrogen/acceptor-hydrogen
    coordination-difference CV, as in :func:`plumed_input_1pt`, and the two
    are kept as separate axes (``ARG=cv_diff1,cv_diff2``) so the free-energy
    surface resolves how the transfers are correlated. Use
    :func:`plumed_input_2pt_1d` instead when only the combined coordinate is
    needed.

    Parameters
    ----------
    geometry : ase.Atoms or openmm.app.Modeller or array_like or str
        The reactant geometry, used to size the switching functions and the
        walls. See :func:`as_positions`.
    idx1, idx2 : list of int
        Three 0-based atom indices each, ordered donor, hydrogen, acceptor,
        for the first and second proton transfer.
    temperature : float or openmm.unit.Quantity
        Simulation temperature in kelvin.
    r_0 : float, optional
        Multiplier on the shorter donor/acceptor-hydrogen distance in each
        pair that sets ``R_0`` of its coordination switching function.
        Default is 1.1.
    wall : float, optional
        Multiplier on the larger of the two donor-acceptor distances that
        sets the upper wall keeping both hydrogen bonds intact. Default is
        1.5.
    angle_lim : float, optional
        Lower wall on each donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in the script's energy unit. None, the default,
        deposits 15 kJ/mol -- written as 15.0 for ``units='plumed'`` and as
        its equivalent 0.155 for ``units='ase'``.
    sigma : float, optional
        Gaussian width of the bias along each axis, in CV units. Default is
        0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid, applied to both axes. Default is -1.1
        and 1.1.
    grid_bin : int, optional
        Number of grid bins per axis. Default is 200.
    kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.
    units : str, optional
        ``'plumed'`` (the default) for nanometres and kJ/mol, or ``'ase'`` for
        angstrom and eV. See the module docstring.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    fes_command : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*.
    """
    body = _two_pt_body(geometry, idx1, idx2, r_0, wall, angle_lim, kappa, units)

    height = _default_height(height, units)
    kelvin, kt = plumed_temperature_pair(temperature, units)
    metad_line, fes_command = plumed_bias_and_fes(
        f_opes,
        "cv_diff1,cv_diff2",
        pace,
        height,
        f"{sigma},{sigma}",
        bias,
        kelvin,
        kt,
        f"{grid_bin},{grid_bin}",
        grid_min=f"{grid_min},{grid_min}",
        grid_max=f"{grid_max},{grid_max}",
    )

    plumed_input = f"""
{plumed_units_header(units)}{body}

# Metadynamics
{metad_line}
PRINT       ARG=cv_diff1,cv_diff2,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, fes_command


def plumed_input_neb_path(
    temperature: Any,
    wall: float = 0.1,
    pace: int = 500,
    height: float | None = None,
    sigma: float = 0.1,
    bias: float = 5.0,
    grid_min: float | None = 0.0,
    grid_max: float | None = 26.0,
    grid_bin: int = 500,
    kappa: float = 500.0,
    lambda_val: float = 250.0,
    neigh_size: int = 8,
    f_opes: bool = False,
    units: str = "plumed",
) -> tuple[str, str]:
    """Build a PLUMED input that biases progress along a reference path.

    When the reaction is not well described by any one geometric coordinate --
    a transfer that only happens after the surroundings rearrange, say -- there
    may be no CV to bias. ``PATHMSD`` sidesteps that by measuring progress
    along a reference path instead: ``path.sss`` says how far along it the
    system is and ``path.zzz`` how far off it has strayed. The bias goes on
    the first and a wall on the second, so the sampling follows the path
    rather than wandering away from it.

    The reference is read from ``neb_path.pdb`` and the alignment template
    from ``index_atoms.pdb``, both by those names in the working directory.
    Build them with :func:`~reactiontools.tools_path.path_from_steered_md` or
    :func:`~reactiontools.tools_io.convert_xyz_to_plumed_ref`.

    Parameters
    ----------
    temperature : float or openmm.unit.Quantity
        Simulation temperature in kelvin.
    wall : float, optional
        Upper wall on ``path.zzz``, the distance off the path, as an absolute
        squared length. Default is 0.1.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in the script's energy unit. None, the default,
        deposits 15 kJ/mol -- written as 15.0 for ``units='plumed'`` and as
        its equivalent 0.155 for ``units='ase'``.
    sigma : float, optional
        Gaussian width of the bias, in units of path node index. Default is
        0.1.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 5.0.
    grid_min, grid_max : float, optional
        Bounds of the ``path.sss`` bias/FES grid, in units of path node
        index. Default is 0.0 and 26.0.
    grid_bin : int, optional
        Number of grid bins. Default is 500.
    kappa : float, optional
        Spring constant of the ``path.zzz`` wall. Default is 500.0.
    lambda_val : float, optional
        ``LAMBDA`` parameter of ``PATHMSD``, controlling how sharply
        ``path.sss`` distinguishes neighbouring frames; see
        :func:`~reactiontools.tools_path.estimate_path_lambda`, which
        recommends one for a given path. Default is 250.0.
    neigh_size : int, optional
        ``NEIGH_SIZE`` of ``PATHMSD``, the number of reference frames
        considered when computing the path distance. Default is 8.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.
    units : str, optional
        ``'plumed'`` (the default) for nanometres and kJ/mol, or ``'ase'`` for
        angstrom and eV. See the module docstring, and note that ``LAMBDA``
        and *wall* both scale with the choice.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    fes_command : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*.
    """
    height = _default_height(height, units)
    kelvin, kt = plumed_temperature_pair(temperature, units)
    metad_line, fes_command = plumed_bias_and_fes(
        f_opes,
        "path.sss",
        pace,
        height,
        sigma,
        bias,
        kelvin,
        kt,
        grid_bin,
        grid_min=grid_min,
        grid_max=grid_max,
        label="metad: ",
    )

    plumed_input = f"""
{plumed_units_header(units)}FIT_TO_TEMPLATE REFERENCE=index_atoms.pdb TYPE=OPTIMAL
path: PATHMSD REFERENCE=neb_path.pdb LAMBDA={lambda_val} NEIGH_SIZE={neigh_size}
{metad_line}
path_limit: UPPER_WALLS ARG=path.zzz AT={wall} KAPPA={kappa}
PRINT ARG=path.sss,path.zzz,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, fes_command
